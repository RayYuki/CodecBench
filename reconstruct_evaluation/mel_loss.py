import numpy as np
import soundfile as sf
import math
import glob
import os
import logging
import torch
from torch import nn
import torchaudio.transforms as T
import librosa.filters
import numpy as np
from tqdm import tqdm
from utils.helpers import find_audio_files, resample_on_gpu

def sqrt_hann_window(window_length, device):
    hann = torch.hann_window(window_length, periodic=True, device=device)
    return torch.sqrt(hann)

def compute_stft_padding(signal_length, window_length, hop_length, match_stride=False):
    if match_stride:
        assert hop_length == window_length // 4, "For match_stride, hop must equal n_fft // 4"
        right_pad = math.ceil(signal_length / hop_length) * hop_length - signal_length
        pad = (window_length - hop_length) // 2
    else:
        right_pad = 0
        pad = 0
    return right_pad, pad

def evaluate_mel_loss(ref_folder, 
             est_folder, 
             target_sr = 16000,            
             n_mels = [150, 80],
             window_lengths = [2048, 512],
             loss_fn = nn.L1Loss(),
             clamp_eps = 1e-5,
             mag_weight = 1.0,
             log_weight = 1.0,
             pow = 2.0,
             match_stride = False,
             mel_fmin = [0.0, 0.0],
             mel_fmax = [None, None],
             window_type = "sqrt_hann",
             device="cuda"
            ):
            
    logging.info(f"Evaluating Mel Loss: ref_folder: {ref_folder}, est_folder: {est_folder}")
    """
    Calculate the Mel Spectrogram distance between pairs of reference and estimated audio files
    located in the given directories, optionally resampling all files to a specified sample rate.
    
    Parameters:
        ref_folder (str): The folder path containing the reference audio files (.wav).
        est_folder (str): The folder path containing the estimated/generated audio files (.wav).
        target_sr (int, optional): The target sample rate to which all audio files will be resampled. If None, no resampling is performed.
        n_fft (int): The number of data points used in each block for the FFT. Default is 2048.
        hop_length (int): The number of audio samples between adjacent STFT columns. Default is 512.
    
    Returns:
        dict: Mel Loss scores for each file pair.
        float: Mean Mel Loss score.
    """
    ref_files = find_audio_files(ref_folder)
    est_files = find_audio_files(est_folder)
    
    if len(ref_files) != len(est_files):
        raise ValueError("The number of reference and estimated files do not match.")
    
    mel_losses = {}
    mean_score = []
    index = 0

    progress_bar = tqdm(zip(ref_files, est_files), desc="Mel Loss evaluating....")

    for ref_path, est_path in progress_bar:
        index += 1
        assert os.path.basename(ref_path) == os.path.basename(est_path), "File names must match!"
        
        ref_audio, ref_rate = sf.read(ref_path)
        est_audio, est_rate = sf.read(est_path)
        
        # GPU 重采样
        if target_sr is not None:
            if ref_rate != target_sr:
                ref_audio = resample_on_gpu(ref_audio, ref_rate, target_sr, device=device)
            if est_rate != target_sr:
                est_audio = resample_on_gpu(est_audio, est_rate, target_sr, device=device)
            ref_rate = target_sr
        elif est_rate != ref_rate:
            est_audio = resample_on_gpu(est_audio, est_rate, ref_rate, device=device)

        min_len = min(ref_audio.shape[0], est_audio.shape[0])
        ref_audio = ref_audio[:min_len]
        est_audio = est_audio[:min_len]
        
        if ref_audio.ndim > 1 and ref_audio.shape[1] > 1:
            ref_audio = np.mean(ref_audio, axis=1)
        if est_audio.ndim > 1 and est_audio.shape[1] > 1:
            est_audio = np.mean(est_audio, axis=1)

        # 确保为一维数组
        ref_audio = np.squeeze(ref_audio)
        est_audio = np.squeeze(est_audio)
        
# 转换为 PyTorch 张量并移到 GPU
        ref_audio_tensor = torch.from_numpy(ref_audio).float().to(device).unsqueeze(0)
        est_audio_tensor = torch.from_numpy(est_audio).float().to(device).unsqueeze(0)
        
        try:
            loss = 0.0
            for n_mel, fmin, fmax, wlen in zip(n_mels, mel_fmin, mel_fmax, window_lengths):
                hop_length = wlen // 4
                right_pad, pad = compute_stft_padding(ref_audio.shape[0], wlen, hop_length, match_stride)
                ref_padded = torch.nn.functional.pad(ref_audio_tensor, (pad, pad + right_pad), mode="reflect")
                est_padded = torch.nn.functional.pad(est_audio_tensor, (pad, pad + right_pad), mode="reflect")

                window = sqrt_hann_window(wlen, device) if window_type == "sqrt_hann" else torch.hann_window(wlen, device=device)
                ref_stft = torch.stft(
                    ref_padded, n_fft=wlen, hop_length=hop_length,
                    window=window, return_complex=True, center=True
                )  # [1, freq, time]
                est_stft = torch.stft(
                    est_padded, n_fft=wlen, hop_length=hop_length,
                    window=window, return_complex=True, center=True
                )
                
                ref_mag = torch.abs(ref_stft)
                est_mag = torch.abs(est_stft)

                mel_basis = librosa.filters.mel(
                    sr=ref_rate, n_fft=wlen, n_mels=n_mel, fmin=fmin, fmax=fmax if fmax is not None else ref_rate/2
                )
                mel_basis = torch.from_numpy(mel_basis).float().to(device)

                ref_mels = torch.matmul(ref_mag.transpose(1, 2), mel_basis.T).transpose(1, 2)
                est_mels = torch.matmul(est_mag.transpose(1, 2), mel_basis.T).transpose(1, 2)

                ref_mels = ref_mels.unsqueeze(1)  # [1, 1, n_mels, time]
                est_mels = est_mels.unsqueeze(1)

                loss += log_weight * loss_fn(
                    ref_mels.clamp(clamp_eps).pow(pow).log10(),
                    est_mels.clamp(clamp_eps).pow(pow).log10(),
                )
                loss += mag_weight * loss_fn(ref_mels, est_mels)
                
            loss = loss.cpu().numpy()
            mean_score.append(loss)
            mel_losses[os.path.basename(ref_path)] = loss

        except Exception as e:
            logging.warning(f"Mel Loss failed for {ref_path}: {str(e)}")
            loss = 0.0

        progress_bar.set_postfix(index=index, ref_path=ref_path, est_path=est_path, mel_loss=loss)

    mean_score = np.nan_to_num(np.asarray(mean_score), 0.0)

    return mel_losses, np.mean(mean_score)