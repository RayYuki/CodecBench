import numpy as np
import soundfile as sf
import librosa
import glob
import os
import logging
from tqdm import tqdm
from utils.helpers import find_audio_files, resample_on_gpu

def evaluate_spectral_convergence(ref_folder, est_folder, target_sr=16000):
    logging.info(f"Evaluating Spectral Convergence: ref_folder: {ref_folder}, est_folder: {est_folder}")
    """
    Calculate Spectral Convergence between reference and estimated audio files.

    Parameters:
        ref_folder (str): Path to reference audio files.
        est_folder (str): Path to estimated/generated audio files.
        target_sr (int): Target sample rate (default 16000).

    Returns:
        dict: Spectral Convergence scores for each file pair.
        float: Mean Spectral Convergence score.
    """
    ref_files = find_audio_files(ref_folder)
    est_files = find_audio_files(est_folder)

    logging.info(f"ref_files num = {len(ref_files)}")
    logging.info(f"est_files num = {len(est_files)}")

    if len(ref_files) != len(est_files):
        raise ValueError("The number of reference and estimated files do not match.")

    sc_scores = {}
    mean_score = []
    index = 0

    progress_bar = tqdm(zip(ref_files, est_files), desc="Spectral Convergence evaluating....")
    for ref_path, est_path in progress_bar:
        index += 1
        assert os.path.basename(ref_path) == os.path.basename(est_path), "File names must match!"

        ref_audio, ref_rate = sf.read(ref_path)
        est_audio, est_rate = sf.read(est_path)

        # GPU 重采样
        if target_sr is not None:
            if ref_rate != target_sr:
                ref_audio = resample_on_gpu(ref_audio, ref_rate, target_sr, device="cuda")
            if est_rate != target_sr:
                est_audio = resample_on_gpu(est_audio, est_rate, target_sr, device="cuda")
            ref_rate = target_sr
        elif est_rate != ref_rate:
            est_audio = resample_on_gpu(est_audio, est_rate, ref_rate, device="cuda")
            
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
        
        try:
            S_ref = np.abs(librosa.stft(ref_audio))
            S_est = np.abs(librosa.stft(est_audio))
            numerator = np.sqrt(np.sum((S_ref - S_est) ** 2))  # Frobenius 范数
            denominator = np.sqrt(np.sum(S_ref ** 2))
            score = numerator / denominator
            mean_score.append(score)
            sc_scores[os.path.basename(ref_path)] = score
        except Exception as e:
            logging.warning(f"Spectral Convergence failed for {ref_path}: {str(e)}")
            score = 0.0

        progress_bar.set_postfix(index=index, ref_path=ref_path, est_path=est_path, sc_score=score)

    mean_score = np.nan_to_num(np.asarray(mean_score), 0.0)
    
    return sc_scores, np.mean(mean_score)