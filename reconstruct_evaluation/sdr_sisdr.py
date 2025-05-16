import numpy as np
import soundfile as sf
from mir_eval.separation import bss_eval_sources
import librosa
import glob
import os
import logging
from tqdm import tqdm
from utils.helpers import find_audio_files, resample_on_gpu

def compute_sisdr(ref, est):
    ref = ref / (np.linalg.norm(ref) + 1e-10)
    est = est / (np.linalg.norm(est) + 1e-10)
    projection = np.dot(ref, est) * ref
    distortion = est - projection
    return 10 * np.log10(np.mean(ref**2) / (np.mean(distortion**2) + 1e-10))

def evaluate_sdr_sisdr(ref_folder, est_folder, target_sr=16000):
    logging.info(f"Evaluating SDR/SISDR: ref_folder: {ref_folder}, est_folder: {est_folder}")
    """
    Calculate SDR and SISDR metrics between reference and estimated audio files.

    Parameters:
        ref_folder (str): Path to reference audio files.
        est_folder (str): Path to estimated/generated audio files.
        target_sr (int): Target sample rate (default 16000).

    Returns:
        mean_sdr, mean_sisdr, np.mean(mean_sdr), np.mean(mean_sisdr)
    """
    ref_files = find_audio_files(ref_folder)
    est_files = find_audio_files(est_folder)

    logging.info(f"ref_files num = {len(ref_files)}")
    logging.info(f"est_files num = {len(est_files)}")

    if len(ref_files) != len(est_files):
        raise ValueError("The number of reference and estimated files do not match.")
        
    sdr_scores = {}
    sisdr_scores = {}
    mean_sdr = []
    mean_sisdr = []
    index = 0
    progress_bar = tqdm(zip(ref_files, est_files), desc="SDR/SISDR evaluating....")

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
            sdr, _, _, _ = bss_eval_sources(ref_audio, est_audio)
            sdr_score = sdr[0]
            sisdr_score = compute_sisdr(ref_audio, est_audio)
            mean_sdr.append(sdr_score)
            mean_sisdr.append(sisdr_score)
            sdr_scores[os.path.basename(ref_path)] = sdr_score
            sisdr_scores[os.path.basename(ref_path)] = sisdr_score
        except Exception as e:
            logging.warning(f"SDR/SISDR failed for {ref_path}: {str(e)}")
            sdr_score = 0.0
            sisdr_score = 0.0
        
        progress_bar.set_postfix(index=index, ref_path=ref_path, est_path=est_path, sdr=sdr_score, sisdr=sisdr_score)
        
    mean_sdr = np.nan_to_num(np.asarray(mean_sdr), 0.0)
    mean_sisdr = np.nan_to_num(np.asarray(mean_sisdr), 0.0)
    
    return sdr_scores, sisdr_scores, np.mean(mean_sdr), np.mean(mean_sisdr)