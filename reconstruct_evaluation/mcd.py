import numpy as np
import soundfile as sf
import librosa
import os
from tqdm import tqdm
import logging
from scipy.signal import firwin, lfilter
import pysptk
import pyworld as pw
from fastdtw import fastdtw
import scipy.spatial.distance as distance
from utils.helpers import find_audio_files, resample_on_gpu

# 低通滤波器
def low_cut_filter(x, fs, cutoff=70):
    nyquist = fs // 2
    norm_cutoff = cutoff / nyquist
    fil = firwin(255, norm_cutoff, pass_zero=False)
    return lfilter(fil, 1, x)

# 提取 WORLD 特征
def world_extract(x, fs, f0min, f0max, mcep_shift=5, mcep_fftl=1024, mcep_dim=39, mcep_alpha=0.466):
    x = x * np.iinfo(np.int16).max  # 缩放到 int16 范围
    x = np.array(x, dtype=np.float64)
    x = low_cut_filter(x, fs)

    f0, time_axis = pw.harvest(x.astype(np.double), fs, f0_floor=f0min, f0_ceil=f0max, frame_period=mcep_shift)
    sp = pw.cheaptrick(x, f0, time_axis, fs, fft_size=mcep_fftl)
    mcep = pysptk.sp2mc(sp, mcep_dim, mcep_alpha)  # 提取 Mel 倒谱系数
    return mcep

# 计算 MCD
def compute_mcd(ref_mcep, est_mcep, use_dtw=True):
    if use_dtw:
        # 使用 DTW 对齐
        _, path = fastdtw(ref_mcep, est_mcep, dist=distance.euclidean)
        path = np.array(path).T
        ref_mcep_aligned = ref_mcep[path[0]]
        est_mcep_aligned = est_mcep[path[1]]
    else:
        # 不使用 DTW，取较短长度
        min_len = min(ref_mcep.shape[0], est_mcep.shape[0])
        ref_mcep_aligned = ref_mcep[:min_len]
        est_mcep_aligned = est_mcep[:min_len]

    # 计算 MCD
    diff2sum = np.sum((ref_mcep_aligned - est_mcep_aligned) ** 2, axis=1)
    mcd = np.mean(10.0 / np.log(10.0) * np.sqrt(2 * diff2sum))
    return mcd

# 主 MCD 计算函数
def evaluate_mcd(ref_folder, 
                est_folder, 
                target_sr=16000, 
                f0min=40, 
                f0max=800, 
                mcep_shift=5, 
                mcep_fftl=1024, 
                mcep_dim=39, 
                mcep_alpha=0.466, 
                seq_mismatch_tolerance=0.1,
                power_threshold=-20,
                use_dtw=True):
    """
    Calculate the Mel Cepstral Distortion (MCD) between pairs of reference and estimated audio files.

    Parameters:
        ref_folder (str): Folder containing reference audio files.
        est_folder (str): Folder containing estimated/generated audio files.
        target_sr (int, optional): Target sample rate for resampling. If None, use original rate.
        f0min (float): Minimum F0 value for WORLD extraction.
        f0max (float): Maximum F0 value for WORLD extraction.
        mcep_shift (float): Frame shift in ms for WORLD analysis.
        mcep_fftl (int): FFT length for spectral analysis.
        mcep_dim (int): Number of Mel cepstral coefficients.
        mcep_alpha (float): All-pass constant for frequency warping.
        use_dtw (bool): Whether to use DTW for frame alignment.

    Returns:
        dict: MCD scores for each file pair.
        float: Mean MCD score.
    """
    logging.info(f"Evaluating MCD: ref_folder: {ref_folder}, est_folder: {est_folder}")

    ref_files = find_audio_files(ref_folder)
    est_files = find_audio_files(est_folder)

    if len(ref_files) != len(est_files):
        raise ValueError("The number of reference and estimated files do not match.")

    mcd_scores = {}
    mean_score = []
    index = 0

    progress_bar = tqdm(zip(ref_files, est_files), desc="MCD evaluating....")

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

        if ref_audio.ndim > 1:
            ref_audio = np.mean(ref_audio, axis=1)
        if est_audio.ndim > 1:
            est_audio = np.mean(est_audio, axis=1)
        
        # 确保为一维数组
        ref_audio = np.squeeze(ref_audio)
        est_audio = np.squeeze(est_audio)

        # 提取 Mel 倒谱系数
        try:
            ref_mcep = world_extract(ref_audio, ref_rate, f0min, f0max, mcep_shift, mcep_fftl, mcep_dim, mcep_alpha)
            est_mcep = world_extract(est_audio, ref_rate, f0min, f0max, mcep_shift, mcep_fftl, mcep_dim, mcep_alpha)
            mcd = compute_mcd(ref_mcep, est_mcep, use_dtw=use_dtw)
            mcd_scores[os.path.basename(ref_path)] = mcd
            mean_score.append(mcd)
        except Exception as e:
            logging.warning(f"MCD failed for {ref_path}: {str(e)}")
            mcd = 0.0

        progress_bar.set_postfix(index=index, ref_path=ref_path, est_path=est_path, mcd=mcd)

    mean_score = np.nan_to_num(np.asarray(mean_score), 0.0)
    return mcd_scores, np.mean(mean_score)