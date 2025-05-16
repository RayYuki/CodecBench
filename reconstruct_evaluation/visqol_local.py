import numpy as np
import soundfile as sf
import librosa
import glob
import os
import logging
from tqdm import tqdm
from visqol import visqol_lib_py
from visqol.pb2 import visqol_config_pb2
from visqol.pb2 import similarity_result_pb2
from utils.helpers import find_audio_files, resample_on_gpu

def evaluate_visqol(ref_folder, est_folder, mode='audio', target_sr=48000):
    logging.info(f"Evaluating ViSQOL-{mode}: ref_folder: {ref_folder}, est_folder: {est_folder}")
    """
    Calculate ViSQOL metric between pairs of reference and estimated audio files.

    Parameters:
        ref_folder (str): Path to reference audio files.
        est_folder (str): Path to estimated/generated audio files.
        mode (str): 'audio' (48kHz) or 'speech' (16kHz).
        target_sr (int): Target sample rate (default 48000 for audio, 16000 for speech).

    Returns:
        dict: ViSQOL scores for each file pair.
        float: Mean ViSQOL score.
    """
    ref_files = find_audio_files(ref_folder)
    est_files = find_audio_files(est_folder)

    logging.info(f"ref_files num = {len(ref_files)}")
    logging.info(f"est_files num = {len(est_files)}")

    if len(ref_files) != len(est_files):
        raise ValueError("The number of reference and estimated files do not match.")

    # Configure ViSQOL
    config = visqol_config_pb2.VisqolConfig()

    if mode == 'audio':
        target_sr = 48000
        config.audio.sample_rate = 48000
        config.options.use_speech_scoring = False
        svr_model_path = "libsvm_nu_svr_model.txt"
    elif mode == 'speech':
        target_sr = 16000
        config.audio.sample_rate = 16000
        config.options.use_speech_scoring = True
        svr_model_path = "lattice_tcditugenmeetpackhref_ls2_nl60_lr12_bs2048_learn.005_ep2400_train1_7_raw.tflite"
    else:
        raise ValueError(f"Unrecognized mode: {mode}")
    
    config.options.svr_model_path = os.path.join(os.path.dirname(visqol_lib_py.__file__), "model", svr_model_path)
    api = visqol_lib_py.VisqolApi()
    api.Create(config)

    visqol_scores = {}
    mean_score = []
    index = 0

    progress_bar = tqdm(zip(ref_files, est_files), desc=f"ViSQOL-{mode} evaluating....")
    for ref_path, est_path in progress_bar:
        index += 1
        assert os.path.basename(ref_path) == os.path.basename(est_path), "File names must match!"

        ref_audio, ref_rate = sf.read(ref_path)
        est_audio, est_rate = sf.read(est_path)

        if ref_rate != target_sr:
            ref_audio = resample_on_gpu(ref_audio, ref_rate, target_sr, device="cuda")
        if est_rate != target_sr:
            est_audio = resample_on_gpu(est_audio, est_rate, target_sr, device="cuda")

        min_len = min(ref_audio.shape[0], est_audio.shape[0])
        ref_audio = ref_audio[:min_len]
        est_audio = est_audio[:min_len]

        # 填充短音频到至少 1 秒
        min_samples = target_sr 
        if ref_audio.shape[0] < min_samples:
            if ref_audio.ndim == 1:
                padding = np.zeros(min_samples - ref_audio.shape[0])
                ref_audio = np.concatenate([ref_audio, padding])
            else:  # 多声道
                padding = np.zeros((min_samples - ref_audio.shape[0], ref_audio.shape[1]))
                ref_audio = np.concatenate([ref_audio, padding], axis=0)
        if est_audio.shape[0] < min_samples:
            if est_audio.ndim == 1:
                padding = np.zeros(min_samples - est_audio.shape[0])
                est_audio = np.concatenate([est_audio, padding])
            else:  # 多声道
                padding = np.zeros((min_samples - est_audio.shape[0], est_audio.shape[1]))
                est_audio = np.concatenate([est_audio, padding], axis=0)

        if ref_audio.ndim > 1 and ref_audio.shape[1] > 1:
            ref_audio = np.mean(ref_audio, axis=1)
        if est_audio.ndim > 1 and est_audio.shape[1] > 1:
            est_audio = np.mean(est_audio, axis=1)
            
        # 确保为一维数组
        ref_audio = np.squeeze(ref_audio)
        est_audio = np.squeeze(est_audio)

        ref_audio = ref_audio.astype(np.float64)
        est_audio = est_audio.astype(np.float64)

        try:
            similarity_result = api.Measure(ref_audio, est_audio)
            score = similarity_result.moslqo
            mean_score.append(score)
            visqol_scores[os.path.basename(ref_path)] = score
        except Exception as e:
            logging.warning(f"ViSQOL failed for {ref_path}: {str(e)}")
            score = 0.0

        progress_bar.set_postfix(index=index, ref_path=ref_path, est_path=est_path, visqol_score=score)

    mean_score = np.nan_to_num(np.asarray(mean_score), 0.0)
    
    return visqol_scores, np.mean(mean_score)