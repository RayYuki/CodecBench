import argparse
import os
import numpy as np
import logging

from utils.sim_utils import ValleEncoder, compute_cosine_similarity

from utils.helpers import find_audio_files


def evaluate_sim(ref_path, syn_path):
    logging.info(
        f"Evaluating Speaker Similarity: ref_path = {ref_path}, syn_path = {syn_path}"
    )
    ref_audio_list = find_audio_files(ref_path)
    syn_audio_list = find_audio_files(syn_path)
    logging.info(f"ref_files num = {len(ref_audio_list)}")
    logging.info(f"syn_files num = {len(syn_audio_list)}")

    model_path = "/remote-home1/model/wavlm_large_finetune/wavlm_large_finetune.pth"
    embedder = ValleEncoder(model_path, use_cuda=True)
    src_embs = embedder(ref_audio_list)
    tgt_embs = embedder(syn_audio_list)
    similarities = compute_cosine_similarity(src_embs, tgt_embs)
    return similarities, np.mean(similarities)
