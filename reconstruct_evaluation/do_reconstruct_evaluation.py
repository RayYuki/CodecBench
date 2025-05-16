import torch
import argparse
import logging
import os
import soundfile as sf

from torch.utils.data import DataLoader
from tqdm import tqdm
from tensorboardX import SummaryWriter

# utils
from utils.helpers import set_logging, count_params_by_module, waiting_for_debug
from utils.spt_utils import load_and_fix_codec_model

# dataset
from reconstruct_evalation_dataset import DatasetForReconstructEvaluation

# metric funcs
from mcd import evaluate_mcd
from mel_loss import evaluate_mel_loss
from mse import evaluate_mse
from pesq_local import evaluate_pesq
from spectral_convergence import evaluate_spectral_convergence
from sdr_sisdr import evaluate_sdr_sisdr
from speaker_similarity import evaluate_sim
from stoi import evaluate_stoi
from visqol_local import evaluate_visqol


def do_evaluation(args):
    # Basic settings
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Tensorboard settings
    writer = SummaryWriter(os.path.join(args.exp_root, args.tag))
    output_audio_dir = os.path.join(args.exp_root, args.tag, "syn_audios")
    os.makedirs(output_audio_dir, exist_ok=True)
    save_to_tensorboard_cnt = args.save_to_tensorboard_cnt

    # Load and fix codec models
    codec_model, _ = load_and_fix_codec_model(args)
    codec_model = codec_model.eval().to(device)
    count_params_by_module("codec_model", codec_model)

    # Dataset
    test_dataset = DatasetForReconstructEvaluation(
        args.test_dataset_dir, target_sample_rate=codec_model.sampling_rate
    )
    test_loader = DataLoader(test_dataset, batch_size=1, shuffle=False, num_workers=1)
    logging.info(f"Number of audios in {args.test_dataset_dir}:  {len(test_loader)}")

    # Generate audios
    cur_saved_to_tensorboard_cnt = 0
    with torch.inference_mode():
        progress_bar = tqdm(test_loader, desc=f"Generate Audios")
        for batchidx, batch in enumerate(progress_bar):
            audiopath, x = batch
            raw_audio = x[0].reshape(1, -1)
            audioname = os.path.basename(audiopath[0])

            # Forward
            x = x.to(device).reshape(1, 1, -1)
            y = codec_model(x)["y"]
            generated_audio = y.reshape(1, -1)

            # Save
            audio_output_path = os.path.join(output_audio_dir, audioname)
            sf.write(
                audio_output_path,
                generated_audio.cpu().transpose(0, 1).numpy(),
                codec_model.sampling_rate,
            )

            # write to tensorboard
            if cur_saved_to_tensorboard_cnt < save_to_tensorboard_cnt:
                cur_saved_to_tensorboard_cnt += 1
                writer.add_audio(
                    f"generate/{cur_saved_to_tensorboard_cnt}",
                    generated_audio.transpose(0, 1).cpu().numpy(),
                    global_step=0,
                    sample_rate=codec_model.sampling_rate,
                )
                writer.add_audio(
                    f"ground_truth/{cur_saved_to_tensorboard_cnt}",
                    raw_audio.transpose(0, 1).cpu().numpy(),
                    global_step=0,
                    sample_rate=codec_model.sampling_rate,
                )

            logging.info(
                f"index = {batchidx}, audio: {audioname} saved to: {audio_output_path}"
            )

    # _, mean_mcd = evaluate_mcd(args.test_dataset_dir, output_audio_dir)
    _, mean_mel_loss = evaluate_mel_loss(args.test_dataset_dir, output_audio_dir)
    _, mean_mse = evaluate_mse(args.test_dataset_dir, output_audio_dir)
    _, mean_pesq_nb = evaluate_pesq(args.test_dataset_dir, output_audio_dir, mode="nb")
    _, mean_pesq_wb = evaluate_pesq(args.test_dataset_dir, output_audio_dir, mode="wb")
    _, mean_sc = evaluate_spectral_convergence(args.test_dataset_dir, output_audio_dir)
    _, _, mean_sdr, mean_sisdr = evaluate_sdr_sisdr(
        args.test_dataset_dir, output_audio_dir
    )
    _, mean_speaker_similarities = evaluate_sim(args.test_dataset_dir, output_audio_dir)
    _, mean_stoi = evaluate_stoi(args.test_dataset_dir, output_audio_dir)
    _, mean_visqol = evaluate_visqol(args.test_dataset_dir, output_audio_dir)

    logging.info(f"Test result:")
    # logging.info(f"\t mcd = {mean_mcd}")
    logging.info(f"\t mel loss = {mean_mel_loss}")
    logging.info(f"\t mse = {mean_mse}")
    logging.info(f"\t pesq-nb = {mean_pesq_nb}")
    logging.info(f"\t pesq-wb = {mean_pesq_wb}")
    logging.info(f"\t spectral convergence = {mean_sc}")
    logging.info(f"\t sdr = {mean_sdr}")
    logging.info(f"\t sisdr = {mean_sisdr}")
    logging.info(f"\t sim = {mean_speaker_similarities}")
    logging.info(f"\t stoi = {mean_stoi}")
    logging.info(f"\t visqol = {mean_visqol}")

    # writer.add_scalar(f"mcd", mean_mcd, global_step=0)
    writer.add_scalar(f"mel loss", mean_mel_loss, global_step=0)
    writer.add_scalar(f"mse", mean_mse, global_step=0)
    writer.add_scalar(f"pesq-nb", mean_pesq_nb, global_step=0)
    writer.add_scalar(f"pesq-wb", mean_pesq_wb, global_step=0)
    writer.add_scalar(f"spectral convergence", mean_sc, global_step=0)
    writer.add_scalar(f"sdr", mean_sdr, global_step=0)
    writer.add_scalar(f"sisdr", mean_sisdr, global_step=0)
    writer.add_scalar(f"sim", mean_speaker_similarities, global_step=0)
    writer.add_scalar(f"stoi", mean_stoi, global_step=0)
    writer.add_scalar(f"visqol", mean_visqol, global_step=0)


def main():
    set_logging()

    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--tag", type=str, required=True)
    parser.add_argument("--exp_root", type=str, default="exp")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--codec_ckpt", type=str, required=True)
    parser.add_argument("--save_to_tensorboard_cnt", type=int, default=20)
    parser.add_argument(
        "--test_dataset_dir",
        type=str,
        default="/remote-home1/share/data/SpeechPretrain/librispeech/LibriSpeech/test-clean/",
    )

    parser.add_argument(
        "--debug", default=0, type=int, nargs="?", help="whether debug or not"
    )
    parser.add_argument("--debug_ip", default="localhost", type=str)
    parser.add_argument("--debug_port", default=32431, type=int)

    args = parser.parse_args()
    if args.debug == 1:
        waiting_for_debug(args.debug_ip, args.debug_port)

    do_evaluation(args)


if __name__ == "__main__":
    main()
