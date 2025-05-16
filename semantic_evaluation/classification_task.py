import json
import argparse
from semantic_evaluation.classification import CodecWrapper
from semantic_evaluation.classification import (
    ESC50,
    VOCALSOUND,
    ChordRecognition,
    GTZAN,
    NSYNTH,
    RAVDESS,
    CREMAD,
    MELD,
)
from utils.spt_utils import load_and_fix_codec_model
from utils.helpers import set_logging
import logging
import torch

device = "cuda" if torch.cuda.is_available() else "cpu"

logging.info("using device: ", device)

# load the dataset information - update this file according to the downloaded dataset(s)


def classification(args):
    MAX_EPOCHS = args.num_epochs
    with open(args.data_config) as f:
        datasets_info = json.load(f)

    codec_model, _ = load_and_fix_codec_model(args)
    target_sr = codec_model.sampling_rate
    codec_model = codec_model.to(device)
    evaluate_dict = {
        "esc50": ESC50,
        "vocalsound": VOCALSOUND,
        "chord_recognition": ChordRecognition,
        "gtzan": GTZAN,
        "nsynth": NSYNTH,
        "ravdess": RAVDESS,
        "crema_d": CREMAD,
        "meld": MELD,
    }

    for dataset_name in evaluate_dict.keys():
        logging.info("Evaluate dataset: ", dataset_name)

        model = CodecWrapper(
            model=codec_model,
            model_name=args.model_type,
            hidden_size=codec_model.hidden_size,
            device=device,
            max_length=datasets_info[dataset_name]["max_length_seconds"] * target_sr,
            sampling_rate=target_sr,
        )

        evaluator = evaluate_dict[dataset_name](
            datasets_info[dataset_name]["path"], verbose=True
        )
        res_dataset = evaluator.evaluate(
            model,
            mode="linear",
            device=device,
            batch_size=args.batch_size,
            num_workers=args.num_workers,
            max_num_epochs=MAX_EPOCHS,
        )

        for metric, value in res_dataset.items():
            logging.info(f"{metric}: {value}")


def main():
    set_logging()
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_type", type=str, required=True)
    parser.add_argument("--tag", type=str, required=True)
    parser.add_argument("--exp_root", type=str, default="exp")
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--codec_ckpt", type=str, required=True)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--num_epochs", type=int, default=20)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument(
        "--data_config", type=str, default="config/semantic_data_config.json"
    )

    args = parser.parse_args()

    classification(args)


if __name__ == "__main__":
    main()
