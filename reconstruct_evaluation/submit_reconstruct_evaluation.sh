#!/usr/bin/env bash

source ~/.bashrc
conda activate CodecBench
which python

work_dir=/remote-home1/CodecBench
cd ${work_dir}
export PYTHONPATH=./

model_type=SpeechTokenizer # ! 需要修改
exp_root=reconstruct_evaluation/exp

tag=${model_type}/speechtokenizer_release # ! 需要修改

config=config/spt_base_cfg.json # ! 需要修改
codec_ckpt=/remote-home1/model/speechtokenizer/SpeechTokenizer.pt # ! 需要修改


# 不需要修改
cmd="python ${work_dir}/reconstruct_evaluation/do_reconstruct_evaluation.py \
--model_type ${model_type} \
--tag ${tag} \
--exp_root ${exp_root} \
--config ${config} \
--codec_ckpt ${codec_ckpt}"

echo "Executing: $cmd"
eval $cmd