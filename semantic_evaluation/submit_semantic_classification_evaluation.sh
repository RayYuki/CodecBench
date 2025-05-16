#!/usr/bin/env bash

source ~/.bashrc
conda activate CodecBench
which python

work_dir=/remote-home1/CodecBench
cd ${work_dir}
export PYTHONPATH=./

model_type=SpeechTokenizer # ! 需要修改
exp_root=semantic_evaluation/exp # ! 需要修改
data_conifg=config/semantic_data_config.json # ! 需要修改

tag=${model_type}/${language}/v1.0/spt1_release # ! 需要修改

config=config/spt_base_cfg.json # ! 需要修改
codec_ckpt=/remote-home1/model/speechtokenizer/SpeechTokenizer.pt # ! 需要修改

# 不需要修改
cmd="python ${work_dir}/semantic_evaluation/classification_task.py \
--model_type ${model_type} \
--tag ${tag} \
--exp_root ${exp_root} \
--config ${config} \
--codec_ckpt ${codec_ckpt} \
--data_config ${data_conifg} \
"
echo "Executing: $cmd"
eval $cmd