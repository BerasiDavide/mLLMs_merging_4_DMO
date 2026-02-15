#!/bin/bash

model=${1:-"qwen2_2b"}
sft_type=${2:-"lora"}
steps=${3:-800} # number of training steps, default is 800, each one with gbs=128

export_models=${4:-"true"} # whether to export models after downloading, default is true

N_samples=$((steps * 128))
budget_id=$((steps / 8))k # 10k, 50k, 100k
exp_name=exp_${model}_${budget_id}

domains=("general" "ocr" "counting" "chart")

for domain in "${domains[@]}"; do
    model_id=${model}_${sft_type}_expert_${domain}-${N_samples}
    echo "Downloading model: $model_id ..."
    hf download --repo-type model daviBera/${model_id} --local-dir "checkpoints/sft_models/${exp_name}/${model_id}"

    if [ "$export_models" == "true" ]; then
        echo "Exporting model: $model_id ..."
        bash scripts/experts/export_expert.sh ${model} ${domain} ${sft_type} ${steps} ${exp_name}
    fi
done
