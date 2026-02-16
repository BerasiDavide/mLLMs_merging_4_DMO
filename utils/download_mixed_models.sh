#!/bin/bash

model=${1:-"qwen2_2b"}
sft_type=${2:-"lora"}
steps=${3:-800} # number of training steps, default is 800, each one with gbs=128
export_models=${4:-"true"} # whether to export models after downloading, default is true

N_samples=$((steps * 128))
budget_id=$((steps / 8))k # 10k, 50k, 100k
exp_name=exp_${model}_${budget_id}

model_id=${model}_${sft_type}_mixed-${N_samples}
echo "Downloading model: $model_id ..."
hf download --repo-type model daviBera/${model_id} --local-dir "checkpoints/sft_models/${exp_name}" --exclude "README.md"

if [ "$export_models" == "true" ]; then

    ## We now export (ie. attach to the base model) the downloaded LoRA adapters. InterVL models are also converted from HF format to the original InterVL format.
    ## To do this, we call the training scripts; if the exported model already exists, the script will skip the training and directly export the model.

    ## If you to export the models as a batch of SLURM jobs, sbatch the scripts directly.
    # sbatch scripts/mixed2/train_mixed2.sh $model "general" "ocr" ${sft_type} ${steps}
    # sbatch scripts/mixed3/train_mixed3.sh $model "counting" "general" "ocr" ${sft_type} ${steps}
    # sbatch scripts/mixed4/train_mixed4.sh $model "chart" "counting" "general" "ocr" ${sft_type} ${steps}

    ## Alternatively, you can run the training and exporting sequentially in a single script as below.
    echo "Exporting the 7 two-domains mixed models..."
    for i in {0..6}; do
        export SLURM_ARRAY_TASK_ID=$i
        bash scripts/mixed2/train_mixed2.sh $model "general" "ocr" ${sft_type} ${steps} ${exp_name}
    done

    echo "Exporting the 21 three-domains mixed models..."
    for i in {0..20}; do
        export SLURM_ARRAY_TASK_ID=$i
        bash scripts/mixed3/train_mixed3.sh $model "counting" "general" "ocr" ${sft_type} ${steps} ${exp_name}
    done

    echo "Exporting the 20 four-domains mixed models..."
    for i in {0..19}; do
        export SLURM_ARRAY_TASK_ID=$i
        bash scripts/mixed4/train_mixed4.sh $model "chart" "counting" "general" "ocr" ${sft_type} ${steps} ${exp_name}
    done
fi
