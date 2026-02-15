#!/bin/bash

#SBATCH --job-name=export_model
#SBATCH -p boost_usr_prod
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -t 0:30:00
#SBATCH -o logs/export_model/%A/%A_%a_%x.out
#SBATCH -e logs/export_model/%A/%A_%a_%x.err


BASE_MODEL=$1
DOMAIN=$2
SFT_STRATEGY=${3:-lora}
STEPS=${4:-800}

N_SAMPLES=$((STEPS * 128))
BUDGET_ID=$((N_SAMPLES / 8))k # 10k, 50k, 100k
EXP_NAME=${5:-exp_${BASE_MODEL}_${BUDGET_ID}}

# === Configuration ===
FINETUNED_ID=${BASE_MODEL}_${SFT_STRATEGY}_expert_${DOMAIN}-${N_SAMPLES}
FINETUNED_PATH=$(realpath checkpoints)/sft_models/${EXP_NAME}/${FINETUNED_ID}
EXPORTED_PATH=$(realpath checkpoints)/exported_models/sft_models/${EXP_NAME}/${FINETUNED_ID}

# === Export ===
bash scripts/simple/export.sh ${BASE_MODEL}_${SFT_STRATEGY} ${FINETUNED_PATH} ${EXPORTED_PATH}
