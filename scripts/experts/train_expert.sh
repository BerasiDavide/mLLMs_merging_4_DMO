#!/bin/bash

#SBATCH --job-name=train_experts
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH -t 2:00:00
#SBATCH -o logs/train_experts/%A_%x.out
#SBATCH -e logs/train_experts/%A_%x.err


BASE_MODEL=$1 # e.g. qwen2_2b
DOMAIN=$2 # e.g. general
SFT_STRATEGY=${3:-lora} # lora, full
MAX_STEPS=${4:-800} # 80, 400, 800

MIXTURE_THRESHOLD=${5:-102400} # Num samples in training mixture

N_SAMPLES=$((MAX_STEPS * 128))
BUDGET_ID=$((MAX_STEPS / 8))k # 10k, 50k, 100k
EXP_NAME=${6:-exp_${BASE_MODEL}_${BUDGET_ID}}

# === Configuration ===
DATASET_NAME=expert_${DOMAIN}-${MIXTURE_THRESHOLD}

FINETUNED_ID=${BASE_MODEL}_${SFT_STRATEGY}_expert_${DOMAIN}-${N_SAMPLES}
FINETUNED_PATH=$(realpath checkpoints)/sft_models/${EXP_NAME}/${FINETUNED_ID}
EXPORTED_PATH=$(realpath checkpoints)/exported_models/sft_models/${EXP_NAME}/${FINETUNED_ID}

# === Train ===
bash scripts/simple/train.sh ${BASE_MODEL}_${SFT_STRATEGY} ${DATASET_NAME} ${MAX_STEPS} ${FINETUNED_PATH}
status=$? # 0 if success

# === Export ===
if [ $status -eq 0 ]; then
  bash scripts/simple/export.sh ${BASE_MODEL}_${SFT_STRATEGY} ${FINETUNED_PATH} ${EXPORTED_PATH}

  # Alternatively, export as a separate job to save a bit of GPU hours if running on a SLURM cluster.
  # LOG_PATH=logs/${EXP_NAME}/export_expert/%A_%x/%A_%a_%x
  # sbatch --job-name=export_${BASE_MODEL}_${SFT_STRATEGY}_${DOMAIN} \
  #   --output=$LOG_PATH.out --error=$LOG_PATH.err \
  #   scripts/simple/export.sh ${BASE_MODEL}_${SFT_STRATEGY} ${FINETUNED_PATH} ${EXPORTED_PATH}
fi
