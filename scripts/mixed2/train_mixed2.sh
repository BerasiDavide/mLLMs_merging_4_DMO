#!/bin/bash

#SBATCH --job-name=train_mixed
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH -t 1:00:00
#SBATCH --array=0-6
#SBATCH -o logs/train_mixed/%A/%A_%a_%x.out
#SBATCH -e logs/train_mixed/%A/%A_%a_%x.err


BASE_MODEL=$1 # e.g. qwen2_2b
DOM1=$2 # e.g. general
DOM2=$3 # e.g. ocr
SFT_STRATEGY=${4:-lora} # lora, full
MAX_STEPS=${5:-800} # 80, 400, 800

MIXTURE_THRESHOLD=${6:-102400} # Num samples in training mixture

N_SAMPLES=$((MAX_STEPS * 128))
BUDGET_ID=$((MAX_STEPS / 8))k # 10k, 50k, 100k
EXP_NAME=${7:-exp_${BASE_MODEL}_${BUDGET_ID}}

MIXTURE_WEIGHTS=(
  '0.125,0.875'
  '0.25,0.75'
  '0.375,0.625'
  '0.5,0.5'
  '0.625,0.375'
  '0.75,0.25'
  '0.875,0.125'
  )

# === Configuration ===
DATASET_NAME=expert_${DOM1}-${MIXTURE_THRESHOLD},expert_${DOM2}-${MIXTURE_THRESHOLD}

CONFIG_ID=${SLURM_ARRAY_TASK_ID}
WEIGHTS=${MIXTURE_WEIGHTS[CONFIG_ID]}

N1=$(printf "%.0f" $(echo "$(echo $WEIGHTS | cut -d',' -f1) * $N_SAMPLES" | bc -l))
N2=$(printf "%.0f" $(echo "$(echo $WEIGHTS | cut -d',' -f2) * $N_SAMPLES" | bc -l))
MIXTURE_ID="mixed_${DOM1}-${N1}++${DOM2}-${N2}"
FINETUNED_ID=${BASE_MODEL}_${SFT_STRATEGY}_${MIXTURE_ID}
FINETUNED_PATH=$(realpath checkpoints)/sft_models/${EXP_NAME}/${FINETUNED_ID}
EXPORTED_PATH=$(realpath checkpoints)/exported_models/sft_models/${EXP_NAME}/${FINETUNED_ID}

# === Train ===
bash scripts/simple/train_mixed.sh ${BASE_MODEL}_${SFT_STRATEGY} ${DATASET_NAME} ${MAX_STEPS} ${WEIGHTS} ${FINETUNED_PATH}
status=$?

# === Export ===
if [ $status -eq 0 ]; then
  bash scripts/simple/export.sh ${BASE_MODEL}_${SFT_STRATEGY} ${FINETUNED_PATH} ${EXPORTED_PATH}

  # Alternatively, export as a separate job to save a bit of GPU hours if running on a SLURM cluster.
  # LOG_PATH=logs/${EXP_NAME}/export_mixed2/%A_%x/%A_%a_%x
  # sbatch --job-name=export_${BASE_MODEL}_${SFT_STRATEGY}_${DOM1}_${DOM2} \
  #   --output=$LOG_PATH.out --error=$LOG_PATH.err \
  #   scripts/simple/export.sh ${BASE_MODEL}_${SFT_STRATEGY} ${FINETUNED_PATH} ${EXPORTED_PATH}
fi