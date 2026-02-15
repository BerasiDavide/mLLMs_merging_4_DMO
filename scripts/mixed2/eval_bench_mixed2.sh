#!/bin/bash

#SBATCH --job-name=eval_mixed
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -t 1:00:00
#SBATCH --array=0-6
#SBATCH -o logs/eval_bench_mixed/%A/%A_%a_%x.out
#SBATCH -e logs/eval_bench_mixed/%A/%A_%a_%x.err

BASE_MODEL=$1 # e.g. qwen2_2b
DOM1=$2 # e.g. general
DOM2=$3 # e.g. ocr
BENCHMARK=$4
SFT_STRATEGY=${5:-lora} # lora, full
STEPS=${6:-800} # 80, 400, 800

N_SAMPLES=$((STEPS * 128))
BUDGET_ID=$((STEPS / 8))k # 10k, 50k, 100k
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
CONFIG_ID=${SLURM_ARRAY_TASK_ID}
WEIGHTS=${MIXTURE_WEIGHTS[CONFIG_ID]}

N1=$(printf "%.0f" $(echo "$(echo $WEIGHTS | cut -d',' -f1) * $N_SAMPLES" | bc -l))
N2=$(printf "%.0f" $(echo "$(echo $WEIGHTS | cut -d',' -f2) * $N_SAMPLES" | bc -l))
MIXTURE_ID="mixed_${DOM1}-${N1}++${DOM2}-${N2}"
FINETUNED_ID=${BASE_MODEL}_${SFT_STRATEGY}_${MIXTURE_ID}
FINETUNED_PATH=$(realpath checkpoints)/sft_models/${EXP_NAME}/${FINETUNED_ID}
EXPORTED_PATH=$(realpath checkpoints)/exported_models/sft_models/${EXP_NAME}/${FINETUNED_ID}

# === Eval Benchmark ===
OUTPUT_FOLDER=$(realpath eval_bench)/${EXP_NAME}/${FINETUNED_ID}
bash scripts/simple/eval_bench.sh ${EXPORTED_PATH} ${BENCHMARK} ${OUTPUT_FOLDER}