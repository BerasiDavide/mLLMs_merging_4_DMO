#!/bin/bash

#SBATCH --job-name=eval_bench_merged
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -t 2:00:00
#SBATCH --array=0-6
#SBATCH -o logs/eval_bench_merged/%A/%A_%a_%x.out
#SBATCH -e logs/eval_bench_merged/%A/%A_%a_%x.err


BASE_MODEL=$1 # e.g. qwen2_2b
DOM1=$2 # e.g. general
DOM2=$3 # e.g. ocr
BENCHMARK=$4
SFT_STRATEGY=${5:-lora} # lora, full
STEPS=${6:-800} # 80, 400, 800
METHOD=${7:-task_arithmetic}
MERGED_TYPE=${8:-merged}

N_SAMPLES=$((STEPS * 128))
BUDGET_ID=$((STEPS / 8))k # 10k, 50k, 100k
EXP_NAME=${9:-exp_${BASE_MODEL}_${BUDGET_ID}}


# === Configuration ===
MIXTURE_WEIGHTS=(
  '0.125,0.875'
  '0.25,0.75'
  '0.375,0.625'
  '0.5,0.5'
  '0.625,0.375'
  '0.75,0.25'
  '0.875,0.125'
  )

# === Load environment ===
#source ${WORK}/miniconda3/etc/profile.d/conda.sh
conda activate merge4DMO

# === Configuration ===
ARRAY_ID=${SLURM_ARRAY_TASK_ID}
WEIGHTS=${MIXTURE_WEIGHTS[ARRAY_ID]}

N1=$(printf "%.0f" $(echo "$(echo $WEIGHTS | cut -d',' -f1) * $N_SAMPLES" | bc -l))
N2=$(printf "%.0f" $(echo "$(echo $WEIGHTS | cut -d',' -f2) * $N_SAMPLES" | bc -l))
MIXTURE_ID="${DOM1}-${N1}++${DOM2}-${N2}"
MERGED_ID=${BASE_MODEL}_${SFT_STRATEGY}_${MERGED_TYPE}_${MIXTURE_ID}--${METHOD}
MODEL_PATH=$(realpath checkpoints)/exported_models/merged_models/${EXP_NAME}/${MERGED_ID}
OUTPUT_FOLDER=$(realpath eval_bench)/${EXP_NAME}/${MERGED_ID}

# === Eval Benchmark ===
bash scripts/simple/eval_bench.sh ${MODEL_PATH} ${BENCHMARK} ${OUTPUT_FOLDER}