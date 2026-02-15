#!/bin/bash

#SBATCH --job-name=eval_experts
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -t 2:00:00
#SBATCH --array=0-$(($(echo $3 | tr ',' '\n' | wc -l) - 1))
#SBATCH -o logs/eval_bench/%A/%A_%a_%x.out
#SBATCH -e logs/eval_bench/%A/%A_%a_%x.err

BASE_MODEL=$1
DOMAIN=$2
BENCHMARKS=$3 # comma-separated list of benchmarks to evaluate on, e.g. "gsm8k,svamp"
SFT_STRATEGY=${4:-lora} # lora, full
STEPS=$5

N_SAMPLES=$((STEPS * 128))
BUDGET_ID=$((STEPS / 8))k # 10k, 50k, 100k
EXP_NAME=${6:-exp_${BASE_MODEL}_${BUDGET_ID}}

EVAL_TYPE=${7:-bench} # bench, loss

# === Configuration ===
FINETUNED_ID=${BASE_MODEL}_${SFT_STRATEGY}_expert_${DOMAIN}-${N_SAMPLES}
MODEL_PATH=$(realpath checkpoints)/exported_models/sft_models/${EXP_NAME}/${FINETUNED_ID}


# Get the benchmark for this array task
IDX=${SLURM_ARRAY_TASK_ID}
BENCHMARK=$(echo $BENCHMARKS | cut -d',' -f$((IDX + 1)))

if [ "$EVAL_TYPE" == "loss" ]; then
    # === Eval Loss ===
    OUTPUT_FOLDER=$(realpath eval_loss)/${EXP_NAME}/${FINETUNED_ID}
    bash scripts/simple/eval_loss.sh ${BASE_MODEL} ${MODEL_PATH} ${BENCHMARK} ${OUTPUT_FOLDER}
    exit 0
elif [ "$EVAL_TYPE" == "bench" ]; then
    # === Eval Benchmark ===
    OUTPUT_FOLDER=$(realpath eval_bench)/${EXP_NAME}/${FINETUNED_ID}
    bash scripts/simple/eval_bench.sh ${MODEL_PATH} ${BENCHMARK} ${OUTPUT_FOLDER}
else
    echo "Unknown EVAL_TYPE: $EVAL_TYPE. Supported types are 'loss' and 'bench'."
    exit 1
fi
