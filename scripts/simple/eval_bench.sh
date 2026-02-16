#!/bin/bash

#SBATCH --job-name=eval_bench_experts
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -t 2:00:00
#SBATCH -o logs/eval_bench/%A/%A_%a_%x.out
#SBATCH -e logs/eval_bench/%A/%A_%a_%x.err

MODEL_PATH=$1
BENCHMARK=$2
OUTPUT_FOLDER=$3
OVERWRITE=${4:-false} # false to skip if results already exist

OUTPUT_PATH=${OUTPUT_FOLDER}/${BENCHMARK}

MODEL_PATH=$(realpath ${MODEL_PATH})


# if output dir exists, skip
if [ -d "$OUTPUT_PATH" ] && [ "$OVERWRITE" != "true" ]; then
    echo "Skipping eval for $MODEL_PATH on $BENCHMARK as results already exist in $OUTPUT_PATH"
    exit 0
fi

echo "--------------------------------"
echo "Evaluating model: ${MODEL_PATH}"
echo "Evaluating tasks: $BENCHMARK"
echo "Saving in: $OUTPUT_PATH"
echo "HF_HOME: $HF_HOME"
echo "--------------------------------"

# === Load environment ===
module load cuda/12.2
source ~/miniconda3/etc/profile.d/conda.sh
conda activate lmms_eval

#export HF_HOME=${SCRATCH}/.cache/huggingface
export WANDB_MODE="offline"
export LMMS_EVAL_USE_CACHE=True
MODEL_SUBPATH=${MODEL_PATH#*checkpoints/}
export LMMS_EVAL_HOME=$(realpath lmms-eval)/eval_cache/${MODEL_SUBPATH}

# === Run the evaluation ===
cd lmms-eval
python -m lmms_eval \
    --model vllm \
    --model_args model=${MODEL_PATH},gpu_memory_utilization=0.85 \
    --tasks ${BENCHMARK} \
    --batch_size 4 --log_samples --log_samples_suffix vllm --output_path ${OUTPUT_PATH} \

#--limit 16
