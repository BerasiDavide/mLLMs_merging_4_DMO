#!/bin/bash

#SBATCH --job-name=eval_loss
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -t 0:30:00
#SBATCH -o logs/eval_loss/%A/%A_%a_%x.out
#SBATCH -e logs/eval_loss/%A/%A_%a_%x.err


EVAL_CONFIG=$1 # e.g., qwen2_2b_lora
MODEL_PATH=$2
EVAL_DATASET=$3
OUTPUT_FOLDER=$4

MODEL_PATH=$(realpath ${MODEL_PATH})
OUTPUT_FOLDER=$(realpath ${OUTPUT_FOLDER})

OVERWRITE=${5:-true}

# keep only the first two parts of EVAL_CONFIG
EVAL_CONFIG=$(echo $EVAL_CONFIG | cut -d'_' -f1-2)
TOKENIZED_PATH=$(realpath LLaMA-Factory)/cache_tokenized/${EVAL_CONFIG}/${EVAL_DATASET}
OUTPUT_PATH=${OUTPUT_FOLDER}/${EVAL_DATASET}

if [ -d "$OUTPUT_PATH" ] && [ "$OVERWRITE" != "true" ]; then
    echo "Output directory $OUTPUT_PATH already exists. Skipping evaluation."
    exit 0
fi

# Check if the path ${MODEL_PATH}-HF exists, and if so, use it instead. Needed for internvl exported models.
if [ -d "${MODEL_PATH}-HF" ]; then
    echo "Found Hugging Face formatted model at ${MODEL_PATH}-HF. Using this path."
    MODEL_PATH="${MODEL_PATH}-HF"
fi

# === Configuration ===
echo "Evaluating model: $MODEL_PATH"
echo "Evaluating tasks: $EVAL_DATASET"
echo "Saving in: $OUTPUT_PATH"
echo "Using tokenized path: $TOKENIZED_PATH"

if [[ $EVAL_DATASET == *","* ]]; then
    echo "Warning: EVAL_DATASET contains a comma. Please ensure this is correct."
    exit 1
fi
if [[ $EVAL_DATASET != *"val"* ]]; then
    echo "Warning: EVAL_DATASET does not contain 'val'. Please ensure this is correct."
fi

# === Load environment ===
module load cuda/12.2
source ~/miniconda3/etc/profile.d/conda.sh
conda activate merge4DMO

# === Run the evaluation ===
export WANDB_MODE="offline"
cd LLaMA-Factory
llamafactory-cli train ../configs/eval_configs/${EVAL_CONFIG}.yaml \
    model_name_or_path=${MODEL_PATH} \
    eval_dataset=${EVAL_DATASET} \
    output_dir=${OUTPUT_PATH} \
    tokenized_path=${TOKENIZED_PATH}