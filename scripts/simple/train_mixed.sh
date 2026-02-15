#!/bin/bash

#SBATCH --job-name=train
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH -t 06:00:00
#SBATCH -o logs/train/%A/%A_%a_%x.out
#SBATCH -e logs/train/%A/%A_%a_%x.err


TRAIN_CONFIG=$1
DATASET_NAME=$2 # eg. expert_ocr-102400,expert_general-102400
MAX_STEPS=$3
WEIGHTS=$4 # eg. 0.25,0.75
OUTPUT_PATH=$5

TRAIN_DATASET_NAME=${DATASET_NAME}
# DATASET_NAME components are separated by commas, add _eval_small to each component for eval dataset
EVAL_DATASET_NAME=$(echo ${DATASET_NAME} | sed 's/,/_eval_small,/g')_eval_small

# Terminate if output path already exists. Exit code as if job correctly completed.
if [ -d "$OUTPUT_PATH" ]; then
  echo "SKIPPING TRAINING: output path $OUTPUT_PATH already exists."
  exit 0
fi

# === Print configuration ===
echo Training config: ${TRAIN_CONFIG}
echo Training with dataset: ${DATASET_NAME}
echo Saving outputs to: ${OUTPUT_PATH}

# === Load environment ===
# module load cuda/12.2
# source ${WORK}/miniconda3/etc/profile.d/conda.sh
# conda activate merge4DMO

# === Run the training ===
export WANDB_MODE="offline"
cd LLaMA-Factory
FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=0,1,2,3 llamafactory-cli train training_configs/${TRAIN_CONFIG}.yaml \
    dataset=${TRAIN_DATASET_NAME} \
    mix_strategy=interleave_under \
    interleave_probs=${WEIGHTS} \
    eval_dataset=${EVAL_DATASET_NAME} \
    output_dir=${OUTPUT_PATH} \
    run_name=${TRAIN_CONFIG}_${DATASET_NAME} \
    max_steps=${MAX_STEPS} \
    save_steps=10000