#!/bin/bash

#SBATCH --job-name=eval_bench_experts
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -t 2:00:00
#SBATCH -o logs/eval_bench/%A/%A_%a_%x.out
#SBATCH -e logs/eval_bench/%A/%A_%a_%x.err

BASE_MODEL_ALIAS=$1
MODEL_PATH=$2
EVAL_DATASETS=$3 # Comma-separated list of datasets
OUTPUT_FOLDER=$4

# Get the benchmark for this array task
IDX=${SLURM_ARRAY_TASK_ID}
EVAL_DATASET=$(echo $EVAL_DATASETS | cut -d',' -f$((IDX + 1)))

bash scripts/simple/eval_loss.sh ${BASE_MODEL_ALIAS} ${MODEL_PATH} ${EVAL_DATASET} ${OUTPUT_FOLDER}