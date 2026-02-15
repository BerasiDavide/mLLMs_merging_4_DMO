#!/bin/bash

#SBATCH --job-name=eval_bench_experts
#SBATCH --nodes=1
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --array=0-13
#SBATCH -t 2:00:00
#SBATCH -o logs/eval_bench/%A/%A_%a_%x.out
#SBATCH -e logs/eval_bench/%A/%A_%a_%x.err

MODEL_PATH=$1
BENCHMARKS=$2 # List of benchmarks separated by commas
OUTPUT_FOLDER=$3

# Get the benchmark for this array task
IDX=${SLURM_ARRAY_TASK_ID}
BENCHMARK=$(echo $BENCHMARKS | cut -d',' -f$((IDX + 1)))

bash scripts/simple/eval_bench.sh ${MODEL_PATH} ${BENCHMARK} ${OUTPUT_FOLDER}
