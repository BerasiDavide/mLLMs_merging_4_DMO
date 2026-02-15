#!/bin/bash

#SBATCH --job-name=train_mixed
#SBATCH --account=iscrb_smiallm
#SBATCH -p boost_usr_prod
#SBATCH --nodes=1
#SBATCH --gres=gpu:4
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=8
#SBATCH -t 1:00:00
#SBATCH --array=0-21
#SBATCH -o logs/train_mixed/%A/%A_%a_%x.out
#SBATCH -e logs/train_mixed/%A/%A_%a_%x.err


BASE_MODEL=$1 # e.g. qwen2_2b
DOM1=$2 # e.g. chart
DOM2=$3 # e.g. counting
DOM3=$4 # e.g. general
DOM4=$5 # e.g. ocr
SFT_STRATEGY=${6:-lora} # lora, full
MAX_STEPS=${7:-800} # 80, 400, 800

MIXTURE_THRESHOLD=${8:-102400} # Num samples in training mixture

N_SAMPLES=$((MAX_STEPS * 128))
BUDGET_ID=$((MAX_STEPS / 8))k # 10k, 50k, 100k
EXP_NAME=${9:-exp_${BASE_MODEL}_${BUDGET_ID}}

# Sampled from Dirichlet distribution with alpha=1.0. See configs/print_names_mixed_configs.py
MIXTURE_WEIGHTS=(
  "0.3247,0.3155,0.322,0.0378"
  "0.0142,0.2392,0.2322,0.5144"
  "0.0347,0.458,0.0308,0.4765"
  "0.4942,0.1104,0.3515,0.0439"
  "0.0532,0.1831,0.5237,0.24"
  "0.275,0.0493,0.4052,0.2705"
  "0.409,0.2601,0.2827,0.0482"
  "0.0713,0.2722,0.1544,0.5021"
  "0.3458,0.1161,0.225,0.3131"
  "0.1867,0.1748,0.4773,0.1612"
  "0.3436,0.3583,0.2793,0.0188"
  "0.2915,0.3481,0.2884,0.0720"
  "0.3722,0.1922,0.411,0.0246"
  "0.0277,0.2177,0.6614,0.0932"
  "0.2422,0.3392,0.3081,0.1105"
  "0.1168,0.1483,0.1625,0.5724"
  "0.391,0.3053,0.1937,0.11"
  "0.0142,0.3056,0.1985,0.4817"
  "0.6082,0.0996,0.0149,0.2773"
  "0.408,0.1312,0.3405,0.1203"
  "0.25,0.25,0.25,0.25"
  )


# === Configuration ===
DATASET_NAME=expert_${DOM1}-${MIXTURE_THRESHOLD},expert_${DOM2}-${MIXTURE_THRESHOLD},expert_${DOM3}-${MIXTURE_THRESHOLD},expert_${DOM4}-${MIXTURE_THRESHOLD}
  
CONFIG_ID=${SLURM_ARRAY_TASK_ID}
WEIGHTS=${MIXTURE_WEIGHTS[CONFIG_ID]}

N1=$(printf "%.0f" $(echo "$(echo $WEIGHTS | cut -d',' -f1) * $N_SAMPLES" | bc -l))
N2=$(printf "%.0f" $(echo "$(echo $WEIGHTS | cut -d',' -f2) * $N_SAMPLES" | bc -l))
N3=$(printf "%.0f" $(echo "$(echo $WEIGHTS | cut -d',' -f3) * $N_SAMPLES" | bc -l))
N4=$(printf "%.0f" $(echo "$(echo $WEIGHTS | cut -d',' -f4) * $N_SAMPLES" | bc -l))
MIXTURE_ID="mixed_${DOM1}-${N1}++${DOM2}-${N2}++${DOM3}-${N3}++${DOM4}-${N4}"
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
  # LOG_PATH=logs/${EXP_NAME}/export_mixed4/%A_%x/%A_%a_%x
  # sbatch --job-name=export_${BASE_MODEL}_${SFT_STRATEGY}_${DOM1}_${DOM2}_${DOM3}_${DOM4} \
  #   --output=$LOG_PATH.out --error=$LOG_PATH.err \
  #   scripts/simple/export.sh ${BASE_MODEL}_${SFT_STRATEGY} ${FINETUNED_PATH} ${EXPORTED_PATH}
fi