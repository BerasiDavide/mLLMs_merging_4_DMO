#!/bin/bash

#SBATCH --job-name=merge_merged
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --mem=128G
#SBATCH --gres=gpu:0
#SBATCH -t 1:00:00
#SBATCH -o logs/merge_merged/%A/%A_%a_%x.out
#SBATCH -e logs/merge_merged/%A/%A_%a_%x.err


BASE_MODEL=$1 # e.g. qwen2_2b
DOM1=$2 # e.g. counting
DOM2=$3 # e.g. general
DOM3=$4 # e.g. ocr
SFT_STRATEGY=${5:-lora} # lora, full
STEPS=${6:-800} # 80, 400, 800
METHOD=${7:-task_arithmetic}
MERGED_TYPE=${8:-merged}

N_SAMPLES=$((STEPS * 128))
BUDGET_ID=$((STEPS / 8))k # 10k, 50k, 100k
EXP_NAME=${9:-exp_${BASE_MODEL}_${BUDGET_ID}}

OVERWRITE=0

MIXTURE_WEIGHTS=(
  "0.125,0.125,0.75"
  "0.125,0.25,0.625"
  "0.125,0.375,0.5"
  "0.125,0.5,0.375"
  "0.125,0.625,0.25"
  "0.125,0.75,0.125"
  "0.25,0.125,0.625"
  "0.25,0.25,0.5"
  "0.25,0.375,0.375"
  "0.25,0.5,0.25"
  "0.25,0.625,0.125"
  "0.375,0.125,0.5"
  "0.375,0.25,0.375"
  "0.375,0.375,0.25"
  "0.375,0.5,0.125"
  "0.5,0.125,0.375"
  "0.5,0.25,0.25"
  "0.5,0.375,0.125"
  "0.625,0.125,0.25"
  "0.625,0.25,0.125"
  "0.75,0.125,0.125"
  "0.333333,0.333333,0.333334"
  )


# === Load environment ===
# source ${WORK}/miniconda3/etc/profile.d/conda.sh
# conda activate merge4DMO

for J in {0..21};
do
    # === Configuration ===
    WEIGHTS=${MIXTURE_WEIGHTS[J]}
    N1=$(printf "%.0f" $(echo "$(echo $WEIGHTS | cut -d',' -f1) * $N_SAMPLES" | bc -l))
    N2=$(printf "%.0f" $(echo "$(echo $WEIGHTS | cut -d',' -f2) * $N_SAMPLES" | bc -l))
    N3=$(printf "%.0f" $(echo "$(echo $WEIGHTS | cut -d',' -f3) * $N_SAMPLES" | bc -l))
    MIXTURE_ID="${DOM1}-${N1}++${DOM2}-${N2}++${DOM3}-${N3}"
    MERGED_ID=${BASE_MODEL}_${SFT_STRATEGY}_${MERGED_TYPE}_${MIXTURE_ID}--${METHOD}

    # === Create merged model ===
    MODEL_PATH=checkpoints/merged_models/${EXP_NAME}/${MERGED_ID}
    if [ $OVERWRITE -eq 1 ]; then
        rm -rf "$MODEL_PATH"
    elif [ -d "$MODEL_PATH" ]; then
        echo "Model $MODEL_PATH already exists. Skipping..."
        continue
    fi
    python model_merging.py --config ${MERGED_ID} \
        --expert-models-folder checkpoints/exported_models/sft_models/${EXP_NAME}/ \
        --merged-models-folder checkpoints/merged_models/${EXP_NAME}/ \
        --expert-size ${N_SAMPLES} \
        --dtype float32 \
        --params_set ${SFT_STRATEGY}

    # === Export merged model ===
    EXPORT_PATH=$(realpath checkpoints)/exported_models/merged_models/${EXP_NAME}/${MERGED_ID}
    bash scripts/simple/export.sh ${BASE_MODEL}_${SFT_STRATEGY} ${MODEL_PATH} ${EXPORT_PATH}
done