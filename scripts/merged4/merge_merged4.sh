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
DOM4=$5 # e.g. chart
SFT_STRATEGY=${6:-lora} # lora, full
STEPS=${7:-800} # 80, 400, 800
METHOD=${8:-task_arithmetic}
MERGED_TYPE=${9:-merged}

N_SAMPLES=$((STEPS * 128))
BUDGET_ID=$((STEPS / 8))k # 10k, 50k, 100k
EXP_NAME=${10:-exp_${BASE_MODEL}_${BUDGET_ID}}

OVERWRITE=0

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


# === Load environment ===
# source ${WORK}/miniconda3/etc/profile.d/conda.sh
# conda activate merge4DMO

for J in {0..20}
do
    # === Configuration ===
    WEIGHTS=${MIXTURE_WEIGHTS[J]}
    N1=$(printf "%.0f" $(echo "$(echo $WEIGHTS | cut -d',' -f1) * $N_SAMPLES" | bc -l))
    N2=$(printf "%.0f" $(echo "$(echo $WEIGHTS | cut -d',' -f2) * $N_SAMPLES" | bc -l))
    N3=$(printf "%.0f" $(echo "$(echo $WEIGHTS | cut -d',' -f3) * $N_SAMPLES" | bc -l))
    N4=$(printf "%.0f" $(echo "$(echo $WEIGHTS | cut -d',' -f4) * $N_SAMPLES" | bc -l))

    MIXTURE_ID="${DOM1}-${N1}++${DOM2}-${N2}++${DOM3}-${N3}++${DOM4}-${N4}"
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