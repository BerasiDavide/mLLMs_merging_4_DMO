#!/bin/bash

#SBATCH --job-name=export_model
#SBATCH --nodes=1
#SBATCH --gres=gpu:0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH -t 0:30:00
#SBATCH -o logs/export_model/%A/%A_%a_%x.out
#SBATCH -e logs/export_model/%A/%A_%a_%x.err

# This scripts prepares a model for eval with lmms_eval: Merge base + adapter and convert internvl models to custom format

CONFIG_NAME=$1 # eg. intern35_2b_lora
ADAPTER_PATH=$2 # eg. checkpoints/sft_models/exp_name/modelname_expert_ocr-102400
EXPORTED_PATH=$3 # eg. checkpoints/exported_models/sft_models/exp_name/modelname_expert_ocr-102400

OVERWRITE=0
# Keep only the fist 3 parts of CONFIG_NAME
CONFIG_NAME=$(echo ${CONFIG_NAME} | cut -d'_' -f1-3)


# Check paths
if [ ! -d "${ADAPTER_PATH}" ]; then
    echo "Adapter path ${ADAPTER_PATH} does not exist. Exiting."
    exit 1
fi
if [ -d "${EXPORTED_PATH}" ] && [ $OVERWRITE -eq 0 ]; then
    echo "Exported model already exists at ${EXPORTED_PATH}. Skipping export."
    exit 0
fi

echo "--------------------------------"
echo "Preparing model for evaluation..."
echo "Export config: ${CONFIG_NAME}"
echo "Adapter path: ${ADAPTER_PATH}"
echo "Exported path: ${EXPORTED_PATH}"
echo "--------------------------------"


# source ${WORK}/miniconda3/etc/profile.d/conda.sh
export WANDB_MODE="offline"

# === Fix QwenVL chat template mismatch ===
# The chat_template.jinja in the final finetuned model is not the same as the one in the base model. The one in the intermediate checkpoints is correct.
qwenvl_chat_template_path=
if [[ $CONFIG_NAME == "qwen"* ]]; then
    echo "Fixing QwenVL chat template mismatch..."
    cp utils/qwenvl_chat_template.jinja ${ADAPTER_PATH}/chat_template.jinja
fi

# === Convert LoRA to full model ===
# If ADAPTER_PATH folder contains adapter_config.json, then it's a LoRA adapter
if [ -f "${ADAPTER_PATH}/adapter_config.json" ]; then
    echo "Exporting LoRA adapter to full model format"
    cd LLaMA-Factory
    conda activate merge4DMO
    llamafactory-cli export ../configs/export_configs/${CONFIG_NAME}.yaml \
        adapter_name_or_path=${ADAPTER_PATH} \
        export_dir=${EXPORTED_PATH}
    cd ..
else
    # Create symlink in exported path to the ADAPTER_PATH (which is actually the full model)
    mkdir -p $(dirname ${EXPORTED_PATH})
    ln -s $(realpath ${ADAPTER_PATH}) ${EXPORTED_PATH}
fi

# === Convert InternVL models from HF to custom format ===
if [[ $CONFIG_NAME == *"intern"* ]]; then
    conda activate lmms_eval

    if [[ $CONFIG_NAME == "intern35_2b"* ]]; then
        REFERENCE_PATH=checkpoints/base_models/InternVL3_5-2B-Pretrained
    elif [[ $CONFIG_NAME == "intern35_8b"* ]]; then
        REFERENCE_PATH=checkpoints/base_models/InternVL3_5-8B-Pretrained
    else
        echo "Unknown InternVL model alias: $CONFIG_NAME"
        exit 1
    fi

    echo "Converting InternVL model from HF to custom format..."
    mv ${EXPORTED_PATH} ${EXPORTED_PATH}-HF
    python utils/internvl_hf2custom.py \
        --custom_path ${REFERENCE_PATH} \
        --hf_path ${EXPORTED_PATH}-HF \
        --save_path ${EXPORTED_PATH}
fi

echo "Model exported and ready for evaluation at: ${EXPORTED_PATH}"