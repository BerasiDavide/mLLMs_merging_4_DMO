#!/bin/bash

model=${1:-"qwen2_2b"}

output_folder="checkpoints/base_models"

if [ "$model" == "qwen2_2b" ]; then
    hf download --repo-type model Qwen/Qwen2-VL-2B --local-dir ${output_folder}/Qwen2-VL-2B
    mv ${output_folder}/Qwen2-VL-2B/chat_template.json ${output_folder}/Qwen2-VL-2B/chat_template_base.json
    hf download --repo-type model Qwen/Qwen2-VL-7B-Instruct chat_template.json --local-dir ${output_folder}/Qwen2-VL-2B/
elif [ "$model" == "qwen2_7b" ]; then
    hf download --repo-type model Qwen/Qwen2-VL-7B --local-dir ${output_folder}/Qwen2-VL-7B
    mv ${output_folder}/Qwen2-VL-7B/chat_template.json ${output_folder}/Qwen2-VL-7B/chat_template_base.json
    hf download --repo-type model Qwen/Qwen2-VL-7B-Instruct chat_template.json --local-dir ${output_folder}/Qwen2-VL-7B
else
    echo "Unsupported model: $model. Supported models are: qwen2_2b, qwen2_7b."
    exit 1
fi