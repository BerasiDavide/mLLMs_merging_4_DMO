BASE_MODELS_ROOT = "checkpoints/base_models"
SAVES_ROOT = "checkpoints/sft_models"
MERGED_MODELS_ROOT = "checkpoints/merged_models"

BASE_MODEL_PATHS = {
    # Qwen2-VL
    'qwen2_2b': 'checkpoints/base_models/Qwen2-VL-2B',
    'qwen2instr_2b': 'checkpoints/base_models/Qwen2-VL-2B-Instruct',
    'qwen2_7b': 'checkpoints/base_models/Qwen2-VL-7B',
    'qwen2instr_7b': 'checkpoints/base_models/Qwen2-VL-7B-Instruct',
    # InternVL3_5
    'intern35_2b': 'checkpoints/base_models/InternVL3_5-2B-Pretrained-HF',
    'intern35instr_2b': 'checkpoints/base_models/InternVL3_5-2B-Instruct-HF',
    'intern35_8b': 'checkpoints/base_models/InternVL3_5-8B-Pretrained-HF',
    'intern35instr_8b': 'checkpoints/base_models/InternVL3_5-8B-Instruct-HF',
}