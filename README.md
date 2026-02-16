<h2 align="center">
Linear Model Merging Unlocks Simple and Scalable Multimodal Data Mixture Optimization
</h2>

<p align="center">
  <a href="https://openreview.net/profile?id=~Davide_Berasi1">Davide Berasi</a>,
  <a href="https://scholar.google.com/citations?user=SxQwDD8AAAAJ&authuser=1">Matteo Farina</a>, 
  <a href="https://scholar.google.com/citations?user=bqTPA8kAAAAJ&authuser=1">Massimiliano Mancini</a>,
  <a href="https://scholar.google.com/citations?user=xf1T870AAAAJ&authuser=1">Elisa Ricci</a>
</p>

<h2 align="center">

[![arXiv](https://img.shields.io/badge/arXiv-2602.04937-b31b1b.svg)](https://www.arxiv.org/pdf/2602.04937)
[![🤗 Model (HuggingFace)](https://img.shields.io/badge/Models-HuggingFace-FFD21E.svg?logo=huggingface&logoColor=yellow)](https://huggingface.co/collections/daviBera/mllms-merging-4-dmo)
[![🤗 Dataset (HuggingFace)](https://img.shields.io/badge/Datasets-HuggingFace-FFD21E.svg?logo=huggingface&logoColor=yellow)](https://huggingface.co/datasets/daviBera/experts_datasets-102400)

</h2>

>**Abstract.**
*Selecting the best data mixture is critical for successful Supervised Fine-Tuning (SFT) of Multimodal Large Language Models. However, determining the optimal mixture weights across multiple domain-specific datasets remains a significant bottleneck due to the combinatorial search space and the high cost associated with even a single training run. This is the so-called Data Mixture Optimization (DMO) problem. On the other hand, model merging unifies domain-specific experts through parameter interpolation. This strategy is efficient, as it only requires a single training run per domain, yet oftentimes leads to suboptimal models. In this work, we take the best of both worlds, studying model merging as an efficient strategy for estimating the performance of different data mixtures. We train domain-specific multimodal experts and evaluate their weighted parameter-space combinations to estimate the efficacy of corresponding data mixtures. We conduct extensive experiments on 14 multimodal benchmarks, and empirically demonstrate that the merged proxy models exhibit a high rank correlation with models trained on actual data mixtures. This decouples the search for optimal mixtures from the resource-intensive training process, thereby providing a scalable and efficient strategy for navigating the complex landscape of mixture weights.*


# 📖 Overview

This repo contains the official implementation of our paper: [Linear Model Merging Unlocks Simple and Scalable Multimodal Data Mixture Optimization](https://www.arxiv.org/pdf/2602.04937).

**TL;DR:**
We study model merging as an efficient proxy for Data Mixture Optimization (DMO) in multimodal LLM supervised fine-tuning.
Instead of training models on many data mixtures, we merge domain-specific experts and use merged models to estimate mixture performance.

### Training and Evaluation
We use [LLamaFactory](https://github.com/hiyouga/LlamaFactory) for training and [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) for evaluation.

### Available models
We release **>150** trained checkpoints on 🤗 [Huggingface](https://huggingface.co/collections/daviBera/mllms-merging-4-dmo).
These are mixture-finetuned and domain-expert checkpoints of Qwen2-VL-2B, Qwen2-VL-7B, InternVL3_5-2B, InternVL3_5-8B. 
In particular, we consider:
- 7 two-domains mixtures (GeneralVQA + OCR)
- 21 three-domains mixtures (GeneralVQA + OCR + Counting)
- 20 four-domains mixtures (GeneralVQA + OCR + Counting + Chart)

# ⚙️ Setup

Clone the repo and enter in it:
```
https://github.com/BerasiDavide/mLLMs_merging_4_DMO.git
cd mLLMs_merging_4_DMO
```
Create the conda environment and activate it:
```
conda create -n merge4DMO python=3.10
conda activate merge4DMO
cd LLaMA-Factory
pip install -e ".[torch,metrics,vllm]" --no-build-isolation
pip install timm
cd ..
```

# 🤖 Model preparation
> Note: Trained (or downloaded) LoRA adapters are then exported into full models. 

<details>
<summary><strong>💾 Storage requirements</strong></summary>

While LoRA adapters are lightweight, full models require significantly more disk space (in float16, 2B models are ~4.2 GB, 7B models are ~15 GB).

Therefore, you may want to symlink the following directories to a larger or temporary storage location (e.g., a scratch partition):

- `checkpoints/exported_models`
- `checkpoints/merged_models`

</details>

1. Download the base model:
```
bash utils/download_base_model.sh "qwen2_2b"   # qwen2_2b, qwen2_7b, intern35_2b, intern35_8b 
```
2. Download and prepare experts:
```
bash utils/download_expert_models.sh "qwen2_2b"
```
3. Merge Experts (Proxy Models):
```
bash scripts/merged2/merge_merged2.sh "qwen2_2b" "general" "ocr"
# bash scripts/merged3/merge_merged3.sh "qwen2_2b" "counting" "general" "ocr"
# bash scripts/merged4/merge_merged4.sh "qwen2_2b" "chart" "counting" "general" "ocr"
```
> **Tip:** You can check `scripts/merged4/merge_merged4_array.sh` to run a batch of Slurm jobs merging across mixing weights.

4. Download and prepare mixture-finetuned models:
```
bash utils/download_mixed_models.sh "qwen2_2b"
```
<!-- This script downloads models trained on 7 mixtures of two domains (General+OCR), 21 of three domains (General+OCR+Counting), 20 on four domains (General+OCR+Counting+OCR).  -->


# 📊 Evaluation

### ⚙️ Setup
Create the conda environment for evaluation:
```
cd lmms-eval
conda create -n lmms_eval -c conda-forge --override-channels
conda activate lmms_eval
pip install -e .
pip install httpx==0.23.3
pip3 install vllm qwen_vl_utils
cd ..
conda activate lmms_eval
```

Download the benchmarks (`gqa`, `vqav2_val_lite`, `vizwiz_vqa_val`, `ok_vqa_val2014`, `textvqa_val`, `ocrbench`, `docvqa_val`, `infovqa_val`, `cv_bench_2d`, `pope`, `chartqa`, `mme`, `vmcbench`, `mmstar`):
```
python utils/download_benchmarks.py`
```

### ▶️ Run evaluation
- Evaluate one model on one benchmark:
```
bash scripts/simple/eval_bench.sh $model_path $benchmark $output_folder
```
<!-- By default, we use the lmms-eval caching system ([link](https://github.com/EvolvingLMMs-Lab/lmms-eval/blob/main/docs/caching.md)) and cache model responses in `lmms-eval/eval_cache/`. Set `LMMS_EVAL_USE_CACHE=False` in the script to avoid caching. -->

- If you have access to a SLURM cluster, you can submit a batch of jobs evaluating merged-proxies and mixture-finetuned models on multiple benchmarks with:
```
bash scripts/job_launchers/launch_eval_bench.sh
```
- Print results:
```
python -m utils.print_results --model "qwen2_2b"
```

# 🔥​ Reproduce training

### 📚 Data preparation
- Every dataset has to be specified in `LLaMA-Factory/data/dataset_info.json`. See the [LLaMA-Factory docs](https://github.com/hiyouga/LlamaFactory/blob/main/data/README.md) for more information.
- You can download our domain-specific jsonl datasets with:
```
hf download --repo-type dataset daviBera/experts_datasets-102400 --local-dir "LLaMA-Factory/data/jsonl_mixtures"
```
- The images we use are included in [Cambrian-10M](https://huggingface.co/datasets/nyu-visionx/Cambrian-10M). We position all image folders in `LLaMA-Factory/data/image_datasets`. 

### ▶️ Run training
> **Note**: You may need to modify the *'Load Environment'* commands in the training scripts in `scripts/simple/` to use them as Slurm scripts.
- Train domain-experts:
```
bash scripts/experts/train_expert.sh "qwen2_2b" "general"
bash scripts/experts/train_expert.sh "qwen2_2b" "ocr"
bash scripts/experts/train_expert.sh "qwen2_2b" "counting"
bash scripts/experts/train_expert.sh "qwen2_2b" "chart"
```
- Train mixture-finetuned models on first mixing ratio:
```
export SLURM_ARRAY_TASK_ID=0
bash scripts/mixed2/train_mixed2.sh "qwen2_2b" "general" "ocr"
```
- Or launch a batch of Slurm jobs training for all mixing ratios:
```
sbatch scripts/mixed2/train_mixed2.sh "qwen2_2b" "general" "ocr"
```

<!-- ## 🚀 Progress:

- [x] Upload models on HF hub (experts, mixture-sft)
- [x] Add merging script
- [x] Add evaluation scripts
- [x] Add script to print results
- [x] Upload datasets on HF hub
- [x] Add training scripts (experts, mixture-sft) -->

## Citation
Please cite this work as follows if you find it useful!
```bibtex
@article{berasi2026linear,
  title={Linear Model Merging Unlocks Simple and Scalable Multimodal Data Mixture Optimization},
  author={Berasi, Davide and Farina, Matteo and Mancini, Massimiliano and Ricci, Elisa},
  journal={arXiv preprint arXiv:2602.04937},
  year={2026}
}
```

