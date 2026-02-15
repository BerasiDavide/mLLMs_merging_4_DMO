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
[![🤗 Model (HuggingFace)](https://img.shields.io/badge/Model-HuggingFace-FFD21E.svg?logo=huggingface&logoColor=yellow)](https://huggingface.co/collections/daviBera/mllms-merging-4-dmo)

</h2>

>**Abstract.**
*Selecting the best data mixture is critical for successful Supervised Fine-Tuning (SFT) of Multimodal Large Language Models. However, determining the optimal mixture weights across multiple domain-specific datasets remains a significant bottleneck due to the combinatorial search space and the high cost associated with even a single training run. This is the so-called Data Mixture Optimization (DMO) problem. On the other hand, model merging unifies domain-specific experts through parameter interpolation. This strategy is efficient, as it only requires a single training run per domain, yet oftentimes leads to suboptimal models. In this work, we take the best of both worlds, studying model merging as an efficient strategy for estimating the performance of different data mixtures. We train domain-specific multimodal experts and evaluate their weighted parameter-space combinations to estimate the efficacy of corresponding data mixtures. We conduct extensive experiments on 14 multimodal benchmarks, and empirically demonstrate that the merged proxy models exhibit a high rank correlation with models trained on actual data mixtures. This decouples the search for optimal mixtures from the resource-intensive training process, thereby providing a scalable and efficient strategy for navigating the complex landscape of mixture weights.*


# 📖 Overview

This repo contains the official implementation of our paper: [Linear Model Merging Unlocks Simple and Scalable Multimodal Data Mixture Optimization](https://www.arxiv.org/pdf/2602.04937).

**TL;DR:**
We study model merging as an efficient proxy for Data Mixture Optimization (DMO) in multimodal LLM supervised fine-tuning.
Instead of training models on many data mixtures, we merge domain-specific experts and use merged models to estimate mixture performance.

### Training and Evaluation
We use [LLamaFactory](https://github.com/hiyouga/LlamaFactory) for training and [lmms-eval](https://github.com/EvolvingLMMs-Lab/lmms-eval) for evaluation. We consider base models from the `Qwen2-VL` and `Intern3.5-VL` families.

### Available models
We release **>200** trained checkpoints on 🤗 [Huggingface](https://huggingface.co/collections/daviBera/mllms-merging-4-dmo). (Not all available yet)

We evaluate 7+21+20=48 different data mixtures of 100k samples:
- 7 two-domains mixtures (GeneralVQA + OCR)
- 21 three-domains mixtures (GeneralVQA + OCR + Counting)
- 20 four-domains mixtures (GeneralVQA + OCR + Counting + Chart)

The corresponding mixture-finetuned models and the domain-experts are available for: Qwen2-VL-2B, Qwen2-VL-7B, InternVL3_5-2B, InternVL3_5-8B.

### 🚀 Progress:

- [x] Upload models on HF hub (experts, mixture-sft)
- [x] Add merging script
- [x] Add evaluation scripts
- [ ] Add script to print results
- [ ] Upload datasets on HF hub
- [ ] Add scripts for data preparation
- [x] Add training scripts (experts, mixture-sft)


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
cd ..
```

# 📦 Model preparation
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
> [!TIP]
> If merging models sequentially is too slow, you can check `scripts/merged4/merge_merged4_array.sh` to merge different models with a batch of SLURM jobs.

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