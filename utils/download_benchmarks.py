import os
from datasets import load_dataset
from huggingface_hub.utils import HfHubHTTPError
import time

def download_dataset(path, name=None, split=None, retries=5):
    for attempt in range(retries):
        try:
            print(f"--- Downloading: {path} {f'| {name}' if name else ''} {f'| split: {split}' if split else ''} ---")
            ds = load_dataset(path, name, split=split)
            print(f"✅ Downloaded {path}.")
            time.sleep(2)
            return
        except HfHubHTTPError as e:
            if e.response is not None and e.response.status_code in {502, 503, 504}:
                wait = 2 ** attempt
                print(f"⚠️ Hub error {e.response.status_code}, retrying in {wait}s...")
                time.sleep(wait)
            else:
                raise
    raise RuntimeError(f"Failed after {retries} retries: {path} / {name}")


print("HF cache set to:", os.environ.get("HF_HOME", "(default)"))

download_dataset("lmms-lab/GQA", "testdev_balanced_images", split="testdev")
download_dataset("lmms-lab/GQA", "testdev_balanced_instructions", split="testdev")
download_dataset("lmms-lab/OK-VQA")
download_dataset("lmms-lab/LMMs-Eval-Lite", "vqav2_val")
download_dataset("lmms-lab/VizWiz-VQA", split="val")
download_dataset("lmms-lab/DocVQA", "DocVQA", split="validation")
download_dataset("lmms-lab/DocVQA", "InfographicVQA", split="validation")
download_dataset("lmms-lab/textvqa", split="test")
download_dataset("lmms-lab/MME", split="test")
download_dataset("echo840/OCRBench", split="test")
download_dataset("nyu-visionx/CV-Bench", "2D", split="test")
download_dataset("lmms-lab/POPE", split="test")
download_dataset("lmms-lab/ChartQA", split="test")
download_dataset("suyc21/VMCBench", split="dev")
download_dataset("Lin-Chen/MMStar", split="val")
