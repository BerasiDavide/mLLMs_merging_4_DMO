import os
import json
import random

from tqdm import tqdm


DATASET_INFO_PATH = "LLaMA-Factory/data/dataset_info.json"


def read_all_jsonl(file_path):
    data = []
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data


def add_to_dataset_info(dataset_name, file_name):
    
    info = {
        "file_name": file_name,
        "formatting": "sharegpt",
        "columns": {
            "messages": "conversations",
            "images": "image"
            }
    }

    with open(DATASET_INFO_PATH, "r") as file:
        info_data = json.load(file)

    if dataset_name in info_data:
        print(f"*** Warning ***: {dataset_name} already exists in dataset_info.json. It will be updated!")

    info_data[dataset_name] = info
    with open(DATASET_INFO_PATH, "w") as file:
        json.dump(info_data, file, indent=4)
    print(f"Updated dataset_info.json with {dataset_name}")


def save_jsonl(samples, output_path, overwrite=True):
    if os.path.exists(output_path):
        if overwrite:
            print(f"File {output_path} already exists. It will be overwritten.")
        else:
            print(f"File {output_path} already exists. Exiting creation.")
            return
            
    with open(output_path, 'w', encoding='utf-8') as f:
        for sample in samples:
            json.dump(sample, f, ensure_ascii=False) # Allow utf-8 characters (ex chinese)
            f.write('\n')
    print(f"Saved jsonl to {output_path}")


def get_dataset_info(dataset_name):
    with open(DATASET_INFO_PATH, "r") as file:
        info_data = json.load(file)
    if dataset_name not in info_data:
        raise ValueError(f"Dataset {dataset_name} not found in dataset_info.json")
    return info_data[dataset_name]