import os
import json
import random

from tqdm import tqdm
from utils.data_utils import read_all_jsonl, save_jsonl, add_to_dataset_info, get_dataset_info


def create_jsonl_from_config(config_path, args):

    mixture_id = os.path.basename(config_path).replace('.json', '')

    # Create mixture
    with open(config_path, 'r') as f:
        config = json.load(f)

    selected_samples = []
    for dataset, num_samples in tqdm(config.items(), desc=f"Creating mixture {mixture_id}"):
        
        dataset_info = get_dataset_info(dataset)

        file_path = os.path.join("LLaMA-Factory/data", dataset_info["file_name"])
        data = read_all_jsonl(file_path)
        samples = data[:num_samples]    # Assumes data is already shuffled
        
        selected_samples += samples

    # Shuffle samples
    random.seed(args.seed)
    random.shuffle(selected_samples)
    
    # Split into train and eval
    if args.eval_size > 0:
        train_samples = selected_samples[:-args.eval_size]
        eval_samples = selected_samples[-args.eval_size:]

        # Create a smaller eval set for faster evaluation during training
        eval_small_size = min(500, len(eval_samples))
        eval_small_samples = eval_samples[:eval_small_size]

    else:
        train_samples = selected_samples
        eval_samples = []

    # Save mixture
    file_name = f"jsonl_mixtures/{mixture_id}.jsonl"
    output_path = os.path.join("LLaMA-Factory/data", file_name)
    save_jsonl(train_samples, output_path, overwrite=args.overwrite)
    add_to_dataset_info(mixture_id, file_name)

    if len(eval_samples) > 0:
        file_name_eval = f"jsonl_mixtures/{mixture_id}_eval.jsonl"
        eval_output_path = os.path.join("LLaMA-Factory/data", file_name_eval)
        save_jsonl(eval_samples, eval_output_path, overwrite=args.overwrite)
        add_to_dataset_info(f"{mixture_id}_eval", file_name_eval)

        file_name_eval_small = f"jsonl_mixtures/{mixture_id}_eval_small.jsonl"
        eval_small_output_path = os.path.join("LLaMA-Factory/data", file_name_eval_small)
        save_jsonl(eval_small_samples, eval_small_output_path, overwrite=args.overwrite)
        add_to_dataset_info(f"{mixture_id}_eval_small", file_name_eval_small)

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Create mixtures.")
    parser.add_argument('--datamixture-config', type=str, required=True, help='Path to a datamixture config json file, or a folder containing multiple json files.')
    parser.add_argument('--eval-size', type=int, default=0, help='If > 0, splits the mixture into train and eval sets with this size for eval.')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--overwrite', action='store_true', help='Whether to overwrite existing files.')
    args = parser.parse_args()

    os.makedirs("LLaMA-Factory/data/jsonl_mixtures", exist_ok=True)
    
    if ".json" in args.datamixture_config:
        create_jsonl_from_config(args.datamixture_config, args)
    else: # assume it's a folder
        config_paths = [os.path.join(args.datamixture_config, f) for f in os.listdir(args.datamixture_config) if f.endswith('.json')]
        for config_path in config_paths:
            create_jsonl_from_config(config_path, args)