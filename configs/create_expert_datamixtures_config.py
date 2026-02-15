import argparse
import os
from math import floor
import numpy as np
import json
from utils.data_utils import get_dataset_info


JSONL_DATA_FOLDER = "LLaMA-Factory/data/jsonl_datasets"

# Define tasks and their corresponding datasets. Dataset names have to be defined in LLaMA-Factory/data/dataset_info.json
task2datasets = {
    # "chart": [
    #     "dvqa_2325k",
    #     "chartqa_28k",
    #     "chart2text_26k",
    #     "vistext_9k"
    # ],
    # "counting": [
    #     "clevr_700k",
    #     "tallyqa_250k"
    # ],
    # "general": [
    #     "vqav2_444k",
    #     "lnqa_302k",
    #     "gqa_72k",
    #     "aokvqa_17k",
    #     "visual7w_14k",
    #     "okvqa_9k"
    # ],
    # "generalv2": [
	#     "allava_laion_500k",
    #     "vqav2_444k",
    #     "lnqa_302k",
    #     "lvis_instruct4v_220k",
    #     "qalign_200k",
    #     "gqa_72k",
    #     "vizwiz_20k",
    #     "visual7w_14k",
    # ],
    # "generalv3": [
	#     "allava_laion_500k",
    #     "vqav2_444k",
    #     "lnqa_302k",
    #     "lvis_instruct4v_220k",
    #     "qalign_200k",
    #     "gqa_72k",
    #     "vizwiz_20k",
    #     "visual7w_14k",
    #     "okvqa_9k"
    # ],
    # "grounding": [
    #     "vg_86k",
    #     "refcocog_25k",
    # ],
    # "groundingInternvl": [
    #     "vgInternvl_86k",
    #     "refcocogInternvl_25k",
    # ],
    "groundingQwenvlnr": [
        "vgQwenvlnr_86k",
        "refcocogQwenvlnr_25k",
    ],
    "groundingInternvlnr": [
        "vgInternvlnr_86k",
        "refcocogInternvlnr_25k",
    ],
    # "math": [
    #     "geo_170k",
    #     "raven_42k",
    #     "geomverse_9k",
    #     "mathvision_3k",
    #     "intergps_1k"
    # ],
    # "mathv2": [
    #     "tallyqa_250k",
    #     "geo_170k",
    #     "raven_42k",
    #     "geomverse_9k",
    #     "mathvision_3k",
    #     "intergps_1k"
    # ],
    # "ocr": [
    #     "synthdog_modified_500k",
    #     "ocr_vqa_80k",
    #     "docvqa_39k",
    #     "textvqa_22k",
    #     "textcaps_22k",
    #     "llavar_20k",
    #     "st_vqa_17k",
    #     "rendered_text_10k",
    #     "infographic_vqa_2k"
    # ],
}

def make_as_uniform_as_possible(N: list, target_sum: int) -> list:
    sorting_idx = np.argsort(N) # Get the indices that would sort the array

    left = target_sum
    taken_N = [0] * len(N)
    for i, idx in enumerate(sorting_idx):   # iterate from smallest to biggest in N
        current_n = N[idx]
        max_size = floor(left / (len(N) - i))
        taken = min(current_n, max_size)
        taken_N[idx] = taken
        left -= taken

    if left > 0:
        print(f"Cannot reach {target_sum} samples as sum(N) = {sum(N)} < {target_sum}")
    return taken_N

def get_proportional_allocation(N: list, target_sum: int) -> list:
    # Ensures that the sum of the returned list is exactly target_sum
    total_size = sum(N)
    raw_allocations = [(n / total_size) * target_sum for n in N]
    floored_allocations = [floor(x) for x in raw_allocations]
    remainder = target_sum - sum(floored_allocations)
    fractional_parts = [x - floor(x) for x in raw_allocations]
    # Distribute the remaining samples based on the largest fractional parts
    for _ in range(remainder):
        max_index = fractional_parts.index(max(fractional_parts))
        floored_allocations[max_index] += 1
        fractional_parts[max_index] = 0  # So we don't pick it again
    return floored_allocations


if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--train-size", type=int, default=102400)
    parser.add_argument("--eval-size", type=int, default=2000)
    parser.add_argument("--tasks", type=str, nargs="+", default=None)
    parser.add_argument("--output-folder", type=str, default="configs/datamixtures_configs/exp_name/experts")
    parser.add_argument("--overwrite", action="store_true", help="Whether to overwrite existing config files")
    parser.add_argument("--distribution", default="uniform", type=str, help="Distribution to use for sampling datasets. Currently only 'uniform' is supported.")
    args = parser.parse_args()

    os.makedirs(args.output_folder, exist_ok=True)
    use_training_sets = True   # Whether to use training sets of the datasets

    if args.tasks is None:
        tasks = sorted(task2datasets.keys())   # Use all tasks if none specified
    else:
        tasks = sorted(args.tasks)   # Ensure consistent ordering
        
    task_size = args.train_size + args.eval_size

    for task in tasks:
        datasets = task2datasets[task]
        if use_training_sets:
            datasets = [f"{d}_train" for d in datasets]

        output_path = os.path.join(args.output_folder, f"expert_{task}-{args.train_size}.json")
        if (os.path.exists(output_path)) and (not args.overwrite):
            print(f"Config for task {task} already exists at {output_path}. Use --overwrite to overwrite it.")
            continue

        # Count samples per dataset
        dataset_sizes = []
        for dataset in datasets:
            dataset_info = get_dataset_info(dataset)
            file_path = os.path.join("LLaMA-Factory/data", dataset_info["file_name"])
            n = len(open(file_path).readlines())
            dataset_sizes.append(n)

        # Determine number of samples per dataset
        if args.distribution == "uniform":
            num_samples = make_as_uniform_as_possible(dataset_sizes, target_sum=task_size)
        elif args.distribution == "proportional":
            num_samples = get_proportional_allocation(dataset_sizes, target_sum=task_size)
        else:
            raise NotImplementedError(f"Distribution {args.distribution} not implemented.")
        config = {dataset: n for dataset, n in zip(datasets, num_samples)}

        # Save configs on file
        with open(output_path, "w") as f:
            f.write(json.dumps(config, indent=4))
        print(f"Config saved to {output_path}.    {sum(num_samples)} / {sum(dataset_sizes)} selected samples")