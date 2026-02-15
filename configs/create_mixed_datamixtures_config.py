'''Creates jsonl mixtures from a jsonl files.'''

import argparse
import os
import numpy as np
import json

from math import floor
from tqdm import tqdm
from itertools import product

TASKS_NAMES = [
    "chart",
    "counting",
    "general",
    "grounding",
    "ocr"
]

def get_mixture_id(tasks, task_sizes) -> str:
    return "++".join([f"{task}-{size}" for task, size in zip(tasks, task_sizes) if size > 0])

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--tasks", type=str, nargs="+", default=["grounding", "ocr"])
    parser.add_argument("--task-size", type=int, default=12_800)
    parser.add_argument("--eval-size", type=int, default=1000)
    parser.add_argument("--num_ticks", default=8, help="e.g. 8 -> [0, 0.125, 0.25, ..., 1.0]")
    parser.add_argument("--grid_subset", type=str, default='diagonal') # 'diagonal' or 'full'
    parser.add_argument("--include_borders", action='store_true', help="Whether to include configs with 0% of a task.")
    parser.add_argument("--output-folder", type=str, default="configs/datamixtures_configs/exp_name")

    args = parser.parse_args()
    os.makedirs(args.output_folder, exist_ok=True)

    tasks = sorted(args.tasks)   # Ensure consistent ordering
    tasks_ids = [f"expert_{task}-{args.task_size}" for task in tasks]

    # Create grid configurations
    grid_ticks = [i / args.num_ticks for i in range(args.num_ticks + 1)]
    grid = product(grid_ticks, repeat=len(tasks))
    if args.grid_subset == 'diagonal':
        configs = [m for m in grid if (sum(m)==1.0)]
    elif args.grid_subset == 'full':
        configs = [m for m in grid if sum(m)==1.0]
    else:
        raise ValueError(f"Unknown grid_subset {args.grid_subset}")

    if not args.include_borders: # exclude configs with 0% of a task
        configs = [m for m in configs if min(m) > 0]

    # Save configs on file
    for config in configs:

        # Create config dict
        config_sizes = [int(ratio * args.task_size) for ratio in config]
        mixed_config = {task_id: size for task_id, size in zip(tasks_ids, config_sizes)}

        # Save configs on file.
        mixture_id = get_mixture_id(tasks, config_sizes)
        file_name = os.path.join(args.output_folder, f"mixed_{mixture_id}.json")
        with open(file_name, "w") as f:
            f.write(json.dumps(mixed_config, indent=4))

        if args.eval_size > 0:
            # Create eval config
            eval_config_sizes = [int(ratio * args.eval_size) for ratio in config]
            mixed_config_eval = {f"{task_id}_eval": size for task_id, size in zip(tasks_ids, eval_config_sizes)}
            
            # Save configs on file.
            file_name = os.path.join(args.output_folder, f"mixed_{mixture_id}_eval.json")
            with open(file_name, "w") as f:
                f.write(json.dumps(mixed_config_eval, indent=4))