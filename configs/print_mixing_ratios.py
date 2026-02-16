import argparse
import os
import numpy as np

from itertools import product


def get_mixture_id(tasks, task_sizes) -> str:
    return "++".join([f"{task}-{size}" for task, size in zip(tasks, task_sizes) if size > 0])

if __name__ == "__main__":

    parser = argparse.ArgumentParser()
    parser.add_argument("--task-train-size", type=int, default=102400)
    parser.add_argument("--eval-size", type=int, default=0)
    parser.add_argument("--ntasks", type=int, default=4)
    parser.add_argument("--nsteps", type=int, default=8)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--nmixtures", type=int, default=30)

    args = parser.parse_args()

    ntasks = args.ntasks
    task_max_size = args.task_train_size

    if args.nmixtures is None:
        ratios = [i / args.nsteps for i in range(args.nsteps + 1)]
        mixture_ratios_list = [m for m in product(ratios, repeat=ntasks) if (sum(m)==1.0 and min(m) > 0)]
        print(f"Created deterministic grid of {len(mixture_ratios_list)} mixtures")

    elif args.nmixtures > 0:
        # Sample random mixtures from Dirichlet
        alpha = [1.0] * ntasks  # Uniform Dirichlet
        rng = np.random.default_rng(seed=args.seed)
        mixture_ratios_list = rng.dirichlet(alpha, size=args.nmixtures).tolist()
        # keep 4 decimal places and guarantee sum to 1.0
        for i in range(len(mixture_ratios_list)):
            r = mixture_ratios_list[i]
            r = [round(v, 4) for v in r]
            diff = 1.0 - sum(r)
            r[-1] += diff  # adjust the last element to ensure sum to 1.0
            mixture_ratios_list[i] = r
        print(f"Created random sampling of {len(mixture_ratios_list)} mixtures from Dirichlet distribution")

    # Print the mixture ratios
    for r in mixture_ratios_list:    # ex: '0.125,0.125,0.75'
        print("\"" + ','.join(map(str, r)) + "\"")

    # Print uniform
    uniform_ratios = [1.0 / ntasks] * ntasks
    # Ensure 4 decimal places
    uniform_ratios = [round(v, 4) for v in uniform_ratios]
    diff = 1.0 - sum(uniform_ratios)
    uniform_ratios[-1] += diff  # adjust the first element to ensure sum to 1.0
    print("Uniform ratios:")
    print("\"" + ','.join(map(str, uniform_ratios)) + "\"")


    # print("\n")
    # for ratios in mixture_ratios_list:
    #     task_sizes = [floor(r * task_max_size) for r in ratios]
    #     mixture_id = get_mixture_id(tasks, task_sizes)
    #     print("\"" + mixture_id + "\"")