'''
Script to print the accuracies of mixed and merged models.
'''

import pandas as pd

from argparse import ArgumentParser
from utils.eval_bench_utils import weights2, weights3, weights4
from utils.eval_bench_utils import get_bench

# Mix configs
mix_configs2 = [{'general': w1, 'ocr': w2} for w1, w2 in weights2]
mix_configs3 = [{'counting': w1, 'general': w2, 'ocr': w3} for w1, w2, w3 in weights3]
mix_configs4 = [{'chart': w1, 'counting': w2, 'general': w3, 'ocr': w4} for w1, w2, w3, w4 in weights4]
mix_configs = mix_configs2 + mix_configs3 + mix_configs4

benchmarks = [
    'gqa',
    'vqav2_val_lite',
    'vizwiz_vqa_val',
    'ok_vqa_val2014',
    'textvqa_val',
    'ocrbench',
    'docvqa_val',
    'infovqa_val',
    'cv_bench_2d-count',
    'pope',
    'chartqa',
    'mme',
    'vmcbench',
    'mmstar',
    ]

model_types = [
    'mixed',
    'merged-task_arithmetic'
]

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument('--model', type=str, default='qwen2_2b')
    parser.add_argument('--sft_strategy', type=str, default='lora')
    parser.add_argument('--steps', type=int, default=800)
    args = parser.parse_args()

    model = args.model
    sft_strategy = args.sft_strategy
    steps = args.steps
    exp_args = {
        'exp_name': f'exp_{model}_{int(steps*0.125)}k',
        'base_model': model,
        'expert_size': steps * 128,
        'sft_strategy': sft_strategy
    }

    # Get results for each mix config and print in a table
    for model_type in model_types:
        table = [] # (models, benchmarks)
        for config in mix_configs:
            model_name = f'{model_type}-' + '-'.join([f"{domain[:3]}{int(weight*100)}" for domain, weight in config.items()])
            accuracies = get_bench(exp_args, config, method=model_type, benchmarks=benchmarks, aggregate=None)
            table.append([model_name] + accuracies)

        # Add average performance column
        df = pd.DataFrame(table)
        bench_names = [b[:8] for b in benchmarks]
        df.columns = ['Model'] + bench_names
        df.set_index('Model', inplace=True)
        df['Avg. Perf.'] = df[bench_names].mean(axis=1)

        # Print results
        print(f"\n*** Experiment: {exp_args['exp_name']}, Base Model: {exp_args['base_model']}, SFT Strategy: {exp_args['sft_strategy']}, Model Type: {model_type} ***")
        print(df.to_string(float_format=lambda x: f"{x * 100:.2f}", na_rep="-", justify="center"))