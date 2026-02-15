
import os
import json
import numpy as np
from utils.model_paths import BASE_MODEL_PATHS

weights2 = [
    (0.125, 0.875),
    (0.25, 0.75),
    (0.375, 0.625),
    (0.5, 0.5),
    (0.625, 0.375),
    (0.75, 0.25),
    (0.875, 0.125),
]
weights3 = [
    (0.125, 0.125, 0.75),
    (0.125, 0.25, 0.625),
    (0.125, 0.375, 0.5),
    (0.125, 0.5, 0.375),
    (0.125, 0.625, 0.25),
    (0.125, 0.75, 0.125),
    (0.25, 0.125, 0.625),
    (0.25, 0.25, 0.5),
    (0.25, 0.375, 0.375),
    (0.25, 0.5, 0.25),
    (0.25, 0.625, 0.125),
    (0.375, 0.125, 0.5),
    (0.375, 0.25, 0.375),
    (0.375, 0.375, 0.25),
    (0.375, 0.5, 0.125),
    (0.5, 0.125, 0.375),
    (0.5, 0.25, 0.25),
    (0.5, 0.375, 0.125),
    (0.625, 0.125, 0.25),
    (0.625, 0.25, 0.125),
    (0.75, 0.125, 0.125),
]
weights4=[
  (0.3247,0.3155,0.322,0.0378),
  (0.0142,0.2392,0.2322,0.5144),
  (0.0347,0.458,0.0308,0.4765),
  (0.4942,0.1104,0.3515,0.0439),
  (0.0532,0.1831,0.5237,0.24),
  (0.275,0.0493,0.4052,0.2705),
  (0.409,0.2601,0.2827,0.0482),
  (0.0713,0.2722,0.1544,0.5021),
  (0.3458,0.1161,0.225,0.3131),
  (0.1867,0.1748,0.4773,0.1612),
  (0.3436,0.3583,0.2793,0.0188),
  (0.2915,0.3481,0.2884,0.0720),
  (0.3722,0.1922,0.411,0.0246),
  (0.0277,0.2177,0.6614,0.0932),
  (0.2422,0.3392,0.3081,0.1105),
  (0.1168,0.1483,0.1625,0.5724),
  (0.391,0.3053,0.1937,0.11),
  (0.0142,0.3056,0.1985,0.4817),
  (0.6082,0.0996,0.0149,0.2773),
  (0.408,0.1312,0.3405,0.1203),
  ]
weights2_edges = [(0.0, 1.0)] + weights2 + [(1.0, 0.0)]
weights3_edges = [(0.0, 0.0, 1.0)] + weights3[:6] + [(0.0, 1.0, 0.0)] + weights3[6:] + [(1.0, 0.0, 0.0)]

def benchmark_to_score_fn(benchmark, result_dict):
    if benchmark == "mme":
        mme_cognition_score = result_dict['mme_cognition_score,none']
        mme_perception_score = result_dict['mme_perception_score,none']
        max = 2800
        normalized_score = (mme_cognition_score + mme_perception_score) / max
        return normalized_score
    if benchmark == "mme_lite":
        mme_cognition_score = result_dict['mme_cognition_score,none']
        mme_perception_score = result_dict['mme_perception_score,none']
        max = 500 # if only 500 samples are used
        normalized_score = (mme_cognition_score + mme_perception_score) / max
        return normalized_score
    elif "gqa" in benchmark:
        return result_dict['exact_match,none']
    elif "ok_vqa" in benchmark:
        return result_dict['exact_match,none']
    elif "seedbench" in benchmark:
        return result_dict['seed_all,none']
    elif "vizwiz_vqa" in benchmark:
        return result_dict['exact_match,none']
    elif "vqav2" in benchmark:
        return result_dict['exact_match,none']
    elif "textvqa" in benchmark:
        return result_dict['exact_match,none']
    elif "docvqa" in benchmark:
        return result_dict['anls,none']
    elif "infovqa" in benchmark:
        return result_dict['anls,none']
    elif "chartqa" in benchmark:
        return result_dict['relaxed_overall,none']
    elif "ocrbench" in benchmark:
        return result_dict['ocrbench_accuracy,none']
    elif "scienceqa" in benchmark:
        return result_dict['exact_match,none']
    elif 'ai2d' in benchmark:
        return result_dict['exact_match,flexible-extract']
    elif 'vmcbench-avg' in benchmark or benchmark == 'vmcbench':
        return result_dict['average,none']
    elif 'vmcbench-reason' in benchmark:
        return result_dict['reason,none']
    elif 'vmcbench-general' in benchmark:
        return result_dict['general,none']
    elif 'vmcbench-doc' in benchmark:
        return result_dict['doc,none']
    elif 'vmcbench-ocr' in benchmark:
        return result_dict['ocr,none']
    elif 'pope' in benchmark:
        return result_dict['pope_accuracy,none']
    elif benchmark == "mmstar-coarse_perc":
        return result_dict['coarse perception,none']
    elif benchmark == "mmstar-fine_perc":
        return result_dict['fine-grained perception,none']
    elif benchmark == "mmstar-instance_reason":
        return result_dict['instance reasoning,none']
    elif benchmark == "mmstar-logical_reason":
        return result_dict['logical reasoning,none']
    elif benchmark == "mmstar-math":
        return result_dict['math,none']
    elif benchmark == "mmstar-science_tech":
        return result_dict['science & technology,none']
    elif benchmark == "mmstar-avg" or benchmark == "mmstar":
        return result_dict['average,none']
    elif benchmark == "pope-acc":
        return result_dict['pope_accuracy,none']
    elif benchmark == "pope-f1":
        return result_dict['pope_f1_score,none']
    elif benchmark == "pope-precision":
        return result_dict['pope_precision,none']
    elif benchmark == "pope-recall":
        return result_dict['pope_recall,none'] 

    elif benchmark == "mathvision_testmini":
        normalized_score = result_dict['mathvision_standard_eval,none'] / 100
        return normalized_score
    elif "refcoco" in benchmark:
        return result_dict['refcoco_ACC@0.5,none']

    else:
        raise ValueError(f"Unknown benchmark: {benchmark}")


def get_bench_json(exp_args, mix_config=None, method='merged-task_arithmetic', sft_strategy='lora', ckpt=None):
    '''Return the results.json for a given configuration'''
    exp_name = exp_args['exp_name']
    base_model = exp_args['base_model']
    expert_size = exp_args['expert_size']
    results_root = "eval_bench/"
    batch_size = 128

    ### Parse method
    if '-' in method:
        type, merging_method = method.split('-')
    else:
        type = method
        merging_method = None

    ### Handle mix_config with floats
    if mix_config is not None:
        mix_config = mix_config.copy()
        for k, v in mix_config.items():
            if isinstance(v, float):
                mix_config[k] = int(v * exp_args['expert_size'])

    ### Determine tasks and num_tasks
    if mix_config is None:
        tasks = []
    else:
        tasks = [t for t, s in mix_config.items() if s > 0]
    num_tasks = len(tasks)

    ### Determine results folder
    if type=='instruct':
        model_key = base_model.replace("_", "instr_")
        model_name = os.path.basename(BASE_MODEL_PATHS[model_key]).replace("-HF", "")
        results_folder = os.path.join(results_root,"base_models", model_name)
        inner_folder = f"models__{model_name}"

    elif type=='base' or num_tasks==0:
        base_model_name = os.path.basename(BASE_MODEL_PATHS[base_model]).replace("-HF", "")
        results_folder = os.path.join(results_root, "base_models", base_model_name)
        inner_folder = f"models__{base_model_name}"

    elif type=='expert' or num_tasks==1:
        ckpt = expert_size // batch_size if ckpt is None else ckpt # Use full expert size checkpoint by default
        file_name = f"{base_model}_{sft_strategy}_expert_{tasks[0]}-{expert_size}/checkpoint-{ckpt}"
        results_folder = os.path.join(results_root, exp_name, file_name)
        inner_folder = file_name.replace("/", "__")

    elif type in {'mixed', 'mixedint'}:
        ckpt = round(sum(mix_config.values()) / batch_size) if ckpt is None else ckpt
        mix_str = '++'.join([f'{task}-{mix_config[task]}' for task in tasks])
        file_name = f"{base_model}_{sft_strategy}_{type}_{mix_str}/checkpoint-{ckpt}"
        results_folder = os.path.join(results_root, exp_name, file_name)
        inner_folder = file_name.replace("/", "__")

    elif type in {'merged', 'alphamerged', 'mergedpeft', 'alphamergedpeft'}:
        mix_str = '++'.join([f'{task}-{mix_config[task]}' for task in tasks])
        file_name = f"{base_model}_{sft_strategy}_{type}_{mix_str}--{merging_method}"
        results_folder = os.path.join(results_root, exp_name, file_name)
        inner_folder = f"{exp_name}__" + file_name.replace("/", "__") + f"{'-CUSTOM' if 'intern' in base_model else ''}"

    else:
        raise ValueError(f"Unknown type: {type}")

    # Load results.json
    results_json_all = {}
    # Examples path: {results_folder}/docvqa_val_lite/intern35_2b_lora_expert_counting-102400__checkpoint-800/20251015_051555_results.json
    for bench in os.listdir(results_folder):
        bench_folder = os.path.join(results_folder, bench, inner_folder)
        if os.path.isdir(bench_folder):
            results_paths = [f for f in os.listdir(bench_folder) if f.endswith('_results.json')]
            if len(results_paths) > 1:
                pass # I may have runned the benchmark multiple times
                #print(f"Warning: Expected exactly one results file in {bench_folder}, but found {len(results_paths)}. Using the first one.")
            results_path = os.path.join(bench_folder, results_paths[0])
            with open(results_path) as f:
                results_json = json.load(f)["results"]
            results_json_all.update(results_json)

    return results_json_all


def get_bench(exp_args, mix_config=None, method='merged-task_arithmetic', sft_strategy='lora', benchmarks='gqa_lite', benchmarks_weights=None, aggregate='mean', raise_notfound_error=False):
    '''Returns the loss for a given configuration and benchmark(s)'''
    results_json = get_bench_json(exp_args, mix_config=mix_config, method=method, sft_strategy=sft_strategy)
    if isinstance(benchmarks, str):
        benchmarks = [benchmarks]

    # Get score values
    values = []
    for bench in benchmarks:
        base_bench = bench.split("-")[0] if "-" in bench else bench
        if base_bench not in results_json:
            if raise_notfound_error:
                raise ValueError(f"Benchmark {bench} not found in results for {mix_config}.")
            else:
                #print(f"Warning: Benchmark {bench} not found in results for {mix_config}.")
                bench_value = 0.0
        else:
            bench_value = benchmark_to_score_fn(bench, results_json[base_bench])
        values.append(bench_value)

    # Reweight losses if needed
    if benchmarks_weights is not None:
        assert len(benchmarks_weights) == len(benchmarks), "Length of benchmarks_weights must match length of benchmarks"
        values = [v * w for v, w in zip(values, benchmarks_weights)]

    # Aggregate the losses
    if aggregate is None:
        return values
    elif aggregate == 'mean':
        return np.mean(values)
    elif aggregate == 'std':
        return np.std(values)
    else:
        raise ValueError(f"Unknown aggregation method: {aggregate}")


def get_configs_bench(exp_args, mix_configs=None, method='merged-task_arithmetic', sft_strategy='lora', benchmarks='gqa_lite', aggregate='mean', benchmarks_weights=None, raise_notfound_error=False):
    '''Return loss for a list of mix_configs'''

    config_scores = []
    for mix_config in mix_configs:
        if method == 'interp':
            raise NotImplementedError("No interp for benchmark")
        else:
            score = get_bench(exp_args, mix_config, method=method, sft_strategy=sft_strategy, benchmarks=benchmarks, benchmarks_weights=benchmarks_weights, aggregate=aggregate, raise_notfound_error=raise_notfound_error)
        config_scores.append(score)
    return config_scores

# Test
# score = get_bench(exp_qwen2_2b_100k, method='base', benchmarks=['gqa_lite', 'ocrbench'], aggregate=None)
# print(score)
# score = get_bench(exp_intern35_2b_100k, method='base', benchmarks=['gqa_lite', 'ocrbench'], aggregate=None)
# print(score)
# score = get_configs_bench(exp_qwen2_2b_100k, mix_configs=mix_configs3, method='mixedint', benchmarks=['gqa_lite', 'ocrbench'], aggregate=None)
# print(score)
# score = get_configs_bench(exp_intern35_2b_100k, mix_configs=mix_configs3, method='mixedint', benchmarks=['gqa_lite', 'ocrbench'], aggregate=None)
# print(score)
# score = get_configs_bench(exp_qwen2_2b_100k, mix_configs=mix_configs2, method='alphamerged-task_arithmetic', benchmarks=['mme'], aggregate=None, raise_notfound_error=True)
# print(score)
# eval_bench/exp_qwen2_2b_100k/qwen2_2b_lora_alphamerged_general-12800++ocr-89600--task_arithmetic/mme/exp_qwen2_2b_100k__qwen2_2b_lora_alphamerged_general-12800++ocr-89600--task_arithmetic/20251101_043758_results.json  
