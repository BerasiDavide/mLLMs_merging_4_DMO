import re
from collections import defaultdict
from typing import Any, Dict, List, Optional


def _extract_answer_letter(text: str) -> str:
    """
    Extract the answer choice letter from a string.

    Examples:
    'A answer1' -> 'A'
    'A) answer2' -> 'A'
    '(B) answer' -> 'B'
    'C' -> 'C'
    '(C)' -> 'C'
    'A.' -> 'A'

    Return an empty string if no letter is found.
    """
    text = text.strip()
    match = re.match(r"[\(\s]*([A-Z])[\)\.\s]*", text, flags=re.IGNORECASE)
    if match:
        return match.group(1).upper()
    return ""

def _extract_answer_letter_softer(text: str, choices) -> str:
    if text in choices:
        # Get the corresponding letter.
        idx = choices.index(text)
        return chr(65 + idx)  # 65 is ASCII for 'A'
    else:
        return _extract_answer_letter(text)


def cv_bench_doc_to_text(doc: dict[str, Any], lmms_eval_specific_kwargs: Optional[dict[str, Any]] = None) -> str:
    if lmms_eval_specific_kwargs is None:
        lmms_eval_specific_kwargs = {}

    num_choices = len(doc["choices"])
    choice_letters = ", ".join([chr(65 + i) for i in range(num_choices)])
    prompt = lmms_eval_specific_kwargs.get("pre_prompt", "").format(choice_letters) + "\n" + doc["prompt"]
    return prompt


def cv_bench_doc_to_visual(doc: dict) -> list:
    return [doc["image"].convert("RGB")]


def cv_bench_process_results(doc: Dict, result: List[str]) -> Dict[str, Dict]:
    key_name = "cv_bench_acc"
    # extract grounded answer
    grounded_output = doc["answer"].strip("()")
    response = result[0]

    # extract predicted answer
    pred_letter = _extract_answer_letter(response)
    flag = pred_letter == grounded_output

    # Add soft matching that removes 'Answer: ' and dots at the end of answers
    response_soft = response.replace("Answer:", "")
    pred_letter_soft = _extract_answer_letter(response_soft)
    flag_soft = pred_letter_soft == grounded_output

    # Add softer matching that retrieves letter from numerical answers
    pred_letter_softer = _extract_answer_letter_softer(response_soft, doc['choices'])
    flag_softer = pred_letter_softer == grounded_output


    cv_bench_submission = {"id": doc["idx"], "gt_content": grounded_output, "pred_parsed": pred_letter, "pred": response, "type": doc["type"], "task": doc["task"], "source": doc["source"], "is_correct": flag}
    cv_bench_submission_soft = {"id": doc["idx"], "gt_content": grounded_output, "pred_parsed": pred_letter_soft, "pred": response_soft, "type": doc["type"], "task": doc["task"], "source": doc["source"], "is_correct": flag_soft}
    return {key_name: cv_bench_submission,
            "count_acc": cv_bench_submission,
            f"{key_name}_soft": cv_bench_submission_soft,
            "count_acc_soft": cv_bench_submission_soft,
            f"{key_name}_softer": {**cv_bench_submission, "pred_parsed": pred_letter_softer, "is_correct": flag_softer},
            "count_acc_softer": {**cv_bench_submission, "pred_parsed": pred_letter_softer, "is_correct": flag_softer},
            }

def cv_bench_aggregate_results(results: List[Dict]):
    total_samples = len(results)
    total_correct = 0

    for sample in results:
        if sample["is_correct"]:
            total_correct += 1

    accuracy = total_correct / total_samples if total_samples > 0 else 0
    return accuracy

## Added for Count task ##
def count_acc_aggregate_results(results: List[Dict]):
    total_samples = 0
    total_correct = 0

    for sample in results:
        if sample['task'] == 'Count':
            total_correct += 1 if sample["is_correct"] else 0
            total_samples += 1

    accuracy = total_correct / total_samples if total_samples > 0 else 0
    return accuracy
## End Added for Count task ##

def cv_bench_default_aggregate_results(results: List[Dict]):
    source_samples = defaultdict(list)
    for elem in results:
        source = elem["source"]
        source_samples[source].append(elem["is_correct"])
    source_accuracies = {source: sum(scores) / len(scores) for source, scores in source_samples.items()}
    ade20k_2d = source_accuracies["ADE20K"]
    coco_2d = source_accuracies["COCO"]
    omni_3d = source_accuracies["Omni3D"]

    # original formula
    cv_bench_accuracy = 1 / 2 * ((ade20k_2d + coco_2d) / 2 + omni_3d)
    return cv_bench_accuracy

