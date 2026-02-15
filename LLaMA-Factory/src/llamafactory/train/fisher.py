import torch
import re
import torch.nn.functional as F

from tqdm import tqdm
from torch import nn

from typing import Optional, Union
from ..extras import logging

logger = logging.get_logger(__name__)


exclude_param_names_regex_lora = [
    '.*visual.*',   # includes mm projetor for Qwen-VL
    '.*embed_tokens.*',
    '.*lm_head.*',
    '.*norm.*',
    '.*bias.*',
    # Added for InternVL3
    '.*vision_tower.*',
    '.*multi_modal_projector.*',
]
exclude_param_names_regex_full = [
    '.*visual.blocks.*',
    '.*visual.patch_embed.*',
    '.*vision_tower.*',
]
exclude_param_names_dict = {
    'lora': exclude_param_names_regex_lora,
    'full': exclude_param_names_regex_full,
}

dtype_map = {
    'float16': torch.float16,
    'float32': torch.float32,
    'float64': torch.float64,
}

def get_param_names_to_merge(input_param_names: list, exclude_param_names_regex: list):
    """
    get the names of parameters that need to be merged
    :param input_param_names: list, names of input parameters
    :param exclude_param_names_regex: list, regular expression of names of parameters that need to be excluded
    :return:
    """
    param_names_to_merge = []
    for param_name in input_param_names:
        exclude = any([re.match(exclude_pattern, param_name) for exclude_pattern in exclude_param_names_regex])
        if not exclude:
            param_names_to_merge.append(param_name)
    return param_names_to_merge

def get_param_squared_gradients(model: nn.Module, param_names_to_merge: list):
    """
    get the squared gradients of parameters
    :param model: nn.Module, model
    :param param_names_to_merge: list, list of parameter names that need to be merged
    :return:
    """
    param_squared_gradients = {param_name: param_value.grad.detach() ** 2 for param_name, param_value in model.named_parameters() if param_name in param_names_to_merge}
    return param_squared_gradients

class FisherAccumulator:
    def __init__(self, model: nn.Module, param_names_to_merge: list, dtype: Optional[Union[str, torch.dtype]] = 'float32'):

        self.model = model
        self.param_names_to_merge = param_names_to_merge
        self.dtype = dtype
        self.fisher_weights = dict()
        for param_name, param_value in model.named_parameters():
            if param_name in param_names_to_merge:
                self.fisher_weights[param_name] = torch.zeros_like(param_value.data, dtype=self.dtype, device='cpu')

    def accumulate(self):
        for param_name, param_value in self.model.named_parameters():
            if param_name in self.param_names_to_merge:
                self.fisher_weights[param_name] += param_value.grad.detach().to(self.dtype).pow(2).to('cpu')


def print_gpu_memory_usage(info: str = ""):
    '''Helper function to print GPU memory usage.'''
    memory_allocated = torch.cuda.memory_allocated() / (1024 ** 3)  # in GB
    memory_total = torch.cuda.get_device_properties(0).total_memory / (1024 ** 3)  # in GB
    print(f"GPU memory usage: {memory_allocated:.2f} GB / {memory_total:.2f} GB - {info}")


def compute_fisher(
        trainer,
        num_fisher_examples: Optional[int] = None,
        params_set: Optional[str] = 'full',
        fisher_dtype: Optional[Union[str, torch.dtype]] = 'float32',
        fisher_method: Optional[str] = 'empirical',
        ):
        fisher_dtype = dtype_map[fisher_dtype]

        # DataLoader (handle multiple eval datasets)
        eval_dataset = trainer.eval_dataset
        if isinstance(eval_dataset, dict):
            if len(eval_dataset) > 1:
                raise NotImplementedError("Computing Fisher for multiple eval datasets is not supported yet.")
            else:
                eval_dataset = list(eval_dataset.values())[0]
        dataloader = trainer.get_eval_dataloader(eval_dataset)
        batch_size = trainer.args.eval_batch_size
 

        # Select parameters to merge
        model_to_merge = trainer.model
        all_param_names = {param_name for param_name, _ in model_to_merge.named_parameters()}
        param_names_to_merge = get_param_names_to_merge(all_param_names, exclude_param_names_dict[params_set])
        assert not model_to_merge.training, "Model should be in eval mode, otherwise dropout and batchnorm will affect the Fisher information."

        # Activate gradients
        for param_name, param_value in model_to_merge.named_parameters():
            if param_name in param_names_to_merge:
                param_value.requires_grad = True
            else:
                param_value.requires_grad = False

        # Initialize fisher weights accumulator. Accumulating avoids OOM for large models
        fisher_accumulator = FisherAccumulator(model=model_to_merge, param_names_to_merge=param_names_to_merge, dtype=fisher_dtype)

        num_computed_examples = 0
        num_total_tokens = 0
        if num_fisher_examples % batch_size != 0:
            print(f"warning: the number of examples for computing fisher cannot be fully divided by the batch size, "
                    "which may lead to a slightly different number of the actually used examples.")
        logger.info(f"Computing Fisher information matrix: num_samples={num_fisher_examples} | batch_size={batch_size} | params_set={params_set} | dtype={fisher_dtype} | method={fisher_method}")
        for step, inputs in tqdm(enumerate(dataloader), desc=f"computing fisher weights"):
            if num_computed_examples >= num_fisher_examples:
                break
           
            inputs = trainer._prepare_inputs(inputs)
            outputs = model_to_merge(**inputs)
            logits = outputs.logits
            B, T, C = logits.shape
            num_valid_tokens = (inputs['labels'] != -100).sum().item()
            
            if fisher_method == 'empirical':
                loss = outputs.loss * num_valid_tokens # cross-entropy loss with ground-truth labels. This is averaged over the batch and tokens
                model_to_merge.zero_grad()
                loss.backward()

            elif fisher_method == 'true_hard':
                
                #num_valid_tokens = (inputs['labels'] != -100).sum(-1) # shape (B,)
                labels_gt = inputs['labels'].view(-1) # shape (B*T)
                logits = logits.float().view(-1, C)[labels_gt != -100,:]  # shape (T', C). T'= number of non-ignored tokens
                log_probs = torch.log_softmax(logits, dim=-1)
                _, target_labels = logits.max(dim=-1) # shape (B*T)

                loss = F.nll_loss(log_probs, target_labels, reduction='mean') * num_valid_tokens
                model_to_merge.zero_grad()
                loss.backward()

            # compute fisher weights for classification task
            elif fisher_method == 'true_soft':
                # Logic from https://github.com/yule-BUAA/MergeLM/blob/main/model_merging_methods/merging_methods.py
                # RK: this implementation is not mathematically correct: it computes (∑_y sqrt(p_y)*∇log p_y)^2 instead of ∑_y p_y*(∇log p_y)^2
                labels_gt = inputs['labels'].view(-1) # shape (B*T)
                logits = logits.float().view(-1, C)[labels_gt != -100,:]  # shape (T', C). T'= number of non-ignored tokens
                # use detach() to detach from the computation graph
                labels_probabilities = torch.softmax(logits, dim=-1).detach() # shape (T', C)
                labels_log_probabilities = torch.log_softmax(logits, dim=-1) # shape (T', C)
                # sqrt labels_probabilities, since torch.sqrt(labels_probabilities) would be squared in the following squared gradients
                labels_expectations = torch.sqrt(labels_probabilities) * labels_log_probabilities
                # sum over label classes and batch dimension
                loss = - labels_expectations.sum(dim=-1).mean(dim=0) * num_valid_tokens
                model_to_merge.zero_grad()
                loss.backward()

            else:
                raise ValueError(f"Unsupported fisher_method: {fisher_method}")


            # accumulate fisher weights
            fisher_accumulator.accumulate()
            num_computed_examples += B
            num_total_tokens += num_valid_tokens
            
            print(f"Step {step}: loss = {loss.item()} | num_valid_tokens = {num_valid_tokens} | num_computed_examples = {num_computed_examples}")

        # mean over batches
        fisher_weights = fisher_accumulator.fisher_weights
        num_batches = step + 1
        for key in fisher_weights.keys():
            #fisher_weights[key] /= num_batches
            fisher_weights[key] /= num_total_tokens  # normalize

        metadata = {
            'num_fisher_examples': num_fisher_examples,
            'params_set': params_set,
            'fisher_dtype': str(fisher_dtype),
            'fisher_method': fisher_method,
            'batch_size': trainer._train_batch_size,
        }

        return fisher_weights, metadata