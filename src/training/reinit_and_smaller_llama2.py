"""
Original size of LLaMA v2 model: 7B parameters:
{
  "_name_or_path": "meta-llama/Llama-2-7b-hf",
  "architectures": [
    "LlamaForCausalLM"
  ],
  "bos_token_id": 1,
  "eos_token_id": 2,
  "hidden_act": "silu",
  "hidden_size": 4096,
  "initializer_range": 0.02,
  "intermediate_size": 11008,
  "max_position_embeddings": 4096,
  "model_type": "llama",
  "num_attention_heads": 32,
  "num_hidden_layers": 32,
  "num_key_value_heads": 32,
  "pretraining_tp": 1,
  "rms_norm_eps": 1e-05,
  "rope_scaling": null,
  "tie_word_embeddings": false,
  "torch_dtype": "float16",
  "transformers_version": "4.31.0.dev0",
  "use_cache": true,
  "vocab_size": 32000
}

"""
import torch
from transformers import AutoModelForCausalLM, AutoConfig
import torch.nn as nn
from pathlib import Path
import datasets
from datasets import load_dataset, interleave_datasets
import torch
from transformers import GPT2LMHeadModel, PreTrainedTokenizer, AutoTokenizer, Trainer, TrainingArguments, AutoConfig
import math
import wandb
import os

def num_params(model: nn.Module) -> int:
    # print("Number of parameters:", sum(p.numel() for p in model.parameters()))
    return sum(p.numel() for p in model.parameters())

def get_weight_norms(model: nn.Module, verbose: bool = False) -> None:
    """
    Prints the L1 norm of the weights of each module in the given PyTorch model.

    Args:
    model (nn.Module): The PyTorch model whose weight norms are to be printed.

    Returns:
    None
    """
    total_weight_norm: float = 0.0
    for name, module in model.named_modules():
        # Check if the module has the 'weight' attribute
        if hasattr(module, 'weight') and isinstance(module.weight, torch.Tensor):
            # Calculate the L1 norm of the weights
            w_norm: float = module.weight.norm(1).item()
            total_weight_norm += w_norm
            if verbose:
                print(f"Norm of weights in module {name}: {w_norm}")
    return total_weight_norm

def reinitialize_weights(model, 
                         std: float = 0.0002,  # 0.02 ref: 
                         ) -> None:
    """
    
    From cs197, we choose std = 0.02 because of these two links:
    Why we chose 0.02 for standard deviation:
    https://github.com/huggingface/transformers/blob/772307be7649e1333a933cfaa229dc0dec2fd331/src/transformers/models/llama/modeling_llama.py#L858
    https://github.com/huggingface/transformers/blob/772307be7649e1333a933cfaa229dc0dec2fd331/src/transformers/models/llama/configuration_llama.py#L127
    Default is set to 0.02 in source code (see line 858 of the first link, and 127 of hte second link)
    """
    for module in model.modules():
        if isinstance(module, nn.Linear):
            # nn.init.normal_(module.weight, mean=0, std=0.02)
            nn.init.normal_(module.weight, mean=0, std=std)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

def reinitialize_weights_xavier(model,):
    """ Reinit with xavier """
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.xavier_normal_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)


def reinitialize_weights_kamming(model,):
    """ Reinit with xavier """
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            nn.init.kaiming_uniform_(module.weight)
            if 'norm' in name.lower() or 'norm' in str(module).lower():
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
def get_microscopic_llama2(verbose: bool = True):
    raise NotImplementedError
    # return get_smaller_llama2(hidden_size=2, num_hidden_layers=3, verbose=verbose)

def get_deafult_smallest_llama2(verbose: bool = True):
    return get_smaller_llama2(hidden_size=32, num_hidden_layers=1, verbose=verbose)

def get_full_llama7b(gpu_idx: int = -1):
    config = AutoConfig.from_pretrained("meta-llama/Llama-2-7b-hf", torch_dtype="auto")
    model = AutoModelForCausalLM.from_config(config)
    if gpu_idx >= 0:
        device = torch.device(f"cuda:{gpu_idx}" if torch.cuda.is_available() else "cpu")
        model = model.to(device)
    return model

def get_smaller_llama2(hidden_size : int = 2048, 
                       num_hidden_layers : int = 12, 
                       verbose : bool = False,):
    config = AutoConfig.from_pretrained("meta-llama/Llama-2-7b-hf")
    config.hidden_size = hidden_size
    config.num_hidden_layers = num_hidden_layers
    model = AutoModelForCausalLM.from_config(config) 
    smaller_model = AutoModelForCausalLM.from_config(config)
    if verbose:
        print(f'config: {config}')
        print("Original number of parameters:", sum(p.numel() for p in model.parameters()))
    return smaller_model

def _test_generate_smaller_model():
    """
    ref: https://stackoverflow.com/questions/76971761/how-to-adapt-llama-v2-model-to-less-than-7b-parameters
    """
    print('Starting to generate a smaller model...')
    # Load the pretrained LLaMA v2 config
    config = AutoConfig.from_pretrained("meta-llama/Llama-2-7b-hf")
    print(f'config: {config} {type(config)}')
    print()
    # Print the original number of parameters 
    model = AutoModelForCausalLM.from_config(config) 
    print("Original number of parameters:", sum(p.numel() for p in model.parameters()))

    # Modify the config to reduce size
    config.hidden_size = 2048
    config.num_hidden_layers = 12

    # Create a new smaller model from the modified config
    smaller_model = AutoModelForCausalLM.from_config(config)
    print("New number of parameters:", sum(p.numel() for p in smaller_model.parameters()))

def _test_reinit_model():
    """ 
    export CUDA_VISIBLE_DEVICES=6
    """
    torch.cuda.empty_cache() 
    print('Starting to reinitialize the model...')
    
    # - Get smaller llama2 model
    # model = get_deafult_smallest_llama2()
    model = get_full_llama7b()
    device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    # - Check norm before reinitialization
    print("-- NORM OF ENTIRE NET BEFORE REINITIALIZATION:")
    total_weight_norm = get_weight_norms(model)
    print(f"Total weight norm: {total_weight_norm}")
    # - Reinitialize weights
    reinitialize_weights(model)
    print("-- NORM OF ENTIRE NET AFTER REINITIALIZATION:")
    total_weight_norm = get_weight_norms(model)
    print(f"Total weight norm: {total_weight_norm}")

if __name__ == '__main__':
    import time
    start = time.time()
    _test_reinit_model()
    print('Done!\a\a\a')