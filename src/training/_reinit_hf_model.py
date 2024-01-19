import torch
from transformers import AutoModelForCausalLM, AutoConfig 
import torch.nn as nn

def main_reinit_model():
    """
    ref: https://stackoverflow.com/questions/76971761/how-to-adapt-llama-v2-model-to-less-than-7b-parameters
    ref: https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py#L721
    ref: https://chat.openai.com/c/977d0cb0-b819-48ac-be5c-6e482ad5e518 
    """
    print('Starting to reinitialize the model...')
    # Load the pretrained LLaMA v2 config
    config = AutoConfig.from_pretrained("meta-llama/Llama-2-7b-hf")
    # print(f'config: {config} {type(config)}')
    # Print the original number of parameters 
    model = AutoModelForCausalLM.from_config(config) 
    # put model on device cuda
    model = model.to('cuda')
    # print the model's device
    print(f'{model.device=}')
    # print(f'{model=}')
    # print("Original number of parameters:", sum(p.numel() for p in model.parameters()))
    # go through all parameters and compute the l1 norm and sum it then print it
    norm_model = sum(p.norm(1) for p in model.parameters())
    # loop through modules of model and reinitialize weights with normal_mean, 0.02 
    print(f'{norm_model=}')
    """
    go through model and print all laters
    """
    # model.init_weights()  # didn't work
    # model._init_weights(module)  # didn't work needs module
    # for name, param in model.named_parameters():
    #     model._init_weights(param)
    # model.post_init()
    reinitialize_weights(model)
    # model._initialize_weights(module)  # didn't work needs module
    # for name, param in model.named_parameters():
    #     print(f'{name=} {param.shape=}')
    norm_model = sum(p.norm(1) for p in model.parameters())
    print(f'{norm_model=}')

def reinitialize_weights(model) -> None:
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

def _init_weights(self, module):
    std = self.config.initializer_range
    if isinstance(module, nn.Linear):
        module.weight.data.normal_(mean=100.0, std=std)
        if module.bias is not None:
            module.bias.data.zero_()
    elif isinstance(module, nn.Embedding):
        module.weight.data.normal_(mean=0.0, std=std)
        if module.padding_idx is not None:
            module.weight.data[module.padding_idx].zero_()

def main_generate_smaller_model():
    """
    ref: https://stackoverflow.com/questions/76971761/how-to-adapt-llama-v2-model-to-less-than-7b-parameters
    """
    print('Starting to reinitialize the model...')
    # Load the pretrained LLaMA v2 config
    config = AutoConfig.from_pretrained("meta-llama/Llama-2-7b-hf")
    print(f'config: {config} {type(config)}')
    # Print the original number of parameters 
    model = AutoModelForCausalLM.from_config(config) 
    print("Original number of parameters:", sum(p.numel() for p in model.parameters()))

    # Modify the config to reduce size
    config.hidden_size = 2048
    config.num_hidden_layers = 12

    # Create new smaller model from modified config
    smaller_model = AutoModelForCausalLM.from_config(config)
    print("New number of parameters:", sum(p.numel() for p in smaller_model.parameters()))

if __name__ == '__main__':
    import time
    start = time.time()
    # main_generate_smaller_model() 
    main_reinit_model()
    print('Done!\a\a\a')