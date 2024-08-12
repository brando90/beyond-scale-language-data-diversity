#%%
import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config

def reinitialize_linear_weights_mutates(
        model, 
        weight_std: float = 0.0002,  # 0.02 ref: Hailey S doesn't recommend this huge value! ref: https://x.com/haileysch__/status/1822758486632997102 
        bias_std: float = 0.0, 
        ) -> None:
    """ Why we chose < 0.02 for standard deviation: https://github.com/alycialee/beyond-scale-language-data-diversity/issues/18   """
    for module in model.modules():
        if isinstance(module, nn.Linear):
            # nn.init.normal_(module.weight, mean=0, std=0.02) # original, evil!
            nn.init.normal_(module.weight, mean=0, std=weight_std)
            if module.bias is not None:
                nn.init.constant_(module.bias, bias_std)

# Step 1: Load the pre-trained GPT-2 XL model
torch.cuda.empty_cache() # Clear CUDA cache to free up memory
torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32 
pretrained_model = AutoModelForCausalLM.from_pretrained("gpt2-xl", torch_dtype=torch_dtype, trust_remote_code=True)
pretrained_tokenizer = AutoTokenizer.from_pretrained("gpt2-xl", padding_side="right", trust_remote_code=True)
pretrained_tokenizer.pad_token = pretrained_tokenizer.eos_token if pretrained_tokenizer.pad_token_id is None else pretrained_tokenizer.pad_token
# Step 2: Calculate the L2 norm of the weights for the pre-trained model
pretrained_weight_norm = sum([torch.norm(param, p=2).item() for param in pretrained_model.parameters()])
print(f"Total L2 norm of pre-trained model weights: {pretrained_weight_norm:.2f}")

# Step 1: Initialize a new GPT-2 model from scratch with custom configuration
config = GPT2Config(
    vocab_size=pretrained_tokenizer.vocab_size,  # Ensure this matches the tokenizer's vocabulary size
    n_ctx=1024,  # Context window size (number of tokens the model can see at once)
    bos_token_id=pretrained_tokenizer.bos_token_id,  # Begin-of-sequence token
    eos_token_id=pretrained_tokenizer.eos_token_id,  # End-of-sequence token
    pad_token_id=pretrained_tokenizer.eos_token_id,  # pad-sequence token
)
model = AutoModelForCausalLM.from_config(config)
# Step 2: Calculate the L2 norm of the weights for the freshly initialized model
scratch_weight_norm = sum([torch.norm(param, p=2).item() for param in model.parameters()])
print(f"Total L2 norm of model initialized from scratch: {scratch_weight_norm:.2f}")

# Step 1: Reinit GPT2 with really small init
reinitialize_linear_weights_mutates(model)
scratch_weight_norm_small_reinit = sum([torch.norm(param, p=2).item() for param in model.parameters()])
print(f"Total L2 norm of model initialized from scratch with small reinit (not default HF config): {scratch_weight_norm_small_reinit:.2f}")

# Justification:
# If the model is truly being initialized from scratch, the weight norm should be much smaller compared to the pre-trained model. 
# This confirms that the training process is starting from a random initialization and not from any pre-existing pre-trained weights.


# #%%
# import torch
# from transformers import GPT2LMHeadModel, GPT2Config, AutoModelForCausalLM

# # Step 1: Load the pre-trained GPT-2 XL model
# pretrained_model = AutoModelForCausalLM.from_pretrained("gpt2-xl")

# # Step 2: Calculate the L2 norm of the weights for the pre-trained model
# pretrained_weight_norm = 0.0
# for param in pretrained_model.parameters():
#     pretrained_weight_norm += torch.norm(param, p=2).item()

# # Print the total L2 norm for the pre-trained model
# print(f"Total L2 norm of pre-trained model weights: {pretrained_weight_norm:.2f}")

# # Step 3: Initialize a new GPT-2 model from scratch with custom configuration
# config = GPT2Config(
#     vocab_size=52000,  # Ensure this matches the tokenizerbdegruviheurrbnr's vocabulary size
#     n_ctx=1024,  # Context window size (number of tokens the model can see at once)
#     bos_token_id=0,  # Begin-of-sequence token
#     eos_token_id=1,  # End-of-sequence token
# )
# model = GPT2LMHeadModel(config)

# # Step 4: Calculate the L2 norm of the weights for the freshly initialized model
# scratch_weight_norm = 0.0
# for param in model.parameters():
#     scratch_weight_norm += torch.norm(param, p=2).item()

# # Print the total L2 norm for the model initialized from scratch
# print(f"Total L2 norm of model initialized from scratch: {scratch_weight_norm:.2f}")

# # Justification:
# # If the model is truly being initialized from scratch, the weight norm should be much smaller compared to the pre-trained model. 
# # This confirms that the training process is starting from a random initialization and not from any pre-existing pre-trained weights.
