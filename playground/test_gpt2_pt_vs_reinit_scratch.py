#%%
        # torch.cuda.empty_cache() # Clear CUDA cache to free up memory
        # torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32 
        # model = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, torch_dtype=torch_dtype, trust_remote_code=True)
        # model = model.to(device)
        # tokenizer = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, padding_side="right", trust_remote_code=True)
        # print(f'{tokenizer=}')
        # print(f'{tokenizer.pad_token=} {tokenizer.eos_token_id=}')
        # tokenizer.pad_token = tokenizer.eos_token if tokenizer.pad_token_id is None else tokenizer.pad_token
        # print(f'{tokenizer.pad_token=} {tokenizer.eos_token_id=}')
        # # get context length for setting max length for training
        # print(f'{model.config=}')
        # if max_length is None:
        #     if hasattr(model.config, "context_length"):
        #         max_length: int = model.config.context_length 
        #         print("Context length:", model.config.context_length)
        #     else:
        #         max_length: int = 1024
        # else:
        #     print(f"Context length not found in model.config, so using your default or hardcoded value. Model is {pretrained_model_name_or_path=}.")
        #     # max_length: int = 4  # for debugging
        #     max_length: int = max_length  # for debugging
        #     # max_length: int = 128_000  # ref: https://huggingface.co/meta-llama/Meta-Llama-3.1-8B

#%%
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, GPT2Config

# Step 1: Load the pre-trained GPT-2 XL model
pretrained_model = AutoModelForCausalLM.from_pretrained("gpt2-xl")
pretrained_tokenizer = AutoTokenizer.from_pretrained("gpt2-xl")

# Step 2: Calculate the L2 norm of the weights for the pre-trained model
pretrained_weight_norm = sum([torch.norm(param, p=2).item() for param in pretrained_model.parameters()])

# Print the total L2 norm for the pre-trained model
print(f"Total L2 norm of pre-trained model weights: {pretrained_weight_norm:.2f}")

# Step 3: Initialize a new GPT-2 model from scratch with custom configuration
config = GPT2Config(
    vocab_size=pretrained_tokenizer.vocab_size,  # Ensure this matches the tokenizer's vocabulary size
    n_ctx=1024,  # Context window size (number of tokens the model can see at once)
    bos_token_id=pretrained_tokenizer.bos_token_id,  # Begin-of-sequence token
    eos_token_id=pretrained_tokenizer.eos_token_id,  # End-of-sequence token
)
model = AutoModelForCausalLM.from_config(config)

# Step 4: Calculate the L2 norm of the weights for the freshly initialized model
scratch_weight_norm = 0.0
for param in model.parameters():
    scratch_weight_norm += torch.norm(param, p=2).item()

# Print the total L2 norm for the model initialized from scratch
print(f"Total L2 norm of model initialized from scratch: {scratch_weight_norm:.2f}")

# Justification:
# If the model is truly being initialized from scratch, the weight norm should be much smaller compared to the pre-trained model. 
# This confirms that the training process is starting from a random initialization and not from any pre-existing pre-trained weights.


#%%
import torch
from transformers import GPT2LMHeadModel, GPT2Config, AutoModelForCausalLM

# Step 1: Load the pre-trained GPT-2 XL model
pretrained_model = AutoModelForCausalLM.from_pretrained("gpt2-xl")

# Step 2: Calculate the L2 norm of the weights for the pre-trained model
pretrained_weight_norm = 0.0
for param in pretrained_model.parameters():
    pretrained_weight_norm += torch.norm(param, p=2).item()

# Print the total L2 norm for the pre-trained model
print(f"Total L2 norm of pre-trained model weights: {pretrained_weight_norm:.2f}")

# Step 3: Initialize a new GPT-2 model from scratch with custom configuration
config = GPT2Config(
    vocab_size=52000,  # Ensure this matches the tokenizerbdegruviheurrbnr's vocabulary size
    n_ctx=1024,  # Context window size (number of tokens the model can see at once)
    bos_token_id=0,  # Begin-of-sequence token
    eos_token_id=1,  # End-of-sequence token
)
model = GPT2LMHeadModel(config)

# Step 4: Calculate the L2 norm of the weights for the freshly initialized model
scratch_weight_norm = 0.0
for param in model.parameters():
    scratch_weight_norm += torch.norm(param, p=2).item()

# Print the total L2 norm for the model initialized from scratch
print(f"Total L2 norm of model initialized from scratch: {scratch_weight_norm:.2f}")

# Justification:
# If the model is truly being initialized from scratch, the weight norm should be much smaller compared to the pre-trained model. 
# This confirms that the training process is starting from a random initialization and not from any pre-existing pre-trained weights.
