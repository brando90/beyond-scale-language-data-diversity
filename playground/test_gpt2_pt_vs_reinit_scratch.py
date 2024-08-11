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
    vocab_size=52000,  # Ensure this matches the tokenizer's vocabulary size
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
