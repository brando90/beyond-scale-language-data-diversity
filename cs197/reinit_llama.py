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

def print_weight_norms(model: nn.Module) -> None:
    """
    Prints the L1 norm of the weights of each module in the given PyTorch model.

    Args:
    model (nn.Module): The PyTorch model whose weight norms are to be printed.

    Returns:
    None
    """
    for name, module in model.named_modules():
        # Check if the module has the 'weight' attribute
        if hasattr(module, 'weight') and isinstance(module.weight, torch.Tensor):
            # Calculate the L1 norm of the weights
            w_norm: float = module.weight.norm(1).item()
            print(f"Norm of weights in module {name}: {w_norm}")


def main_reinit_model():
    torch.cuda.empty_cache() 

    print('Starting to reinitialize the model...')
    
    # Load the pretrained LLaMA v2 config
    config = AutoConfig.from_pretrained("meta-llama/Llama-2-7b-hf", torch_dtype="auto")
    
    # Print the original number of parameters
    model = AutoModelForCausalLM.from_config(config)

    print("NORM OF ENTIRE NET BEFORE REINITIALIZATION:")
   # print_weight_norms(model)
    
    # Reinitialize weights
    # reinitialize_weights(model)
    
    # Move the model to GPU (cuda) after initializing weights
    def get_freest_gpu():
    # Get the index of the GPU with the most free memory
        devices = list(range(torch.cuda.device_count()))
        free_memory = [torch.cuda.memory_reserved(device) - torch.cuda.memory_allocated(device) for device in devices]
        freest_device = devices[free_memory.index(max(free_memory))]
        return freest_device

# Select the GPU with the most free memory
    # freest_device = get_freest_gpu()
    # device = torch.device(f"cuda:{freest_device}" if torch.cuda.is_available() else "cpu")
    device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    
    # Print the model's device
    print(f'{model.device=}')
    
    # Print the original number of parameters
    # print("Original number of parameters:", sum(p.numel() for p in model.parameters()))

    # Print the model architecture
    print(model)

    print("NORM OF ENTIRE NET AFTER REINITIALIZATION:")
    print_weight_norms(model)

        # bf16 or fp32
    bf16=torch.cuda.get_device_capability(torch.cuda.current_device())[0] >= 8,  # if >= 8 ==> brain float 16 available or set to True if you always want fp32
    if bf16:
        torch_dtype = torch.bfloat16
    else: 
        torch_dtype = torch.float32
        # get model

    tokenizer = AutoTokenizer.from_pretrained(
        "meta-llama/Llama-2-7b-hf",
        # cache_dir=args.cache_dir,
        padding_side="right",
        use_fast=False, # Fast tokenizer giving issues.
        tokenizer_type='llama',
        trust_remote_code=True,
        use_auth_token=True,
    )
    # - Ensure padding token is set TODO: how does this not screw up the fine-tuning? e.g., now model doesn't learn to predict eos since it's padded our by mask, ref: https://discuss.huggingface.co/t/why-does-the-falcon-qlora-tutorial-code-use-eos-token-as-pad-token/45954
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    max_length = 512
    def preprocess(examples):
        return tokenizer(examples["text"], padding='max_length', max_length=max_length, truncation=True, return_tensors="pt")
    def map(batch):
        return batch.map(preprocess, batched=True, remove_columns=['text', 'meta'])

    print(f'{torch.cuda.device_count()=} (makes sure GPUs are visible and accesible to Pytorch.)')
    print(f'Model is currently on: {next(iter(model.parameters())).device=}')
    
    # -- Load datasets
    # - Get train data set
    # train_datasets = [load_dataset(path, name, streaming=True, split="train").with_format("torch") for path, name in zip(path, name)]
    # train_datasets = [load_dataset(path, name, data_files=data_file, streaming=streaming, split=split).with_format("torch") for path, name, data_file, split in zip(path, name, data_files, split)]

    uspto_train_dataset = load_dataset('allyc/My-Dataset', 'uspto', split='train[:1%]', cache_dir='/lfs/skampere1/0/allyc/beyond-scale-language-data-diversity/cs197/data/uspto/train').with_format('torch')
    # pubmed_train_dataset = load_dataset('allyc/My-Dataset', 'pubmed', split='train', cache_dir='/lfs/skampere1/0/allyc/beyond-scale-language-data-diversity/cs197/data/pubmed/train').with_format('torch')
    # uspto_eval_dataset = load_dataset('allyc/My-Dataset', 'uspto', split='validation', cache_dir='/lfs/skampere1/0/allyc/beyond-scale-language-data-diversity/cs197/data/uspto/eval').with_format('torch')
    # pubmed_eval_dataset = load_dataset('allyc/My-Dataset', 'pubmed', split='validation', cache_dir='/lfs/skampere1/0/allyc/beyond-scale-language-data-diversity/cs197/data/pubmed/eval').with_format('torch')
    # openwebtext_eval_dataset = load_dataset('suolyer/pile_openwebtext2', split='validation[:1%]', cache_dir='/lfs/skampere1/0/allyc/beyond-scale-language-data-diversity/cs197/data/openwebtext/validation').with_format('torch')  

    uspto_train_dataset = map(uspto_train_dataset)
    # openwebtext_eval_dataset = map(openwebtext_eval_dataset)

    # probabilities = [1.0/len(train_datasets) for _ in train_datasets]  # TODO: perhaps we should change weights to informal and formal have same weight? right now is just in terms of list of data sets perhaps having 2 interleaves one for formal one for informal then use another interleave and do 50/50?. 
    # train_dataset = interleave_datasets(train_datasets, probabilities)
    # TODO: suffle data set False, True, note i've experienced that with shuffle_ds.take(512) is slow...
    # shuffled_dataset = train_dataset.shuffle(buffer_size=buffer_size, seed=seed) if shuffle else train_dataset
    # print(f'{batch=}')
    # column_names = next(iter(batch)).keys()
    def preprocess(examples):
        return tokenizer(examples["text"], padding="max_length", max_length=max_length, truncation=True, return_tensors="pt")
        # return tokenizer(examples["text"], padding="max_length", max_length=model.config.context_length, truncation=True, return_tensors="pt")
    # collate function does this already
    # remove_columns = column_names  # remove all keys that are not tensors to avoid bugs in collate function in task2vec's pytorch data loader
    # def map(batch):
    #     return batch.map(preprocess, batched=True, remove_columns=remove_columns)
    # train_dataset = map(train_dataset)

    # TODO: probably need to write a collate_fn for the eval so that the eval is done right?
    # TODO: we need ppl (and ideally token edit distance for eval, reason explained here: https://arxiv.org/abs/2304.15004)
    
    # ALLY TODO:  remove all keys that are not tensors to avoid bugs in collate function in task2vec's pytorch data loader

    # -- Training arguments and trainer instantiation ref: https://huggingface.co/docs/transformers/v4.31.0/en/main_classes/trainer#transformers.TrainingArguments
    output_dir = Path(f'~/beyond-scale-language-data-diversity/cs197/models/uspto-llama').expanduser()
    training_args = TrainingArguments(
        output_dir=output_dir, 
        num_train_epochs = 1,
        # max_steps=max_steps,  # TODO: hard to fix, see above
        per_device_train_batch_size=2,
        gradient_accumulation_steps=16,
        # gradient_accumulation_steps=gradient_accumulation_steps,  # based on alpaca https://github.com/tatsu-lab/stanford_alpaca, allows to process effective_batch_size = gradient_accumulation_steps * batch_size, num its to accumulate before opt update step
        # gradient_checkpointing = gradient_checkpointing,  # TODO depending on hardware set to true?
        optim= "adamw_bnb_8bit", # "paged_adamw_32bit",  # David hall says to keep 32bit opt https://arxiv.org/pdf/2112.11446.pdf TODO: if we are using brain float 16 bf16 should we be using 32 bit? are optimizers always fb32?  https://discuss.huggingface.co/t/is-there-a-paged-adamw-16bf-opim-option/51284
        warmup_steps=500,  # TODO: once real training starts we can select this number for llama v2, what does llama v2 do to make it stable while v1 didn't?
        warmup_ratio=0.03,  # copying alpaca for now, number of steps for a linear warmup, TODO once real training starts change? 
        # weight_decay=0.01,  # TODO once real training change?
        weight_decay=0.00,  # TODO once real training change?
        learning_rate = .1,  # TODO once real training change? anything larger than -3 I've had terrible experiences with
        max_grad_norm=1.0, # TODO once real training change?
        # lr_scheduler_type="cosine",  # TODO once real training change? using what I've seen most in vision 
        save_steps=2000,  # alpaca does 2000, other defaults were 500
        remove_unused_columns=True,  # TODO don't get why https://stackoverflow.com/questions/76879872/how-to-use-huggingface-hf-trainer-train-with-custom-collate-function/76929999#76929999 , https://claude.ai/chat/475a4638-cee3-4ce0-af64-c8b8d1dc0d90
        # report_to="wandb",  # change to wandb!
        # fp16=False,  # never ever set to True
        # bf16=torch.cuda.get_device_capability(torch.cuda.current_device())[0] >= 8,  # if >= 8 ==> brain float 16 available or set to True if you always want fp32
        evaluation_strategy='steps',
        eval_accumulation_steps=1
    )

    # TODO: might be nice to figure our how llamav2 counts the number of token's they've trained on

    def custom_collate_fn(data: list[dict[str, str]], tokenizer: PreTrainedTokenizer) -> dict[str, torch.Tensor]:
        """ trains on first occurence of eos
        
        ref: https://discuss.huggingface.co/t/why-does-the-falcon-qlora-tutorial-code-use-eos-token-as-pad-token/45954/13?u=brando 
        ref: https://chat.openai.com/share/02d16770-a1f3-4bf4-8fc2-464286daa8a1
        ref: https://claude.ai/chat/80565d1f-ece3-4fad-87df-364ce57aec15 on when to call .clone()
        """
        # we are training full context length forllama so remove code bellow, if it triesto pad hopefully it throws an error
        # -- Ensure tokenizer has a padding token
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        # -- Extract sequences
        # sequences: list[str] = [example.get("text", "") or "" for example in data]
        sequences: list[str] = []
        for idx, example in enumerate(data):
            # Retrieve the value for "text" from the dictionary or default to an empty string if not present or falsy. ref: https://chat.openai.com/share/bead51fe-2acf-4f05-b8f7-b849134bbfd4
            text: str = example.get("text", "") or ""
            sequences.append(text)
        # -- Tokenize the sequences
        tokenized_data = tokenizer(sequences, padding="max_length", max_length=max_length, truncation=True, return_tensors="pt")
        tokenized_data["labels"] = tokenized_data["input_ids"].clone()  # labels is hardcoded in HF so put it!
        # -- Set the mask value for the first eos_token in each sequence to 1
        eos_token_id = tokenizer.eos_token_id
        for idx, input_ids in enumerate(tokenized_data["input_ids"]):
            # Find all occurrences of eos_token
            eos_positions = (input_ids == eos_token_id).nonzero(as_tuple=True)[0]
            if eos_positions.nelement() > 0:  # Check if eos_token is present
                first_eos_position = eos_positions[0]
                tokenized_data["attention_mask"][idx, first_eos_position] = 1  # Set the mask value to 1
                
                # Assert that the label for the first occurrence of eos_token is eos_token_id
                assert tokenized_data["labels"][idx, first_eos_position] == eos_token_id, "The label for the first eos_token is incorrect!"
                
                # For all subsequent occurrences of eos_token, set their labels to -100
                for subsequent_eos_position in eos_positions[1:]:
                    tokenized_data["labels"][idx, subsequent_eos_position] = -100
                    assert tokenized_data["labels"][idx, subsequent_eos_position] == -100, "The label for the subsequent_eos_position incorrect! Should be -100."
        return tokenized_data

    trainer = Trainer(
        model=model,
        args=training_args,  
        train_dataset=uspto_train_dataset,
        # eval_dataset=openwebtext_eval_dataset,
        tokenizer=tokenizer,
        data_collator=lambda data: custom_collate_fn(data, tokenizer=tokenizer)
    )

    # - Train
    cuda_visible_devices = os.environ.get('CUDA_VISIBLE_DEVICES')
    if cuda_visible_devices is not None:
        print(f"CUDA_VISIBLE_DEVICES = {cuda_visible_devices}")
    trainer.train()
    trainer.save_model(output_dir=output_dir)  # TODO is this relaly needed? https://discuss.huggingface.co/t/do-we-need-to-explicity-save-the-model-if-the-save-steps-is-not-a-multiple-of-the-num-steps-with-hf/56745
    print('Done!\a')


def reinitialize_weights(model) -> None:
    for module in model.modules():
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0, std=0.02)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)

def main_generate_smaller_model():
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

if __name__ == '__main__':
    import time
    start = time.time()
    # main_generate_smaller_model() 
    main_reinit_model()
    print('Done!\a\a\a')