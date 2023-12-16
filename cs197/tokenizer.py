from transformers import AutoTokenizer, GPT2Tokenizer
from datasets import load_dataset, Dataset
import torch

# Replace 'bert-base-uncased' with the name or path of the tokenizer you want to use
#tokenizer = AutoTokenizer.from_pretrained('meta-llama/Llama-2-7b-hf',
#                                            # padding_side="right",
 #                                           use_fast=False, 
 #                                           trust_remote_code=True,
 #                                           use_auth_token=True
#)

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

def preprocess(examples):
    tokenized_output = tokenizer(examples['text'], max_length=1024, truncation=True, padding='max_length', return_tensors='pt')
    examples['input_ids'] = tokenized_output['input_ids']
    examples['attention_mask'] = tokenized_output['attention_mask']
    examples['token_length'] = tokenized_output['attention_mask'].sum(dim=1)
    return examples

uspto_dataset = load_dataset('allyc/My-Dataset', 'uspto', split='train', cache_dir='/lfs/skampere1/0/allyc/beyond-scale-language-data-diversity/cs197/data/uspto/train').with_format('torch')
print("---USPTO DATASET LOADED---")
uspto_tokenized_dataset = uspto_dataset.map(preprocess, batched=True)
uspto_total_token_length = sum(examples['token_length'] for examples in uspto_tokenized_dataset)
print("Total token length of uspto dataset: ", uspto_total_token_length)
print(uspto_tokenized_dataset[0])
uspto_tokenized_dataset.save_to_disk("~/beyond-scale-language-data-diversity/cs197/data/uspto/train-tokenized-1024")

pubmed_dataset = load_dataset('allyc/My-Dataset','pubmed', split='train', cache_dir='/lfs/skampere1/0/allyc/beyond-scale-language-data-diversity/cs197/data/pubmed/train').with_format('torch')
print('---PUBMED DATASET LOADED---')
pubmed_tokenized_dataset = pubmed_dataset.map(preprocess, batched=True)
pubmed_total_token_length = sum(examples['token_length'] for examples in pubmed_tokenized_dataset)
print("Total token length of pubmed dataset: ", pubmed_total_token_length)
print(pubmed_tokenized_dataset[0])
pubmed_tokenized_dataset.save_to_disk("~/beyond-scale-language-data-diversity/cs197/data/pubmed/train-tokenized-1024")