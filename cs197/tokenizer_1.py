from transformers import AutoTokenizer, GPT2Tokenizer
from datasets import load_dataset, Dataset

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

if tokenizer.pad_token_id is None:
    tokenizer.pad_token = tokenizer.eos_token

def preprocess(examples):
    tokenized_output = tokenizer(examples['text'], max_length=4096, truncation=True, padding='max_length', return_tensors='pt')
    examples['input_ids'] = tokenized_output['input_ids']
    examples['attention_mask'] = tokenized_output['attention_mask']
    examples['token_length'] = tokenized_output['attention_mask'].sum(dim=1)
    return examples

uspto_dataset = load_dataset('allyc/My-Dataset', 'uspto', split='train', cache_dir='/lfs/skampere1/0/allyc/beyond-scale-language-data-diversity/cs197/data/uspto/train').with_format('torch')
print("---USPTO DATASET LOADED---")
uspto_tokenized_dataset = uspto_dataset.map(preprocess, batched=True)
print("---MAPPING DONE---")
# uspto_total_token_length = sum(examples['token_length'] for examples in uspto_tokenized_dataset)
# print("Total token length of uspto dataset: ", uspto_total_token_length)
print(uspto_tokenized_dataset[0])
uspto_tokenized_dataset.save_to_disk("~/beyond-scale-language-data-diversity/cs197/data/uspto/train-tokenized")
