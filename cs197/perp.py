from transformers import GPT2LMHeadModel, GPT2TokenizerFast

device = "cuda"
model = GPT2LMHeadModel.from_pretrained('/lfs/skampere1/0/allyc/beyond-scale-language-data-diversity/cs197/models/pubmed-uspto-gpt2-medium/checkpoint-68000').to(device)
tokenizer = GPT2TokenizerFast.from_pretrained("gpt2")

from datasets import load_dataset

test = load_dataset("wikitext", "wikitext-2-raw-v1", split="test[:1000]")
# test = load_dataset('venketh/SlimPajama-62B', split='validation[:1000]')
encodings = tokenizer("\n\n".join(test["text"]), return_tensors="pt")

import torch
from tqdm import tqdm

max_length = model.config.n_positions
stride = 1024
seq_len = encodings.input_ids.size(1)

nlls = []
prev_end_loc = 0
for begin_loc in tqdm(range(0, seq_len, stride)):
    end_loc = min(begin_loc + max_length, seq_len)
    trg_len = end_loc - prev_end_loc  # may be different from stride on last loop
    input_ids = encodings.input_ids[:, begin_loc:end_loc].to(device)
    target_ids = input_ids.clone()
    target_ids[:, :-trg_len] = -100

    with torch.no_grad():
        outputs = model(input_ids, labels=target_ids)

        # loss is calculated using CrossEntropyLoss which averages over valid labels
        # N.B. the model only calculates loss over trg_len - 1 labels, because it internally shifts the labels
        # to the left by 1.
        neg_log_likelihood = outputs.loss

    nlls.append(neg_log_likelihood)

    prev_end_loc = end_loc
    if end_loc == seq_len:
        break

ppl = torch.exp(torch.stack(nlls).mean())
print(ppl)