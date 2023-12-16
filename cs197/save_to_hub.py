from transformers import AutoModel

model = AutoModel.from_pretrained('/lfs/skampere1/0/allyc/beyond-scale-language-data-diversity/cs197/models/uspto-gpt2/checkpoint-68000/')

model.push_to_hub("uspto-gpt2")
