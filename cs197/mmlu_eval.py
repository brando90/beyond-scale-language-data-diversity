from datasets import load_dataset, list_metrics
from evaluate import evaluator
from tasksource import list_tasks, load_task
from transformers import GPT2Model
from transformers import AutoTokenizer, GPT2Tokenizer, GPT2LMHeadModel, set_seed
import torch


# data = load_dataset("cais/mmlu", "abstract_algebra", split="test")
# task_evaluator = evaluator("question-answering")
# print(list_metrics())

'''
eval_results = task_evaluator.compute(
    model_or_pipeline="/lfs/skampere1/0/allyc/beyond-scale-language-data-diversity/cs197/models/uspto-gpt2",
    data=data,
    metric="squad"
)
'''
'''
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

model = GPT2Model.from_pretrained('/lfs/skampere1/0/allyc/beyond-scale-language-data-diversity/cs197/models/uspto-gpt2/')
outputs = model(**inputs)
# print(outputs)
last_hidden_states = outputs.last_hidden_state
print(last_hidden_states)
'''

from transformers import pipeline
'''
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2Model.from_pretrained('/lfs/skampere1/0/allyc/beyond-scale-language-data-diversity/cs197/models/uspto-gpt2/checkpoint-68000/')
pipe = pipeline('conversational', model=model, tokenizer=tokenizer, max_new_tokens=100)
print(pipe("This restaurant is awesome"))
'''
model = GPT2LMHeadModel.from_pretrained('/lfs/skampere1/0/allyc/beyond-scale-language-data-diversity/cs197/models/uspto-gpt2/checkpoint-68000/')
tokenizer = GPT2Tokenizer.from_pretrained('gpt2')

generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
set_seed(42)
print(generator("Hello, I'm a language model,", max_length=30, num_return_sequences=1))

