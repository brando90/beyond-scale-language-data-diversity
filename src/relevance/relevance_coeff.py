import os
import time
import datetime

import glob

from diversity.task2vec import Task2Vec 
from diversity import task_similarity
from diversity.div_coeff import cross_diversity_coefficient

from pathlib import Path

import torch
import torch.nn as nn
from transformers import AutoModelForCausalLM , AutoTokenizer
from datasets import load_dataset

import fire

import multiprocessing

from pdb import set_trace as st

def relevance_coeff_task2vec_via_full_embed_dataset(
        dataset_source,
        dataset_target,
        probe_network: nn.Module,
        batch_size: int = 8,
        seed: int = 42, 
        buffer_size: int = 500_000, 
        distance = 'cosine',
        verbose: bool = False,
        debug: bool = False,
        shuffle: bool = True,  # False for faster debugging/testing but it won't be shuffled
        ) -> dict:
    """
    Relevance coefficient with Task2Vec via **full dataset embedding comparison**.
    Given a source (train) dataset and a target (test/eval) dataset/benchmark,
    compute how much the source dataset is relevant/similar/aligned with the target:

        rel_full = Rel_Full(S, T; f_w) = 1 - d(e(D_S; t2v) , e(D_T; t2v))

    by comparing embedding the entire dataset or a large/huge batch. 
    This is in contrast to sampling batches, embedding them via task2vec and comparing a sample of batches from the
    source and target datasets. 

    Note: there is no sense of number of batches here, so num_batches = 1 effectively + if CIs needed need to be with wrt batch examples. 
    """
    # - Get target shuffled data
    shuffled_dataset = dataset_source.shuffle(buffer_size=buffer_size, seed=seed) if shuffle else dataset_source
    batch = shuffled_dataset.take(batch_size)
    
    # - Get Task2Vec embedding for batch
    if not debug:
        embedding_target, loss_target = Task2Vec(probe_network, classifier_opts={'seed': seed}).embed(batch)
    else:
        embedding_target, loss_target = Task2Vec(probe_network, classifier_opts={'break_early': True, 'seed': seed}).embed(batch, epochs=1)  # only for debugging
    print(f'{loss_target=}\n{embedding_target=}\n') if verbose else None

    # - Get source shuffled data
    shuffled_dataset = dataset_target.shuffle(buffer_size=buffer_size, seed=seed)
    batch = shuffled_dataset.take(batch_size)
    
    # - Get Task2Vec embedding for batch
    if not debug:
        embedding_source, loss_source = Task2Vec(probe_network, classifier_opts={'seed': seed}).embed(batch)
    else:
        embedding_source, loss_source = Task2Vec(probe_network, classifier_opts={'break_early': True, 'seed': seed}).embed(batch, epochs=1)  # only for debugging
    print(f'{loss_target=}\n{embedding_target=}\n') if verbose else None

    # - Store results 
    embeddings, losses = [], []
    embeddings.append({'embedding_target': embedding_target, 'embedding_source': embedding_source})
    losses.append({'loss_target': loss_target, 'loss_source': loss_source})
 
    # - Compute relevance/alignment coeff
    distance_matrix = task_similarity.pdist([embedding_target, embedding_source], distance=distance)
    rel = 1 - distance_matrix[0, 1]
    rel_ci = task_similarity.stats_of_distance_matrix(distance_matrix)[1]

    # - Results
    results: dict = {'rel': rel, 'rel_ci': rel_ci,
                    'embeddings': embeddings,
                    'distance_matrix': distance_matrix,
                    'losses': losses,
                    'batch_size': batch_size}
    return results

# -- Tests

def load_mdl_and_tok(pretrained_model_name_or_path):
    torch_dtype = torch.bfloat16 if torch.cuda.is_bf16_supported() else torch.float32 
    mdl = AutoModelForCausalLM.from_pretrained(pretrained_model_name_or_path, torch_dtype=torch_dtype, trust_remote_code=True)
    device = torch.device(f"cuda:{0}" if torch.cuda.is_available() else "cpu")
    mdl = mdl.to(device)
    tok = AutoTokenizer.from_pretrained(pretrained_model_name_or_path, padding_side="right", trust_remote_code=True)
    tok.pad_token = tok.eos_token if tok.pad_token_id is None else tok.pad_token
    return mdl, tok

def load_math_test(dataset_source, tokenizer, num_proc=None, batched: bool = True, max_length=128, truncation=True):
    jsonl_files: list[str] = glob.glob(os.path.expanduser(dataset_source), recursive=True)
    assert len(jsonl_files) > 1, f'You have some error only 1 file, likely you are missing /**/*.json from your path to the data set {dataset_source=}.'
    dataset_source = load_dataset('json', data_files=jsonl_files, split='test')
    ds_src = load_dataset(dataset_source)
    # to train str
    raw_str_2_train_str = lambda examples : {'text': [f'problem: {prob}\n\nsolution: {sol}' for prob, sol in zip(examples['problems'], examples['solutions'])]}
    train_str_ds = ds_src.map(raw_str_2_train_str, batched=batched, num_proc=num_proc)
    # to train tokenized ds
    tok_ds = tok_ds.with_format('torch')  
    tokenize_function = lambda examples: tokenizer(examples['text'], padding='max_length', max_length=max_length, truncation=truncation, return_tensors='pt')
    tok_ds = train_str_ds.map(tokenize_function, batched=batched, remove_columns=train_str_ds.column_names, num_proc=num_proc)
    # get lm dataset
    # usually you do group_texts, packing, or mask extra eos's
    lm_ds = tok_ds
    return lm_ds, lm_ds

def _test_sanity_check_align_to_yourself_is_very_high_or_1(
        dataset_source: str = '~/beyond-scale-language-data-diversity/data/MATH/test/**/*.json',
        dataset_target: str = '~/beyond-scale-language-data-diversity/data/MATH/test/**/*.json',

        pretrained_model_name_or_path: str = 'gpt2',

        batch_size: int = 8,
        max_length: int = 128,

        num_proc: int = ((6 * multiprocessing.cpu_count()) // 8),
):
    # -- Get probe network
    mdl, tok = load_mdl_and_tok(pretrained_model_name_or_path)

    # -- Load data set
    print(f'\n-Load the dataset')
    src_dataset = load_math_test(dataset_source, tok, max_length=max_length, num_proc=num_proc)
    target_dataset = load_math_test(dataset_target, tok, max_length=max_length, num_proc=num_proc)

    # -- Compute alignment
    results = relevance_coeff_task2vec_via_full_embed_dataset(src_dataset, target_dataset, mdl, batch_size=batch_size)
    print(f'{results=}')

def main():
    _test_sanity_check_align_to_yourself_is_very_high_or_1()

if __name__ == "__main__":
    import time
    start_time = time.time()
    fire.Fire(main)
    print(f"Time taken: {time.time() - start_time:.2f} seconds, or {(time.time() - start_time) / 60:.2f} minutes, or {(time.time() - start_time) / 3600:.2f} hours.\a")
