"""
Compute baseline diversity metrics (Vendi Score, N-gram diversity, mean embedding cosine)
alongside Task2Vec diversity coefficient on the same datasets and batches.

Addresses OpenReview criticism: "no comparison to simpler diversity baselines."

Usage:
    conda activate beyond_scale_div_coeff
    pip install vendi-score  # for Vendi Score
    python experiments/02_baseline_diversity_metrics/compute_baseline_metrics.py

Output: experiments/02_baseline_diversity_metrics/expt_results/baseline_comparison.csv
"""
import argparse
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from scipy.spatial.distance import cosine
from transformers import AutoModel, AutoTokenizer, GPT2LMHeadModel

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from datasets import load_dataset
from diversity.div_coeff import get_diversity_coefficient
from diversity.task2vec import Task2Vec

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
DATASETS_CONFIG = {
    "c4": {"path": "allenai/c4", "name": "en", "split": "train", "streaming": True},
    "wikitext": {"path": "wikitext", "name": "wikitext-103-v1", "split": "train", "streaming": True},
    "the_pile": {"path": "monology/pile-uncopyrighted", "split": "train", "streaming": True},
    "pile_cc": {"path": "conceptofmind/pile_cc", "split": "train", "streaming": True},
    "pubmed": {"path": "UDACA/PileSubsets", "name": "pubmed", "split": "train", "streaming": True},
    "uspto": {"path": "UDACA/PileSubsets", "name": "uspto", "split": "train", "streaming": True},
    "hacker_news": {"path": "conceptofmind/pile_hacker_news", "split": "train", "streaming": True},
    "nih_exporter": {"path": "conceptofmind/pile_nih-exporter", "split": "train", "streaming": True},
    "openwebtext": {"path": "Skylion007/openwebtext", "split": "train", "streaming": True},
}

OUT_DIR = Path(__file__).parent / "expt_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)

NUM_BATCHES = 100  # number of batches to sample per dataset
BATCH_SIZE = 512   # tokens per batch
MAX_SEQ_LEN = 128  # max sequence length for tokenization
SEED = 42


# ---------------------------------------------------------------------------
# 1. N-gram diversity
# ---------------------------------------------------------------------------
def ngram_diversity(token_ids_list: list[list[int]], n: int = 2) -> float:
    """Compute distinct N-grams / total N-grams across all sequences.

    Args:
        token_ids_list: list of tokenized sequences (each a list of int)
        n: N-gram order (1=unigram, 2=bigram, etc.)

    Returns:
        ratio of distinct N-grams to total N-grams (0 to 1)
    """
    all_ngrams = []
    for token_ids in token_ids_list:
        for i in range(len(token_ids) - n + 1):
            all_ngrams.append(tuple(token_ids[i:i + n]))
    if not all_ngrams:
        return 0.0
    return len(set(all_ngrams)) / len(all_ngrams)


def compute_ngram_diversity_batched(
    batches: list[list[list[int]]], ns: list[int] = [1, 2, 3, 4]
) -> dict[str, float]:
    """Compute N-gram diversity for each N, averaged across batches.

    For fair comparison with Task2Vec (which measures inter-batch diversity),
    we compute diversity within each batch then average, AND compute pairwise
    N-gram overlap between batches.
    """
    results = {}

    # Within-batch diversity (averaged)
    for n in ns:
        per_batch = [ngram_diversity(batch, n) for batch in batches]
        results[f"ngram_{n}_within"] = float(np.mean(per_batch))

    # Cross-batch diversity: pairwise Jaccard distance of N-gram sets
    for n in ns:
        batch_ngram_sets = []
        for batch in batches:
            ngrams = set()
            for token_ids in batch:
                for i in range(len(token_ids) - n + 1):
                    ngrams.add(tuple(token_ids[i:i + n]))
            batch_ngram_sets.append(ngrams)

        # Pairwise Jaccard distance
        distances = []
        for i in range(len(batch_ngram_sets)):
            for j in range(i + 1, len(batch_ngram_sets)):
                intersection = len(batch_ngram_sets[i] & batch_ngram_sets[j])
                union = len(batch_ngram_sets[i] | batch_ngram_sets[j])
                if union > 0:
                    distances.append(1.0 - intersection / union)
        results[f"ngram_{n}_cross_jaccard"] = float(np.mean(distances)) if distances else 0.0

    return results


# ---------------------------------------------------------------------------
# 2. Mean embedding cosine distance
# ---------------------------------------------------------------------------
def _get_batch_embedding(
    texts: list[str],
    model: AutoModel,
    tokenizer: AutoTokenizer,
    device: str,
    max_seq_len: int,
) -> np.ndarray:
    """Get a single embedding vector for a batch by mean-pooling per-sample embeddings.

    Uses the last token embedding (causal LM style) instead of mean-pooling over
    sequence length, which produces more discriminative batch representations.
    """
    model.eval()
    sample_embeddings = []
    sub_batch_size = 32

    for start in range(0, len(texts), sub_batch_size):
        sub_batch = texts[start:start + sub_batch_size]
        inputs = tokenizer(
            sub_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_len,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            # Use last non-padding token embedding (more discriminative for causal LMs)
            seq_lens = inputs["attention_mask"].sum(dim=1) - 1  # index of last token
            hidden = outputs.last_hidden_state
            last_token_embs = hidden[torch.arange(hidden.size(0)), seq_lens]
            sample_embeddings.append(last_token_embs.cpu().numpy())

    all_emb = np.concatenate(sample_embeddings, axis=0)
    # L2-normalize each sample, then average
    norms = np.linalg.norm(all_emb, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    all_emb_normed = all_emb / norms
    return all_emb_normed.mean(axis=0)


def compute_mean_embedding_diversity(
    batches: list[list[str]],
    model: AutoModel,
    tokenizer: AutoTokenizer,
    device: str = "cuda",
    max_seq_len: int = 128,
) -> float:
    """Compute pairwise cosine distance of mean hidden-state embeddings per batch.

    For each batch: tokenize → forward pass → get last-token embeddings → L2-normalize →
    mean-pool → get one vector per batch. Then compute mean pairwise cosine distance.
    This is the "mean embedding cosine" baseline that reviewers ask about.
    """
    batch_embeddings = []
    for batch_texts in batches:
        emb = _get_batch_embedding(batch_texts, model, tokenizer, device, max_seq_len)
        batch_embeddings.append(emb)

    # Pairwise cosine distance
    n = len(batch_embeddings)
    distances = []
    for i in range(n):
        for j in range(i + 1, n):
            d = cosine(batch_embeddings[i], batch_embeddings[j])
            distances.append(d)

    return float(np.mean(distances)) if distances else 0.0


# ---------------------------------------------------------------------------
# 3. Vendi Score
# ---------------------------------------------------------------------------
def compute_vendi_score(
    batches: list[list[str]],
    model: AutoModel,
    tokenizer: AutoTokenizer,
    device: str = "cuda",
    max_seq_len: int = 128,
    max_samples: int = 500,
) -> float:
    """Compute Vendi Score using GPT-2 embeddings.

    Vendi Score = exp(Shannon entropy of eigenvalues of normalized kernel matrix).
    Uses mean-pooled GPT-2 embeddings as feature representations.

    We subsample to max_samples for tractability (O(N^2) kernel + eigendecomp).
    """
    try:
        from vendi_score import vendi
    except ImportError:
        print("WARNING: vendi-score not installed. pip install vendi-score")
        return float("nan")

    model.eval()
    all_embeddings = []

    # Flatten batches and subsample
    all_texts = []
    for batch_texts in batches:
        all_texts.extend(batch_texts)
    if len(all_texts) > max_samples:
        rng = np.random.RandomState(SEED)
        indices = rng.choice(len(all_texts), max_samples, replace=False)
        all_texts = [all_texts[i] for i in indices]

    # Embed using last-token representation (more discriminative for causal LMs)
    sub_batch_size = 32
    for start in range(0, len(all_texts), sub_batch_size):
        sub_batch = all_texts[start:start + sub_batch_size]
        inputs = tokenizer(
            sub_batch,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=max_seq_len,
        ).to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            seq_lens = inputs["attention_mask"].sum(dim=1) - 1
            hidden = outputs.last_hidden_state
            last_token_embs = hidden[torch.arange(hidden.size(0)), seq_lens]
            all_embeddings.append(last_token_embs.cpu().numpy())

    X = np.concatenate(all_embeddings, axis=0)

    # Normalize for cosine similarity kernel
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.maximum(norms, 1e-8)
    X_normed = X / norms

    # Cosine similarity kernel
    K = X_normed @ X_normed.T

    # Vendi Score from kernel matrix
    score = vendi.score_K(K)
    return float(score)


# ---------------------------------------------------------------------------
# 4. Sample batches from dataset
# ---------------------------------------------------------------------------
def sample_batches_from_dataset(
    dataset_name: str,
    tokenizer: AutoTokenizer,
    num_batches: int = NUM_BATCHES,
    batch_size: int = BATCH_SIZE,
    max_seq_len: int = MAX_SEQ_LEN,
    seed: int = SEED,
) -> tuple[list[list[str]], list[list[list[int]]]]:
    """Sample text batches from a dataset. Returns (text_batches, token_batches).

    Each batch is a list of text strings / token id lists.
    """
    config = DATASETS_CONFIG[dataset_name]
    load_kwargs = {"path": config["path"], "split": config["split"], "streaming": config["streaming"]}
    if "name" in config:
        load_kwargs["name"] = config["name"]
    if config["streaming"]:
        load_kwargs["trust_remote_code"] = True

    print(f"  Loading {dataset_name}...")
    ds = load_dataset(**load_kwargs)

    # Determine text column
    if config["streaming"]:
        first = next(iter(ds))
    else:
        first = ds[0]
    text_col = "text" if "text" in first else list(first.keys())[0]

    # Shuffle and collect samples
    if config["streaming"]:
        ds = ds.shuffle(seed=seed, buffer_size=10_000)

    text_batches = []
    token_batches = []
    current_text_batch = []
    current_token_batch = []

    for sample in ds:
        text = sample[text_col]
        if not text or len(text.strip()) < 10:
            continue
        tokens = tokenizer.encode(text, truncation=True, max_length=max_seq_len)
        if len(tokens) < 5:
            continue

        current_text_batch.append(text[:max_seq_len * 5])  # rough char limit
        current_token_batch.append(tokens)

        if len(current_text_batch) >= batch_size:
            text_batches.append(current_text_batch)
            token_batches.append(current_token_batch)
            current_text_batch = []
            current_token_batch = []

            if len(text_batches) >= num_batches:
                break

    print(f"  Sampled {len(text_batches)} batches of {batch_size} samples from {dataset_name}")
    return text_batches, token_batches


# ---------------------------------------------------------------------------
# 5. Main
# ---------------------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(description="Compute baseline diversity metrics")
    parser.add_argument("--datasets", nargs="+", default=list(DATASETS_CONFIG.keys()),
                        help="Datasets to evaluate")
    parser.add_argument("--num_batches", type=int, default=NUM_BATCHES)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--skip_task2vec", action="store_true",
                        help="Skip Task2Vec (use if already computed)")
    parser.add_argument("--skip_vendi", action="store_true",
                        help="Skip Vendi Score (slow, O(N^2))")
    args = parser.parse_args()

    np.random.seed(SEED)
    torch.manual_seed(SEED)

    print(f"Device: {args.device}")
    print(f"Datasets: {args.datasets}")
    print(f"Batches: {args.num_batches} x {args.batch_size}")

    # Load models
    model_name = "gpt2"
    print(f"\nLoading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    tokenizer.pad_token = tokenizer.eos_token
    embedding_model = AutoModel.from_pretrained(model_name).to(args.device)

    all_results = []

    for dataset_name in args.datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*60}")

        try:
            text_batches, token_batches = sample_batches_from_dataset(
                dataset_name, tokenizer, args.num_batches, args.batch_size
            )
        except Exception as e:
            print(f"  ERROR loading {dataset_name}: {e}")
            continue

        result = {"dataset": dataset_name}

        # --- N-gram diversity ---
        print("  Computing N-gram diversity...")
        ngram_results = compute_ngram_diversity_batched(token_batches)
        result.update(ngram_results)
        for k, v in ngram_results.items():
            print(f"    {k}: {v:.4f}")

        # --- Mean embedding cosine ---
        print("  Computing mean embedding cosine diversity...")
        emb_div = compute_mean_embedding_diversity(
            text_batches, embedding_model, tokenizer, args.device
        )
        result["mean_embedding_cosine"] = emb_div
        print(f"    mean_embedding_cosine: {emb_div:.4f}")

        # --- Vendi Score ---
        if not args.skip_vendi:
            print("  Computing Vendi Score...")
            vendi = compute_vendi_score(
                text_batches, embedding_model, tokenizer, args.device
            )
            result["vendi_score"] = vendi
            print(f"    vendi_score: {vendi:.4f}")

        # --- Task2Vec diversity coefficient ---
        if not args.skip_task2vec:
            print("  Computing Task2Vec diversity coefficient...")
            try:
                probe = GPT2LMHeadModel.from_pretrained(model_name).to(args.device)
                # Use existing API — need to load dataset fresh for Task2Vec
                config = DATASETS_CONFIG[dataset_name]
                load_kwargs = {"path": config["path"], "split": config["split"],
                               "streaming": config["streaming"]}
                if "name" in config:
                    load_kwargs["name"] = config["name"]
                load_kwargs["trust_remote_code"] = True
                ds = load_dataset(**load_kwargs)

                def tokenize_map(example):
                    return tokenizer(
                        example["text"] if "text" in example else example[list(example.keys())[0]],
                        truncation=True, max_length=MAX_SEQ_LEN, padding="max_length",
                        return_tensors="pt"
                    )

                t2v_results = get_diversity_coefficient(
                    ds, tokenize_map, probe, tokenizer,
                    batch_size=args.batch_size,
                    num_batches=args.num_batches,
                    seed=SEED,
                )
                result["task2vec_div_coeff"] = t2v_results["div_coeff"]
                result["task2vec_ci"] = t2v_results["div_coeff_ci"]
                print(f"    task2vec_div_coeff: {t2v_results['div_coeff']:.4f} "
                      f"± {t2v_results['div_coeff_ci']:.4f}")
                del probe
                torch.cuda.empty_cache()
            except Exception as e:
                print(f"    Task2Vec ERROR: {e}")
                result["task2vec_div_coeff"] = float("nan")

        all_results.append(result)

    # Save results
    df = pd.DataFrame(all_results)
    csv_path = OUT_DIR / "baseline_comparison.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved → {csv_path}")
    print(df.to_string(index=False))

    # Compute rank correlations between metrics
    print("\n" + "=" * 60)
    print("Rank correlations between metrics")
    print("=" * 60)
    from scipy import stats as sp_stats

    metric_cols = [c for c in df.columns if c != "dataset" and not c.endswith("_ci")]
    rank_corrs = []
    for i, m1 in enumerate(metric_cols):
        for m2 in metric_cols[i + 1:]:
            valid = df[[m1, m2]].dropna()
            if len(valid) < 3:
                continue
            sr, sp = sp_stats.spearmanr(valid[m1], valid[m2])
            rank_corrs.append({"metric_1": m1, "metric_2": m2,
                               "spearman_rho": sr, "p_value": sp, "n": len(valid)})
            print(f"  {m1} vs {m2}: Spearman ρ = {sr:.3f} (p={sp:.3e}, n={len(valid)})")

    if rank_corrs:
        rc_df = pd.DataFrame(rank_corrs)
        rc_path = OUT_DIR / "rank_correlation.csv"
        rc_df.to_csv(rc_path, index=False)
        print(f"\nSaved → {rc_path}")


if __name__ == "__main__":
    main()
