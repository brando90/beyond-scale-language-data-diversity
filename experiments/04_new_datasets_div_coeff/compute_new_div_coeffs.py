"""
Compute Task2Vec diversity coefficients for additional pre-training datasets
to expand Table 1: FineWeb, FineWeb-Edu, Dolma, RedPajama v2.

Addresses reviewer N6rW (ICLR 2025): "should use standard LLM pre-training
datasets like C4, OpenWebText, The Pile, RedPajama, SlimPajama, RefinedWeb,
Dolma, FineWeb, DCLM."

Table 1 already has 10 datasets. This adds 4 more for 14 total.

Usage:
    conda activate beyond_scale_div_coeff
    CUDA_VISIBLE_DEVICES=0 python experiments/04_new_datasets_div_coeff/compute_new_div_coeffs.py
    # Or run a single dataset:
    CUDA_VISIBLE_DEVICES=0 python experiments/04_new_datasets_div_coeff/compute_new_div_coeffs.py --dataset fineweb
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from transformers import AutoTokenizer, GPT2LMHeadModel

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from datasets import load_dataset
from diversity.div_coeff import get_diversity_coefficient

# ---------------------------------------------------------------------------
# Configuration — matches existing Table 1 parameters
# ---------------------------------------------------------------------------
OUTPUT_DIR = Path(__file__).parent / "expt_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

PROBE_MODEL = "gpt2"  # GPT-2 small, same as all existing measurements
BATCH_SIZE = 512
NUM_BATCHES = 600      # same as existing Table 1 computations
MAX_SEQ_LEN = 128
SEED = 42
BUFFER_SIZE = 500_000

# New datasets to compute diversity coefficients for
NEW_DATASETS = {
    "fineweb": {
        "path": "HuggingFaceFW/fineweb",
        "name": "default",
        "split": "train",
        "streaming": True,
        "text_column": "text",
        "description": "FineWeb — 15T token web crawl (HuggingFace)",
    },
    "fineweb_edu": {
        "path": "HuggingFaceFW/fineweb-edu",
        "name": "default",
        "split": "train",
        "streaming": True,
        "text_column": "text",
        "description": "FineWeb-Edu — educational subset of FineWeb",
    },
    "dolma": {
        "path": "allenai/dolma",
        "name": "default",
        "split": "train",
        "streaming": True,
        "text_column": "text",
        "description": "Dolma — OLMo pre-training data (Allen AI)",
    },
    "redpajama_v2": {
        "path": "togethercomputer/RedPajama-Data-V2",
        "name": "default",
        "split": "train",
        "streaming": True,
        "text_column": "raw_content",
        "description": "RedPajama v2 — 30T token web corpus (Together)",
    },
}

# Existing Table 1 values for reference/validation
EXISTING_DIV_COEFFS = {
    "c4": 0.208,
    "wikitext": 0.207,
    "the_pile": 0.230,
    "pile_cc": 0.230,
    "pubmed": 0.168,
    "uspto": 0.158,
    "hacker_news": 0.172,
    "nih_exporter": 0.164,
    "slim_pajama": 0.211,
    "openwebtext": 0.199,
}


def make_tokenize_map(tokenizer, text_column: str, max_seq_len: int):
    """Create a tokenization map function compatible with get_diversity_coefficient.

    The map function receives the result of dataset.take(batch_size), which is a
    HuggingFace IterableDataset. It must return a dataset-like object that PyTorch
    DataLoader can iterate over, yielding dicts with 'input_ids' and 'attention_mask'.

    This matches the pattern in src/diversity/main.py:178:
        tokenized_task_dataset = task_dataset.map(preprocess_function, batched=True, ...)
    """
    def preprocess(examples):
        """Tokenize examples. Do NOT use return_tensors='pt' inside .map() —
        HF .map() expects plain python lists. DataLoader handles tensorization."""
        texts = examples.get(text_column, examples.get("text", [""]))
        if isinstance(texts, str):
            texts = [texts]
        return tokenizer(
            texts,
            truncation=True,
            max_length=max_seq_len,
            padding="max_length",
        )

    def tokenize_map(take_dataset):
        # take_dataset is an IterableDataset from .take()
        # Materialize into a list, then create a regular Dataset so DataLoader can
        # call len() on it (required by Task2Vec's tqdm progress bar).
        from datasets import Dataset as HFDataset
        examples = list(take_dataset)
        texts = [ex.get(text_column, ex.get("text", "")) or "" for ex in examples]
        tokenized = tokenizer(
            texts,
            truncation=True,
            max_length=max_seq_len,
            padding="max_length",
            return_tensors="pt",
        )
        return HFDataset.from_dict({
            "input_ids": tokenized["input_ids"],
            "attention_mask": tokenized["attention_mask"],
        }).with_format("torch")

    return tokenize_map


def compute_single_dataset(dataset_name: str, config: dict, probe, tokenizer, device: str,
                           num_batches: int = NUM_BATCHES, batch_size: int = BATCH_SIZE) -> dict:
    """Compute diversity coefficient for a single dataset."""
    print(f"\n{'='*60}")
    print(f"Computing diversity coefficient for: {dataset_name}")
    print(f"  {config['description']}")
    print(f"  path={config['path']}, split={config['split']}")
    print(f"  num_batches={num_batches}, batch_size={batch_size}, seed={SEED}")
    print(f"{'='*60}")

    # Load dataset
    load_kwargs = {
        "path": config["path"],
        "split": config["split"],
        "streaming": config["streaming"],
        "trust_remote_code": True,
    }
    if config.get("name") and config["name"] != "default":
        load_kwargs["name"] = config["name"]

    try:
        ds = load_dataset(**load_kwargs)
    except Exception as e:
        print(f"  ERROR loading dataset: {e}")
        # Try without name
        load_kwargs.pop("name", None)
        try:
            ds = load_dataset(**load_kwargs)
        except Exception as e2:
            print(f"  ERROR (retry without name): {e2}")
            return {"dataset": dataset_name, "div_coeff": float("nan"),
                    "div_coeff_ci": float("nan"), "error": str(e2)}

    # Create tokenize map
    tokenize_fn = make_tokenize_map(tokenizer, config["text_column"], MAX_SEQ_LEN)

    # Compute diversity coefficient
    t_start = time.time()
    try:
        results = get_diversity_coefficient(
            ds,
            tokenize_fn,
            probe,
            tokenizer,
            batch_size=batch_size,
            num_batches=num_batches,
            seed=SEED,
            buffer_size=BUFFER_SIZE,
            streaming=config["streaming"],
        )
        elapsed = time.time() - t_start

        div_coeff = results["div_coeff"]
        div_coeff_ci = results["div_coeff_ci"]

        print(f"\n  RESULT: {dataset_name} div_coeff = {div_coeff:.4f} ± {div_coeff_ci:.4f}")
        print(f"  Time: {elapsed/60:.1f} minutes")

        # Save embeddings for reproducibility
        emb_path = OUTPUT_DIR / f"{dataset_name}_embeddings.npy"
        embeddings_array = np.array([e.hessian for e in results["embeddings"]])
        np.save(emb_path, embeddings_array)
        print(f"  Embeddings saved → {emb_path}")

        # Save distance matrix
        dist_path = OUTPUT_DIR / f"{dataset_name}_distance_matrix.npy"
        np.save(dist_path, results["distance_matrix"])

        return {
            "dataset": dataset_name,
            "div_coeff": div_coeff,
            "div_coeff_ci": div_coeff_ci,
            "num_batches": num_batches,
            "batch_size": batch_size,
            "seed": SEED,
            "elapsed_minutes": elapsed / 60,
            "error": None,
        }

    except Exception as e:
        elapsed = time.time() - t_start
        print(f"  ERROR computing div coeff: {e}")
        import traceback
        traceback.print_exc()
        return {
            "dataset": dataset_name,
            "div_coeff": float("nan"),
            "div_coeff_ci": float("nan"),
            "num_batches": num_batches,
            "batch_size": batch_size,
            "seed": SEED,
            "elapsed_minutes": elapsed / 60,
            "error": str(e),
        }


def main():
    parser = argparse.ArgumentParser(description="Compute div coeffs for new datasets")
    parser.add_argument("--dataset", type=str, default=None,
                        help="Single dataset to compute (default: all)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--num_batches", type=int, default=NUM_BATCHES)
    parser.add_argument("--batch_size", type=int, default=BATCH_SIZE)
    args = parser.parse_args()

    num_batches = args.num_batches
    batch_size = args.batch_size

    print(f"Device: {args.device}")
    print(f"Probe model: {PROBE_MODEL}")
    print(f"num_batches={num_batches}, batch_size={batch_size}, seed={SEED}")

    # Load probe model and tokenizer
    tokenizer = AutoTokenizer.from_pretrained(PROBE_MODEL)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    probe = GPT2LMHeadModel.from_pretrained(PROBE_MODEL).to(args.device)

    # Select datasets
    if args.dataset:
        if args.dataset not in NEW_DATASETS:
            print(f"Unknown dataset: {args.dataset}")
            print(f"Available: {list(NEW_DATASETS.keys())}")
            return
        datasets_to_run = {args.dataset: NEW_DATASETS[args.dataset]}
    else:
        datasets_to_run = NEW_DATASETS

    # Compute diversity coefficients
    all_results = []
    for name, config in datasets_to_run.items():
        # Check if already computed
        csv_path = OUTPUT_DIR / "new_datasets_div_coeff.csv"
        if csv_path.exists():
            existing = pd.read_csv(csv_path)
            if name in existing["dataset"].values:
                existing_row = existing[existing["dataset"] == name].iloc[0]
                if not np.isnan(existing_row["div_coeff"]):
                    print(f"\n  SKIP: {name} already computed (div_coeff={existing_row['div_coeff']:.4f})")
                    all_results.append(existing_row.to_dict())
                    continue

        result = compute_single_dataset(name, config, probe, tokenizer, args.device,
                                        num_batches=num_batches, batch_size=batch_size)
        all_results.append(result)

        # Save incrementally (in case of crash)
        df = pd.DataFrame(all_results)
        csv_path = OUTPUT_DIR / "new_datasets_div_coeff.csv"
        df.to_csv(csv_path, index=False)
        print(f"  Incremental save → {csv_path}")

    # Final summary
    print(f"\n{'='*60}")
    print("FINAL RESULTS")
    print(f"{'='*60}")

    df = pd.DataFrame(all_results)
    csv_path = OUTPUT_DIR / "new_datasets_div_coeff.csv"
    df.to_csv(csv_path, index=False)
    print(f"Saved → {csv_path}")

    print(f"\n{'Dataset':<20s} {'Div Coeff':>10s} {'CI':>8s} {'Time (min)':>10s}")
    print("-" * 52)
    for _, row in df.iterrows():
        err = f"  ERROR: {row['error']}" if row.get("error") else ""
        print(f"{row['dataset']:<20s} {row['div_coeff']:>10.4f} {row.get('div_coeff_ci', float('nan')):>8.4f} "
              f"{row.get('elapsed_minutes', 0):>10.1f}{err}")

    # Compare with existing datasets
    print(f"\n{'='*60}")
    print("COMPARISON WITH EXISTING TABLE 1 VALUES")
    print(f"{'='*60}")
    print(f"{'Dataset':<20s} {'Div Coeff':>10s}")
    print("-" * 32)
    for name, val in sorted(EXISTING_DIV_COEFFS.items(), key=lambda x: x[1]):
        print(f"{name:<20s} {val:>10.3f}")
    print("-" * 32)
    for _, row in df.iterrows():
        if not np.isnan(row["div_coeff"]):
            print(f"{row['dataset']:<20s} {row['div_coeff']:>10.4f}  ← NEW")

    # GPU cleanup
    del probe
    torch.cuda.empty_cache()
    print("\nGPU memory freed.")


if __name__ == "__main__":
    main()
