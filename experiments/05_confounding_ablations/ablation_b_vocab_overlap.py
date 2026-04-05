"""
Ablation B: Vocabulary overlap between training and evaluation datasets.

Computes token-level Jaccard similarity between each training set (USPTO,
PubMed, USPTO+PubMed) and evaluation sets (C4, OpenWebText). If high-diversity
training datasets don't have higher overlap with eval data, then vocabulary
overlap is ruled out as a confounding factor for the diversity→performance
correlation.

Usage:
    python experiments/05_confounding_ablations/ablation_b_vocab_overlap.py
"""
from pathlib import Path
from collections import Counter

import pandas as pd
from datasets import load_dataset
from transformers import AutoTokenizer

OUTPUT_DIR = Path(__file__).parent / "expt_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Training datasets with their diversity coefficients
TRAIN_DATASETS = {
    "USPTO": {
        "path": "UDACA/PileSubsets", "name": "uspto",
        "div_coeff": 0.158,
    },
    "PubMed": {
        "path": "UDACA/PileSubsets", "name": "pubmed",
        "div_coeff": 0.168,
    },
}

# Evaluation datasets
EVAL_DATASETS = {
    "C4": {
        "path": "allenai/c4", "name": "en",
    },
    "OpenWebText2": {
        "path": "openwebtext", "name": None,
    },
}

NUM_SAMPLES = 5000  # samples per dataset for vocabulary computation
MAX_SEQ_LEN = 512
SEED = 42


def get_token_set(dataset_config: dict, tokenizer, num_samples: int) -> tuple[set, Counter]:
    """Get the set and frequency distribution of tokens in a dataset."""
    ds = load_dataset(
        dataset_config["path"],
        dataset_config.get("name"),
        split="train",
        streaming=True,
        trust_remote_code=True,
    )

    token_counter = Counter()
    n = 0
    for example in ds:
        text = example.get("text", "")
        if not text or len(text.strip()) < 20:
            continue
        tokens = tokenizer.encode(text[:MAX_SEQ_LEN * 4], add_special_tokens=False)[:MAX_SEQ_LEN]
        token_counter.update(tokens)
        n += 1
        if n >= num_samples:
            break

    return set(token_counter.keys()), token_counter


def jaccard_similarity(set_a: set, set_b: set) -> float:
    """Compute Jaccard similarity between two sets."""
    intersection = len(set_a & set_b)
    union = len(set_a | set_b)
    return intersection / union if union > 0 else 0.0


def weighted_jaccard(counter_a: Counter, counter_b: Counter) -> float:
    """Compute weighted Jaccard similarity (min/max of counts)."""
    all_tokens = set(counter_a.keys()) | set(counter_b.keys())
    min_sum = sum(min(counter_a.get(t, 0), counter_b.get(t, 0)) for t in all_tokens)
    max_sum = sum(max(counter_a.get(t, 0), counter_b.get(t, 0)) for t in all_tokens)
    return min_sum / max_sum if max_sum > 0 else 0.0


def main():
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    print(f"Tokenizer vocab size: {tokenizer.vocab_size}")

    # Compute token sets for all datasets
    print("\n--- Loading training datasets ---")
    train_data = {}
    for name, config in TRAIN_DATASETS.items():
        print(f"  {name}...")
        token_set, token_counter = get_token_set(config, tokenizer, NUM_SAMPLES)
        train_data[name] = {"set": token_set, "counter": token_counter}
        print(f"    Unique tokens: {len(token_set)}, Total tokens: {sum(token_counter.values())}")

    # Simulate USPTO+PubMed by combining
    combined_set = train_data["USPTO"]["set"] | train_data["PubMed"]["set"]
    combined_counter = train_data["USPTO"]["counter"] + train_data["PubMed"]["counter"]
    train_data["USPTO+PubMed"] = {"set": combined_set, "counter": combined_counter}
    print(f"  USPTO+PubMed (combined): Unique tokens: {len(combined_set)}, Total tokens: {sum(combined_counter.values())}")

    print("\n--- Loading evaluation datasets ---")
    eval_data = {}
    for name, config in EVAL_DATASETS.items():
        print(f"  {name}...")
        token_set, token_counter = get_token_set(config, tokenizer, NUM_SAMPLES)
        eval_data[name] = {"set": token_set, "counter": token_counter}
        print(f"    Unique tokens: {len(token_set)}, Total tokens: {sum(token_counter.values())}")

    # Compute Jaccard similarities
    print("\n--- Computing Jaccard similarities ---")
    rows = []
    div_coeffs = {
        "USPTO": 0.158, "PubMed": 0.168, "USPTO+PubMed": 0.195,
    }

    for train_name, td in train_data.items():
        for eval_name, ed in eval_data.items():
            jacc = jaccard_similarity(td["set"], ed["set"])
            w_jacc = weighted_jaccard(td["counter"], ed["counter"])

            row = {
                "train_dataset": train_name,
                "eval_dataset": eval_name,
                "div_coeff": div_coeffs[train_name],
                "jaccard_type_overlap": jacc,
                "weighted_jaccard_overlap": w_jacc,
                "train_unique_tokens": len(td["set"]),
                "eval_unique_tokens": len(ed["set"]),
                "shared_tokens": len(td["set"] & ed["set"]),
            }
            rows.append(row)
            print(f"  {train_name} → {eval_name}: "
                  f"Jaccard={jacc:.4f}, Weighted={w_jacc:.4f}, "
                  f"Shared={row['shared_tokens']}")

    df = pd.DataFrame(rows)
    csv_path = OUTPUT_DIR / "ablation_b_vocab_overlap.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved → {csv_path}")

    # Analysis: Does higher diversity correlate with higher vocab overlap?
    print(f"\n{'='*60}")
    print("ANALYSIS: Does diversity correlate with vocab overlap?")
    print(f"{'='*60}")
    for eval_name in EVAL_DATASETS:
        sub = df[df["eval_dataset"] == eval_name].sort_values("div_coeff")
        print(f"\nEval on {eval_name}:")
        for _, row in sub.iterrows():
            print(f"  {row['train_dataset']:15s} div={row['div_coeff']:.3f}  "
                  f"Jaccard={row['jaccard_type_overlap']:.4f}  "
                  f"Weighted={row['weighted_jaccard_overlap']:.4f}")

        # Check if overlap increases with diversity
        overlaps = sub["jaccard_type_overlap"].values
        divs = sub["div_coeff"].values
        if overlaps[-1] > overlaps[0]:
            print(f"  ⚠ WARNING: Overlap increases with diversity "
                  f"({overlaps[0]:.4f} → {overlaps[-1]:.4f})")
            print(f"    This means vocab overlap could be a confounder")
        else:
            print(f"  ✓ Overlap does NOT increase with diversity "
                  f"({overlaps[0]:.4f} → {overlaps[-1]:.4f})")
            print(f"    Vocab overlap is ruled out as a confounder")


if __name__ == "__main__":
    main()
