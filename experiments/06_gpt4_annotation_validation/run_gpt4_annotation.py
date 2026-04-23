"""
GPT-4 Annotation Validation of Task2Vec Diversity Metric.

Samples 30 batch pairs from datasets with known different diversity coefficients.
Presents text samples to GPT-4 and asks which batch is more diverse.
Compares GPT-4 rankings with Task2Vec diversity coefficient rankings.

Usage:
    export OPENAI_API_KEY=$(cat ~/keys/openai_bm_key_koyejolab.txt)
    python experiments/06_gpt4_annotation_validation/run_gpt4_annotation.py
"""
import json
import os
import random
import time
from pathlib import Path

import pandas as pd
from datasets import load_dataset

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------
NUM_PAIRS = 30
SAMPLES_PER_BATCH = 8  # number of text snippets shown per batch
MAX_CHARS = 500  # truncate each sample to this many chars
SEED = 42
OUTPUT_DIR = Path(__file__).parent / "expt_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Datasets and their known Task2Vec diversity coefficients
DATASET_CONFIGS = {
    "c4": {
        "path": "allenai/c4", "name": "en", "split": "train",
        "streaming": True, "text_field": "text",
        "div_coeff": 0.208,
    },
    "wikitext": {
        "path": "wikitext", "name": "wikitext-103-raw-v1", "split": "train",
        "streaming": False, "text_field": "text",
        "div_coeff": 0.207,
    },
    "openwebtext": {
        "path": "openwebtext", "name": None, "split": "train",
        "streaming": True, "text_field": "text",
        "div_coeff": 0.199,
    },
    "bookcorpus": {
        "path": "bookcorpus", "name": None, "split": "train",
        "streaming": True, "text_field": "text",
        "div_coeff": 0.160,  # books are topically narrow per domain
    },
}

# Pairs to compare: (dataset_A, dataset_B) where A should be LESS diverse than B
# according to Task2Vec. We'll present both orderings randomly to GPT-4.
CROSS_DATASET_PAIRS = [
    ("bookcorpus", "c4"),          # 0.160 vs 0.208 — large gap
    ("bookcorpus", "wikitext"),    # 0.160 vs 0.207 — large gap
    ("bookcorpus", "openwebtext"), # 0.160 vs 0.199 — medium gap
    ("openwebtext", "c4"),         # 0.199 vs 0.208 — small gap
    ("wikitext", "c4"),            # 0.207 vs 0.208 — very small gap
]


def load_samples(dataset_name: str, num_samples: int, offset: int = 0) -> list[str]:
    """Load text samples from a dataset."""
    cfg = DATASET_CONFIGS[dataset_name]
    ds = load_dataset(cfg["path"], cfg.get("name"), split=cfg["split"],
                      streaming=cfg.get("streaming", False),
                      trust_remote_code=True)

    samples = []
    text_field = cfg["text_field"]

    if cfg.get("streaming"):
        for i, example in enumerate(ds):
            if i < offset:
                continue
            if i >= offset + num_samples * 3:  # sample more to filter
                break
            text = example.get(text_field, "")
            if text and len(text.strip()) > 50:  # skip very short samples
                samples.append(text.strip()[:MAX_CHARS])
            if len(samples) >= num_samples:
                break
    else:
        indices = list(range(offset, min(offset + num_samples * 3, len(ds))))
        for idx in indices:
            text = ds[idx].get(text_field, "")
            if text and len(text.strip()) > 50:
                samples.append(text.strip()[:MAX_CHARS])
            if len(samples) >= num_samples:
                break

    return samples[:num_samples]


def build_prompt(batch_a: list[str], batch_b: list[str]) -> str:
    """Build GPT-4 prompt for diversity comparison."""
    batch_a_text = "\n---\n".join(f"Sample {i+1}: {s}" for i, s in enumerate(batch_a))
    batch_b_text = "\n---\n".join(f"Sample {i+1}: {s}" for i, s in enumerate(batch_b))

    return f"""You are evaluating the diversity of two collections of text samples. Diversity means variety in topics, writing styles, vocabulary, and subject matter within a collection.

**Collection A:**
{batch_a_text}

**Collection B:**
{batch_b_text}

Which collection exhibits greater topical and stylistic diversity? Consider:
1. How many distinct topics or subject areas are covered
2. Variation in writing style (formal, informal, technical, narrative)
3. Vocabulary diversity
4. Range of domains represented

Answer with ONLY one of: "A" or "B" (the collection that is MORE diverse).
Then on the next line, give a brief one-sentence explanation."""


def query_gpt4(prompt: str, api_key: str, model: str = "gpt-4o") -> dict:
    """Query GPT-4 API and return response."""
    import openai
    client = openai.OpenAI(api_key=api_key)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": "You are a helpful assistant that evaluates text diversity. Be concise."},
            {"role": "user", "content": prompt},
        ],
        temperature=0.0,
        max_tokens=150,
    )

    content = response.choices[0].message.content.strip()
    # Parse response: first line should be "A" or "B"
    lines = content.strip().split("\n")
    choice = lines[0].strip().upper()
    if choice not in ("A", "B"):
        # Try to find A or B in the first line
        if "A" in lines[0] and "B" not in lines[0]:
            choice = "A"
        elif "B" in lines[0] and "A" not in lines[0]:
            choice = "B"
        else:
            choice = "UNCLEAR"

    explanation = lines[1].strip() if len(lines) > 1 else ""

    return {
        "choice": choice,
        "explanation": explanation,
        "raw_response": content,
    }


def main():
    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        key_file = Path.home() / "keys" / "openai_bm_key_koyejolab.txt"
        if key_file.exists():
            api_key = key_file.read_text().strip()
        else:
            raise ValueError("Set OPENAI_API_KEY or place key in ~/keys/openai_bm_key_koyejolab.txt")

    random.seed(SEED)

    # Pre-load samples from each dataset
    print("Loading dataset samples...")
    dataset_samples = {}
    for name in DATASET_CONFIGS:
        print(f"  Loading {name}...")
        # Load enough samples for multiple batches
        dataset_samples[name] = load_samples(name, num_samples=SAMPLES_PER_BATCH * 15)
        print(f"  Got {len(dataset_samples[name])} samples from {name}")

    # Generate 30 pairs
    pairs = []
    pair_idx = 0

    # Cross-dataset pairs: 5 pairs × 6 offsets = 30 pairs
    for cross_pair in CROSS_DATASET_PAIRS:
        ds_low, ds_high = cross_pair
        div_low = DATASET_CONFIGS[ds_low]["div_coeff"]
        div_high = DATASET_CONFIGS[ds_high]["div_coeff"]

        for offset_mult in range(6):  # 6 different sample offsets per pair type
            if pair_idx >= NUM_PAIRS:
                break

            offset = offset_mult * SAMPLES_PER_BATCH
            batch_low = dataset_samples[ds_low][offset:offset + SAMPLES_PER_BATCH]
            batch_high = dataset_samples[ds_high][offset:offset + SAMPLES_PER_BATCH]

            if len(batch_low) < SAMPLES_PER_BATCH or len(batch_high) < SAMPLES_PER_BATCH:
                continue

            # Randomly assign which is "A" and which is "B"
            if random.random() < 0.5:
                batch_a, batch_b = batch_low, batch_high
                correct_answer = "B"  # B is more diverse
                a_dataset, b_dataset = ds_low, ds_high
            else:
                batch_a, batch_b = batch_high, batch_low
                correct_answer = "A"  # A is more diverse
                a_dataset, b_dataset = ds_high, ds_low

            pairs.append({
                "pair_idx": pair_idx,
                "a_dataset": a_dataset,
                "b_dataset": b_dataset,
                "a_div_coeff": DATASET_CONFIGS[a_dataset]["div_coeff"],
                "b_div_coeff": DATASET_CONFIGS[b_dataset]["div_coeff"],
                "correct_answer": correct_answer,
                "batch_a": batch_a,
                "batch_b": batch_b,
            })
            pair_idx += 1

    print(f"\nGenerated {len(pairs)} pairs for annotation")

    # Query GPT-4 for each pair
    results = []
    for p in pairs:
        print(f"\nPair {p['pair_idx']+1}/{len(pairs)}: "
              f"{p['a_dataset']} (div={p['a_div_coeff']:.3f}) vs "
              f"{p['b_dataset']} (div={p['b_div_coeff']:.3f})")

        prompt = build_prompt(p["batch_a"], p["batch_b"])

        try:
            response = query_gpt4(prompt, api_key)
        except Exception as e:
            print(f"  ERROR: {e}")
            response = {"choice": "ERROR", "explanation": str(e), "raw_response": ""}
            time.sleep(5)
            continue

        is_correct = response["choice"] == p["correct_answer"]
        print(f"  GPT-4: {response['choice']} | Correct: {p['correct_answer']} | "
              f"{'AGREE' if is_correct else 'DISAGREE'}")
        print(f"  Reason: {response['explanation']}")

        results.append({
            "pair_idx": p["pair_idx"],
            "a_dataset": p["a_dataset"],
            "b_dataset": p["b_dataset"],
            "a_div_coeff": p["a_div_coeff"],
            "b_div_coeff": p["b_div_coeff"],
            "correct_answer": p["correct_answer"],
            "gpt4_choice": response["choice"],
            "agrees": is_correct,
            "explanation": response["explanation"],
        })

        # Rate limit
        time.sleep(1)

    # Save results
    df = pd.DataFrame(results)
    csv_path = OUTPUT_DIR / "gpt4_annotations.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved annotations → {csv_path}")

    # Compute agreement statistics
    valid = df[df["gpt4_choice"].isin(["A", "B"])]
    agreement_rate = valid["agrees"].mean()
    n_agree = valid["agrees"].sum()
    n_total = len(valid)

    # Cohen's kappa (comparing to random baseline of 0.5)
    # For binary agreement: kappa = (po - pe) / (1 - pe)
    # pe = 0.5 (random chance), po = agreement_rate
    pe = 0.5
    kappa = (agreement_rate - pe) / (1 - pe) if agreement_rate != pe else 0.0

    agreement_data = {
        "agreement_rate": float(agreement_rate),
        "n_agree": int(n_agree),
        "n_total": int(n_total),
        "n_errors": int(len(df) - len(valid)),
        "cohens_kappa": float(kappa),
    }

    # Per-pair-type breakdown
    pair_type_stats = {}
    for _, row in valid.iterrows():
        key = f"{row['a_dataset']}_vs_{row['b_dataset']}"
        if key not in pair_type_stats:
            pair_type_stats[key] = {"agree": 0, "total": 0}
        pair_type_stats[key]["total"] += 1
        if row["agrees"]:
            pair_type_stats[key]["agree"] += 1

    for key, stats in pair_type_stats.items():
        stats["rate"] = stats["agree"] / stats["total"] if stats["total"] > 0 else 0
    agreement_data["per_pair_type"] = pair_type_stats

    json_path = OUTPUT_DIR / "annotation_agreement.json"
    with open(json_path, "w") as f:
        json.dump(agreement_data, f, indent=2)
    print(f"Saved agreement stats → {json_path}")

    print(f"\n{'='*60}")
    print(f"RESULTS SUMMARY")
    print(f"{'='*60}")
    print(f"Agreement rate: {n_agree}/{n_total} = {agreement_rate:.1%}")
    print(f"Cohen's kappa: {kappa:.3f}")
    print(f"\nPer pair type:")
    for key, stats in pair_type_stats.items():
        print(f"  {key}: {stats['agree']}/{stats['total']} = {stats['rate']:.1%}")
    print(f"\nInterpretation:")
    if agreement_rate > 0.8:
        print("  Strong agreement — Task2Vec captures human-recognizable diversity")
    elif agreement_rate > 0.6:
        print("  Moderate agreement — Task2Vec partially aligns with human judgment")
    elif agreement_rate > 0.5:
        print("  Weak agreement — slightly better than random")
    else:
        print("  No agreement — Task2Vec does not match human diversity intuition")


if __name__ == "__main__":
    main()
