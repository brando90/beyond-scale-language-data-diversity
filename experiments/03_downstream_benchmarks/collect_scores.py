"""
Collect downstream benchmark scores from lm-evaluation-harness output and map to diversity coefficients.

Reads: /dfs/scratch0/brando9/data/beyond_scale/eval_results/<model>_downstream/
Output: experiments/03_downstream_benchmarks/expt_results/downstream_benchmarks.csv

Usage:
    python experiments/03_downstream_benchmarks/collect_scores.py
"""
import json
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Model → diversity coefficient / family mapping (same as experiment 00)
# ---------------------------------------------------------------------------
DIVERSITY_COEFFICIENTS = {
    "GPT2_51M_1.31B_USPTO":              0.158,
    "GPT2_51M_1.31B_PubMedAbs":          0.168,
    "GPT2_51M_1.31B_USPTOAndPubMedAbs":  0.195,
    "GPT2_51M_557M_USPTO":               0.158,
    "GPT2_51M_557M_PubMedAbs":           0.168,
    "GPT2_51M_557M_USPTOAndPubMedAbs":   0.195,
    "GPT2_117M_2.2B_USPTO":              0.158,
    "GPT2_117M_2.2B_PubMedAbs":          0.168,
    "GPT2_117M_2.2B_USPTOAndPubMedAbs":  0.195,
    "GPT2_204M_USPTO":                   0.158,
    "GPT2_204M_PubMedAbs":               0.168,
    "GPT2_204M_USPTOAndPubMedAbs":       0.195,
    "GPT2_345M_2.2B_USPTO":              0.158,
    "GPT2_345M_2.2B_PubMedAbs":          0.168,
    "GPT2_345M_2.2B_USPTOAndPubMedAbs":  0.195,
    "GPT2_810M_PubMedAbs":               0.168,
    "GPT2_810M_2.2B_USPTOAndPubMedAbs":  0.195,
    "GPT2_1.5B_180M_USPTO":              0.158,
    "GPT2_1.5B_180M_PubMedAbs":          0.168,
    "GPT2_1.5B_180M_USPTOAndPubMedAbs":  0.195,
    "LLama2_Uspto_Ckpt_1":               0.158,
    "LLama2_Pubmed_Ckpt_2":              0.168,
    "LLama2_Pubmed_Ckpt_7":              0.168,
    "LLama2_Uspto_Pubmed_Ckpt_3":        0.195,
    "LLama2_Uspto_Pubmed_Ckpt_4":        0.195,
    "LLama2_Uspto_Pubmed_Ckpt_5":        0.195,
    "LLama2_Uspto_Pubmed_Ckpt_6":        0.195,
}

MODEL_FAMILY = {
    k: "LLaMA2-7B" if "LLama" in k else
    "GPT2-51M" if "51M" in k else
    "GPT2-117M" if "117M" in k else
    "GPT2-204M" if "204M" in k else
    "GPT2-345M" if "345M" in k else
    "GPT2-810M" if "810M" in k else
    "GPT2-1.5B" if "1.5B" in k else "unknown"
    for k in DIVERSITY_COEFFICIENTS
}

BENCHMARKS = ["arc_easy", "hellaswag", "winogrande", "lambada_openai"]
RESULTS_DIR = Path("/dfs/scratch0/brando9/data/beyond_scale/eval_results")
OUT_DIR = Path(__file__).parent / "expt_results"
OUT_DIR.mkdir(parents=True, exist_ok=True)


def extract_benchmark_scores(model_dir: Path) -> dict[str, float]:
    """Extract accuracy scores from lm-eval results JSON."""
    scores = {}

    # lm-eval outputs results_*.json with structure: {"results": {"benchmark": {"acc,none": val}}}
    results_files = list(model_dir.glob("**/results_*.json"))
    if not results_files:
        return scores

    for rf in results_files:
        with open(rf) as f:
            data = json.load(f)
        if "results" not in data:
            continue
        for bench_name, bench_data in data["results"].items():
            # Try acc_norm first (normalized accuracy), then acc
            for key in ["acc_norm,none", "acc,none", "acc_norm", "acc"]:
                if key in bench_data:
                    scores[bench_name] = float(bench_data[key])
                    break
    return scores


def main():
    rows = []

    for model_name, div_coeff in DIVERSITY_COEFFICIENTS.items():
        # Try both naming patterns
        for suffix in ["_downstream", ""]:
            model_dir = RESULTS_DIR / f"{model_name}{suffix}"
            if model_dir.exists():
                break
        else:
            print(f"MISSING: {model_name}")
            continue

        scores = extract_benchmark_scores(model_dir)
        if not scores:
            print(f"NO RESULTS: {model_name} (dir exists but no results JSON)")
            continue

        row = {
            "model": model_name,
            "family": MODEL_FAMILY[model_name],
            "div_coeff": div_coeff,
        }
        for bench in BENCHMARKS:
            row[bench] = scores.get(bench, None)

        rows.append(row)
        bench_str = "  ".join(f"{b}={scores.get(b, 'N/A')}" for b in BENCHMARKS)
        print(f"{model_name:45s}  div={div_coeff:.3f}  {bench_str}")

    df = pd.DataFrame(rows)
    csv_path = OUT_DIR / "downstream_benchmarks.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved → {csv_path}")
    print(f"Models with results: {len(df)}")
    print(df.to_string(index=False))


if __name__ == "__main__":
    main()
