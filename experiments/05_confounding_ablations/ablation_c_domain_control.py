"""
Ablation C: Domain control analysis.

Analyzes whether domain match with evaluation data confounds the
diversity→performance relationship. Uses existing UDACA evaluation results
to compare models trained on datasets that differ in domain vs diversity.

Key comparison:
  - USPTO (div=0.158) vs PubMed (div=0.168): different domains, different diversity
  - USPTO+PubMed (div=0.195): different domain mix, highest diversity
  - If diversity drives performance (not domain match), then USPTO+PubMed
    should outperform both individual datasets regardless of eval domain.

Usage:
    python experiments/05_confounding_ablations/ablation_c_domain_control.py
"""
import json
from pathlib import Path

import pandas as pd
import numpy as np

OUTPUT_DIR = Path(__file__).parent / "expt_results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Existing MMLU results directory
EVAL_RESULTS_DIR = Path("/dfs/scratch0/brando9/data/beyond_scale/eval_results")

# Model groups for domain analysis
# Each entry: (model_dir_name, diversity, training_domain, family, tokens)
MODELS = [
    # GPT2-117M family (same size, same tokens)
    ("GPT2_117M_2.2B_USPTO", 0.158, "patents", "GPT2-117M", "2.2B"),
    ("GPT2_117M_2.2B_PubMedAbs", 0.168, "medical", "GPT2-117M", "2.2B"),
    ("GPT2_117M_2.2B_USPTOAndPubMedAbs", 0.195, "mixed", "GPT2-117M", "2.2B"),
    # GPT2-345M family
    ("GPT2_345M_2.2B_USPTO", 0.158, "patents", "GPT2-345M", "2.2B"),
    ("GPT2_345M_2.2B_PubMedAbs", 0.168, "medical", "GPT2-345M", "2.2B"),
    ("GPT2_345M_2.2B_USPTOAndPubMedAbs", 0.195, "mixed", "GPT2-345M", "2.2B"),
    # GPT2-51M (1.31B tokens)
    ("GPT2_51M_1.31B_USPTO", 0.158, "patents", "GPT2-51M-1.31B", "1.31B"),
    ("GPT2_51M_1.31B_PubMedAbs", 0.168, "medical", "GPT2-51M-1.31B", "1.31B"),
    ("GPT2_51M_1.31B_USPTOAndPubMedAbs", 0.195, "mixed", "GPT2-51M-1.31B", "1.31B"),
]


def extract_mmlu_accuracy(model_dir: Path) -> float | None:
    """Extract average MMLU accuracy from lm-eval results."""
    results_files = list(model_dir.glob("**/results_*.json"))
    if not results_files:
        return None

    accuracies = []
    for rf in results_files:
        with open(rf) as f:
            data = json.load(f)
        if "results" not in data:
            continue
        for bench_name, bench_data in data["results"].items():
            for key in ["acc,none", "acc_norm,none", "acc", "acc_norm"]:
                if key in bench_data:
                    accuracies.append(float(bench_data[key]))
                    break

    return np.mean(accuracies) if accuracies else None


def main():
    print("=" * 70)
    print("ABLATION C: Domain Control Analysis")
    print("=" * 70)

    rows = []
    for model_name, div_coeff, domain, family, tokens in MODELS:
        model_dir = EVAL_RESULTS_DIR / model_name
        if not model_dir.exists():
            print(f"  MISSING: {model_name}")
            continue

        mmlu_acc = extract_mmlu_accuracy(model_dir)
        if mmlu_acc is None:
            print(f"  NO RESULTS: {model_name}")
            continue

        row = {
            "model": model_name,
            "family": family,
            "div_coeff": div_coeff,
            "training_domain": domain,
            "tokens": tokens,
            "avg_mmlu_acc": mmlu_acc,
        }
        rows.append(row)
        print(f"  {model_name:45s} div={div_coeff:.3f} domain={domain:8s} MMLU={mmlu_acc:.4f}")

    if not rows:
        print("No results found. Ensure MMLU evaluations have been run.")
        return

    df = pd.DataFrame(rows)
    csv_path = OUTPUT_DIR / "ablation_c_domain_control.csv"
    df.to_csv(csv_path, index=False)
    print(f"\nSaved → {csv_path}")

    # Analysis per family
    print(f"\n{'='*70}")
    print("DOMAIN CONTROL ANALYSIS")
    print(f"{'='*70}")

    for family in df["family"].unique():
        sub = df[df["family"] == family].sort_values("div_coeff")
        print(f"\n{family} ({sub['tokens'].iloc[0]} tokens):")
        print(f"  {'Training Data':30s} {'Domain':10s} {'Div':>6s} {'MMLU':>8s}")
        print(f"  {'-'*60}")
        for _, row in sub.iterrows():
            print(f"  {row['model']:30s} {row['training_domain']:10s} "
                  f"{row['div_coeff']:6.3f} {row['avg_mmlu_acc']:8.4f}")

        # Check: does diversity predict performance within this family?
        if len(sub) >= 3:
            sorted_by_div = sub.sort_values("div_coeff")
            sorted_by_perf = sub.sort_values("avg_mmlu_acc")

            div_rank = sorted_by_div["model"].tolist()
            perf_rank = sorted_by_perf["model"].tolist()

            if div_rank == perf_rank:
                print(f"  → Diversity ranking matches performance ranking")
            else:
                print(f"  → Diversity ranking: {[d.split('_')[-1] for d in div_rank]}")
                print(f"  → Performance ranking: {[d.split('_')[-1] for d in perf_rank]}")

            # Key test: Does mixed always beat individual?
            mixed = sub[sub["training_domain"] == "mixed"]
            individuals = sub[sub["training_domain"] != "mixed"]
            if not mixed.empty and not individuals.empty:
                mixed_perf = mixed["avg_mmlu_acc"].iloc[0]
                max_individual = individuals["avg_mmlu_acc"].max()
                if mixed_perf > max_individual:
                    print(f"  ✓ Mixed (div=0.195) outperforms both individual datasets")
                    print(f"    Mixed: {mixed_perf:.4f} > Best individual: {max_individual:.4f}")
                else:
                    print(f"  ✗ Mixed does NOT outperform best individual")
                    print(f"    Mixed: {mixed_perf:.4f} vs Best individual: {max_individual:.4f}")

    # Overall conclusion
    print(f"\n{'='*70}")
    print("CONCLUSION")
    print(f"{'='*70}")
    mixed = df[df["training_domain"] == "mixed"]
    patents = df[df["training_domain"] == "patents"]
    medical = df[df["training_domain"] == "medical"]

    print(f"  Average MMLU by training domain:")
    print(f"    Patents (USPTO, div=0.158): {patents['avg_mmlu_acc'].mean():.4f}")
    print(f"    Medical (PubMed, div=0.168): {medical['avg_mmlu_acc'].mean():.4f}")
    print(f"    Mixed (USPTO+PubMed, div=0.195): {mixed['avg_mmlu_acc'].mean():.4f}")
    print()

    # Neither patents nor medical is a "domain match" for MMLU (which is general
    # knowledge). If mixed outperforms both, it's because of diversity, not domain.
    if mixed["avg_mmlu_acc"].mean() > max(patents["avg_mmlu_acc"].mean(),
                                            medical["avg_mmlu_acc"].mean()):
        print("  The mixed dataset (highest diversity) outperforms both individual")
        print("  datasets on MMLU, despite none being a domain match for MMLU.")
        print("  This supports diversity (not domain match) as the driver.")
    else:
        print("  Results are mixed — further analysis needed.")


if __name__ == "__main__":
    main()
