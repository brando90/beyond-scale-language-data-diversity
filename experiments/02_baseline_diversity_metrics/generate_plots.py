"""
Generate comparison plots for baseline diversity metrics vs Task2Vec.

Reads: experiments/02_baseline_diversity_metrics/expt_results/baseline_comparison.csv
Output: experiments/02_baseline_diversity_metrics/expt_results/*.{png,pdf}
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

EXPT_DIR = Path(__file__).parent / "expt_results"

METRIC_DISPLAY = {
    "task2vec_div_coeff": "Task2Vec Div Coeff",
    "mean_embedding_cosine": "Mean Embedding Cosine",
    "vendi_score": "Vendi Score",
    "ngram_1_cross_jaccard": "Unigram Jaccard Distance",
    "ngram_2_cross_jaccard": "Bigram Jaccard Distance",
    "ngram_3_cross_jaccard": "Trigram Jaccard Distance",
    "ngram_4_cross_jaccard": "4-gram Jaccard Distance",
    "ngram_1_within": "Unigram Diversity (within-batch)",
    "ngram_2_within": "Bigram Diversity (within-batch)",
}


def main():
    csv_path = EXPT_DIR / "baseline_comparison.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found. Run compute_baseline_metrics.py first.")
        return
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} datasets from {csv_path}")

    # --- Plot 1: Bar chart comparing all metrics across datasets ---
    key_metrics = ["task2vec_div_coeff", "mean_embedding_cosine",
                   "ngram_2_cross_jaccard", "vendi_score"]
    key_metrics = [m for m in key_metrics if m in df.columns and df[m].notna().any()]

    if key_metrics:
        fig, axes = plt.subplots(1, len(key_metrics), figsize=(5 * len(key_metrics), 6))
        if len(key_metrics) == 1:
            axes = [axes]

        for ax, metric in zip(axes, key_metrics):
            sorted_df = df.sort_values(metric, ascending=True).dropna(subset=[metric])
            colors = plt.cm.viridis(np.linspace(0.2, 0.8, len(sorted_df)))
            ax.barh(sorted_df["dataset"], sorted_df[metric], color=colors)
            ax.set_xlabel(METRIC_DISPLAY.get(metric, metric))
            ax.set_title(METRIC_DISPLAY.get(metric, metric), fontsize=10)
            ax.tick_params(axis="y", labelsize=8)

        plt.suptitle("Diversity Metrics Comparison Across Datasets", fontsize=13, y=1.02)
        plt.tight_layout()
        for ext in ("png", "pdf"):
            out = EXPT_DIR / f"baseline_comparison_bars.{ext}"
            plt.savefig(out, dpi=150, bbox_inches="tight")
            print(f"Saved → {out}")
        plt.close()

    # --- Plot 2: Rank comparison scatter (Task2Vec vs each baseline) ---
    baselines = [m for m in key_metrics if m != "task2vec_div_coeff"]
    if "task2vec_div_coeff" in df.columns and baselines:
        fig, axes = plt.subplots(1, len(baselines), figsize=(5.5 * len(baselines), 5))
        if len(baselines) == 1:
            axes = [axes]

        for ax, baseline in zip(axes, baselines):
            valid = df[["dataset", "task2vec_div_coeff", baseline]].dropna()
            if len(valid) < 3:
                continue

            ax.scatter(valid["task2vec_div_coeff"], valid[baseline],
                       s=80, zorder=3, color="steelblue")

            # Annotate points with dataset names
            for _, row in valid.iterrows():
                ax.annotate(row["dataset"], (row["task2vec_div_coeff"], row[baseline]),
                            fontsize=7, xytext=(4, 4), textcoords="offset points")

            # Spearman correlation
            sr, sp = stats.spearmanr(valid["task2vec_div_coeff"], valid[baseline])
            ax.set_title(f"Spearman ρ = {sr:.3f} (p={sp:.3f})", fontsize=10)
            ax.set_xlabel("Task2Vec Diversity Coefficient")
            ax.set_ylabel(METRIC_DISPLAY.get(baseline, baseline))
            ax.grid(True, alpha=0.3)

        plt.suptitle("Task2Vec vs Baseline Metrics — Dataset Ranking Agreement",
                      fontsize=13, y=1.02)
        plt.tight_layout()
        for ext in ("png", "pdf"):
            out = EXPT_DIR / f"task2vec_vs_baselines_scatter.{ext}"
            plt.savefig(out, dpi=150, bbox_inches="tight")
            print(f"Saved → {out}")
        plt.close()

    # --- Plot 3: Rank correlation heatmap ---
    rc_path = EXPT_DIR / "rank_correlation.csv"
    if rc_path.exists():
        rc_df = pd.read_csv(rc_path)
        all_metrics = sorted(set(rc_df["metric_1"]) | set(rc_df["metric_2"]))
        n = len(all_metrics)
        idx_map = {m: i for i, m in enumerate(all_metrics)}
        corr_matrix = np.eye(n)

        for _, row in rc_df.iterrows():
            i, j = idx_map[row["metric_1"]], idx_map[row["metric_2"]]
            corr_matrix[i, j] = row["spearman_rho"]
            corr_matrix[j, i] = row["spearman_rho"]

        fig, ax = plt.subplots(figsize=(8, 7))
        im = ax.imshow(corr_matrix, cmap="RdYlGn", vmin=-1, vmax=1, aspect="auto")
        labels = [METRIC_DISPLAY.get(m, m) for m in all_metrics]
        ax.set_xticks(range(n))
        ax.set_yticks(range(n))
        ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
        ax.set_yticklabels(labels, fontsize=8)

        for i in range(n):
            for j in range(n):
                ax.text(j, i, f"{corr_matrix[i,j]:.2f}",
                        ha="center", va="center", fontsize=8,
                        color="white" if abs(corr_matrix[i, j]) > 0.6 else "black")

        plt.colorbar(im, label="Spearman ρ")
        plt.title("Pairwise Rank Correlation Between Diversity Metrics")
        plt.tight_layout()
        for ext in ("png", "pdf"):
            out = EXPT_DIR / f"rank_correlation_heatmap.{ext}"
            plt.savefig(out, dpi=150, bbox_inches="tight")
            print(f"Saved → {out}")
        plt.close()

    print("\nDone.")


if __name__ == "__main__":
    main()
