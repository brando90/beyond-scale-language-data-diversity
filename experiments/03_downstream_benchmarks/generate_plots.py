"""
Generate scatter plots: diversity coefficient vs downstream benchmark scores.

Reads: experiments/03_downstream_benchmarks/expt_results/downstream_benchmarks.csv
Output: experiments/03_downstream_benchmarks/expt_results/div_coeff_vs_*.{png,pdf}

Follows the same style as experiments/00_div_vs_benchmark_scores/ plots.
"""
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from sklearn.metrics import r2_score

EXPT_DIR = Path(__file__).parent / "expt_results"

FAMILY_COLORS = {
    "GPT2-51M":  "royalblue",
    "GPT2-117M": "deepskyblue",
    "GPT2-204M": "darkturquoise",
    "GPT2-345M": "mediumslateblue",
    "GPT2-810M": "rebeccapurple",
    "GPT2-1.5B": "darkviolet",
    "LLaMA2-7B": "crimson",
}

BENCHMARKS = {
    "arc_easy": "ARC-Easy Accuracy",
    "hellaswag": "HellaSwag Accuracy (norm)",
    "winogrande": "WinoGrande Accuracy",
    "lambada_openai": "LAMBADA Accuracy",
}

DIV_LABELS = [(0.158, "USPTO"), (0.168, "PubMed"), (0.195, "USPTO+PubMed")]


def correlations(x, y):
    slope, intercept = np.polyfit(x, y, 1)
    y_pred = slope * x + intercept
    r2 = r2_score(y, y_pred)
    pr, pp = stats.pearsonr(x, y)
    sr, sp = stats.spearmanr(x, y)
    kt, kp = stats.kendalltau(x, y)
    return dict(slope=slope, intercept=intercept, r2=r2,
                pearson_r=pr, pearson_p=pp,
                spearman_r=sr, spearman_p=sp,
                kendall_t=kt, kendall_p=kp)


def main():
    csv_path = EXPT_DIR / "downstream_benchmarks.csv"
    if not csv_path.exists():
        print(f"ERROR: {csv_path} not found. Run collect_scores.py first.")
        return
    df = pd.read_csv(csv_path)
    print(f"Loaded {len(df)} models from {csv_path}")

    # --- Individual benchmark plots ---
    for bench_key, bench_label in BENCHMARKS.items():
        valid = df[["model", "family", "div_coeff", bench_key]].dropna()
        if len(valid) < 3:
            print(f"SKIP {bench_key}: only {len(valid)} data points")
            continue

        fig, axes = plt.subplots(1, 2, figsize=(18, 6))
        x_all = valid["div_coeff"].values
        y_all = valid[bench_key].values

        # Left: all models
        ax = axes[0]
        c_all = correlations(x_all, y_all)
        for family, gdf in valid.groupby("family"):
            color = FAMILY_COLORS.get(family, "gray")
            ax.scatter(gdf["div_coeff"], gdf[bench_key],
                       label=family, color=color, s=80, zorder=3)
        x_line = np.linspace(x_all.min() - 0.003, x_all.max() + 0.003, 100)
        ax.plot(x_line, c_all["slope"] * x_line + c_all["intercept"],
                "k--", linewidth=1.5)
        ax.annotate(
            f"y = {c_all['slope']:.3f}x + {c_all['intercept']:.3f}\n"
            f"R² = {c_all['r2']:.3f}\n"
            f"Pearson r = {c_all['pearson_r']:.3f} (p={c_all['pearson_p']:.3f})\n"
            f"Spearman ρ = {c_all['spearman_r']:.3f}",
            xy=(0.03, 0.97), xycoords="axes fraction",
            va="top", ha="left", fontsize=8,
            bbox=dict(boxstyle="round,pad=0.3", fc="white", alpha=0.85),
        )
        for dv, dl in DIV_LABELS:
            ax.axvline(dv, color="black", linestyle="dotted", linewidth=0.8)
        ax.set_xlabel("Task2Vec Diversity Coefficient")
        ax.set_ylabel(bench_label)
        ax.set_title(f"Div Coeff vs. {bench_label} (all models)")
        ax.legend(fontsize=8, loc="lower right")
        ax.grid(True, alpha=0.3)

        # Right: per-family fits
        ax2 = axes[1]
        for family, gdf in valid.groupby("family"):
            color = FAMILY_COLORS.get(family, "gray")
            fx, fy = gdf["div_coeff"].values, gdf[bench_key].values
            ax2.scatter(fx, fy, color=color, s=80, zorder=3)
            if len(gdf) >= 2:
                fc = correlations(fx, fy)
                r2_str = f" R²={fc['r2']:.2f}" if len(gdf) >= 3 else ""
                ax2.plot(np.sort(fx),
                         np.poly1d(np.polyfit(fx, fy, 1))(np.sort(fx)),
                         color=color, linewidth=1.5,
                         label=f"{family}{r2_str}")
        for dv, dl in DIV_LABELS:
            ax2.axvline(dv, color="black", linestyle="dotted", linewidth=0.8)
        ax2.set_xlabel("Task2Vec Diversity Coefficient")
        ax2.set_ylabel(bench_label)
        ax2.set_title(f"Div Coeff vs. {bench_label} (per family)")
        ax2.legend(fontsize=7, loc="lower right")
        ax2.grid(True, alpha=0.3)

        plt.tight_layout()
        for ext in ("png", "pdf"):
            out = EXPT_DIR / f"div_coeff_vs_{bench_key}.{ext}"
            plt.savefig(out, dpi=150, bbox_inches="tight")
            print(f"Saved → {out}")
        plt.close()

    # --- Combined panel: all benchmarks in one figure ---
    available = [b for b in BENCHMARKS if b in df.columns and df[b].notna().sum() >= 3]
    if available:
        ncols = len(available)
        fig, axes = plt.subplots(1, ncols, figsize=(5.5 * ncols, 5))
        if ncols == 1:
            axes = [axes]

        for ax, bench_key in zip(axes, available):
            valid = df[["family", "div_coeff", bench_key]].dropna()
            for family, gdf in valid.groupby("family"):
                color = FAMILY_COLORS.get(family, "gray")
                ax.scatter(gdf["div_coeff"], gdf[bench_key],
                           label=family, color=color, s=60, zorder=3)
            x_all = valid["div_coeff"].values
            y_all = valid[bench_key].values
            c = correlations(x_all, y_all)
            x_line = np.linspace(x_all.min() - 0.003, x_all.max() + 0.003, 100)
            ax.plot(x_line, c["slope"] * x_line + c["intercept"], "k--", linewidth=1.2)
            ax.set_xlabel("Div Coeff")
            ax.set_ylabel(BENCHMARKS[bench_key])
            ax.set_title(f"R²={c['r2']:.3f}, ρ={c['spearman_r']:.3f}", fontsize=10)
            ax.grid(True, alpha=0.3)
            if ax == axes[0]:
                ax.legend(fontsize=6, loc="lower right")

        plt.suptitle("Diversity Coefficient vs. Downstream Benchmarks", fontsize=13, y=1.02)
        plt.tight_layout()
        for ext in ("png", "pdf"):
            out = EXPT_DIR / f"div_coeff_vs_all_benchmarks.{ext}"
            plt.savefig(out, dpi=150, bbox_inches="tight")
            print(f"Saved → {out}")
        plt.close()

    # --- Print correlation summary ---
    print("\n" + "=" * 70)
    print("CORRELATION SUMMARY")
    print("=" * 70)
    for bench_key in available:
        valid = df[["div_coeff", bench_key]].dropna()
        c = correlations(valid["div_coeff"].values, valid[bench_key].values)
        print(f"\n{BENCHMARKS[bench_key]}:")
        print(f"  R² = {c['r2']:.4f}  |  Pearson r = {c['pearson_r']:.4f} (p={c['pearson_p']:.4e})")
        print(f"  Spearman ρ = {c['spearman_r']:.4f}  |  Kendall τ = {c['kendall_t']:.4f}")


if __name__ == "__main__":
    main()
