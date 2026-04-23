# Claude Code Runbook: Experiment 02 — Baseline Diversity Metrics

**Goal:** Compute Vendi Score, N-gram diversity, and mean embedding cosine distance alongside Task2Vec on the same datasets. Compare rankings.

---

## Step 0 — Environment setup

```bash
hostname
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader

conda activate beyond_scale_div_coeff
pip install vendi-score  # for Vendi Score
```

## Step 1 — Run baseline metrics computation

```bash
cd ~/beyond-scale-language-data-diversity

# Full run (all datasets, all metrics — ~2-4 hours on 1 GPU)
CUDA_VISIBLE_DEVICES=0 python experiments/02_baseline_diversity_metrics/compute_baseline_metrics.py \
    --num_batches 100 --batch_size 512 --device cuda

# Quick test (2 datasets, skip slow metrics)
CUDA_VISIBLE_DEVICES=0 python experiments/02_baseline_diversity_metrics/compute_baseline_metrics.py \
    --datasets pubmed uspto --num_batches 20 --batch_size 128 --skip_task2vec
```

## Step 2 — Generate plots

```bash
python experiments/02_baseline_diversity_metrics/generate_plots.py
```

## Step 3 — Verify

- [ ] `expt_results/baseline_comparison.csv` has all datasets × all metrics
- [ ] `expt_results/rank_correlation.csv` has Spearman ρ between metric pairs
- [ ] Plots saved as PNG + PDF
- [ ] No NaN values (or explained why missing)
- [ ] Report whether Task2Vec ranking agrees or disagrees with baselines

## Step 4 — Push to W&B

```bash
python experiments/02_baseline_diversity_metrics/push_to_wandb.py
```

## Interpretation guide

- **High Spearman ρ between Task2Vec and baselines:** Simpler metrics capture similar information. Paper should explain why Task2Vec is still preferred (e.g., theoretical grounding, robustness to paraphrasing).
- **Low Spearman ρ:** Task2Vec captures structural diversity that simpler metrics miss. Show specific datasets where rankings diverge and explain why.
- **Either outcome is publishable** — the point is to have the comparison.
