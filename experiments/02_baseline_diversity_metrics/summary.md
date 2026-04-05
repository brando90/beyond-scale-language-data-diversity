# Experiment 02: Baseline Diversity Metrics Comparison

**TL;DR:** Implement Vendi Score, N-gram diversity, and mean embedding cosine distance as baselines. Run all four metrics (including Task2Vec) on the same datasets and batches. Compare rankings to show whether Task2Vec captures something the simpler metrics miss — or validate that they agree (either outcome is informative for the paper).

---

## Motivation (from OpenReview)

Every full-conference review (ICLR 2024: FQiZ, kREU; ICLR 2025: v5Te, JTBn, N6rW, Bhvz) and both meta-reviews flag the same issue: **no comparison to simpler diversity baselines**. The paper must show that Task2Vec is preferable to N-gram diversity, mean embedding cosine, and Vendi Score — or at minimum, discuss the trade-offs with empirical evidence.

---

## Metrics to implement

| Metric | Definition | Compute cost |
|--------|-----------|-------------|
| **Task2Vec div coeff** (existing) | Mean pairwise cosine distance of FIM diagonals across batches | Medium (gradient computation) |
| **Vendi Score** | exp(Shannon entropy of eigenvalues of kernel similarity matrix) | O(N² + eigendecomposition) |
| **N-gram diversity** | Distinct N-grams / Total N-grams for N=1,2,3,4 | Cheap (counting) |
| **Mean embedding cosine** | Pairwise cosine distance of mean GPT-2 hidden-state embeddings per batch | Cheap (forward pass only) |

---

## Datasets

Same datasets already in Table 1 (so results are directly comparable):

| Dataset | Task2Vec Div Coeff (known) |
|---------|---------------------------|
| C4 (en) | ~0.231 |
| WikiText-103 | ~0.206 |
| The Pile | ~0.246 |
| Pile-CC | ~0.230 |
| PubMed | ~0.168 |
| USPTO | ~0.158 |
| HackerNews | ~0.172 |
| NIH ExPorter | ~0.164 |
| SlimPajama | ~0.228 |
| OpenWebText | ~0.227 |

---

## Expected outputs

1. `expt_results/baseline_comparison.csv` — all metrics × all datasets
2. `expt_results/rank_correlation.csv` — Spearman ρ between each metric pair
3. `expt_results/baseline_comparison_table.png` — formatted comparison table
4. `expt_results/metric_rank_comparison.png` — bar chart showing rank agreement
5. W&B Report with all results

---

## Relevant files

```
experiments/02_baseline_diversity_metrics/compute_baseline_metrics.py  # main script
experiments/02_baseline_diversity_metrics/generate_plots.py            # plotting
experiments/02_baseline_diversity_metrics/push_to_wandb.py             # W&B logging
src/diversity/div_coeff.py                                             # existing Task2Vec API
src/diversity/task2vec.py                                              # embedding computation
src/diversity/task_similarity.py                                       # distance metrics
```
