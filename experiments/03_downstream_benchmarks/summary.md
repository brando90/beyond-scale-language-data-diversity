# Experiment 03: Downstream Task Benchmarks (Beyond CE Loss)

**TL;DR:** Run ARC-Easy, HellaSwag, WinoGrande, and LAMBADA on all existing UDACA checkpoints (GPT-2 51M–1.5B + LLaMA-2 7B, each at 3 diversity levels). Plot diversity coefficient vs. each benchmark score. This addresses the #1 evaluation complaint: cross-entropy loss alone is not convincing as a downstream performance proxy.

---

## Motivation (from OpenReview)

Reviewers JTBn, N6rW, and Bhvz (ICLR 2025) and the meta-review all flag: "Evaluation based solely on cross-entropy; not strong/convincing." The ICLR 2025 meta-review specifically says: "the empirical evidence for the correlation between diversity and performance is limited." Adding standard LM benchmarks directly addresses this.

Experiment 00 already added MMLU log-likelihood. This experiment extends to 4 more benchmarks for a total of 5 downstream tasks.

---

## Benchmarks

| Benchmark | Type | Metric | Why chosen |
|-----------|------|--------|------------|
| ARC-Easy | Multiple choice science | acc_norm | Standard reasoning benchmark |
| HellaSwag | Sentence completion | acc_norm | Tests commonsense reasoning |
| WinoGrande | Coreference resolution | acc | Tests commonsense (Winograd schema) |
| LAMBADA | Next-word prediction | acc | Tests long-range context understanding |
| MMLU | Multiple choice knowledge | acc | Already done in Experiment 00 |

---

## Models (all already on HuggingFace under UDACA org)

3 diversity levels × 8 model families = up to 24 models:

| Training Data | Div Coeff |
|--------------|-----------|
| USPTO | 0.158 |
| PubMed | 0.168 |
| USPTO+PubMed | 0.195 |

Model families: GPT2-51M (1.31B tokens), GPT2-51M (557M tokens), GPT2-117M, GPT2-204M, GPT2-345M, GPT2-810M, GPT2-1.5B, LLaMA2-7B.

---

## Expected outputs

1. `expt_results/downstream_benchmarks.csv` — model × benchmark scores + div_coeff
2. `expt_results/div_coeff_vs_arc_easy.{png,pdf}` — scatter + linear fit + R²
3. `expt_results/div_coeff_vs_hellaswag.{png,pdf}`
4. `expt_results/div_coeff_vs_winogrande.{png,pdf}`
5. `expt_results/div_coeff_vs_lambada.{png,pdf}`
6. `expt_results/div_coeff_vs_all_benchmarks.{png,pdf}` — combined panel
7. W&B Report with all results

---

## Relevant files

```
experiments/03_downstream_benchmarks/run_benchmarks.sh         # lm_eval runner
experiments/03_downstream_benchmarks/collect_scores.py         # aggregate results
experiments/03_downstream_benchmarks/generate_plots.py         # scatter plots
experiments/03_downstream_benchmarks/push_to_wandb.py          # W&B logging
experiments/00_div_vs_benchmark_scores/                        # reference (MMLU)
src/data_analysis/plot_div_coeff_vs_mmlu_acc.py               # reference plot style
```
