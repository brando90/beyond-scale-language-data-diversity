# Claude Code Runbook: Experiment 03 — Downstream Task Benchmarks

**Goal:** Run ARC-Easy, HellaSwag, WinoGrande, LAMBADA on all UDACA models. Plot diversity coefficient vs. benchmark scores. Extends Experiment 00 (MMLU) to address reviewer demand for downstream evaluations beyond cross-entropy.

---

## Step 0 — Environment setup

```bash
hostname
nvidia-smi --query-gpu=index,memory.used,memory.free --format=csv,noheader

conda activate eleuther_lm_eval_harness_20240927
# if env doesn't exist:
conda create -n eleuther_lm_eval_harness_20240927 python=3.11 -y
conda activate eleuther_lm_eval_harness_20240927
pip install lm-eval
```

## Step 1 — Run benchmark evaluations

```bash
cd ~/beyond-scale-language-data-diversity

# Run all models on GPU 0 (~8-12 hours for all 27 models)
bash experiments/03_downstream_benchmarks/run_benchmarks.sh 0

# Or run on multiple GPUs in parallel (open separate terminals):
bash experiments/03_downstream_benchmarks/run_benchmarks.sh 0 &  # subset 1
bash experiments/03_downstream_benchmarks/run_benchmarks.sh 1 &  # subset 2
# (modify run_benchmarks.sh to split MODELS array between GPUs)
```

Results go to: `/dfs/scratch0/brando9/data/beyond_scale/eval_results/<model>_downstream/`

## Step 2 — Collect and aggregate scores

```bash
python experiments/03_downstream_benchmarks/collect_scores.py
```

Output: `experiments/03_downstream_benchmarks/expt_results/downstream_benchmarks.csv`

## Step 3 — Generate plots

```bash
python experiments/03_downstream_benchmarks/generate_plots.py
```

Output: `experiments/03_downstream_benchmarks/expt_results/div_coeff_vs_*.{png,pdf}`

## Step 4 — Verify

- [ ] All 27 models have results for all 4 benchmarks
- [ ] CSV has no unexpected NaN values
- [ ] Scatter plots show per-family linear fits with R²
- [ ] Report overall and per-family correlation direction

## Step 5 — Push to W&B and create Report

```bash
python experiments/03_downstream_benchmarks/push_to_wandb.py
```

## Key question to answer

> Does higher Task2Vec diversity coefficient of training data correlate with higher downstream benchmark accuracy?

If yes for all/most benchmarks: strong evidence for the paper's main claim.
If mixed: report honestly. Some benchmarks may not benefit from general diversity (e.g., WinoGrande requires specific commonsense knowledge).
