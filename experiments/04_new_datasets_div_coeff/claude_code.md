# Claude Code Runbook: Experiment 04 — New Datasets Div Coeff

**Goal:** Compute diversity coefficients for FineWeb, FineWeb-Edu, Dolma, RedPajama to expand Table 1.

---

## Step 0 — Environment

```bash
conda activate beyond_scale_div_coeff
cd ~/beyond-scale-language-data-diversity
```

## Step 1 — Compute

```bash
# FineWeb
CUDA_VISIBLE_DEVICES=0 python src/diversity/main.py \
    --task_name c4 \
    --output_dir ./experiments/04_new_datasets_div_coeff/expt_results/fineweb \
    --num_tasks 600 --batch_size 512 --pretrained --finetune \
    --buffer_size 500000
# NOTE: You'll need to modify main.py to add FineWeb as a dataset option,
# or write a wrapper script that loads FineWeb and calls get_diversity_coefficient() directly.
```

For datasets not in main.py's supported list, use the API directly:

```python
from datasets import load_dataset
from transformers import GPT2LMHeadModel, AutoTokenizer
from diversity.div_coeff import get_diversity_coefficient

ds = load_dataset("HuggingFaceFW/fineweb", split="train", streaming=True)
tokenizer = AutoTokenizer.from_pretrained("gpt2")
probe = GPT2LMHeadModel.from_pretrained("gpt2").cuda()

results = get_diversity_coefficient(ds, tokenize_map, probe, tokenizer,
    batch_size=512, num_batches=600, seed=42)
print(f"FineWeb div_coeff: {results['div_coeff']:.4f} ± {results['div_coeff_ci']:.4f}")
```

## Step 2 — Aggregate and update Table 1

Append new values to `expt_results/new_datasets_div_coeff.csv`.

## Verification

- [ ] Each dataset has div_coeff ± CI
- [ ] Values are in expected range (0.15-0.25 for general web data)
- [ ] Results are reproducible (same seed → same value)
