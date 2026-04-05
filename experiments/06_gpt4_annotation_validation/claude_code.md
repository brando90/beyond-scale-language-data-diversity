# Claude Code Runbook: Experiment 06 — GPT-4 Annotation Validation

**Goal:** Validate Task2Vec diversity metric against GPT-4 judgments on 30 batch pairs.

---

## Step 0 — Setup

```bash
conda activate beyond_scale_div_coeff
pip install openai  # for GPT-4 API
export OPENAI_API_KEY=$(cat ~/keys/openai_api_key.txt)
```

## Step 1 — Generate batch pairs

Sample 30 pairs of text batches where one batch comes from a higher-diversity dataset and the other from a lower-diversity dataset. Mix of:
- Cross-dataset pairs (e.g., C4 batch vs PubMed batch)
- Within-dataset pairs with different measured diversity (different random subsets)

## Step 2 — Run GPT-4 annotation

```bash
python experiments/06_gpt4_annotation_validation/run_gpt4_annotation.py
```

## Step 3 — Compute agreement

```bash
python experiments/06_gpt4_annotation_validation/compute_agreement.py
```

## Verification

- [ ] 30 pairs annotated
- [ ] Agreement rate computed
- [ ] Cohen's kappa reported
- [ ] Results interpretable (>70% agreement is good, >50% is better than random)
