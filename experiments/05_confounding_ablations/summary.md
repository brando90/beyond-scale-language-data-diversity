# Experiment 05: Confounding Factor Ablations

**TL;DR:** Ablations isolating diversity's effect from dataset size, domain similarity, and vocabulary overlap. Addresses reviewer concern that merging datasets changes more than just diversity.

---

## Motivation

Reviewer Z6o3 (ICLR 2024): "The Pile contains both USPTO & PubMed, so PubMed+USPTO improvement not purely diversity."
Reviewer Bhvz (ICLR 2025): "When merging datasets, characteristics beyond diversity influence performance."
Reviewer N6rW (ICLR 2025): "PPL on OpenWebText2/C4 as evaluation metric unclear; does PubMed have lower PPL due to diversity or because it's similar to C4/OpenWebText?"

---

## Three ablations

### Ablation A — Size control
**Question:** Is the performance gain from merging datasets just because of more data?
**Method:** Subsample PubMed+USPTO to exactly match PubMed-only token count. If merged data still wins at equal size, diversity (not size) is the driver.

### Ablation B — Vocabulary overlap
**Question:** Does performance improve because training data overlaps more with eval data?
**Method:** Compute token-level Jaccard similarity between each training set and eval sets (C4, OpenWebText2). Report overlap alongside diversity coefficient. If high-diversity datasets don't have higher overlap, the confounder is ruled out.

### Ablation C — Domain control
**Question:** Does domain match with eval set confound the diversity-performance relationship?
**Method:** Compare Pile subsets that have similar diversity but different domains, or different diversity but similar domains. E.g., HackerNews (0.172) vs PubMed (0.168) — similar diversity, different domain.

---

## Expected outputs

1. `expt_results/ablation_a_size_control.csv` — performance at equal token counts
2. `expt_results/ablation_b_vocab_overlap.csv` — Jaccard overlap for each train-eval pair
3. `expt_results/ablation_c_domain_control.csv` — domain-matched comparisons
4. `expt_results/confounding_ablations_summary.png` — visualization of ablation results
