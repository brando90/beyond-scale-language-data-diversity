# Experiment 06: GPT-4 Annotation Validation of Diversity Metric

**TL;DR:** Use GPT-4 to annotate 30 batch pairs as "which is more diverse?" and compare with Task2Vec diversity coefficient rankings. Validates that the metric captures something an intelligent observer recognizes as diversity.

---

## Motivation

Reviewer v5Te (ICLR 2025): "Human annotation study missing; synthetic dataset experiments valuable but conclusions stronger with human annotation validation."

GPT-4 serves as a scalable proxy for human annotation (faster, cheaper, still validates interpretability).

---

## Method

1. Sample 30 batch pairs from existing datasets with known different diversity coefficients
2. For each pair, present raw text samples to GPT-4
3. Ask: "Which collection of text samples (A or B) exhibits greater topical and stylistic diversity?"
4. Compare GPT-4's ranking with Task2Vec ranking
5. Compute agreement rate and Cohen's kappa

---

## Expected outputs

1. `expt_results/gpt4_annotations.csv` — GPT-4 judgments per pair
2. `expt_results/annotation_agreement.json` — agreement rate, kappa
3. Section for paper: "GPT-4 agreed with Task2Vec ranking in X/30 cases (κ = Y)"
