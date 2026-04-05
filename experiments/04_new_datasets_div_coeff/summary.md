# Experiment 04: Diversity Coefficients for Additional Pre-training Datasets

**TL;DR:** Compute Task2Vec diversity coefficients for FineWeb, Dolma, and RedPajama to expand Table 1 beyond the 10 datasets already measured. Addresses reviewer complaint that "only 2-3 unrepresentative datasets" are used (note: Table 1 already has 10, but reviewers want more standard LLM pre-training datasets).

---

## Motivation

Reviewer N6rW (ICLR 2025): "Two pre-training datasets [are] highly unusual/unrepresentative; should use: C4, OpenWebText, The Pile, RedPajama, SlimPajama, RefinedWeb, Dolma, FineWeb, DCLM."

Table 1 already includes: C4, WikiText-103, The Pile, Pile-CC, PubMed, USPTO, HackerNews, NIH ExPorter, SlimPajama, OpenWebText. Missing: FineWeb, Dolma, RedPajama.

---

## Datasets to add

| Dataset | HF path | Notes |
|---------|---------|-------|
| FineWeb | `HuggingFaceFW/fineweb` | 15T tokens, web crawl |
| FineWeb-Edu | `HuggingFaceFW/fineweb-edu` | Educational subset |
| Dolma | `allenai/dolma` | OLMo pre-training data |
| RedPajama v2 | `togethercomputer/RedPajama-Data-V2` | 30T tokens |

---

## Method

Same as existing: `get_diversity_coefficient()` with pretrained GPT-2 small probe, batch_size=512, num_batches=600, seed=42.

## Expected output

Updated Table 1 with 13-14 datasets total. Append to `expt_results/new_datasets_div_coeff.csv`.
