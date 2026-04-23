# Experiment 01: Addressing OpenReview Feedback — Master Plan

**TL;DR:** Consolidated plan to address all reviewer concerns from ICLR 2024 (reject), ICLR 2025 (reject), and workshop acceptances (ICML 2023, DMLR@ICLR 2024). 10 tasks total, 5 critical experiments + 3 high-priority writing fixes + 2 medium improvements.

---

## Submission History

| Venue | Decision | Forum |
|-------|----------|-------|
| ICML 2023 Workshop (DeployableGenerativeAI) | Accept | https://openreview.net/forum?id=oCYjN48axE |
| DMLR @ ICLR 2024 Workshop | Accept (Poster) | https://openreview.net/forum?id=tgkWxsOapD |
| ICLR 2024 Main Track | Reject | https://openreview.net/forum?id=506Sxc0Adp |
| ICLR 2025 Main Track | Reject | https://openreview.net/forum?id=kDakBhOaBV |

---

## Sub-experiments

| # | Experiment | Priority | Status |
|---|-----------|----------|--------|
| 02 | Baseline diversity metrics (Vendi, N-gram, embedding) | CRITICAL | Scripts written |
| 03 | Downstream benchmarks (ARC, HellaSwag, WinoGrande, LAMBADA) | CRITICAL | Scripts written |
| 04 | New datasets div coeff (FineWeb, Dolma, RedPajama) | CRITICAL | Scaffold ready |
| 05 | Confounding ablations (size, domain, vocab overlap) | CRITICAL | Scaffold ready |
| 06 | GPT-4 annotation validation | MEDIUM | Scaffold ready |

## Writing tasks (no separate experiment folder needed)

| # | Task | Priority |
|---|------|----------|
| 6 | Improve Task2Vec methodology + pseudocode in 02_method.tex | HIGH |
| 7 | Update related work with 10+ missing citations | HIGH |
| 8 | Tone down overclaiming, remove "paradigm shift" | HIGH |
| 10 | Fix 18+ presentation issues | MEDIUM |
| 4 | Address inconsistent diversity→performance in 05_discussion.tex | CRITICAL |

---

## Execution order

1. **Expt 03** (downstream benchmarks) — run overnight, mostly automated lm_eval
2. **Expt 02** (baseline metrics) — run in parallel on different GPU
3. **Writing tasks 6, 7, 8** — can be done while experiments run
4. **Expt 04** (new datasets) — straightforward, uses existing API
5. **Expt 05** (confounders) — requires some new training
6. **Writing task 4** (failure analysis) — needs experiment results first
7. **Expt 06** (GPT-4 annotation) — lower priority, quick to run
8. **Writing task 10** (formatting) — do last, after all content changes

---

## Reference files

```
experiments/01_addressing_openreviews/deep_research.md         # ChatGPT + Gemini analysis
experiments/01_addressing_openreviews/iclr2024_reviews.md      # ICLR 2024 reviews
experiments/01_addressing_openreviews/iclr2025_reviews.md      # ICLR 2025 reviews
experiments/01_addressing_openreviews/00_master_suggested_prompts.md  # original prompt list
```
