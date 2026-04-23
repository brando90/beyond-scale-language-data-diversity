# Architecture — beyond-scale-language-data-diversity

## `src/diversity/` — Core module

| File | Role |
|------|------|
| `main.py` | CLI entry point; loads dataset, runs Task2Vec embedding loop |
| `div_coeff.py` | `get_diversity_coefficient()`, `cross_diversity_coefficient()` — main API |
| `task2vec.py` | Task2Vec class; computes diagonal FIM via montecarlo/variational/autoregressive methods |
| `task_similarity.py` | `pdist()` (pairwise cosine), `stats_of_distance_matrix()`, `plot_distance_matrix()` |
| `data_mixtures.py` | Mixture definitions: Uniform, DoReMi, LLaMA v1 for C4+WikiText; 5-subset Pile |
| `utils.py` | `AverageMeter`, `get_error()` (autoregressive loss), `seed_everything()` |
| `main_ginc.py` | GINC-specific entry point |
| `scripts/` | Shell runners for paper experiments |
| `notebooks/` | `plot.ipynb` reproduces paper figures; `plot-ginc.ipynb` for GINC plots |

Comments marked `## LLM DIV` indicate modifications from the original CV Task2Vec for language model use.

## `src/alignment/` — Alignment/relevance coefficients

- `align_t2v_coeff.py`: `relevance_coeff_task2vec_via_full_embed_dataset()`, `alignment_with_diversity_coefficient()`
- `_align.py`: Alignment framework

## `src/training/` — LM fine-tuning

HuggingFace Trainer + TRL/PEFT (LoRA/QLoRA). Supports GPT-2, LLaMA-2, Mistral, C4, UDACA PileSubsets.

## `src/ginc/` — Synthetic in-context learning data

Generates datasets using HMMs with varying number of symbols/HMMs. Has its own conda env (`conda-env.yml`) and runner scripts.

## `src/data_analysis/` — Paper figures

Scripts correlating diversity coefficient vs. cross-entropy loss and perplexity (R² analysis).
