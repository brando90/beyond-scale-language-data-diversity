# Overview — beyond-scale-language-data-diversity

Official implementation of the **Task2Vec Diversity Coefficient** — a metric for measuring natural language data diversity. The paper shows LLMs are pre-trained on formally diverse data.

- **Paper:** https://arxiv.org/abs/2306.13840
- **Core idea:** Compute Task2Vec embeddings (diagonal of Fisher Information Matrix) for dataset batches, then measure pairwise cosine distances to get a diversity coefficient.
- **Probe network:** GPT-2 (by default)
- **Supported datasets:** C4, WikiText-103, The Pile (and its sub-datasets), GINC (synthetic)

---

## Key API Usage

```python
from diversity.div_coeff import get_diversity_coefficient

results = get_diversity_coefficient(
    dataset,
    probe_network,  # typically GPT-2
    num_tasks=200,
    batch_size=512,
)
print(results['div_coeff'], results['div_coeff_ci'])
```

`get_diversity_coefficient()` returns a dict with: `div_coeff`, `div_coeff_ci`, `embeddings`, `distance_matrix`, `losses`, `num_batches`.
