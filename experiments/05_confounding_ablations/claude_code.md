# Claude Code Runbook: Experiment 05 — Confounding Ablations

**Goal:** Three ablations to isolate diversity's effect from size, vocab overlap, and domain.

---

## Ablation A — Size control

```bash
# 1. Count tokens in PubMed-only training set
python -c "
from datasets import load_dataset
from transformers import AutoTokenizer
t = AutoTokenizer.from_pretrained('gpt2')
ds = load_dataset('UDACA/PileSubsets', 'pubmed', split='train', streaming=True)
total = sum(len(t.encode(ex['text'])) for i, ex in enumerate(ds) if i < 50000)
print(f'PubMed tokens (first 50k examples): {total}')
"

# 2. Create size-matched PubMed+USPTO subset (subsample to same token count)
# 3. Compute diversity coefficient on subsampled mix
# 4. Train GPT-2 small on subsampled mix (same hyperparams as original)
# 5. Evaluate on C4/OWT2 and downstream benchmarks
# 6. Compare: if subsampled mix still beats PubMed-only, diversity matters beyond size
```

## Ablation B — Vocabulary overlap

```python
# Compute Jaccard overlap of token vocabularies between training and eval sets
from datasets import load_dataset
from transformers import AutoTokenizer

tokenizer = AutoTokenizer.from_pretrained('gpt2')

def get_vocab_set(dataset, n_samples=10000):
    tokens = set()
    for i, ex in enumerate(dataset):
        if i >= n_samples: break
        tokens.update(tokenizer.encode(ex['text']))
    return tokens

# Load training sets
pubmed_tokens = get_vocab_set(load_dataset('UDACA/PileSubsets', 'pubmed', split='train', streaming=True))
uspto_tokens = get_vocab_set(load_dataset('UDACA/PileSubsets', 'uspto', split='train', streaming=True))
mixed_tokens = pubmed_tokens | uspto_tokens

# Load eval sets
c4_tokens = get_vocab_set(load_dataset('allenai/c4', 'en', split='validation', streaming=True))
owt_tokens = get_vocab_set(load_dataset('Skylion007/openwebtext', split='train', streaming=True))

# Jaccard similarity
for name, train_set in [('PubMed', pubmed_tokens), ('USPTO', uspto_tokens), ('Mixed', mixed_tokens)]:
    for eval_name, eval_set in [('C4', c4_tokens), ('OWT', owt_tokens)]:
        jaccard = len(train_set & eval_set) / len(train_set | eval_set)
        print(f'{name} vs {eval_name}: Jaccard = {jaccard:.4f}')
```

## Ablation C — Domain control

Compare Pile subsets with similar diversity but different domains:
- HackerNews (0.172) vs PubMed (0.168): similar div, different domain
- Pile-CC (0.230) vs C4 (0.231): similar div, similar domain (both web)

If similar-diversity datasets from different domains produce different performance, domain (not just diversity) matters. Report this honestly.

## Verification

- [ ] Ablation A: size-matched comparison shows diversity effect beyond size
- [ ] Ablation B: vocab overlap table for all train-eval pairs
- [ ] Ablation C: domain-matched comparison results
