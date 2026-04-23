# Build & Dev — beyond-scale-language-data-diversity

## Installation

```bash
# Conda (recommended)
conda create -n beyond_scale_div_coeff python=3.11 -y
conda activate beyond_scale_div_coeff
pip install -e ~/beyond-scale-language-data-diversity

# Or venv
python3.11 -m venv ~/.virtualenvs/beyond_scale_div_coeff
source ~/.virtualenvs/beyond_scale_div_coeff/bin/activate
pip install -e ~/beyond-scale-language-data-diversity
```

The `install.sh` script installs via conda and also sets up dependencies (`ultimate-utils`, `ultimate-anatome`).

---

## Running Experiments

**Compute diversity coefficient (main workflow):**
```bash
python src/diversity/main.py \
  --task_name c4        # or wikitext, the_pile
  --num_tasks 200 \
  --batch_size 512 \
  --buffer_size 500000 \
  --finetune --pretrained \
  --output_dir ./output_dir \
  --cache_dir ./cache_dir
```

**Batch runners (used in paper):**
```bash
# C4, WikiText-103, The Pile — 200 tasks each
bash src/diversity/scripts/runner.sh

# Individual Pile sub-datasets
bash src/diversity/scripts/runner_thepile_subdataset.sh

# GINC diversity
bash src/diversity/scripts/runner_ginc.sh
```

**GINC synthetic data:**
```bash
# Generate datasets (HMMs with varying symbols)
bash src/ginc/scripts/runner_generate.sh

# Train GPT-2 on GINC
bash src/ginc/scripts/runner_train.sh

# Or directly:
python src/diversity/main_ginc.py \
  --batch_size 512 --finetune --pretrained \
  --cache_dir ./cache_dir --n_hmms=10 --n_symbols=50
```

**Model training (LLaMA-2, GPT-2, Mistral via HF Trainer + TRL/PEFT):**
```bash
python src/training/train.py
python src/training/eval.py
```
