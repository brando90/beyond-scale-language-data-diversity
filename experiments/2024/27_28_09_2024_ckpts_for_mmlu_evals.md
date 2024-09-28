# Checkpoints to eval -log P^Vocab(CorrectChoice)

```bash
# whole table: https://wandb.ai/brando/beyond-scale/table?nw=nwuserbrando


# Note: example ckpt path
# /lfs/ampere9/0/brando9/data/results_2024-m02-d03-t16h_32m_24s/checkpoint-1551

# - ckpt 1
# https://wandb.ai/brando/beyond-scale/runs/ghjkc8tc/logs
# /lfs/ampere9/0/brando9/data/results_2024-m02-d04-t02h_02m_02s
# TODO find full ckpt path
# uspto (https://huggingface.co/datasets/UDACA/PileSubsets)


# - ckpt 2
# https://wandb.ai/brando/beyond-scale/runs/7jqujyv1/logs
# /lfs/ampere9/0/brando9/data/results_2024-m02-d04-t01h_50m_22s
# pubmed (https://huggingface.co/datasets/UDACA/PileSubsets)

# - ckpt 3
# https://wandb.ai/brando/beyond-scale/runs/s9ou7l1n/logs
# /lfs/ampere9/0/brando9/data/results_2024-m02-d04-t01h_47m_30s
# uspto + pubmed

# - ckpt 4
# https://wandb.ai/brando/beyond-scale/runs/3o05mvz6/logs
# /lfs/ampere9/0/brando9/data/results_2024-m02-d04-t01h_45m_48s
# uspto + pubmed

# - ckpt 5
# https://wandb.ai/brando/beyond-scale/runs/ad2f1yew/logs
# /lfs/ampere9/0/brando9/data/results_2024-m02-d03-t16h_34m_01s
# uspto + pubmed

# - ckpt 6
# https://wandb.ai/brando/beyond-scale/runs/2dy8rrcc/logs 
# /lfs/ampere9/0/brando9/data/results_2024-m01-d29-t16h_01m_30s
# uspto + pubmed

# - ckpt 7
# https://wandb.ai/brando/beyond-scale/runs/fj5xd2kj?nw=nwuserbrando
# /lfs/ampere9/0/brando9/data/results_2024-m01-d29-t16h_00m_55s
# pubmed
```

Let's push the ckpts
```bash
ssh brando9@ampere9.stanford.edu 

conda activate beyond_scale_div_coeff

python ~/beyond-scale-language-data-diversity/src/push_hf_models_to_hf.py
```

Eval from RS:
```bash

conda create -n eleuther_lm_eval_harness_20240927 python=3.11

conda activate eleuther_lm_eval_harness_20240927

git clone --depth 1 https://github.com/EleutherAI/lm-evaluation-harness
cd lm-evaluation-harness && pip install -e . && cd ..
```