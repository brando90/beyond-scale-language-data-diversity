#!/bin/bash
# Launch downstream benchmarks on ampere1 across 4 GPUs.
# Each GPU processes a subset of models sequentially.
# Run from skampere2: bash experiments/03_downstream_benchmarks/launch_ampere1.sh

set -euo pipefail

OUTPUT_DIR="/dfs/scratch0/brando9/data/beyond_scale/eval_results"
TASKS="arc_easy,hellaswag,winogrande,lambada_openai"

# Split 27 models into 4 groups for GPUs 0, 1, 3, 4
# Group 0 (7 models): GPT2 51M variants + 117M USPTO
GPU0_MODELS=(
    "UDACA/gpt2-51M-1.31B-USPTO"
    "UDACA/gpt2-51M-1.31B-PubMedAbs"
    "UDACA/gpt2-51M-1.31B-USPTOAndPubMedAbs"
    "UDACA/gpt2-51M-557M-USPTO"
    "UDACA/gpt2-51M-557M-PubMedAbs"
    "UDACA/gpt2-51M-557M-USPTOAndPubMedAbs"
    "UDACA/gpt2-117M-2.2B-USPTO"
)

# Group 1 (7 models): 117M PubMed/Combined, 204M, 345M USPTO
GPU1_MODELS=(
    "UDACA/gpt2-117M-2.2B-PubMedAbs"
    "UDACA/gpt2-117M-2.2B-USPTOAndPubMedAbs"
    "UDACA/gpt2-204M-USPTO"
    "UDACA/gpt2-204M-PubMedAbs"
    "UDACA/gpt2-204M-USPTOandPubMedAbs"
    "UDACA/gpt2-345M-2.2B-USPTO"
    "UDACA/gpt2-345M-2.2B-PubMedAbs"
)

# Group 2 (6 models): 345M Combined, 810M, 1.5B
GPU3_MODELS=(
    "UDACA/gpt2-345M-2.2B-USPTOandPubMedAbs"
    "UDACA/gpt2-810M-PubMedAbs"
    "UDACA/gpt2-810M-2.2B-USPTOAndPubMedAbs"
    "UDACA/gpt2-1.5B-180M-USPTO"
    "UDACA/gpt2-1.5B-180M-PubMedAbs"
    "UDACA/gpt2-1.5B-180M-USPTOAndPubMedAbs"
)

# Group 3 (7 models): All LLaMA-2 7B models
GPU4_MODELS=(
    "UDACA/llama2-uspto-ckpt-1"
    "UDACA/llama2-pubmed-ckpt-2"
    "UDACA/llama2-pubmed-ckpt-7"
    "UDACA/llama2-uspto-pubmed-ckpt-3"
    "UDACA/llama2-uspto-pubmed-ckpt-4"
    "UDACA/llama2-uspto-pubmed-ckpt-5"
    "UDACA/llama2-uspto-pubmed-ckpt-6"
)

run_gpu_group() {
    local GPU_INDEX=$1
    shift
    local MODELS=("$@")

    echo "[GPU $GPU_INDEX] Starting ${#MODELS[@]} models at $(date)"

    for MODEL in "${MODELS[@]}"; do
        MODEL_SHORT=$(echo "$MODEL" | sed 's|UDACA/||')
        MODEL_OUT="$OUTPUT_DIR/${MODEL_SHORT}_downstream"

        if [ -d "$MODEL_OUT" ] && [ "$(find "$MODEL_OUT" -name 'results_*.json' 2>/dev/null | head -1)" ]; then
            echo "[GPU $GPU_INDEX] SKIP (exists): $MODEL_SHORT"
            continue
        fi

        BATCH_SIZE=16
        if [[ "$MODEL" == *"llama"* ]]; then
            BATCH_SIZE=4
        fi

        echo "[GPU $GPU_INDEX] START: $MODEL_SHORT (batch_size=$BATCH_SIZE) at $(date)"
        CUDA_VISIBLE_DEVICES=$GPU_INDEX lm_eval \
            --model hf \
            --model_args "pretrained=$MODEL,trust_remote_code=True" \
            --tasks "$TASKS" \
            --device cuda \
            --batch_size "$BATCH_SIZE" \
            --output_path "$MODEL_OUT" \
            --log_samples \
            2>&1
        echo "[GPU $GPU_INDEX] DONE: $MODEL_SHORT at $(date)"
    done

    echo "[GPU $GPU_INDEX] All models complete at $(date)"
}

# Launch all 4 GPU groups in parallel
run_gpu_group 0 "${GPU0_MODELS[@]}" &
run_gpu_group 1 "${GPU1_MODELS[@]}" &
run_gpu_group 3 "${GPU3_MODELS[@]}" &
run_gpu_group 4 "${GPU4_MODELS[@]}" &

echo "Launched 4 GPU groups in background. Waiting for all to complete..."
wait
echo "=== All evaluations complete at $(date) ==="
