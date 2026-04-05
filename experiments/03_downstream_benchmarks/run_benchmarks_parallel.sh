#!/bin/bash
# Run downstream benchmarks on multiple GPUs in parallel.
# Splits 27 UDACA models across available GPUs on ampere1.
#
# Usage: bash experiments/03_downstream_benchmarks/run_benchmarks_parallel.sh

set -euo pipefail

OUTPUT_DIR="/dfs/scratch0/brando9/data/beyond_scale/eval_results"
TASKS="arc_easy,hellaswag,winogrande,lambada_openai"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

mkdir -p "$OUTPUT_DIR"

run_model() {
    local GPU_INDEX=$1
    local MODEL=$2
    local MODEL_SHORT
    MODEL_SHORT=$(echo "$MODEL" | sed 's|UDACA/||')
    local MODEL_OUT="$OUTPUT_DIR/${MODEL_SHORT}_downstream"

    # Skip if results already exist
    if [ -d "$MODEL_OUT" ] && [ "$(find "$MODEL_OUT" -name 'results_*.json' 2>/dev/null | head -1)" ]; then
        echo "[GPU $GPU_INDEX] SKIP (exists): $MODEL_SHORT"
        return 0
    fi

    # GPT-2 models: batch_size 16; LLaMA models: batch_size 4
    local BATCH_SIZE=16
    if [[ "$MODEL" == *"LLama"* ]]; then
        BATCH_SIZE=4
    fi

    echo "[GPU $GPU_INDEX] START: $MODEL_SHORT"
    CUDA_VISIBLE_DEVICES=$GPU_INDEX lm_eval \
        --model hf \
        --model_args "pretrained=$MODEL,trust_remote_code=True" \
        --tasks "$TASKS" \
        --device cuda \
        --batch_size "$BATCH_SIZE" \
        --output_path "$MODEL_OUT" \
        --log_samples \
        2>&1 | tee "$MODEL_OUT.log"
    echo "[GPU $GPU_INDEX] DONE: $MODEL_SHORT"
}

# All 27 UDACA models (correct HuggingFace IDs: lowercase, hyphens)
ALL_MODELS=(
    "UDACA/gpt2-51M-1.31B-USPTO"
    "UDACA/gpt2-51M-1.31B-PubMedAbs"
    "UDACA/gpt2-51M-1.31B-USPTOAndPubMedAbs"
    "UDACA/gpt2-51M-557M-USPTO"
    "UDACA/gpt2-51M-557M-PubMedAbs"
    "UDACA/gpt2-51M-557M-USPTOAndPubMedAbs"
    "UDACA/gpt2-117M-2.2B-USPTO"
    "UDACA/gpt2-117M-2.2B-PubMedAbs"
    "UDACA/gpt2-117M-2.2B-USPTOAndPubMedAbs"
    "UDACA/gpt2-204M-USPTO"
    "UDACA/gpt2-204M-PubMedAbs"
    "UDACA/gpt2-204M-USPTOandPubMedAbs"
    "UDACA/gpt2-345M-2.2B-USPTO"
    "UDACA/gpt2-345M-2.2B-PubMedAbs"
    "UDACA/gpt2-345M-2.2B-USPTOandPubMedAbs"
    "UDACA/gpt2-810M-PubMedAbs"
    "UDACA/gpt2-810M-2.2B-USPTOAndPubMedAbs"
    "UDACA/gpt2-1.5B-180M-USPTO"
    "UDACA/gpt2-1.5B-180M-PubMedAbs"
    "UDACA/gpt2-1.5B-180M-USPTOAndPubMedAbs"
    "UDACA/llama2-uspto-ckpt-1"
    "UDACA/llama2-pubmed-ckpt-2"
    "UDACA/llama2-pubmed-ckpt-7"
    "UDACA/llama2-uspto-pubmed-ckpt-3"
    "UDACA/llama2-uspto-pubmed-ckpt-4"
    "UDACA/llama2-uspto-pubmed-ckpt-5"
    "UDACA/llama2-uspto-pubmed-ckpt-6"
)

# Available GPUs on ampere1
GPUS=(0 1 3 4)
NUM_GPUS=${#GPUS[@]}

echo "=== Parallel Downstream Benchmark Evaluation ==="
echo "GPUs: ${GPUS[*]}"
echo "Tasks: $TASKS"
echo "Models: ${#ALL_MODELS[@]}"
echo ""

# Round-robin assign models to GPUs
for i in "${!ALL_MODELS[@]}"; do
    GPU_IDX=$((i % NUM_GPUS))
    GPU=${GPUS[$GPU_IDX]}
    run_model "$GPU" "${ALL_MODELS[$i]}" &

    # Wait for all GPU slots to finish before starting next batch
    if (( (i + 1) % NUM_GPUS == 0 )); then
        wait
    fi
done
wait

echo ""
echo "=== All evaluations complete ==="
echo "Results in: $OUTPUT_DIR"
echo "Next: python experiments/03_downstream_benchmarks/collect_scores.py"
