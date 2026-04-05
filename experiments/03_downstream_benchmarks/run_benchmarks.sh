#!/bin/bash
# Run downstream benchmarks (ARC-Easy, HellaSwag, WinoGrande, LAMBADA) on all UDACA models.
#
# Prerequisites:
#   conda activate eleuther_lm_eval_harness_20240927
#   # or: pip install lm-eval
#
# Usage:
#   bash experiments/03_downstream_benchmarks/run_benchmarks.sh [GPU_INDEX]
#
# Output: /dfs/scratch0/brando9/data/beyond_scale/eval_results/<model_name>/

set -euo pipefail

GPU_INDEX=${1:-0}
OUTPUT_DIR="/dfs/scratch0/brando9/data/beyond_scale/eval_results"
TASKS="arc_easy,hellaswag,winogrande,lambada_openai"

mkdir -p "$OUTPUT_DIR"

# All UDACA models to evaluate (correct HuggingFace IDs: lowercase, hyphens)
MODELS=(
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

echo "=== Downstream Benchmark Evaluation ==="
echo "GPU: $GPU_INDEX"
echo "Tasks: $TASKS"
echo "Output: $OUTPUT_DIR"
echo "Models: ${#MODELS[@]}"
echo ""

for MODEL in "${MODELS[@]}"; do
    MODEL_SHORT=$(echo "$MODEL" | sed 's|UDACA/||')
    MODEL_OUT="$OUTPUT_DIR/${MODEL_SHORT}_downstream"

    # Skip if results already exist
    if [ -d "$MODEL_OUT" ] && [ "$(find "$MODEL_OUT" -name 'results_*.json' 2>/dev/null | head -1)" ]; then
        echo "SKIP (exists): $MODEL_SHORT"
        continue
    fi

    echo ""
    echo ">>> Running: $MODEL_SHORT"
    echo "    Output:  $MODEL_OUT"

    # GPT-2 models: batch_size 16; LLaMA models: batch_size 4
    BATCH_SIZE=16
    if [[ "$MODEL" == *"LLama"* ]]; then
        BATCH_SIZE=4
    fi

    CUDA_VISIBLE_DEVICES=$GPU_INDEX lm_eval \
        --model hf \
        --model_args "pretrained=$MODEL,trust_remote_code=True" \
        --tasks "$TASKS" \
        --device cuda \
        --batch_size "$BATCH_SIZE" \
        --output_path "$MODEL_OUT" \
        --log_samples \
        2>&1 | tee "$MODEL_OUT.log"

    echo "DONE: $MODEL_SHORT"
done

echo ""
echo "=== All evaluations complete ==="
echo "Results in: $OUTPUT_DIR"
echo "Next: python experiments/03_downstream_benchmarks/collect_scores.py"
