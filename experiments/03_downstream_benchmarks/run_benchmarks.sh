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

# All UDACA models to evaluate
MODELS=(
    "UDACA/GPT2_51M_1.31B_USPTO"
    "UDACA/GPT2_51M_1.31B_PubMedAbs"
    "UDACA/GPT2_51M_1.31B_USPTOAndPubMedAbs"
    "UDACA/GPT2_51M_557M_USPTO"
    "UDACA/GPT2_51M_557M_PubMedAbs"
    "UDACA/GPT2_51M_557M_USPTOAndPubMedAbs"
    "UDACA/GPT2_117M_2.2B_USPTO"
    "UDACA/GPT2_117M_2.2B_PubMedAbs"
    "UDACA/GPT2_117M_2.2B_USPTOAndPubMedAbs"
    "UDACA/GPT2_204M_USPTO"
    "UDACA/GPT2_204M_PubMedAbs"
    "UDACA/GPT2_204M_USPTOAndPubMedAbs"
    "UDACA/GPT2_345M_2.2B_USPTO"
    "UDACA/GPT2_345M_2.2B_PubMedAbs"
    "UDACA/GPT2_345M_2.2B_USPTOAndPubMedAbs"
    "UDACA/GPT2_810M_PubMedAbs"
    "UDACA/GPT2_810M_2.2B_USPTOAndPubMedAbs"
    "UDACA/GPT2_1.5B_180M_USPTO"
    "UDACA/GPT2_1.5B_180M_PubMedAbs"
    "UDACA/GPT2_1.5B_180M_USPTOAndPubMedAbs"
    "UDACA/LLama2_Uspto_Ckpt_1"
    "UDACA/LLama2_Pubmed_Ckpt_2"
    "UDACA/LLama2_Pubmed_Ckpt_7"
    "UDACA/LLama2_Uspto_Pubmed_Ckpt_3"
    "UDACA/LLama2_Uspto_Pubmed_Ckpt_4"
    "UDACA/LLama2_Uspto_Pubmed_Ckpt_5"
    "UDACA/LLama2_Uspto_Pubmed_Ckpt_6"
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
