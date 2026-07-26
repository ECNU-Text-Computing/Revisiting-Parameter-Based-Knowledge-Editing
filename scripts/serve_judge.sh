#!/bin/bash
# Launch vLLM serving Qwen2.5-72B-Instruct for LLM-as-a-Judge evaluation.
# The judge is used for semantic consistency scoring (Section 5).
#
# Usage: bash scripts/serve_judge.sh

MODEL_PATH="${MODEL_PATH:-/home/wyren/.cache/huggingface/hub/models--Qwen--Qwen2.5-72B-Instruct/snapshots/495f39366efef23836d0cfae4fbe635880d2be31}"
VLLM_PORT="${VLLM_PORT:-21910}"
TP_SIZE="${TP_SIZE:-4}"

echo "Starting vLLM judge service on port ${VLLM_PORT}..."
echo "Model: ${MODEL_PATH}"

vllm serve "$MODEL_PATH" \
    --tensor-parallel-size "$TP_SIZE" \
    --gpu-memory-utilization 0.9 \
    --host 0.0.0.0 \
    --port "$VLLM_PORT" \
    --served-model-name Qwen2.5-72B-Instruct \
    --max-model-len 8192 \
    --dtype bfloat16 &

sleep 10
while ! curl -s "http://localhost:${VLLM_PORT}/v1/models" > /dev/null; do
    sleep 10
    echo "Waiting for vLLM service..."
done
echo "vLLM judge service ready on port ${VLLM_PORT}"
