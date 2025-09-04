#!/bin/bash

# Example usage:
# Quick run:
# SAMPLES_PER_CATEGORY=5 CONCURRENT_REQUESTS=4 VLLM_MODELS="openai/gpt-oss-20b" ROUTER_MODELS="auto" ./run_bench.sh
# Long run:
# SAMPLES_PER_CATEGORY=100 CONCURRENT_REQUESTS=4 VLLM_MODELS="openai/gpt-oss-20b" ROUTER_MODELS="auto" ./run_bench.sh

set -x -e

export ROUTER_API_KEY="${ROUTER_API_KEY:-1234567890}"
export VLLM_API_KEY="${VLLM_API_KEY:-1234567890}"
export ROUTER_ENDPOINT="${ROUTER_ENDPOINT:-http://localhost:8801/v1}"
export VLLM_ENDPOINT="${VLLM_ENDPOINT:-http://localhost:8000/v1}"
export ROUTER_MODELS="${ROUTER_MODELS:-auto}"
export VLLM_MODELS="${VLLM_MODELS:-openai/gpt-oss-20b}"
export SAMPLES_PER_CATEGORY="${SAMPLES_PER_CATEGORY:-5}"
export CONCURRENT_REQUESTS="${CONCURRENT_REQUESTS:-4}"

# Run the benchmark
python router_reason_bench.py \
  --run-router \
  --router-endpoint "$ROUTER_ENDPOINT" \
  --router-api-key "$ROUTER_API_KEY" \
  --router-models "$ROUTER_MODELS" \
  --run-vllm \
  --vllm-endpoint "$VLLM_ENDPOINT" \
  --vllm-api-key "$VLLM_API_KEY" \
  --vllm-models "$VLLM_MODELS" \
  --samples-per-category "$SAMPLES_PER_CATEGORY" \
  --vllm-exec-modes NR XC \
  --concurrent-requests "$CONCURRENT_REQUESTS" \
  --output-dir results/reasonbench

# Generate plots
echo "Processing model paths..."
echo "VLLM_MODELS: $VLLM_MODELS"
echo "ROUTER_MODELS: $ROUTER_MODELS"

# Get first model name and make it path-safe
VLLM_MODEL_FIRST=$(echo "$VLLM_MODELS" | cut -d' ' -f1)
ROUTER_MODEL_FIRST=$(echo "$ROUTER_MODELS" | cut -d' ' -f1)
echo "First models: VLLM=$VLLM_MODEL_FIRST, Router=$ROUTER_MODEL_FIRST"

# Replace / with _ for path safety
VLLM_MODELS_SAFE=$(echo "$VLLM_MODEL_FIRST" | tr '/' '_')
ROUTER_MODELS_SAFE=$(echo "$ROUTER_MODEL_FIRST" | tr '/' '_')
echo "Safe paths: VLLM=$VLLM_MODELS_SAFE, Router=$ROUTER_MODELS_SAFE"

# Construct the full paths
VLLM_SUMMARY="results/reasonbench/vllm::${VLLM_MODELS_SAFE}/summary.json"
ROUTER_SUMMARY="results/reasonbench/router::${ROUTER_MODELS_SAFE}/summary.json"
echo "Looking for summaries at:"
echo "VLLM: $VLLM_SUMMARY"
echo "Router: $ROUTER_SUMMARY"

python bench_plot.py \
  --summary "$VLLM_SUMMARY" \
  --router-summary "$ROUTER_SUMMARY"
