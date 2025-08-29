#!/bin/bash

set -x 

export ROUTER_API_KEY="1234567890"
export VLLM_API_KEY="1234567890"
export ROUTER_ENDPOINT="http://localhost:8801/v1"
export VLLM_ENDPOINT="http://localhost:8000/v1"
export ROUTER_MODELS="auto"
export VLLM_MODELS="openai/gpt-oss-20b"

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
  --samples-per-category 5 \
  --vllm-exec-modes NR XC \
  --concurrent-requests 4 \
  --output-dir results/reasonbench

# Generate plots
VLLM_MODEL_FIRST="${VLLM_MODELS%% *}"
ROUTER_MODEL_FIRST="${ROUTER_MODELS%% *}"
VLLM_MODELS_SAFE="${VLLM_MODEL_FIRST//\//_}"
ROUTER_MODELS_SAFE="${ROUTER_MODEL_FIRST//\//_}"
python bench_plot.py \
  --summary "results/reasonbench/vllm::${VLLM_MODELS_SAFE}/summary.json" \
  --router-summary "results/reasonbench/router::${ROUTER_MODELS_SAFE}/summary.json"
