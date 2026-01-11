#!/bin/bash

#docker exec vllm-dev bash -c 'pkill -9 -f "python.*vllm" 2>/dev/null; rm -rf /root/.cache/flashinfer/0.6.0/121a/cached_ops/fused_moe_120'
docker exec vllm-dev bash -c 'export PYTHONPATH=/workspace/flashinfer:/workspace/vllm && \
  cd /workspace/vllm && \
  python3 -m vllm.entrypoints.openai.api_server \
    --model openai/gpt-oss-120b \
    --host 0.0.0.0 \
    --port 8000 \
    --served-model-name gpt-oss-120b \
    --quantization mxfp4 \
    --tensor-parallel-size 1 \
    --gpu-memory-utilization 0.70 \
    --max-model-len 131072 \
    --max-num-seqs 2 \
    --max-num-batched-tokens 8192 \
    --enforce-eager \
    --enable-prefix-caching \
    --load-format fastsafetensors'
