#!/bin/bash

#docker exec vllm-dev bash -c 'rm -rf ~/.cache/flashinfer/*'
docker exec vllm-dev bash -c 'export VLLM_MOE_WARMUP_FORCE_UNIFORM=1 && export VLLM_MOE_ROUTING_LOG=1 && export VLLM_MXFP8_QUANT_LOG=1 && export VLLM_LOGGING_LEVEL=DEBUG && export FLASHINFER_LOGLEVEL=3 && export VLLM_FLASHINFER_CALL_LOG=1 && export PYTHONPATH=/workspace/flashinfer:/workspace/vllm && cd /workspace/vllm && CUDA_VISIBLE_DEVICES=0 python3 -m vllm.entrypoints.openai.api_server \
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
    --load-format fastsafetensors 2>&1' 
