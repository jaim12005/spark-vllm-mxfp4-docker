#!/usr/bin/env python3
"""Check if CUDA is initialized during vLLM startup."""
import sys
import os

sys.path.insert(0, "/workspace/flashinfer")
sys.path.insert(0, "/workspace/vllm")

import torch


def check_cuda():
    init = torch.cuda.is_initialized()
    if init:
        alloc = torch.cuda.memory_allocated() / 1e9
        res = torch.cuda.memory_reserved() / 1e9
        print(f"  CUDA initialized: alloc={alloc:.2f}GB, reserved={res:.2f}GB")
    else:
        print("  CUDA not initialized")


print("=== Checking CUDA initialization during vLLM startup ===")
check_cuda()

from vllm.engine.arg_utils import EngineArgs

print("After EngineArgs import:")
check_cuda()

engine_args = EngineArgs(
    model="openai/gpt-oss-120b",
    quantization="mxfp4",
    tensor_parallel_size=1,
    gpu_memory_utilization=0.70,
    max_model_len=8192,
    max_num_seqs=2,
    max_num_batched_tokens=4096,
    enforce_eager=True,
    enable_prefix_caching=True,
    load_format="fastsafetensors",
)
print("After EngineArgs creation:")
check_cuda()

print("--- Creating VllmConfig ---")
vllm_config = engine_args.create_engine_config()
print("After VllmConfig:")
check_cuda()

print("--- Get executor class ---")
from vllm.v1.executor.abstract import Executor

executor_cls = Executor.get_class(vllm_config)
print(f"Executor: {executor_cls}")
check_cuda()

print("--- Check mp context ---")
from vllm.utils.system_utils import get_mp_context

print(
    f"VLLM_WORKER_MULTIPROC_METHOD before: {os.environ.get('VLLM_WORKER_MULTIPROC_METHOD', 'not set')}"
)
ctx = get_mp_context()
print(
    f"VLLM_WORKER_MULTIPROC_METHOD after: {os.environ.get('VLLM_WORKER_MULTIPROC_METHOD', 'not set')}"
)
print(f"Context method: {ctx.get_start_method()}")
check_cuda()
