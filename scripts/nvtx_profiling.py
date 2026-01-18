#!/usr/bin/env python3
"""
NVTX profiling markers for vLLM.

Usage:
    # Before starting vLLM, import this module:
    import nvtx_profiling
    nvtx_profiling.patch_vllm()
    
    # Or set environment variable:
    VLLM_NVTX_PROFILING=1 vllm serve ...
"""

import os
import functools
import torch.cuda.nvtx as nvtx


def wrap_execute_model(original_fn):
    """Wrap execute_model with NVTX markers for prefill/decode."""
    
    @functools.wraps(original_fn)
    def wrapped(self, scheduler_output, intermediate_tensors=None):
        num_scheduled = scheduler_output.total_num_scheduled_tokens
        num_reqs = len(scheduler_output.num_scheduled_tokens) if scheduler_output.num_scheduled_tokens else 1
        
        # Heuristic: prefill has more tokens per request
        avg_tokens = num_scheduled / max(num_reqs, 1)
        phase = "prefill" if avg_tokens > 2 else "decode"
        
        with nvtx.range(f"{phase}_reqs={num_reqs}_tokens={num_scheduled}"):
            return original_fn(self, scheduler_output, intermediate_tensors)
    
    return wrapped


def patch_vllm():
    """Apply NVTX profiling patches to vLLM."""
    try:
        from vllm.v1.worker.gpu_model_runner import GPUModelRunner
        
        # Wrap execute_model
        original = GPUModelRunner.execute_model
        GPUModelRunner.execute_model = wrap_execute_model(original)
        
        print("[NVTX] Patched GPUModelRunner.execute_model with prefill/decode markers")
        return True
    except ImportError as e:
        print(f"[NVTX] Could not patch vLLM: {e}")
        return False


# Auto-patch if environment variable is set
if os.environ.get("VLLM_NVTX_PROFILING", "").lower() in ("1", "true", "yes"):
    patch_vllm()
