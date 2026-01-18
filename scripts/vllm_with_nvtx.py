#!/usr/bin/env python3
"""
Wrapper to start vLLM with NVTX profiling markers.
Patches vLLM before starting the server.
"""

import sys
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


def patch_and_run():
    # Import vLLM components
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner
    
    # Apply NVTX patch
    original = GPUModelRunner.execute_model
    GPUModelRunner.execute_model = wrap_execute_model(original)
    print("[NVTX] Patched GPUModelRunner.execute_model with prefill/decode markers", 
          file=sys.stderr)
    
    # Now run the actual vLLM server
    from vllm.entrypoints.openai.api_server import run_server
    from vllm.entrypoints.openai.cli_args import make_arg_parser
    
    parser = make_arg_parser()
    args = parser.parse_args()
    run_server(args)


if __name__ == "__main__":
    patch_and_run()
