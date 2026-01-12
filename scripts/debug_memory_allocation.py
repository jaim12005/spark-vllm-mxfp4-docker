#!/usr/bin/env python3
"""Debug GPU memory allocation during vLLM model loading.

Traces memory usage at key points to identify where the 100GB
allocation occurs before the EngineCore fork.
"""

import gc
import os
import sys

# Set up environment
os.environ["PYTHONPATH"] = "/workspace/flashinfer:/workspace/vllm"
sys.path.insert(0, "/workspace/flashinfer")
sys.path.insert(0, "/workspace/vllm")

import torch


def get_memory_mb():
    """Get current GPU memory usage in MB."""
    if not torch.cuda.is_available():
        return 0, 0
    allocated = torch.cuda.memory_allocated() / 1e6
    reserved = torch.cuda.memory_reserved() / 1e6
    return allocated, reserved


def log_memory(stage: str):
    """Log memory usage at a stage."""
    allocated, reserved = get_memory_mb()
    print(f"[MEMORY] {stage}: allocated={allocated:.0f}MB, reserved={reserved:.0f}MB")
    return allocated, reserved


def main():
    print("=" * 60)
    print("GPU Memory Allocation Debug")
    print("=" * 60)
    
    log_memory("Initial (before any imports)")
    
    # Step 1: Import vLLM
    print("\n--- Importing vLLM ---")
    from vllm import LLM, SamplingParams
    log_memory("After vLLM import")
    
    # Step 2: Import config classes
    print("\n--- Importing config classes ---")
    from vllm.config import VllmConfig
    from vllm.engine.arg_utils import EngineArgs
    log_memory("After config imports")
    
    # Step 3: Parse engine args (no model loading yet)
    print("\n--- Parsing engine args ---")
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
    log_memory("After EngineArgs creation")
    
    # Step 4: Create VllmConfig
    print("\n--- Creating VllmConfig ---")
    vllm_config = engine_args.create_engine_config()
    log_memory("After VllmConfig creation")
    
    # Step 5: Check model config
    print("\n--- Model config details ---")
    print(f"  Model: {vllm_config.model_config.model}")
    print(f"  Quantization: {vllm_config.model_config.quantization}")
    print(f"  Max model len: {vllm_config.model_config.max_model_len}")
    log_memory("After model config access")
    
    # Step 6: Try to load model weights directly (this is where it might blow up)
    print("\n--- Attempting weight loader initialization ---")
    try:
        from vllm.model_executor.model_loader.loader import get_model_loader
        loader = get_model_loader(vllm_config.load_config)
        log_memory("After get_model_loader")
        
        # Check what the loader will do
        print(f"  Loader type: {type(loader).__name__}")
        print(f"  Load format: {vllm_config.load_config.load_format}")
    except Exception as e:
        print(f"  Error: {e}")
        log_memory("After loader error")
    
    # Step 7: Try fastsafetensors parsing specifically
    print("\n--- Testing fastsafetensors weight parsing ---")
    try:
        from vllm.model_executor.model_loader.weight_utils import (
            initialize_dummy_weights,
            np_cache_weights_iterator,
        )
        log_memory("After weight_utils import")
        
        # Check if we can get the model path
        from huggingface_hub import snapshot_download
        model_path = snapshot_download(
            "openai/gpt-oss-120b",
            allow_patterns=["*.safetensors", "*.json"],
            local_files_only=True,  # Don't download, just find local
        )
        print(f"  Model path: {model_path}")
        log_memory("After snapshot_download (local)")
        
    except Exception as e:
        print(f"  Error: {e}")
        log_memory("After weight parsing error")
    
    # Step 8: Check if there's pre-loading happening
    print("\n--- Checking for pre-allocated buffers ---")
    gc.collect()
    torch.cuda.empty_cache()
    log_memory("After gc.collect and empty_cache")
    
    # Step 9: Try minimal model initialization to see where allocation happens
    print("\n--- Testing model class initialization ---")
    try:
        from vllm.model_executor.models import get_model_architecture
        model_cls = get_model_architecture(vllm_config.model_config)
        print(f"  Model class: {model_cls}")
        log_memory("After get_model_architecture")
    except Exception as e:
        print(f"  Error: {e}")
    
    # Final summary
    print("\n" + "=" * 60)
    print("Summary")
    print("=" * 60)
    final_alloc, final_reserved = log_memory("Final state")
    
    if final_reserved > 1000:  # More than 1GB reserved
        print(f"\n⚠️  WARNING: {final_reserved/1000:.1f}GB reserved before model loading!")
        print("This memory will be inherited by forked child processes.")
    else:
        print(f"\n✓ Memory usage is reasonable ({final_reserved:.0f}MB)")


if __name__ == "__main__":
    main()
