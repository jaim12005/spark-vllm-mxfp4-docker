#!/usr/bin/env python3
"""Test SM120 MoE tile expansion compilation.

This script validates that both TILE_M=64 and TILE_M=128 variants
compile successfully on SM121 (GB10).
"""

import os
import sys
import time

# Ensure we use local FlashInfer
os.environ["PYTHONPATH"] = "/workspace/flashinfer:/workspace/vllm"
sys.path.insert(0, "/workspace/flashinfer")
sys.path.insert(0, "/workspace/vllm")


def test_module_loading():
    """Test that FlashInfer modules can be imported."""
    print("=" * 60)
    print("Testing FlashInfer module import...")
    print("=" * 60)
    
    import flashinfer
    print(f"FlashInfer path: {flashinfer.__file__}")
    
    from flashinfer.fused_moe.core import (
        get_cutlass_fused_moe_module,
        select_tile_m_for_sm120,
        SM120_VALID_TILE_M,
    )
    
    print(f"Valid SM120 tile sizes: {SM120_VALID_TILE_M}")
    print(f"select_tile_m_for_sm120(1) = {select_tile_m_for_sm120(1)}")
    print(f"select_tile_m_for_sm120(64) = {select_tile_m_for_sm120(64)}")
    print(f"select_tile_m_for_sm120(65) = {select_tile_m_for_sm120(65)}")
    print(f"select_tile_m_for_sm120(1000) = {select_tile_m_for_sm120(1000)}")
    print()


def test_jit_compile_tile_128():
    """Test compilation of TILE_M=128 (default prefill-optimized)."""
    print("=" * 60)
    print("Testing TILE_M=128 (prefill-optimized) JIT compilation...")
    print("=" * 60)
    
    from flashinfer.fused_moe.core import get_cutlass_fused_moe_module
    
    start = time.time()
    try:
        # This will JIT compile if not cached
        module_128 = get_cutlass_fused_moe_module(
            backend="121",  # SM121 for GB10
            use_fast_build=False,
            tile_m=128,
        )
        elapsed = time.time() - start
        print(f"✓ TILE_M=128 module loaded successfully ({elapsed:.1f}s)")
        print(f"  Module type: {type(module_128)}")
        return True
    except Exception as e:
        elapsed = time.time() - start
        print(f"✗ TILE_M=128 compilation FAILED ({elapsed:.1f}s)")
        print(f"  Error: {e}")
        return False


def test_jit_compile_tile_64():
    """Test compilation of TILE_M=64 (decode-optimized)."""
    print()
    print("=" * 60)
    print("Testing TILE_M=64 (decode-optimized) JIT compilation...")
    print("=" * 60)
    
    from flashinfer.fused_moe.core import get_cutlass_fused_moe_module
    
    start = time.time()
    try:
        # This will JIT compile if not cached
        module_64 = get_cutlass_fused_moe_module(
            backend="121",  # SM121 for GB10
            use_fast_build=False,
            tile_m=64,
        )
        elapsed = time.time() - start
        print(f"✓ TILE_M=64 module loaded successfully ({elapsed:.1f}s)")
        print(f"  Module type: {type(module_64)}")
        return True
    except Exception as e:
        elapsed = time.time() - start
        print(f"✗ TILE_M=64 compilation FAILED ({elapsed:.1f}s)")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return False


def main():
    print("\n" + "=" * 60)
    print(" SM120 MoE Tile Expansion Test")
    print("=" * 60 + "\n")
    
    # Clear MoE kernel cache to force recompilation
    import subprocess
    cache_dir = "/root/.cache/flashinfer"
    print(f"Note: JIT cache at {cache_dir}")
    print("To force recompile, run: rm -rf ~/.cache/flashinfer/*/cached_ops/fused_moe_*")
    print()
    
    test_module_loading()
    
    success_128 = test_jit_compile_tile_128()
    success_64 = test_jit_compile_tile_64()
    
    print()
    print("=" * 60)
    print(" Test Summary")
    print("=" * 60)
    print(f"  TILE_M=128: {'✓ PASS' if success_128 else '✗ FAIL'}")
    print(f"  TILE_M=64:  {'✓ PASS' if success_64 else '✗ FAIL'}")
    print()
    
    if success_128 and success_64:
        print("All tile expansion tests PASSED!")
        return 0
    else:
        print("Some tile expansion tests FAILED!")
        return 1


if __name__ == "__main__":
    sys.exit(main())
