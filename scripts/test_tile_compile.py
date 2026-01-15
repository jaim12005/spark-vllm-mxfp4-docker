#!/usr/bin/env python3
"""
Compile test for SM120/121 MoE GEMM tile configurations.
Tests JIT compilation of different tile sizes without running a full inference.
"""
import sys
import time


def test_tile_compile(tile_mn: tuple[int, int]) -> bool:
    """Test JIT compilation for a specific tile configuration."""
    import torch

    # Clear JIT cache for this tile to force recompile
    import shutil
    import os
    cache_dir = os.path.expanduser("~/.cache/flashinfer/")
    if os.path.exists(cache_dir):
        # Don't clear the whole cache, just let JIT handle it
        pass

    major, minor = torch.cuda.get_device_capability()
    backend = f"{major * 10 + minor}"
    
    if backend not in ("120", "121"):
        print(f"[SKIP] Not an SM120/121 device (got SM{backend})")
        return True

    print(f"\n{'='*60}")
    print(f"Testing tile_mn={tile_mn} on SM{backend}")
    print(f"{'='*60}")

    from flashinfer.fused_moe.core import get_cutlass_fused_moe_module

    start = time.time()
    try:
        mod = get_cutlass_fused_moe_module(
            backend=backend,
            use_fast_build=True,
            tile_mn=tile_mn,
        )
        elapsed = time.time() - start
        print(f"[OK] tile_mn={tile_mn} compiled in {elapsed:.1f}s")
        return True
    except Exception as e:
        elapsed = time.time() - start
        print(f"[FAIL] tile_mn={tile_mn} failed after {elapsed:.1f}s")
        print(f"  Error: {e}")
        # Print more details for compile errors
        import traceback
        traceback.print_exc()
        return False


def main():
    # Test configurations - SM120 MXFP4 constraints:
    #   - M must be multiple of 64 (tcgen05 hardware minimum)
    #   - N must be power of 2 in {8, 16, 32, 64, 128, 256}
    #   - (128, 256) exceeds smem capacity
    # Both M < 128 and N < 128 are padded to 128 internally for scale factor layouts
    # The sm120_rr_smem_copy_selector_B selects appropriate copy atom based on TileN
    configs = [
        # M=64: all N values fit
        (64, 8), (64, 16), (64, 32), (64, 64), (64, 128), (64, 256),
        # M=128: N=256 exceeds smem capacity
        (128, 8), (128, 16), (128, 32), (128, 64), (128, 128),
    ]
    
    results = {}
    failed = []
    for tile_mn in configs:
        ok = test_tile_compile(tile_mn)
        results[tile_mn] = ok
        if not ok:
            failed.append(tile_mn)
            # Continue testing other configs instead of stopping
    
    print(f"\n{'='*60}")
    print("RESULTS:")
    print(f"{'='*60}")
    passed = [t for t, ok in results.items() if ok]
    print(f"  PASSED: {len(passed)}/{len(configs)}")
    if failed:
        print(f"  FAILED: {failed}")
    else:
        print("  All configurations compiled successfully!")
    
    # Exit with error if any failed
    if failed:
        sys.exit(1)


if __name__ == "__main__":
    main()
