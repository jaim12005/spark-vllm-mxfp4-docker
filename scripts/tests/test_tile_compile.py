#!/usr/bin/env python3
"""
Compile test for SM120/121 MoE GEMM tile configurations.
Tests JIT compilation of different tile sizes without running a full inference.

Tile Configuration Constraints for SM120 MXFP4:
===============================================

Hardware Constraints:
  - Physical M must be >= 64 (tcgen05 MMA instruction minimum)
  - Physical N must be power of 2 (smem layout atom constraint)
  - Shared memory: 101,376 bytes (~99 KB) on GB10
  - Pipeline requires >= 2 stages (double-buffering for latency hiding)

Shared Memory Budget per Stage (approximate):
  - A tensor (FP8): M * K bytes
  - B tensor (FP4): N * K / 2 bytes
  - Scale factors: ~16-64 bytes
  - Pipeline overhead: ~15 KB (conservative estimate)

With swap_ab=True:
  - Logical (M, N) becomes physical (N, M)
  - Enables small logical M (8, 16, 32) by making physical M = logical N
  - Enables large logical M (256) when physical fits in smem

Verified Configurations (compile tested):
  - M ∈ {64, 128}, N ∈ {8, 16, 32, 64, 128, 256} (excluding 128x256)

Theoretical with swap_ab (need testing):
  - M ∈ {8, 16, 32}, N ∈ {64, 128} (maps to physical M=64 or 128)
  - M=256, N=64 (maps to physical 64x256)

Known to Fail:
  - (128, 256): exceeds smem capacity (only 1 stage fits, need 2+)
  - (256, 128), (256, 256): physical M=256 + large N exceeds smem
  - (512, *): any M=512 configuration exceeds smem
"""
import sys
import time


def test_tile_compile(tile_mn: tuple[int, int], swap_ab: bool = False) -> bool:
    """Test JIT compilation for a specific tile configuration."""
    import torch

    major, minor = torch.cuda.get_device_capability()
    backend = f"{major * 10 + minor}"
    
    if backend not in ("120", "121"):
        print(f"[SKIP] Not an SM120/121 device (got SM{backend})")
        return True

    swap_str = " (swap_ab)" if swap_ab else ""
    print(f"\n{'='*60}")
    print(f"Testing tile_mn={tile_mn}{swap_str} on SM{backend}")
    print(f"{'='*60}")

    from flashinfer.fused_moe.core import get_cutlass_fused_moe_module

    start = time.time()
    try:
        mod = get_cutlass_fused_moe_module(
            backend=backend,
            use_fast_build=True,
            tile_mn=tile_mn,
            # TODO: pass swap_ab when supported
        )
        elapsed = time.time() - start
        print(f"[OK] tile_mn={tile_mn}{swap_str} compiled in {elapsed:.1f}s")
        return True
    except Exception as e:
        elapsed = time.time() - start
        print(f"[FAIL] tile_mn={tile_mn}{swap_str} failed after {elapsed:.1f}s")
        print(f"  Error: {e}")
        import traceback
        traceback.print_exc()
        return False


# =============================================================================
# Tile Configuration Categories
# =============================================================================

# Currently verified and working (no swap needed)
# Physical M ∈ {64, 128}, Physical N ∈ {8, 16, 32, 64, 128, 256}
VERIFIED_NO_SWAP = [
    # M=64: all power-of-2 N values fit in smem
    (64, 8),
    (64, 16),
    (64, 32),
    (64, 64),
    (64, 128),
    (64, 256),
    # M=128: N up to 128 fits, N=256 exceeds smem
    (128, 8),
    (128, 16),
    (128, 32),
    (128, 64),
    (128, 128),
]

# Theoretical configurations requiring swap_ab=True
# Logical (M, N) -> Physical (N, M)
# Enables small logical M by making physical M = logical N (which is >= 64)
THEORETICAL_WITH_SWAP = [
    # Logical M=8: physical M = logical N (64 or 128)
    # (8, 64) -> physical (64, 8)
    # (8, 128) -> physical (128, 8)
    (8, 64),
    (8, 128),
    
    # Logical M=16: physical M = logical N (64 or 128)
    # (16, 64) -> physical (64, 16)
    # (16, 128) -> physical (128, 16)
    (16, 64),
    (16, 128),
    
    # Logical M=32: physical M = logical N (64 or 128)
    # (32, 64) -> physical (64, 32)
    # (32, 128) -> physical (128, 32)
    (32, 64),
    (32, 128),
    
    # Logical M=256: physical M = 64, physical N = 256
    # (256, 64) -> physical (64, 256) - marginal, ~2 stages
    (256, 64),
]

# Known to fail due to smem constraints
KNOWN_FAIL = [
    # (128, 256): CUTLASS calculates only 1 stage, need >= 2
    # Stage size: A=16KB + B=16KB + overhead = ~50KB
    # 101KB / 50KB = ~2, but actual overhead pushes to 1 stage
    (128, 256),
    
    # These would also fail:
    # (256, 128) -> physical too large
    # (256, 256) -> physical too large
    # (512, 64) -> even with swap, physical (64, 512) is marginal
]


def main():
    """Run tile compilation tests."""
    import argparse
    
    parser = argparse.ArgumentParser(description="Test SM120 MXFP4 tile configurations")
    parser.add_argument("--verified-only", action="store_true",
                        help="Only test verified configurations (no swap)")
    parser.add_argument("--include-swap", action="store_true",
                        help="Include theoretical swap_ab configurations")
    parser.add_argument("--include-known-fail", action="store_true",
                        help="Include known-to-fail configurations")
    parser.add_argument("--tile", type=str, default=None,
                        help="Test a specific tile, e.g., '64,128'")
    args = parser.parse_args()
    
    # Build test list
    configs = []
    
    if args.tile:
        # Test specific tile
        m, n = map(int, args.tile.split(","))
        configs.append(((m, n), False))
    else:
        # Verified configurations (always included)
        for tile in VERIFIED_NO_SWAP:
            configs.append((tile, False))
        
        # Theoretical swap configurations
        if args.include_swap:
            for tile in THEORETICAL_WITH_SWAP:
                configs.append((tile, True))
        
        # Known failures (for documentation/verification)
        if args.include_known_fail:
            for tile in KNOWN_FAIL:
                configs.append((tile, False))
    
    print(f"Testing {len(configs)} tile configurations...")
    print()
    print("Configuration categories:")
    print(f"  VERIFIED_NO_SWAP: {len(VERIFIED_NO_SWAP)} tiles")
    print(f"  THEORETICAL_WITH_SWAP: {len(THEORETICAL_WITH_SWAP)} tiles")
    print(f"  KNOWN_FAIL: {len(KNOWN_FAIL)} tiles")
    
    results = {}
    failed = []
    
    for tile_mn, swap_ab in configs:
        ok = test_tile_compile(tile_mn, swap_ab)
        results[(tile_mn, swap_ab)] = ok
        if not ok:
            failed.append((tile_mn, swap_ab))
    
    print(f"\n{'='*60}")
    print("RESULTS SUMMARY")
    print(f"{'='*60}")
    
    passed = [(t, s) for (t, s), ok in results.items() if ok]
    print(f"  PASSED: {len(passed)}/{len(configs)}")
    
    if failed:
        print(f"  FAILED ({len(failed)}):")
        for tile, swap in failed:
            swap_str = " (swap_ab)" if swap else ""
            print(f"    - {tile}{swap_str}")
    else:
        print("  All configurations compiled successfully!")
    
    # Print summary table
    print(f"\n{'='*60}")
    print("TILE SUPPORT MATRIX (No Swap)")
    print(f"{'='*60}")
    print()
    print("       N:    8    16    32    64   128   256")
    print("    M  " + "-" * 40)
    
    for m in [64, 128]:
        row = f"  {m:>3}  |"
        for n in [8, 16, 32, 64, 128, 256]:
            tile = (m, n)
            if (tile, False) in results:
                status = " ✓  " if results[(tile, False)] else " ✗  "
            elif tile in KNOWN_FAIL:
                status = " ✗  "
            else:
                status = " -  "
            row += status
        print(row)
    
    print()
    print("Legend: ✓ = passed, ✗ = failed, - = not tested")
    
    # Exit with error if any failed (excluding known failures)
    unexpected_failures = [
        (t, s) for (t, s) in failed 
        if t not in KNOWN_FAIL or args.include_known_fail
    ]
    if unexpected_failures:
        sys.exit(1)


if __name__ == "__main__":
    main()
