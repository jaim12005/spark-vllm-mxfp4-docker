#!/usr/bin/env python3
"""Analysis of SM120/SM121 tile constraints for MXFP4 decode optimization.

This script documents the 128x128 tile constraint for SM120 MoE operations.

Key Finding:
    While CUTLASS supports M=64 tiles for block-scaled GEMM (with ScaleGranularityM=1),
    the FlashInfer MoE kernel uses kSm12xBlockScaleGranularity=128, making M=64 tiles
    incompatible due to TMA layout constraints.
    
What We Tried:
    - Added M=64 tiles to FlashInfer's SM120 config
    - Got TMA error: "CTA_Tile and SLayout top-level size equivalence"
    
Why It Failed:
    - MoE kernel uses 128-element scale granularity
    - TMA requires tile alignment with scale factor layout
    - M=64 < 128 causes layout mismatch
    
References:
    - flashinfer/csrc/nv_internal/.../sm12x_arch_config.h (kSm12xBlockScaleGranularity=128)
    - flashinfer/csrc/nv_internal/.../sm12x_activation_quantizer.cuh
"""

import argparse


def analyze_tile_efficiency():
    """Calculate compute efficiency for different M values with 128x128 tiles."""
    
    print("=" * 70)
    print("SM120/SM121 Block-Scaled GEMM Tile Efficiency Analysis")
    print("=" * 70)
    print()
    
    # Tile size is fixed at 128x128 for SM120 FP4
    TILE_M = 128
    TILE_N = 128
    
    print(f"Current FlashInfer MoE tile size: {TILE_M}×{TILE_N}")
    print()
    print("MoE-SPECIFIC LIMITATION:")
    print("  - MoE kernel uses kSm12xBlockScaleGranularity = 128 (scale per 128 elements)")
    print("  - TMA requires tile alignment with scale factor layout")
    print("  - M=64 tiles fail: 'CTA_Tile and SLayout top-level size equivalence' error")
    print("  - CUTLASS examples use ScaleGranularityM=1 (different layout, incompatible)")
    print()
    
    print("Compute Efficiency by Batch Size (M):")
    print("-" * 50)
    print(f"{'M (tokens)':<12} {'Tiles Needed':<15} {'Efficiency':<15} {'Waste'}")
    print("-" * 50)
    
    test_m_values = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512]
    
    for m in test_m_values:
        tiles_needed = (m + TILE_M - 1) // TILE_M
        total_rows_processed = tiles_needed * TILE_M
        efficiency = m / total_rows_processed * 100
        waste = 100 - efficiency
        
        print(f"{m:<12} {tiles_needed:<15} {efficiency:>6.2f}%        {waste:>6.2f}%")
    
    print("-" * 50)
    print()
    
    print("IMPACT ON DECODE (M=1):")
    print("  - Single token decode uses 1 row of a 128×128 tile")
    print("  - Compute efficiency: 0.78% (127 rows wasted)")
    print("  - Per-token MoE overhead: ~381μs (FC1: 266μs + FC2: 115μs)")
    print("  - 60 layers × 381μs = 22.9ms MoE time per token")
    print()
    
    print("COMPARISON WITH COMPETITORS:")
    print("-" * 50)
    print("  llama.cpp:  58 tok/s  (likely uses different kernel strategy)")
    print("  SGLang:     52 tok/s")
    print("  vLLM/FI:    29 tok/s  (limited by 128×128 tile constraint)")
    print("-" * 50)
    print()
    
    print("ATTEMPTED SOLUTION (FAILED):")
    print("  We tried adding M=64 tiles but got TMA layout errors.")
    print("  The MoE kernel's scale factor granularity (128) makes smaller tiles incompatible.")
    print()
    print("VIABLE ALTERNATIVES:")
    print()
    print("1. Speculative decoding:")
    print("   - Generate multiple candidate tokens → M > 1")
    print("   - vLLM has built-in support")
    print()
    print("2. Token batching:")
    print("   - Accumulate decode tokens before MoE processing")
    print("   - Trades latency for throughput")
    print()
    print("3. Reduce scale granularity (HIGH EFFORT):")
    print("   - Change kSm12xBlockScaleGranularity from 128 to 64")
    print("   - Redesign scale factor packing")
    print("   - Change weight quantization format")
    print()
    print("Why llama.cpp achieves 58 tok/s vs our 29 tok/s:")
    print("   - Likely uses custom GEMV with on-the-fly dequantization")
    print("   - Not constrained by TMA block-scaled GEMM")
    print()


def show_code_references():
    """Show why M=64 tiles don't work for MoE."""
    
    print("=" * 70)
    print("Why M=64 Tiles Don't Work for MoE")
    print("=" * 70)
    print()
    
    print("1. MoE kernel uses fixed 128-element scale granularity:")
    print("-" * 50)
    print("""
// sm12x_arch_config.h
constexpr int kSm12xBlockScaleGranularity = 128;

// sm12x_activation_quantizer.cuh
static constexpr int kBlkMN = kSm12xBlockScaleGranularity;  // 128
""")
    
    print("2. CUTLASS example uses DIFFERENT scale layout (ScaleGranularityM=1):")
    print("-" * 50)
    print("""
// examples/87_blackwell_geforce_gemm_blockwise/87b_*.cu
using PingpongMmaTileShape_MNK = Shape<_64, _128, _128>;

constexpr int ScaleGranularityM = 1;   // <-- Per-element, NOT 128!
constexpr int ScaleGranularityN = 128;
constexpr int ScaleGranularityK = 128;

// This is INCOMPATIBLE with MoE's 128-element blocks
""")
    
    print("3. TMA error when trying M=64 with 128-element scale granularity:")
    print("-" * 50)
    print("""
// Compilation error:
// static assertion failed with "TMA requires CTA_Tile and 
// SLayout top-level size equivalence."

// The scale factor layout (SLayout) is designed for 128 elements.
// When tile_m=64 < 128, TMA can't map the scale factors.
""")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Analyze SM120/SM121 tile constraints for MXFP4"
    )
    parser.add_argument(
        "--show-code",
        action="store_true",
        help="Show code references for the tile constraint"
    )
    args = parser.parse_args()
    
    analyze_tile_efficiency()
    
    if args.show_code:
        print()
        show_code_references()

