#!/usr/bin/env python3
"""
Analyze MoE kernel tile configurations for SM121 decode optimization.

This script helps understand the tile waste at different batch sizes (M values).
Use this to guide optimization decisions for M=1 decode performance.

Usage:
    python3 scripts/analyze_moe_tiles.py
"""

# SM120/SM121 FP4 tile configurations from FlashInfer/TRT-LLM
# Format: (M, N, K_bytes) - K is in bytes, not elements
SM120_FP4_TILES = [
    ("CtaShape64x128x256B", 64, 128, 256),
    ("CtaShape128x128x128B", 128, 128, 128),  # Primary validated tile
    ("CtaShape128x128x256B", 128, 128, 256),
    ("CtaShape256x128x128B", 256, 128, 128),
]

# gpt-oss-120b MoE dimensions
GPT_OSS_CONFIG = {
    "num_experts": 128,
    "hidden_dim": 2944,
    "intermediate_dim": 5888,
    "topk": 8,
    "num_layers": 60,
    "moe_calls_per_layer": 2,  # FC1 + FC2
}


def analyze_tile_waste(m_values: list[int], tiles: list[tuple]) -> None:
    """Analyze compute waste for different M (batch) sizes."""
    
    print("=" * 70)
    print("MoE Tile Efficiency Analysis for SM121 (gpt-oss-120b)")
    print("=" * 70)
    print()
    
    config = GPT_OSS_CONFIG
    print(f"Model: {config['num_experts']} experts, {config['hidden_dim']} hidden, {config['intermediate_dim']} intermediate")
    print(f"MoE calls per token: {config['num_layers']} layers × {config['moe_calls_per_layer']} calls = {config['num_layers'] * config['moe_calls_per_layer']}")
    print()
    
    # Find best tile for each M
    print("Tile Efficiency by Batch Size (M):")
    print("-" * 70)
    print(f"{'M':>6} | {'Best Tile':>25} | {'Tile M':>8} | {'Waste %':>8} | {'Effective Compute':>18}")
    print("-" * 70)
    
    for m in m_values:
        # Find smallest tile that fits M
        best_tile = None
        best_waste = 100.0
        
        for name, tile_m, tile_n, tile_k in tiles:
            if tile_m >= m:
                waste = (tile_m - m) / tile_m * 100
                if waste < best_waste:
                    best_waste = waste
                    best_tile = (name, tile_m, tile_n, tile_k)
        
        if best_tile:
            name, tile_m, _, _ = best_tile
            effective = 100 - best_waste
            print(f"{m:>6} | {name:>25} | {tile_m:>8} | {best_waste:>7.1f}% | {effective:>17.1f}%")
        else:
            print(f"{m:>6} | No valid tile | - | - | -")
    
    print()
    
    # Recommendations
    print("=" * 70)
    print("Recommendations:")
    print("=" * 70)
    print()
    print("1. CRITICAL: For M=1 (single token decode), 128×128 tiles waste 99.2% compute!")
    print("   → Need smaller tiles: 8×128, 16×128, or 32×128")
    print()
    print("2. GEMV Path: For M≤8, a GEMV kernel would be more efficient")
    print("   → Memory-bound operation, no tensor core waste")
    print()
    print("3. Token Batching: Group decode tokens to improve utilization")
    print("   → M=8 with 128×128 tile = 93.75% waste (vs 99.2% for M=1)")
    print("   → M=32 with 128×128 tile = 75% waste")
    print("   → M=128 with 128×128 tile = 0% waste")
    print()
    print("4. Current SM120 FP4 tiles (from cutlass_heuristic.cpp):")
    print("   - Only CtaShape128x128x128B is validated for SM120 FP4")
    print("   - Adding smaller tiles requires CUTLASS kernel instantiation")
    print()


def estimate_decode_performance(current_tok_per_sec: float, tiles: list[tuple]) -> None:
    """Estimate potential decode performance with better tiles."""
    
    print("=" * 70)
    print("Potential Performance Improvement")
    print("=" * 70)
    print()
    
    # Current: M=1 with 128×128 tile
    current_waste = 127 / 128 * 100  # 99.2%
    current_effective = 100 - current_waste
    
    print(f"Current decode: {current_tok_per_sec:.1f} tok/s (with {current_waste:.1f}% compute waste)")
    print()
    
    # Projected with better tiles
    scenarios = [
        ("M=1 with 8×128 tile", 1, 8),
        ("M=1 with 16×128 tile", 1, 16),
        ("M=1 with GEMV kernel", 1, 1),  # Perfect efficiency assumed
        ("M=8 batched with 8×128 tile", 8, 8),
        ("M=8 batched with 128×128 tile", 8, 128),
    ]
    
    print("Projected performance with optimizations:")
    print("-" * 50)
    
    for name, m, tile_m in scenarios:
        new_waste = max(0, (tile_m - m) / tile_m * 100)
        new_effective = 100 - new_waste
        
        # Rough estimate: performance scales with effective compute ratio
        # This is simplified - real gains depend on many factors
        improvement_factor = new_effective / current_effective
        projected_tok_per_sec = current_tok_per_sec * improvement_factor
        
        print(f"  {name}:")
        print(f"    Waste: {new_waste:.1f}% → Effective: {new_effective:.1f}%")
        print(f"    Projected: ~{projected_tok_per_sec:.0f} tok/s ({improvement_factor:.1f}× improvement)")
        print()


if __name__ == "__main__":
    # Test batch sizes
    m_values = [1, 2, 4, 8, 16, 32, 64, 128, 256, 512, 1024, 2048]
    
    analyze_tile_waste(m_values, SM120_FP4_TILES)
    
    # Estimate performance improvement (using current ~29 tok/s baseline)
    estimate_decode_performance(29.0, SM120_FP4_TILES)


