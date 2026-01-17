// Debug script to print actual CUTLASS SharedStorage sizes
// Compile with: nvcc -std=c++17 -I/path/to/cutlass/include debug_smem_sizes.cu -o debug_smem_sizes

#include <cstdio>

// Minimal includes to get the size calculations
#include <cute/layout.hpp>
#include <cutlass/arch/arch.h>
#include <cutlass/arch/barrier.h>
#include <cutlass/pipeline/sm90_pipeline.hpp>
#include <cutlass/pipeline/sm100_pipeline.hpp>

using namespace cute;

int main() {
    printf("CUTLASS Shared Memory Size Analysis\n");
    printf("====================================\n\n");
    
    // Barrier sizes
    printf("Barrier sizes:\n");
    printf("  ClusterBarrier: %zu bytes\n", sizeof(cutlass::arch::ClusterBarrier));
    printf("  ClusterTransactionBarrier: %zu bytes\n", sizeof(cutlass::arch::ClusterTransactionBarrier));
    printf("\n");
    
    // Pipeline SharedStorage sizes for different stage counts
    printf("PipelineTmaAsync SharedStorage:\n");
    printf("  1 stage: %zu bytes\n", sizeof(cutlass::PipelineTmaAsync<1>::SharedStorage));
    printf("  2 stages: %zu bytes\n", sizeof(cutlass::PipelineTmaAsync<2>::SharedStorage));
    printf("  3 stages: %zu bytes\n", sizeof(cutlass::PipelineTmaAsync<3>::SharedStorage));
    printf("  4 stages: %zu bytes\n", sizeof(cutlass::PipelineTmaAsync<4>::SharedStorage));
    printf("\n");
    
    // TMA descriptor size
    printf("TmaDescriptor: %zu bytes\n", sizeof(cute::TmaDescriptor));
    printf("\n");
    
    // SM120 capacity
    printf("SM120 smem capacity: %d bytes\n", cutlass::arch::sm120_smem_capacity_bytes);
    printf("SM100 smem capacity: %d bytes\n", cutlass::arch::sm100_smem_capacity_bytes);
    printf("\n");
    
    // Calculate for (128, 256, 128) tile
    constexpr int TileM = 128;
    constexpr int TileN = 256;
    constexpr int TileK = 128;
    constexpr int Stages = 2;
    
    // A tensor: FP8 stored as uint8_t
    constexpr size_t smem_A_size = TileM * TileK * Stages;
    
    // B tensor: FP4 stored as uint8_t (NOT packed!)
    constexpr size_t smem_B_size = TileN * TileK * Stages;
    
    // Scale factors (approximate)
    constexpr size_t smem_SFA_size = 512 * Stages;
    constexpr size_t smem_SFB_size = 1024 * Stages;
    
    // TensorMapStorage: 4 TMA descriptors
    constexpr size_t tensormap_size = 4 * sizeof(cute::TmaDescriptor);
    
    // Pipeline storage
    constexpr size_t pipeline_size = sizeof(cutlass::PipelineTmaAsync<Stages>::SharedStorage);
    
    printf("Estimated sizes for (%d, %d, %d) with %d stages:\n", TileM, TileN, TileK, Stages);
    printf("  smem_A (alignas 1024): %zu bytes\n", smem_A_size);
    printf("  smem_B (alignas 1024): %zu bytes\n", smem_B_size);
    printf("  smem_SFA: %zu bytes\n", smem_SFA_size);
    printf("  smem_SFB: %zu bytes\n", smem_SFB_size);
    printf("  TensorMapStorage: %zu bytes\n", tensormap_size);
    printf("  PipelineStorage: %zu bytes\n", pipeline_size);
    printf("\n");
    
    // With alignment
    auto align_up = [](size_t val, size_t align) {
        return ((val + align - 1) / align) * align;
    };
    
    size_t offset = 0;
    offset = align_up(offset, 1024);  // alignas(1024) smem_A
    offset += smem_A_size;
    offset = align_up(offset, 1024);  // alignas(1024) smem_B
    offset += smem_B_size;
    offset += smem_SFA_size;
    offset += smem_SFB_size;
    
    size_t tensor_storage = align_up(offset, 128);  // aligned_struct<128>
    size_t tensormap_storage = align_up(tensormap_size, 128);
    size_t pipeline_storage = align_up(pipeline_size, 16);
    
    size_t total = tensor_storage + tensormap_storage + pipeline_storage;
    
    printf("With alignment:\n");
    printf("  TensorStorage: %zu bytes\n", tensor_storage);
    printf("  TensorMapStorage: %zu bytes\n", tensormap_storage);
    printf("  PipelineStorage: %zu bytes\n", pipeline_storage);
    printf("  TOTAL: %zu bytes\n", total);
    printf("\n");
    
    if (total <= cutlass::arch::sm120_smem_capacity_bytes) {
        printf("Result: FITS in SM120 smem\n");
    } else {
        printf("Result: DOESN'T FIT - over by %zu bytes\n", 
               total - cutlass::arch::sm120_smem_capacity_bytes);
    }
    
    return 0;
}
