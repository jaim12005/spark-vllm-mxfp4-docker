#include <cstdio>
#include <cstdlib>
#include <cstdint>
#include <vector>

#include <cuda_runtime.h>

// Pull in the actual FlashInfer launcher implementation (includes CUTLASS + internal headers).
// NOTE: This must be compiled inside the vllm-dev container where /workspace/flashinfer exists.
#include "csrc/nv_internal/tensorrt_llm/kernels/cutlass_kernels/moe_gemm/launchers/moe_gemm_sm120_mixed_input_launcher.inl"

using namespace cute;

static inline void ck(cudaError_t st, char const* what) {
  if (st != cudaSuccess) {
    std::fprintf(stderr, "CUDA ERROR: %s: %s\n", what, cudaGetErrorString(st));
    std::exit(1);
  }
}

template <typename T>
static inline T* cuda_malloc_elems(size_t elems) {
  void* p = nullptr;
  ck(cudaMalloc(&p, elems * sizeof(T)), "cudaMalloc");
  return reinterpret_cast<T*>(p);
}

static inline void* cuda_malloc_bytes(size_t bytes) {
  void* p = nullptr;
  ck(cudaMalloc(&p, bytes), "cudaMalloc");
  return p;
}

int main(int argc, char** argv) {
  int num_groups = 1;
  int64_t M = 256;
  int64_t N = 5888;
  int64_t K = 2944;
  if (argc >= 2) num_groups = std::atoi(argv[1]);
  if (argc >= 5) {
    M = std::atoll(argv[2]);
    N = std::atoll(argv[3]);
    K = std::atoll(argv[4]);
  }

  using namespace tensorrt_llm::kernels;
  using namespace tensorrt_llm::kernels::cutlass_kernels_oss;

  using HopperInput = cutlass_kernels::TmaWarpSpecializedGroupedGemmInput;
  using ProblemShape = HopperInput::ProblemShape;
  using UnderlyingProblemShape = typename ProblemShape::UnderlyingProblemShape;

  // Match the simple shapes used in our earlier minimal CUTLASS repros.
  using CTAShape = Shape<_128, _128, _128>;
  using ClusterShape = Shape<_1, _1, _1>;
  using EpilogueTag = void;

  // These template args match the MXFP4 path:
  // - T must be FP8 here because the launcher instantiation configures TMA descriptors
  //   based on T (and will static-assert for unsupported TMA formats like BF16).
  using T = __nv_fp8_e4m3;
  using WeightType = __nv_fp4_e2m1;
  using OutType = cutlass::bfloat16_t;

  // Allocate device memory for one group (we point all groups to the same buffers for simplicity).
  auto* A = cuda_malloc_elems<T>(static_cast<size_t>(M * K));
  void* B = cuda_malloc_bytes((static_cast<size_t>(N * K) / 2));
  auto* D = cuda_malloc_elems<OutType>(static_cast<size_t>(M * N));

  // Identity scale buffers (filled with 0x7F).
  size_t sfa_bytes = computeSm120IdentitySFABufferSize(M, N, K, /*L=*/1);
  size_t sfb_bytes = computeSm120IdentitySFBBufferSize(M, N, K, /*L=*/1);
  uint8_t* sfa = acquireSm120IdentitySFABuffer(sfa_bytes);
  uint8_t* sfb = acquireSm120IdentitySFBBuffer(sfb_bytes);

  // Device-side pointer arrays for A/B/D, strides, layouts, scales.
  auto* d_problem_shapes = cuda_malloc_elems<UnderlyingProblemShape>(num_groups);
  auto* d_A_ptr = cuda_malloc_elems<void const*>(num_groups);
  auto* d_B_ptr = cuda_malloc_elems<void const*>(num_groups);
  auto* d_D_ptr = cuda_malloc_elems<void*>(num_groups);

  // Strides: TmaWarpSpecializedGroupedGemmInput uses non-pointer stride values, but the launcher
  // expects grouped-GEMM pointer-to-stride arrays. We follow the minimal repro convention and pass
  // device arrays of packed stride objects.
  using StrideAVal = HopperInput::StrideA;
  using StrideBVal = HopperInput::StrideB;
  using StrideDVal = HopperInput::StrideD;
  auto* d_stride_A = cuda_malloc_elems<StrideAVal>(num_groups);
  auto* d_stride_B = cuda_malloc_elems<StrideBVal>(num_groups);
  auto* d_stride_D = cuda_malloc_elems<StrideDVal>(num_groups);

  // Layout objects for scale factors (device array).
  // We don’t know the concrete LayoutSFA/LayoutSFB types here, so we reuse the launcher helpers
  // that already compute correct layouts via Sm1xxBlockScaledConfig<32>.
  using SfConfig = HopperInput::MXFPXBlockScaledConfig;  // vec=32
  auto layout_sfa = SfConfig::tile_atom_to_shape_SFA(cute::make_shape((int)M, (int)N, (int)K, 1));
  auto layout_sfb = SfConfig::tile_atom_to_shape_SFB(cute::make_shape((int)M, (int)N, (int)K, 1));
  using LayoutSFAVal = decltype(layout_sfa);
  using LayoutSFBVal = decltype(layout_sfb);
  auto* d_layout_sfa = cuda_malloc_elems<LayoutSFAVal>(num_groups);
  auto* d_layout_sfb = cuda_malloc_elems<LayoutSFBVal>(num_groups);

  // Device-side pointer arrays for scale buffers (all groups point to same identity buffer).
  auto* d_sfa_ptrs = cuda_malloc_elems<uint8_t const*>(num_groups);
  auto* d_sfb_ptrs = cuda_malloc_elems<uint8_t const*>(num_groups);

  std::vector<UnderlyingProblemShape> h_shapes(num_groups);
  std::vector<void const*> h_A_ptr(num_groups);
  std::vector<void const*> h_B_ptr(num_groups);
  std::vector<void*> h_D_ptr(num_groups);
  std::vector<StrideAVal> h_stride_A(num_groups);
  std::vector<StrideBVal> h_stride_B(num_groups);
  std::vector<StrideDVal> h_stride_D(num_groups);
  std::vector<LayoutSFAVal> h_layout_sfa(num_groups);
  std::vector<LayoutSFBVal> h_layout_sfb(num_groups);
  std::vector<uint8_t const*> h_sfa_ptrs(num_groups);
  std::vector<uint8_t const*> h_sfb_ptrs(num_groups);

  for (int i = 0; i < num_groups; ++i) {
    h_shapes[i] = UnderlyingProblemShape(M, N, K);
    h_A_ptr[i] = A;
    h_B_ptr[i] = B;
    h_D_ptr[i] = D;
    h_stride_A[i] = cutlass::make_cute_packed_stride(StrideAVal{}, cute::Shape<int, int, int>{(int)M, (int)K, 1});
    h_stride_B[i] = cutlass::make_cute_packed_stride(StrideBVal{}, cute::Shape<int, int, int>{(int)N, (int)K, 1});
    h_stride_D[i] = cutlass::make_cute_packed_stride(StrideDVal{}, cute::Shape<int, int, int>{(int)M, (int)N, 1});
    h_layout_sfa[i] = layout_sfa;
    h_layout_sfb[i] = layout_sfb;
    h_sfa_ptrs[i] = sfa;
    h_sfb_ptrs[i] = sfb;
  }

  ck(cudaMemcpy(d_problem_shapes, h_shapes.data(), sizeof(h_shapes[0]) * h_shapes.size(), cudaMemcpyHostToDevice),
     "H2D problem_shapes");
  ck(cudaMemcpy(d_A_ptr, h_A_ptr.data(), sizeof(h_A_ptr[0]) * h_A_ptr.size(), cudaMemcpyHostToDevice),
     "H2D A_ptr");
  ck(cudaMemcpy(d_B_ptr, h_B_ptr.data(), sizeof(h_B_ptr[0]) * h_B_ptr.size(), cudaMemcpyHostToDevice),
     "H2D B_ptr");
  ck(cudaMemcpy(d_D_ptr, h_D_ptr.data(), sizeof(h_D_ptr[0]) * h_D_ptr.size(), cudaMemcpyHostToDevice),
     "H2D D_ptr");
  ck(cudaMemcpy(d_stride_A, h_stride_A.data(), sizeof(h_stride_A[0]) * h_stride_A.size(), cudaMemcpyHostToDevice),
     "H2D stride_A");
  ck(cudaMemcpy(d_stride_B, h_stride_B.data(), sizeof(h_stride_B[0]) * h_stride_B.size(), cudaMemcpyHostToDevice),
     "H2D stride_B");
  ck(cudaMemcpy(d_stride_D, h_stride_D.data(), sizeof(h_stride_D[0]) * h_stride_D.size(), cudaMemcpyHostToDevice),
     "H2D stride_D");
  ck(cudaMemcpy(d_layout_sfa, h_layout_sfa.data(), sizeof(h_layout_sfa[0]) * h_layout_sfa.size(), cudaMemcpyHostToDevice),
     "H2D layout_sfa");
  ck(cudaMemcpy(d_layout_sfb, h_layout_sfb.data(), sizeof(h_layout_sfb[0]) * h_layout_sfb.size(), cudaMemcpyHostToDevice),
     "H2D layout_sfb");
  ck(cudaMemcpy(d_sfa_ptrs, h_sfa_ptrs.data(), sizeof(h_sfa_ptrs[0]) * h_sfa_ptrs.size(), cudaMemcpyHostToDevice),
     "H2D sfa_ptrs");
  ck(cudaMemcpy(d_sfb_ptrs, h_sfb_ptrs.data(), sizeof(h_sfb_ptrs[0]) * h_sfb_ptrs.size(), cudaMemcpyHostToDevice),
     "H2D sfb_ptrs");

  // Workspace: allocate a generously aligned buffer.
  size_t ws_bytes = cutlass_kernels::TmaWarpSpecializedGroupedGemmInput::workspaceSize(
      num_groups, HopperInput::FpXBlockScalingType::MXFPX);
  void* ws = cuda_malloc_bytes(ws_bytes + 256);
  auto ws_aligned = reinterpret_cast<uint8_t*>((reinterpret_cast<uintptr_t>(ws) + 255) & ~uintptr_t(255));

  HopperInput in{};
  in.shape_info = ProblemShape{num_groups, d_problem_shapes, nullptr};
  in.ptr_act = reinterpret_cast<void const**>(d_A_ptr);
  in.ptr_weight = reinterpret_cast<void const**>(d_B_ptr);
  in.ptr_d = reinterpret_cast<void**>(d_D_ptr);
  in.stride_act = d_stride_A;
  in.stride_weight = d_stride_B;
  in.stride_d = d_stride_D;
  in.fpX_block_scaling_type = HopperInput::FpXBlockScalingType::MXFPX;
  in.fpX_block_scaling_factors_act = reinterpret_cast<uint8_t const**>(d_sfa_ptrs);
  in.fpX_block_scaling_factors_weight = reinterpret_cast<uint8_t const**>(d_sfb_ptrs);
  in.fpX_block_scaling_factors_stride_act = d_layout_sfa;
  in.fpX_block_scaling_factors_stride_weight = d_layout_sfb;
  in.gemm_workspace = ws_aligned;
  in.gemm_workspace_size = ws_bytes;

  int device_id = 0;
  ck(cudaGetDevice(&device_id), "cudaGetDevice");
  int sm_count = 0;
  ck(cudaDeviceGetAttribute(&sm_count, cudaDevAttrMultiProcessorCount, device_id),
     "cudaDevAttrMultiProcessorCount");

  std::printf("[launcher_repro] groups=%d M=%lld N=%lld K=%lld ws_bytes=%zu sm_count=%d\n",
              num_groups, (long long)M, (long long)N, (long long)K, ws_bytes, sm_count);

  // This should reach gemm.initialize() inside the launcher.
  sm120_mixed_input_moe_gemm_kernelLauncher<T, WeightType, OutType, EpilogueTag, CTAShape, ClusterShape, true>(
      in, num_groups, sm_count, /*stream=*/0, /*kernel_occupancy=*/nullptr, /*workspace_size=*/nullptr);

  ck(cudaDeviceSynchronize(), "cudaDeviceSynchronize");
  auto last = cudaGetLastError();
  std::printf("[launcher_repro] last_error=%s\n", cudaGetErrorString(last));

  return 0;
}

