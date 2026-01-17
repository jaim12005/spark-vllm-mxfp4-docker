# TensorRT-LLM Analysis

Analysis of TensorRT-LLM's strengths and techniques we can learn from to improve gpt-oss-120b decode performance on DGX Spark.

## Executive Summary

TensorRT-LLM has several sophisticated optimizations that could benefit our vLLM implementation:

1. **Runtime Autotuning** - Profiles GEMM tactics at startup, selects optimal kernels per workload
2. **Min-Latency Mode** - Specialized code path for single-token decode
3. **DeepGEMM** - Custom grouped GEMM implementation for block FP8
4. **Fused Finalize** - Combines scaling/reduction with GEMM epilogue
5. **Eagle3 Integration** - Speculative decoding with gpt-oss-120b specifically

---

## 1. Runtime Autotuning

**Location**: `tensorrt_llm/_torch/autotuner.py`

TRT-LLM implements a sophisticated autotuning system that profiles kernel tactics at model load time.

### Key Features

```python
@dataclass
class TuningConfig:
    dynamic_tensor_specs: Tuple[DynamicTensorSpec, ...]  # Which dims to tune
    tune_max_num_tokens: int  # Max tokens to consider
    use_cold_l2_cache: bool   # Simulate cold cache for realistic timings
    use_cuda_graph: bool      # Profile with CUDA graphs
```

### Tactic Selection Process

1. At model initialization, `finalize_tactic()` is called
2. AutoTuner profiles multiple GEMM implementations for both GEMM1 (w3_w1) and GEMM2 (w2)
3. Best tactics are stored and reused during inference

```python
# From moe_op_cutlass.py
_, gemm_tactic_1 = tuner.choose_one(
    "trtllm::fused_moe::gemm1",
    [self.moe_runner],
    MoERunner.tuning_config,
    [...],
    gemm_idx=1,
)
```

### What We Can Learn

- **FlashInfer has an autotuner** but we're not using it for tile selection
- TRT-LLM tunes per-GEMM (separate tactics for gate/up vs down projection)
- Cold L2 cache simulation gives more realistic decode latencies

**Action Item**: Wire up FlashInfer's autotuner to select optimal tiles at runtime rather than using static heuristics.

---

## 2. Min-Latency Mode

**Purpose**: Optimized path for single-token decode (batch size 1)

TRT-LLM explicitly separates decode-optimized paths:

```python
# Select the appropriate run method based on latency mode
run_moe = (self.moe_runner.fused_moe_runner.run_moe_min_latency 
           if min_latency_mode 
           else self.moe_runner.fused_moe_runner.run_moe)
```

### Characteristics

- Avoids overhead of batched operations
- Uses different memory access patterns optimized for small M
- Potentially different tile shapes

**Note**: DeepGEMM does not support min_latency_mode, suggesting it's already optimized for single-token.

### What We Can Learn

Our 64×128 tile selection is a step in this direction, but TRT-LLM suggests having a completely separate code path may be beneficial.

---

## 3. DeepGEMM

**Location**: `tensorrt_llm/_torch/modules/fused_moe/fused_moe_deepgemm.py`

DeepGEMM is TRT-LLM's custom grouped GEMM implementation for GB200 block FP8.

### Key Features

```python
def deepgemm_fp8_group_blockwise_gemm(
    d: torch.Tensor,  # Output [G, M, N]
    a: torch.Tensor,  # Input [G, M, K] - FP8
    b: torch.Tensor,  # Weight [G, N, K] - FP8
    sfa: torch.Tensor,  # Scale factors A
    sfb: torch.Tensor,  # Scale factors B
    masked_m: torch.Tensor,  # Tokens per expert
    expected_m: int,  # Expected M for all experts
)
```

### Block FP8 Quantization

- Uses group size 128 for block-wise quantization
- Scale factors in E8M0 format (exponent only, no mantissa)
- Triton kernel for fused quantization + index copy:

```python
# Quantization with per-block absmax
_absmax = tl.maximum(tl.max(tl.abs(input_data)), eps)
output_s = _absmax / fp8_max
output_s = tl.exp2(tl.ceil(tl.log2(tl.abs(output_s))))  # Round to power of 2
output_q = tl.clamp(input_data / output_s, -fp8_max, fp8_max)
```

### What We Can Learn

- **Fused quantization + permute** eliminates separate kernel launch
- **Power-of-2 scales** simplify hardware multiplication
- **Masked operations** handle variable expert counts efficiently

**Action Item**: Investigate fusing our `mxfp8_quantize` into the MoE permute operation.

---

## 4. Fused Finalize

TRT-LLM combines the final scaling and reduction step with the GEMM epilogue:

```python
use_fused_finalize: bool = True  # Default enabled
```

This eliminates a separate kernel for:
- Applying routing weights (token_final_scales)
- Reducing across experts (when top_k > 1)
- Type conversion to output dtype

### What We Can Learn

FlashInfer has `FINALIZE` fusion mode but our SM120 launcher doesn't use it:

> "Looking at the SM120 launcher, it builds a fixed epilogue using cutlass::epilogue::TmaWarpSpecialized. It doesn't seem to handle the FINALIZE fusion mode."

**Action Item**: Add FINALIZE fusion support to SM120 launcher.

---

## 5. Eagle3 Speculative Decoding

**Location**: `examples/eagle/`, perf configs

TRT-LLM has working Eagle3 integration for gpt-oss-120b:

```yaml
# From gpt_oss_120b_fp4_grace_blackwell.yaml
- name: "gpt_oss_fp4_tp4_eagle3_1k1k"
  model_name: "gpt_oss_120b_fp4"
  speculative_config:
    decoding_type: 'Eagle'
    eagle3_layers_to_capture: [-1]
    max_draft_len: 3
    speculative_model_dir: "gpt_oss/gpt-oss-120b-Eagle3"
```

### Key Parameters

- `max_draft_len: 3` - Draft 3 tokens at a time
- `eagle3_layers_to_capture: [-1]` - Use last layer features
- Tree-based verification with dynamic tree generation (Eagle-2)

### What We Can Learn

1. Eagle3 draft model exists for gpt-oss-120b at `gpt_oss/gpt-oss-120b-Eagle3`
2. Short draft lengths (3) work well - minimizes verification overhead
3. Can potentially 2-3x single-stream decode throughput

**Action Item**: Investigate vLLM's speculative decoding support and Eagle3 integration.

---

## 6. Wide Expert Parallelism

TRT-LLM supports extensive expert parallelism configurations:

```yaml
tensor_parallel_size: 4
moe_expert_parallel_size: 4  # EP separate from TP
enable_attention_dp: true    # Attention data parallelism
```

### AlltoAll Methods

```python
class AlltoallMethodType(IntEnum):
    NVLinkOneSided = 1
    NVLinkTwoSided = 2
    DeepEP = 3           # CUDA graph compatible
    DeepEPLowLatency = 4 # Lowest latency
```

### What We Can Learn

For single-GPU DGX Spark, not directly applicable. But for future multi-GPU:
- Separate EP from TP allows better load balancing
- DeepEP integration for efficient expert routing

---

## 7. FP4 Weight Quantization

TRT-LLM supports multiple FP4 modes:

```python
weight_view_dtype = module.w3_w1_weight.dtype
if getattr(module, 'has_w4afp8', False):
    weight_view_dtype = torch.quint4x2  # W4A8
elif module.has_w4a16_mxfp4:
    weight_view_dtype = torch.uint8     # MXFP4
```

### gpt-oss-120b Configurations

From perf configs, they test:
- `gpt_oss_120b_fp4` on B200/B300 (Blackwell)
- `tensor_parallel_size: 2` minimum for their tests
- `kv_cache_config.dtype: 'fp8'` - Same as our approach

---

## 8. Memory Buffer Management

TRT-LLM uses a centralized buffer pool for workspace allocation:

```python
class DeepGemmMoEOp(MoEOp):
    buffers = get_memory_buffers()
    
    def _get_deepgemm_workspace(self, module, m_max, group_size):
        workspace["workspace_0"] = self.buffers.get_buffer(
            [expert_size_per_partition, m_max, fp8_dim],
            dtype=torch.float8_e4m3fn,
            buffer_name='workspace_0',
            reserve_buffer=capture_graph  # Reserve during graph capture
        )
```

### Benefits

- Avoids allocation during inference
- Compatible with CUDA graph capture
- Reuses buffers across layers

---

## Summary: Priority Actions

| Priority | Action | Expected Impact |
|----------|--------|-----------------|
| **HIGH** | Wire up FlashInfer autotuner for tile selection | 5-10% decode improvement |
| **HIGH** | Fuse quantization into MoE permute | Reduce kernel launches |
| **MEDIUM** | Add FINALIZE fusion to SM120 launcher | 2-5% improvement |
| **MEDIUM** | Investigate Eagle3 speculative decoding | 2-3x single-stream |
| **LOW** | Implement proper per-block scale computation | Better numerical accuracy |

---

## References

- TensorRT-LLM source: `~/projects/TensorRT-LLM`
- gpt-oss-120b FP4 config: `tests/scripts/perf-sanity/gpt_oss_120b_fp4_grace_blackwell.yaml`
- DeepGEMM implementation: `tensorrt_llm/_torch/modules/fused_moe/fused_moe_deepgemm.py`
- Autotuner: `tensorrt_llm/_torch/autotuner.py`
- Eagle3 example: `examples/eagle/README.md`
