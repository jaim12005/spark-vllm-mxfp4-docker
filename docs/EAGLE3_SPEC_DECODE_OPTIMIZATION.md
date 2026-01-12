# Eagle3 Speculative Decoding Optimization Plan

## Summary

We're making vLLM's Eagle3 tree-based speculative decoding **"best-in-class"** for decode throughput on the quantized gpt-oss-120b model.

**The Problem**: Draft token selection uses raw `argmax`/`top-k` on logits, completely ignoring the sampling transforms (temperature, top-p, penalties, etc.) that the verifier applies. This causes unnecessary token rejections because drafts don't match what the verifier would actually sample.

**The Fix (Three Pillars)**:
1. **Align drafting with sampling** - Apply the same transforms to draft logits so we predict what the verifier will actually output
2. **Eliminate overhead** - Remove repeated allocations (`torch.cat` in loops) and unnecessary tensor copies (`.contiguous()` per step)
3. **Diagnose the quantization gap** - The verifier uses MXFP4 weights/MXFP8 activations, but Eagle3 was trained on BF16 hidden states. We need telemetry to measure this mismatch and mitigations to close it.

**Target**: ≥52 tok/s decode (matching SGLang/llama.cpp) vs current 29 tok/s.

---

## Current State Analysis

### Key Files (in vLLM repo)

| Component | File | Issue |
|-----------|------|-------|
| Draft Proposer | `vllm/v1/spec_decode/eagle.py` | `propose_tree()` uses raw argmax/top-k |
| Tree Attention | `vllm/v1/attention/backends/tree_attn.py` | `.contiguous()` called per draft step |
| Rejection Sampler | `vllm/v1/sample/rejection_sampler.py` | Handles verification with full transforms |
| GPU Speculator | `vllm/v1/worker/gpu/spec_decode/eagle.py` | Uses Gumbel sampling with temperature only |
| Metrics | `vllm/v1/spec_decode/metrics.py` | Basic acceptance rate tracking |

### Critical Issues Identified

1. **Draft selection ignores sampling transforms**: `propose_tree()` uses `logits.argmax()` and `torch.topk()` on raw logits, ignoring temperature, top-p, top-k, min_p, penalties, and logit bias
2. **Logits processors explicitly disabled for spec decode**: `build_logitsprocs()` returns empty `LogitsProcessors()` when speculative decoding is enabled
3. **Per-step allocations in `propose_tree`**: `torch.cat()` in loop (lines 765-768), `repeat_interleave()` per level
4. **Tree attention bias `.contiguous()` per step**: Line 226 creates new tensor each drafting step
5. **No mismatch diagnostics**: No telemetry for *why* tokens are rejected

---

## Prioritized Backlog

### P0: Critical (Week 1-2) - Direct TPS/Acceptance Impact

| ID | Task | Complexity | Expected Impact |
|----|------|------------|-----------------|
| P0.1 | Sampling-aligned draft selection | Medium | +15-30% acceptance |
| P0.2 | Branch-aware penalty application | Medium | +5-10% acceptance |
| P0.3 | Eliminate `torch.cat` in `propose_tree` | Low | -10-20% overhead |
| P0.4 | Cache tree attention bias slices | Low | -5-10% overhead |

### P1: Important (Week 3-4) - Stability and Observability

| ID | Task | Complexity | Expected Impact |
|----|------|------------|-----------------|
| P1.1 | Acceptance debugging telemetry | Medium | Diagnostic enabler |
| P1.2 | RNG isolation for draft sampling | Low | Correctness guarantee |
| P1.3 | Exact mode for baseline preservation | Medium | Quality guarantee |
| P1.4 | Sensible tree structure defaults | Low | +5% acceptance |

### P2: Enhancement (Week 5-6) - Quantization Mismatch

| ID | Task | Complexity | Expected Impact |
|----|------|------------|-----------------|
| P2.1 | Quantization impact measurement | Medium | Baseline understanding |
| P2.2 | No-retrain mitigations | High | +5-15% acceptance |
| P2.3 | Speculator adaptation training | Very High | +10-25% acceptance |

---

## Detailed Implementation

### P0.1: Sampling-Aligned Draft Selection

**Problem**: Draft tokens selected via `argmax(raw_logits)` mismatch verifier's post-processed distribution.

**Solution**: Apply same transforms as `Sampler.apply_logits_processors()` + `Sampler.sample()` before draft selection.

**Files to modify**:
- `vllm/v1/spec_decode/eagle.py`: `EagleProposer.propose_tree()`

**Algorithm**:

```python
def select_draft_tokens(logits, sampling_metadata, num_children):
    # 1. Apply penalties (if any)
    if not sampling_metadata.no_penalties:
        logits = apply_branch_aware_penalties(logits, ...)
    
    # 2. Apply logit bias
    if sampling_metadata.logit_bias:
        apply_logit_bias(logits, ...)
    
    # 3. Apply temperature scaling (for greedy: keep logits, for random: scale)
    if not sampling_metadata.all_greedy:
        logits = logits / temperature.unsqueeze(-1)
    
    # 4. Apply top-k/top-p mask
    if sampling_metadata.top_k or sampling_metadata.top_p:
        logits = apply_top_k_top_p_mask(logits, top_k, top_p)
    
    # 5. Select drafts
    if num_children == 1:
        return logits.argmax(dim=-1)  # Best single token
    else:
        return logits.topk(num_children, dim=-1).indices  # Best N tokens
```

**Key design decisions**:
- Single-child: argmax on post-processed logits (maximize match probability)
- Multi-child: top-N on post-processed logits (maximize coverage mass)
- Stochastic drafting as optional mode with separate RNG (see P1.2)

---

### P0.2: Branch-Aware Penalty Application

**Problem**: Penalties depend on prefix tokens; tree branches have divergent prefixes.

**Solution**: Incremental penalty state with per-branch delta updates.

**Files to modify**:
- `vllm/v1/spec_decode/eagle.py`: New `BranchPenaltyState` class
- `vllm/v1/sample/ops/penalties.py`: Export incremental API

**Data structure**:

```python
@dataclass
class BranchPenaltyState:
    # Token counts per request (shared base)
    base_token_counts: torch.Tensor  # [batch, vocab_size]
    # Per-branch delta: tokens added in this branch
    branch_deltas: dict[tuple[int, ...], torch.Tensor]  # path -> [batch, vocab]
    
    def apply_penalty_for_branch(self, logits, branch_path):
        counts = self.base_token_counts + self.branch_deltas.get(branch_path, 0)
        return apply_penalties_from_counts(logits, counts, ...)
```

---

### P0.3: Eliminate `torch.cat` in `propose_tree`

**Problem**: Lines 765-768 concatenate tensors in loop, causing repeated allocations.

**Solution**: Preallocate buffers sized for total drafts, fill via slicing.

**Changes to `vllm/v1/spec_decode/eagle.py`**:

```python
def propose_tree(self, ...):
    total_drafts = self.cu_drafts_per_level[-1]  # Total tree nodes
    
    # Preallocate once
    tree_input_ids = torch.empty(batch_size, total_drafts, ...)
    tree_positions = torch.empty(batch_size, total_drafts, ...)
    tree_hidden_states = torch.empty(batch_size, total_drafts, hidden_size, ...)
    
    write_offset = 0
    for level in range(tree_depth - 1):
        level_drafts = self.cu_drafts_per_level[level] - (
            self.cu_drafts_per_level[level-1] if level > 0 else 0
        )
        # Write directly into preallocated buffer
        tree_input_ids[:, write_offset:write_offset+level_drafts] = draft_token_ids
        tree_positions[:, write_offset:write_offset+level_drafts] = draft_positions
        tree_hidden_states[:, write_offset:write_offset+level_drafts] = draft_hidden_states
        write_offset += level_drafts
        ...
```

---

### P0.4: Cache Tree Attention Bias Slices

**Problem**: Line 226 calls `.contiguous()` every drafting step.

**Solution**: Precompute and cache all required slices during initialization.

**Changes to `vllm/v1/attention/backends/tree_attn.py`**:

```python
class TreeAttentionMetadataBuilder:
    def __init__(self, ...):
        ...
        # Precompute bias slices for all possible query lengths
        self.tree_attn_bias_slices = {}
        for query_len in range(1, len(tree_choices) + 2):
            start, end = 1, 1 + query_len
            self.tree_attn_bias_slices[query_len] = (
                self.tree_attn_bias[start:end, start:end].contiguous()
            )
    
    def build_for_drafting(self, common_attn_metadata, draft_index):
        if draft_index == 0:
            tree_attn_bias = torch.empty(0)
        else:
            tree_attn_bias = self.tree_attn_bias_slices[
                common_attn_metadata.max_query_len
            ]
        ...
```

---

### P1.1: Acceptance Debugging Telemetry

**Problem**: No visibility into *why* mismatches occur.

**Solution**: Env-var gated capture of mismatch diagnostics.

**New file**: `vllm/v1/spec_decode/telemetry.py`

```python
@dataclass
class MismatchDiagnostic:
    request_id: str
    position: int
    draft_token: int
    target_token: int
    draft_logit: float
    target_logit: float
    entropy: float
    top_p_mass: float  # Mass within top-p
    active_processors: list[str]
    penalty_applied: bool
    token_in_allowed_mask: bool

class AcceptanceTelemetry:
    enabled: bool = os.environ.get("VLLM_SPEC_DECODE_TELEMETRY", "0") == "1"
    
    def capture_mismatch(self, ...):
        if not self.enabled:
            return
        # Capture diagnostic, write to buffer
        ...
    
    def get_summary(self) -> dict:
        # Return aggregated stats:
        # - Mean mismatch position
        # - "Draft outside allowed mask" count
        # - "Penalty mismatch" count
        # - Entropy distribution at mismatch points
```

**Integration points**:
- `vllm/v1/sample/rejection_sampler.py`: Inject capture after rejection decision
- `vllm/v1/spec_decode/metrics.py`: Add new Prometheus counters

---

### P1.2: RNG Isolation for Draft Sampling

**Problem**: Draft sampling must not consume/perturb verifier RNG stream.

**Solution**: Separate stateless RNG keyed by (request_seed, absolute_position, branch_id).

**Changes to `vllm/v1/spec_decode/eagle.py`**:

```python
def get_draft_rng(request_seed: int, position: int, branch_id: int) -> torch.Generator:
    """Deterministic RNG that doesn't affect verifier stream."""
    combined_seed = hash((request_seed, position, branch_id)) & ((1 << 63) - 1)
    gen = torch.Generator(device='cuda')
    gen.manual_seed(combined_seed)
    return gen
```

---

### P1.3: Exact Mode for Baseline Preservation

**Problem**: Need to verify spec decode doesn't change output distribution.

**Solution**: Add `--spec-decode-exact-mode` that forces greedy drafting and logs any distribution deviation.

**New config flag** in `vllm/config/speculative.py`:

```python
exact_mode: bool = False  # When True, use argmax drafting only, validate outputs match
```

---

### P1.4: Sensible Tree Structure Defaults

**Problem**: Chain (depth=N, width=1) underperforms for long context.

**Solution**: Adaptive tree structure based on model characteristics.

**Default tree template**:

```python
# Wide near root, narrower later
DEFAULT_TREE = [
    (0,), (1,), (2,), (3,),          # 4 children at root
    (0,0), (0,1), (1,0), (1,1),      # 2 children each
    (0,0,0), (0,1,0), (1,0,0), (1,1,0),  # 1 child each
]
```

Optional: Adaptive width based on nucleus size (if entropy is high, use more children).

---

### P2.1-P2.3: Quantization Mismatch Plan

#### P2.1: Measurement Protocol

Create benchmark comparing acceptance rates:
- **Baseline**: BF16 verifier + Eagle3 speculator
- **Quantized**: MXFP4 verifier + same Eagle3 speculator

**Metrics to capture**:
- Per-position acceptance rate
- KL divergence of hidden states
- Logit correlation

#### P2.2: No-Retrain Mitigations

1. **Normalization alignment**: Add LayerNorm before speculator input
2. **Rescaling**: Learn scalar multipliers for hidden state channels
3. **Higher precision for speculator inputs**: Keep specific verifier layer outputs at FP8/BF16

```python
# In Eagle3LlamaForCausalLM.combine_hidden_states()
if self.quant_alignment_layer is not None:
    hidden_states = self.quant_alignment_layer(hidden_states)
```

#### P2.3: Speculator Adaptation Training (If Needed)

**Recipe**:
1. Collect hidden states from quantized verifier on calibration set
2. Fine-tune Eagle3 head with distillation loss:
   - Teacher: BF16 Eagle3 logits
   - Student input: Quantized verifier hidden states
3. Training data: 1-10B tokens from target domain
4. Expected training time: 2-8 GPU-hours

---

## Benchmarking Protocol

### Workloads

| Workload | Prompt Len | Output Len | Concurrency | Sampling |
|----------|-----------|------------|-------------|----------|
| Decode-heavy | 2048 | 128 | 1, 4, 8 | greedy |
| Mixed | 2048 | 128 | 1, 4, 8 | temp=0.7, top_p=0.9 |
| Penalty | 2048 | 128 | 1 | rep_pen=1.1, freq_pen=0.5 |
| Long context | 16384 | 256 | 1 | greedy |

### KPIs

| Metric | Target | Measurement |
|--------|--------|-------------|
| tg TPS (greedy) | >=52 tok/s | `llama-benchy --output-lengths 128` |
| Mean acceptance length | >=4.0 | Prometheus metrics |
| Verifier forwards/token | <=1.25 | `1 / mean_acceptance_length` |
| Draft overhead | <=15% of forward | Profiler breakdown |
| Memory bandwidth util | >=80% | nsight |
| Kernel launch count | Minimize | nsight |

### Overhead Breakdown

Profile these phases:
1. **Draft generation**: `propose_tree()` wall time
2. **Metadata build**: `build_for_drafting()` wall time
3. **Tree attention**: Forward pass with tree mask
4. **Accept kernel**: `rejection_sample()` wall time
5. **Post-processing**: Token gathering, output assembly

---

## Testing Requirements

1. **Unit tests for sampling alignment**:
   - Verify draft selection matches verifier for all transform combinations
   - Test branch-aware penalties produce correct token counts

2. **Integration tests**:
   - `test_level2_correctness.sh`: Output matches non-spec-decode baseline
   - `test_level3_stress.sh`: No memory growth under sustained load

3. **Regression tests**:
   - TPS must not regress from current baseline
   - Acceptance rate must improve with sampling alignment

---

## Risk Mitigation

| Risk | Mitigation |
|------|------------|
| Sampling alignment overhead cancels acceptance gains | Profile early; use fused Triton kernels |
| Tree bias caching OOMs for large trees | Lazy allocation, max tree size limit |
| Quantization mismatch unfixable without retraining | Prioritize P2.2 mitigations first |
| RNG isolation changes output distribution | Extensive seed-based reproducibility tests |

---

## Implementation Status

- [ ] P0.1: Sampling-aligned draft selection in propose_tree()
- [ ] P0.2: Branch-aware penalty application for tree mode
- [ ] P0.3: Eliminate torch.cat in propose_tree with preallocated buffers
- [ ] P0.4: Cache tree attention bias slices in TreeAttentionMetadataBuilder
- [ ] P1.1: Acceptance debugging telemetry with env-var gating
- [ ] P1.2: Isolated RNG for stochastic draft sampling
- [ ] P1.3: Exact mode flag for baseline preservation testing
- [ ] P1.4: Sensible tree structure defaults (wide near root)
- [ ] P2.1: Benchmark for quantization impact measurement
- [ ] P2.2: No-retrain mitigations (norm alignment, rescaling)
- [ ] P2.3: Speculator adaptation training recipe
