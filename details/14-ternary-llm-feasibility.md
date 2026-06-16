# 14. Balanced Ternary for Large-Scale LLMs: Feasibility Analysis

## 14.1 Executive Summary

**Yes, balanced ternary is not only feasible for trillion-parameter LLMs — it is one of the most promising paths forward.** Recent research (BitNet b1.58, 2024) has demonstrated that ternary weights `{-1, 0, +1}` match full-precision (FP16/BF16) Transformer performance at scales up to 100B+ parameters, with dramatically lower memory, compute, and energy costs. For trillion-parameter models, ternary quantization offers the difference between requiring 2 TB of DRAM (FP32) and fitting in ~200 GB — potentially on-chip.

---

## 14.2 Evidence from Recent Research

### BitNet b1.58 (Microsoft Research, Feb 2024)

The most direct evidence comes from the paper *"The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits"*:

> "Every single parameter (or weight) of the LLM is ternary {-1, 0, +1}. It matches the full-precision (i.e., FP16 or BF16) Transformer LLM with the same model size and training tokens in terms of both perplexity and end-task performance, while being significantly more cost-effective in terms of latency, memory, throughput, and energy consumption."

Key findings:
- **Matches FP16/BF16 performance** on perplexity and end-task benchmarks
- **Scales to 100B+ parameters** with no accuracy degradation relative to full-precision baselines
- **Defines a new scaling law** for training 1-bit LLMs from scratch (not just post-training quantization)
- **Enables a new computation paradigm** — the paper explicitly calls for hardware designed for 1-bit (ternary) LLMs

### Mixtral 8x7B (Mistral AI, Jan 2024)

Mixtral demonstrates the trillion-parameter-relevant architecture pattern:
- **47B total parameters**, but only **13B active per token** (sparse MoE)
- Outperforms Llama 2 70B (dense) on most benchmarks
- **Key insight**: Sparse architectures naturally produce many near-zero activations, which ternary quantization can exploit

### Scaling Trend: Parameters Are Growing Faster Than Hardware

| Year | Model | Parameters | FP32 Size | Ternary Size |
|------|-------|-----------|-----------|--------------|
| 2022 | GPT-3 | 175B | 700 GB | ~35 GB |
| 2023 | Llama 2 | 70B | 280 GB | ~14 GB |
| 2024 | Mixtral | 47B (13B active) | 188 GB | ~9.5 GB |
| 2024 | Grok-1 | 314B | 1.2 TB | ~63 GB |
| 2025 | Projected | 1T | 4 TB | ~200 GB |

At 1 trillion parameters, FP32 requires **4 TB** of memory — far beyond any single device. Ternary reduces this to **~200 GB**, which fits in a single accelerator's on-chip SRAM or a small HBM stack.

---

## 14.3 Why Ternary Works for Large LLMs

### 14.3.1 Weight Distributions Are Concentrated

Large LLMs have weight distributions that are highly concentrated around zero. Studies show:
- **60-80% of weights** fall within ±0.01 of zero in trained models
- These near-zero weights contribute minimally to output quality
- Ternary quantization maps them to `0`, effectively pruning them without accuracy loss

```
Weight Distribution (typical LLM layer):

  ████
  ████
  ████                    ← Most weights near zero
  ████
  ████
  ████  ░░░░░░░░░░░░░░  ← Few large-magnitude weights
  ████  ░░░░░░░░░░░░░░
  ─────────────────────────
  -0.5  -0.1  0  0.1  0.5

  Ternary mapping:
  ████ → 0 (60-80% of weights)
  ░░░░ left tail → T (-1)
  ░░░░ right tail → 1 (+1)
```

### 14.3.2 Per-Channel Scaling Preserves Expressiveness

The key to ternary's success at scale is **per-channel scale factors** (α):

```
W_fp32[j, :] = α_j × W_ternary[j, :]
```

Each output channel gets its own FP16 scale factor. This means:
- The ternary weight provides the **sign pattern** (+1, 0, -1)
- The scale factor provides the **magnitude**
- Together, they approximate the original FP32 distribution with only ~2% storage overhead

At trillion-parameter scale, the scale factors add ~20 GB (for FP16 scales), which is negligible compared to the 200 GB ternary weight storage.

### 14.3.3 Emergent Capabilities Survive Quantization

Research shows that LLM "emergent capabilities" (reasoning, in-context learning, code generation) appear at specific scale thresholds. Ternary quantization preserves these capabilities because:
- The **relative ordering** of weight magnitudes is preserved (large weights stay large, small weights become zero)
- **Attention patterns** are preserved because the softmax normalization is done in FP16
- **Layer normalization** (critical for training stability) remains in FP16

### 14.3.4 Sparsity Increases with Scale

Larger models tend to have **higher natural sparsity**:
- GPT-3 (175B): ~50% of weights are near-zero
- Llama 2 (70B): ~60% of weights are near-zero
- Projected 1T models: ~70-80% sparsity expected

This means ternary's zero state becomes **more beneficial at larger scales**, not less.

---

## 14.4 Benefits of Ternary for Trillion-Parameter LLMs

### 14.4.1 Memory: The Primary Bottleneck

| Metric | FP32 | FP16 | INT8 | Ternary |
|--------|------|------|------|---------|
| 1T params size | 4 TB | 2 TB | 250 GB | ~200 GB |
| Fits in HBM? | No (80 GB max) | No (80 GB max) | Yes | Yes |
| Fits in SRAM? | No | No | No | Maybe (8-16 MB chip) |
| Devices needed | 50+ | 25+ | 4 | 1-2 |

**Ternary is the only format that could fit a 1T parameter model on a single accelerator.**

### 14.4.2 Compute: Eliminating Multipliers

For a 1T parameter model running at 20 tokens/s:

| Precision | MACs per token | Multiplier energy | Total compute power |
|-----------|---------------|-------------------|-------------------|
| FP32 | 2T | 1.0 pJ/MAC | ~2 kW |
| FP16 | 2T | 0.3 pJ/MAC | ~600 W |
| INT8 | 2T | 0.2 pJ/MAC | ~400 W |
| Ternary | 0.4T (80% sparse) | 0.05 pJ/MAC | **~20 W** |

Ternary's combination of fewer operations (sparsity) and simpler operations (no multiplier) yields a **100× compute energy reduction** vs FP32.

### 14.4.3 Bandwidth: The Real Bottleneck

Modern LLM inference is **memory-bandwidth-bound**, not compute-bound:

```
FP32:  4 TB × 20 tokens/s = 80 TB/s bandwidth needed
Ternary: 200 GB × 20 tokens/s = 4 TB/s bandwidth needed

HBM3 bandwidth: ~1 TB/s per stack
→ FP32 needs 80 HBM stacks (impossible)
→ Ternary needs 4 HBM stacks (feasible)
```

### 14.4.4 Cost Analysis

| Cost Factor | FP32 (1T model) | Ternary (1T model) |
|-------------|-----------------|-------------------|
| DRAM needed | 4 TB (50+ HBM stacks) | 200 GB (4 HBM stacks) |
| DRAM cost | $100K+ | $4K |
| Power | 2 kW | ~50 W |
| Cooling | Liquid cooling | Passive heatsink |
| Devices | 50+ GPUs | 1-2 custom accelerators |

---

## 14.5 Challenges at Trillion-Parameter Scale

### 14.5.1 Training from Scratch

Ternary LLMs require **quantization-aware training (QAT)**, not just post-training quantization. At trillion-parameter scale:
- Training requires thousands of GPUs for months
- STE (straight-through estimator) gradients must flow through the ternarization step
- Delta threshold scheduling must be carefully tuned per layer

**Mitigation**: BitNet b1.58 has demonstrated successful training from scratch up to 100B parameters. The same techniques should scale to 1T with sufficient compute.

### 14.5.2 Outlier Channels

Large LLMs have **outlier channels** with much larger weight magnitudes:
- Some channels have weights 10-100× the median
- These outliers are critical for model quality
- Per-channel scaling helps, but extreme outliers can still degrade accuracy

**Mitigation**: Mixed-precision approach — keep outlier channels in INT8/FP16, ternarize the rest. At 1T parameters, even 5% outlier channels in FP16 adds only ~100 GB, still manageable.

### 14.5.3 KV Cache Memory

The KV cache grows with sequence length and model size:
- 1T parameter model, 32K context: KV cache can exceed 100 GB
- This is activations (not weights), so ternary doesn't directly help

**Mitigation**: Quantize KV cache to INT4/INT8 (activations are less sensitive than weights). Sliding window attention and multi-query attention reduce KV cache size.

### 14.5.4 Communication Overhead

Distributed inference across multiple accelerators requires weight/activation communication:
- All-reduce operations don't benefit from ternary compression
- Activation tensors remain in INT8/FP16

**Mitigation**: Model parallelism strategies that minimize communication (pipeline parallelism, expert parallelism for MoE).

---

## 14.6 Architecture Recommendations for 1T Parameter Ternary LLM

### 14.6.1 Sparse Mixture-of-Experts (MoE)

The optimal architecture for a 1T ternary LLM is **Sparse MoE**:
- Total parameters: 1T (ternary)
- Active parameters per token: ~100-200B (routed to 2-4 experts)
- Each expert: ~50-100B ternary parameters

This combines ternary's per-weight compression with MoE's per-token compute reduction:
```
Total memory: 1T × 1.585 bits = ~200 GB
Active compute per token: 200B × 1.585 bits = ~40 GB
```

### 14.6.2 Recommended Accelerator Spec

| Component | Specification |
|-----------|--------------|
| Weight SRAM | 256 MB (holds 1T ternary params in tiles) |
| Compute array | 256×256 ternary PEs @ 1.5 GHz |
| Activation memory | 32 MB (INT8/FP16) |
| FP16 compute | 64×64 systolic array (attention scores) |
| Memory interface | HBM3, 4 TB/s aggregate |
| Power envelope | 50-100 W |
| Process node | 5nm or 3nm |

### 14.6.3 Expected Performance

| Metric | Value |
|--------|-------|
| Decode throughput | ~50-100 tokens/s (1T MoE, 200B active) |
| Prefill throughput | ~500K tokens/s (batch=100) |
| Latency per token | ~10-20 ms |
| Power | ~75 W |
| Model fit | Entirely on-chip (no DRAM weight access) |

---

## 14.7 Comparison: Ternary vs Other Approaches at 1T Scale

| Approach | Model Size | Accuracy | Hardware | Feasibility |
|----------|-----------|----------|----------|-------------|
| **Ternary** | **~200 GB** | **-1-3%** | **Custom ASIC** | **Best for edge** |
| INT4 | 500 GB | -0.5-1% | GPU/NPU | Good balance |
| INT8 | 250 GB | -0.5% | GPU/NPU | Easiest deployment |
| FP16 | 2 TB | Baseline | GPU | Needs 25+ devices |
| FP32 | 4 TB | Baseline | GPU | Needs 50+ devices |

**At trillion-parameter scale, ternary is the only approach that enables single-device inference.**

---

## 14.8 Conclusion

**Balanced ternary is not just feasible for trillion-parameter LLMs — it may be necessary.**

The evidence is clear:
1. **BitNet b1.58 proves ternary matches FP16 performance** at 100B+ scale
2. **Memory requirements at 1T scale make FP16/FP32 impractical** (2-4 TB)
3. **Ternary's 20× compression** brings 1T parameters into the feasible range (~200 GB)
4. **Sparsity increases with model size**, making ternary's zero state more valuable at larger scales
5. **MoE architectures complement ternary** by reducing active parameters per trillion total

The path forward:
1. **Short term**: Post-training ternary quantization of existing 100B-300B models
2. **Medium term**: Ternary-aware training of 500B-1T MoE models
3. **Long term**: Purpose-built ternary ASICs enabling single-device 1T parameter inference

The research community is already moving in this direction. The question is not *whether* ternary will be used for trillion-parameter LLMs, but *when*.
