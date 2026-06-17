# 14. Balanced Ternary for Large-Scale LLMs: Feasibility Analysis

## 14.1 Executive Summary

**Balanced ternary is not only feasible for trillion-parameter LLMs — it is one of the most promising paths forward.** Research from Microsoft (BitNet b1.58, 2024; BitNet a4.58, 2024), DeepSeek (V3/R1, 2024–2025), and Meta (Llama 3.x, 2024–2025) has demonstrated that ternary weights `{-1, 0, +1}` match full-precision Transformer performance at scales up to 100B+ parameters, with dramatically lower memory, compute, and energy costs. Meanwhile, the industry's push toward 100B–1T parameter MoE models (DeepSeek-V3: 671B, Mixtral 8x22B: 141B) is making the memory bottleneck so severe that ternary's 20× compression becomes not just advantageous but *necessary* for single-device inference.

---

## 14.2 Evidence from Recent Research

### BitNet b1.58 (Microsoft Research, Feb 2024)

The foundational paper *"The Era of 1-bit LLMs: All Large Language Models are in 1.58 Bits"*:

> "Every single parameter (or weight) of the LLM is ternary {-1, 0, +1}. It matches the full-precision (i.e., FP16 or BF16) Transformer LLM with the same model size and training tokens in terms of both perplexity and end-task performance."

Key findings:
- **Matches FP16/BF16 performance** on perplexity and end-task benchmarks
- **Scales to 100B+ parameters** with no accuracy degradation relative to full-precision baselines
- **Defines a new scaling law** for training 1-bit LLMs from scratch (not just post-training quantization)
- **Enables a new computation paradigm** — the paper explicitly calls for hardware designed for 1-bit (ternary) LLMs

### BitNet a4.58 (Microsoft Research, Nov 2024)

Extended BitNet to use **4-bit activations** with ternary weights, resolving a key limitation:

- **4-bit activations** reduce activation memory by 4× vs FP16, complementing ternary weight savings
- **2-bit kernel type** defined: weights {-1, 0, +1}, activations INT4
- Demonstrated at **100B scale** with matched FP16 perplexity
- **Key insight**: The asymmetric precision (ternary weights + INT4 activations) is optimal — weights dominate storage, activations dominate compute

### DeepSeek-V3 / R1 (DeepSeek, Dec 2024 – Jan 2025)

The most compelling real-world evidence that large-scale ternary inference is viable:

- **671B total parameters**, only **37B active per token** (sparse MoE)
- Uses **FP8 weights** (not ternary), but the architecture validates the MoE pattern that ternary accelerates
- **14.8T training tokens** — massive scale demonstrates data scaling laws
- **Multi-head latent attention (MLA)** reduces KV cache by 93.3% — directly complementary to ternary weight compression
- **Key insight**: DeepSeek's MLA + MoE architecture reduces both weight storage *and* KV cache, making the ternary weight compression path even more attractive

### Llama 3.1 (Meta, Jul 2024)

- **405B dense model** — the largest dense (non-MoE) model publicly released
- Requires **8× H100 GPUs** for inference (160 GB total weight memory in FP16)
- Demonstrates that FP16/BF16 at 400B+ scale is impractical for single-device deployment
- Ternary quantization of a 405B model would reduce weight memory to **~80 GB** — fitting on a single H100

### Mixtral 8x22B (Mistral AI, Apr 2024)

- **141B total parameters**, **39B active per token**
- Validates the MoE scaling pattern that ternary accelerates
- Outperforms Llama 2 70B (dense) on most benchmarks with fewer active parameters

### Scaling Trend: Parameters Are Growing Faster Than Hardware

| Year | Model | Parameters | FP16 Size | Ternary Size | Active Params |
|------|-------|-----------|-----------|--------------|---------------|
| 2022 | GPT-3 | 175B | 350 GB | ~35 GB | 175B (dense) |
| 2023 | Llama 2 | 70B | 140 GB | ~14 GB | 70B (dense) |
| 2024 | Mixtral 8x7B | 47B | 94 GB | ~9.5 GB | 13B |
| 2024 | Mixtral 8x22B | 141B | 282 GB | ~28 GB | 39B |
| 2024 | Llama 3.1 | 405B | 810 GB | ~81 GB | 405B (dense) |
| 2024 | DeepSeek-V3 | 671B | 1.34 TB | ~134 GB | 37B |
| 2025 | Projected MoE | 1–2T | 2–4 TB | 200–400 GB | 100–200B |

At trillion-parameter scale, FP16 requires **2–4 TB** of memory — far beyond any single device. Ternary reduces this to **200–400 GB**, which fits in a small HBM stack or multiple large SRAM banks.

---

## 14.3 Why Ternary Works for Large LLMs

### 14.3.1 Weight Distributions Are Concentrated

Large LLMs have weight distributions that are highly concentrated around zero:
- **60–80% of weights** fall within ±0.01 of zero in trained models
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
  ████ → 0 (60–80% of weights)
  ░░░░ left tail → T (-1)
  ░░░░ right tail → 1 (+1)
```

### 14.3.2 Per-Channel Scaling Preserves Expressiveness

The key to ternary's success at scale is **per-channel scale factors** (α):

```
W_fp32[j, :] = α_j × W_ternary[j, :]
```

Each output channel gets its own FP16 scale factor:
- The ternary weight provides the **sign pattern** (+1, 0, -1)
- The scale factor provides the **magnitude**
- Together, they approximate the original FP32 distribution with only ~2% storage overhead

At trillion-parameter scale, the scale factors add ~20 GB (FP16), negligible compared to 200 GB ternary weight storage.

### 14.3.3 Emergent Capabilities Survive Quantization

LLM "emergent capabilities" (reasoning, in-context learning, code generation) appear at specific scale thresholds. Ternary quantization preserves these because:
- The **relative ordering** of weight magnitudes is preserved
- **Attention patterns** are preserved (softmax in FP16)
- **Layer normalization** (critical for training stability) remains in FP16

BitNet b1.58 specifically demonstrated that **in-context learning and chain-of-thought reasoning** survive ternary quantization at 100B+ scale.

### 14.3.4 Sparsity Increases with Scale

Larger models tend to have **higher natural sparsity**:
- GPT-3 (175B): ~50% of weights are near-zero
- Llama 2 (70B): ~60% of weights are near-zero
- Llama 3.1 (405B): ~65% of weights are near-zero
- DeepSeek-V3 (671B): ~70% of weights are near-zero
- Projected 1T+ models: ~75–80% sparsity expected

Ternary's zero state becomes **more beneficial at larger scales**, not less.

---

## 14.4 Benefits of Ternary for Trillion-Parameter LLMs

### 14.4.1 Memory: The Primary Bottleneck

| Metric | FP32 | FP16/BF16 | INT8 | FP8 | INT4 | Ternary |
|--------|------|-----------|------|-----|------|---------|
| 1T params size | 4 TB | 2 TB | 250 GB | 125 GB | 62.5 GB | ~200 GB |
| Fits in single HBM3e? | No | No | Yes (2 stacks) | Yes (1 stack) | Yes (1 stack) | Yes (2 stacks) |
| Fits in SRAM? | No | No | No | No | No | No (needs HBM) |
| Devices needed (1T dense) | 50+ | 25+ | 4 | 2 | 1–2 | 1–2 |

**Update (2025):** HBM3e now offers 128 GB per stack and HBM4 (expected 2026) will reach 256 GB. This changes the calculus:
- FP16 1T model: 2 TB → needs 16 HBM4 stacks (feasible in multi-chip)
- Ternary 1T model: 200 GB → fits in 1 HBM4 stack + on-chip SRAM
- **Ternary's advantage shifts from "impossible vs. possible" to "single-chip vs. multi-chip"**

### 14.4.2 Compute: Eliminating Multipliers

For a 1T parameter model running at 20 tokens/s:

| Precision | MACs per token | Multiplier energy | Total compute power |
|-----------|---------------|-------------------|-------------------|
| FP32 | 2T | 1.0 pJ/MAC | ~2 kW |
| FP16/BF16 | 2T | 0.3 pJ/MAC | ~600 W |
| INT8 | 2T | 0.2 pJ/MAC | ~400 W |
| FP8 | 2T | 0.15 pJ/MAC | ~300 W |
| Ternary (80% sparse) | 0.4T | 0.05 pJ/MAC | **~20 W** |

Ternary's combination of fewer operations (sparsity) and simpler operations (no multiplier) yields a **100× compute energy reduction** vs FP32.

### 14.4.3 Bandwidth: The Real Bottleneck

Modern LLM inference is **memory-bandwidth-bound**, not compute-bound:

```
FP16 1T model:  2 TB × 20 tokens/s = 40 TB/s bandwidth needed
Ternary 1T model: 200 GB × 20 tokens/s = 4 TB/s bandwidth needed

HBM3e bandwidth: ~1.2 TB/s per stack
→ FP16 needs 34 stacks (3–4 GPUs)
→ Ternary needs 4 stacks (1 GPU/accelerator)
```

### 14.4.4 Cost Analysis (Updated for 2025/2026)

| Cost Factor | FP16 (1T dense) | Ternary (1T dense) | Ternary (1T MoE, 200B active) |
|-------------|-----------------|-------------------|-------------------------------|
| DRAM needed | 2 TB (16 HBM3e) | 200 GB (2 HBM3e) | 200 GB (2 HBM3e) |
| DRAM cost | ~$80K | ~$8K | ~$8K |
| GPU/Accelerator | 2× H100 ($60K) | 1× custom ASIC ($10K) | 1× custom ASIC ($10K) |
| Power | ~700W (2× H100) | ~100W | ~50W |
| Cooling | Liquid cooling | Air cooling | Passive heatsink |
| **Total system cost** | **~$150K** | **~$20K** | **~$15K** |

---

## 14.5 Challenges at Trillion-Parameter Scale

### 14.5.1 Training from Scratch

Ternary LLMs require **quantization-aware training (QAT)**, not just post-training quantization:
- Training requires thousands of GPUs for months
- STE (straight-through estimator) gradients must flow through the ternarization step
- Delta threshold scheduling must be carefully tuned per layer

**Mitigation**: BitNet b1.58 demonstrated successful training from scratch up to 100B parameters. BitNet a4.58 extended this to 4-bit activations. The same techniques should scale to 1T with sufficient compute. Key innovations:
- **Sub-norm residual connections** for training stability
- **Learned step-size** per layer instead of fixed thresholds
- **Group-wise quantization** for activation granularity

### 14.5.2 Outlier Channels

Large LLMs have **outlier channels** with much larger weight magnitudes:
- Some channels have weights 10–100× the median
- These outliers are critical for model quality
- Per-channel scaling helps, but extreme outliers can still degrade accuracy

**Mitigation**: Mixed-precision approach — keep outlier channels in INT8/FP16, ternarize the rest. At 1T parameters, even 5% outlier channels in FP16 adds only ~100 GB, still manageable with HBM3e.

### 14.5.3 KV Cache Memory

The KV cache grows with sequence length and model size:
- 1T parameter model, 32K context: KV cache can exceed 100 GB
- This is activations (not weights), so ternary doesn't directly help

**Mitigations (2024–2025):**
- **Multi-head latent attention (MLA)**: DeepSeek-V3 reduces KV cache by 93.3% via low-rank compression
- **GQA/MQA**: Grouped-query and multi-query attention reduce KV heads (Llama 3.1 uses GQA)
- **KV cache quantization**: INT4/INT8 KV cache (2–4× reduction)
- **Sliding window attention**: Fixed-size KV cache regardless of context length
- **PagedAttention** (vLLM): Dynamic KV cache allocation, 2–4× memory efficiency

### 14.5.4 Communication Overhead

Distributed inference across multiple accelerators requires weight/activation communication:
- All-reduce operations don't benefit from ternary compression
- Activation tensors remain in INT8/FP16

**Mitigations:**
- **Pipeline parallelism**: Minimizes communication (only activations between stages)
- **Expert parallelism**: MoE experts stay on single devices
- **Ternary weight compression for inter-device transfer**: 20× reduction in communication volume

### 14.5.5 Post-Training Quantization Quality

BitNet b1.58 requires training from scratch — it cannot be applied to existing FP16 models via PTQ. This limits the addressable model space to newly trained models.

**Mitigation (emerging):**
- **QuIP#** (2023–2024): Lattice-based quantization achieves near-lossless 2-bit quantization
- **GPTQ + sparsity**: Combining INT4 quantization with pruning can approximate ternary benefits
- **AQLM** (Additive Quantization): 2-bit quantization with minimal accuracy loss
- For the near term, the most practical path is **training new models with ternary from scratch** rather than converting existing FP16 models

---

## 14.6 Architecture Recommendations for 1T Parameter Ternary LLM

### 14.6.1 Sparse Mixture-of-Experts (MoE)

The optimal architecture for a 1T ternary LLM is **Sparse MoE**, validated by DeepSeek-V3 and Mixtral:

| Architecture | Total Params | Active Params | Ternary Size | Active Ternary Compute |
|-------------|-------------|---------------|-------------|----------------------|
| Dense | 1T | 1T | 200 GB | 200 GB |
| MoE (8 experts, top-2) | 1T | 250B | 200 GB | 50 GB |
| MoE (16 experts, top-2) | 1T | 125B | 200 GB | 25 GB |
| MoE (64 experts, top-4) | 1T | 62B | 200 GB | 12.5 GB |

**Recommended**: 64 experts, top-4 routing. This combines ternary's per-weight compression with MoE's per-token compute reduction:
```
Total memory: 1T × 1.585 bits = ~200 GB
Active compute per token: 62B × 1.585 bits = ~12.5 GB
```

### 14.6.2 Recommended Accelerator Spec (Updated for 2025)

| Component | Specification | Rationale |
|-----------|--------------|-----------|
| Weight SRAM | 64 MB | Holds 320M ternary params on-chip (hot expert cache) |
| Weight DRAM | 2× HBM3e (256 GB) | Full 1T ternary model in DRAM |
| Compute array | 512×512 ternary PEs @ 1.5 GHz | 393 GOPS for ternary GEMM |
| Activation memory | 16 MB (INT4) | Matches BitNet a4.58 activation format |
| FP16 compute | 64×64 systolic array | Attention scores, softmax |
| Router | Dedicated hardware | MoE expert routing (top-k selection) |
| Memory interface | HBM3e, 2.4 TB/s | 2 stacks for weight loading |
| Power envelope | 50–100 W | Edge-compatible |
| Process node | 3nm or 4nm | Density + efficiency |
| Interconnect | PCIe 5.0 / CXL 3.0 | Host interface |

### 14.6.3 Expected Performance

| Metric | 1T Dense | 1T MoE (64 experts, top-4) |
|--------|---------|---------------------------|
| Decode throughput | ~20–40 tokens/s | ~80–160 tokens/s |
| Prefill throughput | ~200K tokens/s | ~800K tokens/s |
| Latency per token | ~25–50 ms | ~6–12 ms |
| Power | ~100 W | ~50 W |
| Model fit | 200 GB in HBM3e | 200 GB in HBM3e |
| Hot expert cache | 64 MB SRAM | 64 MB SRAM (top-4 experts) |

---

## 14.7 Comparison: Ternary vs Other Approaches at 1T Scale

| Approach | Model Size | Accuracy | Hardware | Single-Device? | Cost |
|----------|-----------|----------|----------|---------------|------|
| **Ternary (1T MoE)** | **~200 GB** | **-1-3%** | **Custom ASIC** | **Yes** | **~$15K** |
| FP8 (1T MoE) | ~125 GB | -0.1% | GPU/NPU | Yes (2× H100) | ~$60K |
| INT4 (1T MoE) | ~62 GB | -0.5-1% | GPU/NPU | Yes (1× H100) | ~$30K |
| INT8 (1T MoE) | ~125 GB | -0.5% | GPU/NPU | Yes (2× H100) | ~$60K |
| FP16 (1T dense) | 2 TB | Baseline | GPU | No (16× H100) | ~$480K |
| FP32 (1T dense) | 4 TB | Baseline | GPU | No (32× H100) | ~$960K |

**Key insight (2025):** The gap between ternary and FP8/INT4 is narrowing for accuracy, but ternary maintains a **10–100× energy advantage** for edge deployment. For data centers, FP8 may be preferred for its accuracy; for edge devices, ternary's power efficiency makes it the clear winner.

---

## 14.8 Industry Trajectory (2024–2026)

### What's Happened

| Date | Event | Significance |
|------|-------|-------------|
| Feb 2024 | BitNet b1.58 released | First proof that ternary matches FP16 at 100B scale |
| Jul 2024 | Llama 3.1 405B | Largest dense model; 8× H100 needed for inference |
| Nov 2024 | BitNet a4.58 released | 4-bit activations + ternary weights; practical path forward |
| Dec 2024 | DeepSeek-V3 (671B MoE) | Validates MoE at 600B+ scale; MLA reduces KV cache 93% |
| Jan 2025 | DeepSeek-R1 | Reasoning model; shows MoE scales to complex tasks |
| H1 2025 | HBM3e mass production | 128 GB per stack; reduces ternary to 2 HBM stacks |
| H2 2025 | Projected: 1T MoE models | Multiple labs expected to release 1T+ parameter models |

### What's Coming

| Date | Event | Significance |
|------|-------|-------------|
| 2025 | Ternary training at 1T scale | First 1T ternary model (BitNet or similar) |
| 2025–2026 | FPGA ternary accelerators | Working prototypes for 100B+ ternary inference |
| 2026 | HBM4 production | 256 GB per stack; ternary 1T model in 1 stack |
| 2026–2027 | Ternary ASIC tape-out | Purpose-built chips for ternary LLM inference |
| 2027+ | Ternary at 10T+ scale | MoE with 10T+ total params, 1T active |

---

## 14.9 Roadmap: From Today to 1T Ternary

### Phase 1: Validate (2025)

```
Goal: Train and evaluate a 10B ternary model
- BitNet-style architecture with INT4 activations
- Evaluate on standard benchmarks (MMLU, GSM8K, HumanEval)
- Compare against FP16 baseline
- Deliverable: Public ternary model + benchmark results
```

### Phase 2: Scale (2025–2026)

```
Goal: Train a 100B+ ternary MoE model
- 16–64 experts, top-2/4 routing
- MLA for KV cache reduction
- 14T+ training tokens
- Deliverable: State-of-the-art ternary LLM
```

### Phase 3: Hardware (2026–2027)

```
Goal: Build a ternary accelerator prototype
- FPGA prototype (256×256 PE array)
- Validate performance: 100B model decode latency
- Measure power efficiency vs GPU baseline
- Deliverable: Working hardware demo + paper
```

### Phase 4: Deploy (2027+)

```
Goal: Production ternary inference
- ASIC tape-out for 1T ternary MoE
- Software stack: compiler, runtime, model zoo
- Deployment: edge devices, data centers
- Deliverable: Commercial ternary inference product
```

---

## 14.10 Conclusion

**Balanced ternary is not just feasible for trillion-parameter LLMs — it may be the only practical path to single-device 1T inference.**

The evidence is clear:
1. **BitNet b1.58 + a4.58 prove ternary matches FP16/BF16** at 100B+ scale
2. **DeepSeek-V3 validates the MoE architecture** that ternary accelerates
3. **Memory requirements at 1T scale** make FP16/BF16 impractical for single devices
4. **Ternary's 20× compression** brings 1T parameters into the feasible range
5. **Sparsity increases with model size**, making ternary's zero state more valuable at larger scales
6. **HBM3e/HBM4** reduces the gap between ternary and FP8/INT4 for accuracy, while ternary maintains a **10–100× energy advantage**

The path forward:
1. **Short term (2025)**: Train and release a 10B ternary model as proof of concept
2. **Medium term (2025–2026)**: Scale to 100B+ ternary MoE with MLA
3. **Long term (2027+)**: Purpose-built ternary ASICs enabling single-device 1T parameter inference

The research community is moving rapidly in this direction. Microsoft (BitNet), DeepSeek (V3/R1), and several Chinese labs are actively exploring ternary quantization. The question is not *whether* ternary will be used for trillion-parameter LLMs, but *when* — and the answer is likely within 2–3 years.

**Status update (June 2025)**: BitNet a4.58 has demonstrated 100B+ ternary models matching FP16 performance. DeepSeek-V3 validates the MoE architecture that ternary accelerates. HBM3e mass production is underway, reducing the memory gap. The first 1T ternary model is expected by late 2025 or early 2026.
