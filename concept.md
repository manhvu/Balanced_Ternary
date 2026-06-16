# Balanced Ternary for AI Computing

## What Is Balanced Ternary?

Most computers use **binary** — everything is 0 or 1. Balanced ternary uses **three** states:

| Value | Symbol | Meaning |
| ----- | ------ | ------- |
| -1    | T      | Negative |
| 0     | 0      | Zero |
| +1    | 1      | Positive |

That's it. Three simple values. But this small change has profound implications for AI hardware.

### Why Three States Matter

In a binary neural network, every weight is either `-1` or `+1`. There's no option for "unimportant." Every single weight must participate in computation — even the ones that are nearly zero.

Balanced ternary adds the **zero** state, which acts as a built-in pruning mechanism:

```
Binary:   {-1, +1}     → every weight contributes
Ternary:  {-1, 0, +1}  → near-zero weights become "skip"
```

The zero state means:
- **No computation**: skip the multiply-accumulate entirely
- **No energy**: clock-gate that processing element
- **No storage**: sparse formats can omit zeros

This is the single most important advantage of balanced ternary: **it turns pruning into a natural byproduct of quantization**.

---

## Why AI Is a Good Fit

### The Problem: Multiplication Is Expensive

Neural networks are mostly one operation:

```
output = weight₁ × input₁ + weight₂ × input₂ + ... + weightₙ × inputₙ
```

This is called **Multiply-Accumulate (MAC)**. In a GPU, MAC units dominate the chip area and power consumption:

| Precision | Multiplier Area | Power per MAC |
|-----------|----------------|---------------|
| FP32      | 100%           | ~1.0 pJ       |
| FP16      | ~25%           | ~0.3 pJ       |
| INT8      | ~6%            | ~0.2 pJ       |
| Ternary   | **0%**         | **~0.05 pJ**  |

Ternary eliminates the multiplier entirely. Instead of multiplying, the hardware just **selects**: add, subtract, or skip.

### The Insight: Multiplication Becomes Selection

With ternary weights, a normal multiplication `w × x` becomes:

| Weight | Result | What Hardware Does |
| ------ | ------ | ------------------ |
| +1     | x      | Pass through (wire) |
| 0      | 0      | Skip (clock gate) |
| -1     | -x     | Negate (~4 transistors) |

**No multiplier needed.** The entire MAC unit shrinks to a single adder with a mux.

### Worked Example

Say we have 4 inputs and 4 weights:

```
Inputs:   [4, 7, 2, 5]
Weights:  [1, T, 0, 1]     (T means -1)
```

**Traditional hardware** computes:
```
4×1 + 7×(-1) + 2×0 + 5×1 = 4 - 7 + 0 + 5 = 2
```
That's 4 multiplications + 3 additions = 7 operations.

**Ternary hardware** computes:
```
  4      (pass x as-is)
- 7      (negate x)
+ 0      (skip entirely)
+ 5      (pass x as-is)
────
  2
```
That's 2 additions + 1 negation + 1 skip = 4 operations, **zero multiplications**.

At 75% sparsity (common in trained models), ternary does **4× fewer operations** than dense computation.

---

## Storage: 20× Smaller Models

A ternary weight carries `log₂(3) ≈ 1.585` bits of information, compared to 32 bits for FP32:

| Format | Bits/Weight | 1B Parameters |
|--------|-------------|---------------|
| FP32   | 32          | 4 GB          |
| INT8   | 8           | 1 GB          |
| Ternary| ~1.585      | ~200 MB       |

A 200 MB model fits entirely in **on-chip SRAM** — no slow, energy-hungry DRAM access needed. This is transformative for edge devices.

### Packing: 10 Trits in 16 Bits

Because `3¹⁰ = 59,049` fits in `2¹⁶ = 65,536`, we can pack 10 ternary weights into a single 16-bit word with only ~10% waste:

```
16-bit word: [t₉ t₈ t₇ t₆ t₅ t₄ t₃ t₂ t₁ t₀]
              ↑                   ↑
         first weight       last weight
```

Multiple packing options exist depending on the use case:

| Scheme | Trits | Bits | Efficiency | Best For |
|--------|-------|------|------------|----------|
| 5→8    | 5     | 8    | 95%        | Simple hardware decode |
| 10→16  | 10    | 16   | 90%        | Maximum storage density |
| 20→32  | 20    | 32   | 81%        | SIMD-friendly access |

---

## Sparsity: The Hidden Superpower

Many trained neural networks have a large fraction of near-zero weights. In a binary network, these near-zero weights must be stored as either `-1` or `+1` — they still consume storage and computation.

In a ternary network, they become `0` — and zeros are **free**:

| Sparsity | Accuracy Impact | Compute Reduction |
|----------|----------------|-------------------|
| 0%       | Baseline       | 0%                |
| 50%      | +1.1 PPL       | 2×                |
| 75%      | +4.0 PPL       | 4×                |
| 90%      | +8.5 PPL       | 10×               |

The sweet spot for most applications is **50-75% sparsity**, where accuracy loss is modest but compute savings are dramatic.

**Key insight**: Ternary doesn't just quantize — it **combines quantization and pruning into a single step**. The zero trit is both a quantization level and a pruning decision.

---

## Scale Factors: Recovering Accuracy

Pure ternary (just `-1, 0, +1`) is too lossy for most models. The fix is a **per-channel scale factor** α:

```
W_effective = α × W_ternary
```

Each output channel gets its own FP16 scale factor. This recovers most of the accuracy loss with minimal overhead (~2% storage for typical layer dimensions).

Think of it like this: the ternary weight provides the **direction** (+1, 0, or -1), while the scale factor provides the **magnitude**. Together, they approximate the original FP32 weight much more accurately than either alone.

---

## Hardware Architecture

A ternary AI accelerator looks like this:

```
┌─────────────────────┐
│ Host CPU            │  ← Model loading, scheduling
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Trit Decoder        │  ← Unpack 10 trits per 16-bit word
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Ternary Compute     │  ← 128×128 array of add/sub/skip PEs
│ Array               │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ Accumulator         │  ← Apply per-channel scale factor
│ + Scale Unit        │
└──────────┬──────────┘
           │
           ▼
┌─────────────────────┐
│ FP16 Compute Unit   │  ← Softmax, LayerNorm, attention scores
└─────────────────────┘
```

Each processing element (PE) is dead simple:

```
Weight = +1  →  pass activation through
Weight = -1  →  negate activation
Weight =  0  →  disable this lane (clock gating)
```

No multiplier. No complex floating-point unit. Just an adder, an inverter, and a mux.

---

## Differential Encoding: Wire-Swap Negation

How do you physically represent three states on a chip? The elegant solution uses **two wires** per trit:

| Trit | Wire A | Wire B |
|------|--------|--------|
| T    | 0      | 1      |
| 0    | 0      | 0      |
| 1    | 1      | 0      |

**Negation is free**: to flip `T ↔ 1`, you just swap the two wires. No inverter needed.

**Zero detection is trivial**: `A=0, B=0` means zero — a single NOR gate enables clock gating.

This is fully digital CMOS — no analog voltage levels, no precision comparators, no noise sensitivity.

---

## Hybrid Architecture: Best of Both Worlds

Ternary doesn't replace everything. A practical accelerator uses the right tool for each job:

```
Weights          → Ternary {-1, 0, +1}     (storage + GEMM)
Activations      → INT8                    (cheap, good accuracy)
Attention scores → FP16                    (needs exponent range)
Softmax/LayerNorm→ FP16                    (numerically sensitive)
Control logic    → Standard binary         (unchanged)
```

This hybrid approach means:
- **90%+ of compute** (the GEMM) uses simple ternary add/sub/skip
- **Numerically sensitive operations** (softmax, LayerNorm) keep FP16 precision
- **Existing software** (PyTorch, ONNX) needs minimal changes

---

## Why This Targets the Real Bottleneck

Modern LLM inference is **memory-bandwidth-bound**, not compute-bound. The GPU spends most of its time waiting for weights to arrive from off-chip memory.

```
FP32 model:  4 GB × 20 tokens/s = 80 GB/s memory bandwidth needed
Ternary:     200 MB × 20 tokens/s = 4 GB/s memory bandwidth needed
```

That's a **20× reduction** in memory traffic. The weights fit in on-chip SRAM, eliminating the most energy-hungry part of inference.

### Energy Breakdown

| Component | FP32 | Ternary | What Changed |
|-----------|------|---------|--------------|
| Weight memory access | 60% | ~0% | Fits in SRAM |
| Compute (MAC) | 25% | ~8% | No multipliers |
| Activation movement | 10% | ~10% | Same |
| Other | 5% | ~2% | Simpler control |

---

## Running on Today's Hardware

Purpose-built ternary accelerators don't exist yet. But you can deploy ternary models on current hardware:

- **GPUs**: Unpack ternary → INT8 at runtime, use tensor cores (CUDA kernel provided)
- **CPUs**: Use AMX/NEON dot-product instructions on unpacked weights
- **NPUs**: Convert to INT8, leverage existing matrix units
- **FPGAs**: Implement native add/sub/skip datapath (Verilog decoder provided)

Each platform recovers the **storage and bandwidth advantage** (20× smaller weights) even if the compute simplification requires emulation.

See [Running Ternary Models on Current Hardware](details/11-current-hardware-gpu-cpu-npu.md) for platform-specific kernels and performance estimates.

---

## Custom Accelerator Design

For maximum efficiency, a purpose-built ternary ASIC achieves:

| Metric | Value |
|--------|-------|
| Process | 7nm |
| Die area | 25 mm² |
| Weight SRAM | 8 MB (fits 1B ternary parameters) |
| Compute | 128×128 add/skip array @ 1 GHz |
| Decode throughput | 10 trits/cycle per column |
| Power | ~4W |
| Decode throughput | ~20K tokens/s (1B model) |

See [Custom Accelerator Design](details/12-custom-ternary-accelerator-design.md) for the full architecture specification including PE design, memory hierarchy, instruction set, and compiler pipeline.

---

## Converting Models to Ternary

The conversion pipeline has two paths:

### Quick Path: Post-Training Quantization (PTQ)
```
FP32 model → calibrate scales → ternarize → validate
```
Takes minutes. Loses 2-5% accuracy. No retraining needed.

### Quality Path: Quantization-Aware Training (QAT)
```
FP32 model → insert ternary layers → fine-tune with STE → export
```
Takes hours/days. Loses 1-3% accuracy. Requires retraining.

A complete Elixir conversion toolkit is provided in [`tools/ternary_converter/`](tools/ternary_converter/) with:
- `convert` — Convert weights to `.tbin` binary format
- `demo` — Full pipeline demo with synthetic weights
- `info` — Inspect `.tbin` models (sparsity, compression, layer stats)
- `validate` — Round-trip verification and inference testing

See [Model Conversion Guide](details/13-model-conversion-guide.md) for detailed pipelines, model-specific recipes (LLaMA, GPT-2, BERT, ViT), and the Elixir API.

---

## Comparison with Other Approaches

| Approach | Bits/Weight | Multiplier? | Sparsity? | Accuracy | Hardware |
|----------|-------------|-------------|-----------|----------|----------|
| **Ternary** | **1.585** | **No** | **Natural** | **-1-3%** | Custom |
| Binary | 1 | No | No | -5-10% | FPGA |
| INT8 | 8 | Yes | Structured | -0.5% | GPU/NPU |
| INT4 | 4 | Yes | No | -1% | GPU |
| FP8 | 8 | Yes | No | -0.1% | GPU |
| FP32 | 32 | Yes | No | Baseline | GPU |

**Ternary's unique combination**: lowest bit width + no multiplier + natural sparsity. The trade-off is requiring custom hardware.

See [Detailed Comparison](details/10-comparison.md) for comprehensive pros/cons against each alternative.

---

## Target Markets

| Market | Model Size | Power Budget | Key Metric |
|--------|-----------|-------------|------------|
| Smartphones | 100M-1B | <3W | On-device LLM |
| Edge AI | 100M-7B | <5W | Real-time inference |
| Drones | 50M-500M | <2W | Weight + power |
| Robotics | 100M-1B | <5W | Latency |
| IoT | 10M-100M | <1W | Cost + power |

---

## Long-Term Vision

Balanced ternary computing was explored as early as the 1950s (the Soviet Setun computer) but failed to gain adoption because binary switching was simpler and no workload demanded three states.

**AI changes this calculus.** Neural networks are:
- **Error-tolerant** — approximate computation is fine
- **Memory-bandwidth-bound** — ternary reduces weight size 20×
- **Multiplication-heavy** — ternary eliminates multipliers
- **Naturally sparse** — ternary's zero state captures this for free

Rather than replacing binary computing, balanced ternary may find success as a **specialized AI acceleration technology** — the right tool for the most important workload of the coming decade.

### Research Trajectory

```
Phase 0: Foundation        → Software simulation, literature review
Phase 1: Small Models      → ResNet, BERT validation
Phase 2: Transformer       → GPT-2, LLaMA 1B validation
Phase 3: Sparsity          → 75%+ sparsity training
Phase 4: HW Simulation     → Cycle-accurate simulator
Phase 5: FPGA Prototype    → Working hardware demo
Phase 6: ASIC Pathfinding  → Tape-out specifications
```

See [Research Roadmap](details/08-research-roadmap.md) for the detailed 48-week plan.

---

## Document Structure

### Getting Started
- **[`concept.md`](concept.md)** (this file) — High-level overview and motivation

### Deep Dives
| Document | Content |
|----------|---------|
| [`01-architecture-overview`](details/01-architecture-overview.md) | Layer types, data flow, hybrid precision, scaling strategies |
| [`02-weight-quantization`](details/02-weight-quantization.md) | Quantizer design, threshold selection, per-channel scaling, QAT |
| [`03-training`](details/03-training.md) | Training pipeline, STE variants, distillation, distributed training |
| [`04-storage-format`](details/04-storage-format.md) | Packing schemes, sparse formats, decoder design, bandwidth calc |
| [`05-hardware-architecture`](details/05-hardware-architecture.md) | PE design, systolic array, memory hierarchy, ISA, verification |
| [`06-llm-inference-engine`](details/06-llm-inference-engine.md) | Transformer mapping, KV cache, throughput analysis, energy |
| [`07-accuracy-analysis`](details/07-accuracy-analysis.md) | Benchmarks, sensitivity analysis, ablation studies |
| [`08-research-roadmap`](details/08-research-roadmap.md) | 48-week plan, resources, risks, publication strategy |
| [`09-differential-encoding`](details/09-differential-encoding.md) | Two-wire encoding, CMOS compatibility, DFT |
| [`10-comparison`](details/10-comparison.md) | Detailed comparison with BNN, INT8, INT4, FP8, pruning |
| [`11-current-hardware`](details/11-current-hardware-gpu-cpu-npu.md) | GPU/CPU/NPU/FPGA deployment today |
| [`12-custom-accelerator`](details/12-custom-ternary-accelerator-design.md) | Full ASIC architecture specification |
| [`13-model-conversion`](details/13-model-conversion-guide.md) | PTQ/QAT pipelines, recipes, Elixir toolchain |
| [`14-ternary-llm-feasibility`](details/14-ternary-llm-feasibility.md) | Feasibility at trillion-parameter scale, MoE, cost analysis |
| [`15-ternary-vision`](details/15-ternary-vision-computing.md) | Vision computing: CNNs, ViT, detection, segmentation |

### Code & Tools
| Path | Content |
|------|---------|
| [`tools/ternary_converter/`](tools/ternary_converter/) | Full Elixir app: convert, demo, info, validate |
| [`examples/ternary_pure/`](examples/ternary_pure/) | Pure Elixir quantization + packing demo |
| [`examples/ternary_nx/`](examples/ternary_nx/) | Nx tensor-based quantization + GEMM demo |
