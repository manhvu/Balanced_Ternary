# 1. Balanced-Ternary Neural Network Architecture Overview

## 1.1 The Core Idea

A standard neural-network layer computes:

```
yⱼ = Σᵢ wᵢⱼ × xᵢ + bⱼ
```

If every weight `wᵢⱼ` is constrained to `{-1, 0, +1}`, multiplication simplifies to:

| Weight | Result     | Hardware Action |
|--------|------------|-----------------|
| +1     | +xᵢ        | Add activation  |
| 0      | 0          | Skip            |
| -1     | -xᵢ        | Subtract activation |

The entire multiply-accumulate (MAC) unit becomes an add/subtract/skip unit. No multiplier is needed.

### Elixir: Core Ternary Operation

```elixir
defmodule TernaryMAC do
  @type trit :: -1 | 0 | 1
  @type activation :: number()

  @doc """
  Ternary multiply: weight (-1, 0, +1) × activation.
  No real multiplication needed — just add, subtract, or skip.
  """
  @spec trit_mul(trit(), activation()) :: activation()
  def trit_mul(1, x), do: x
  def trit_mul(-1, x), do: -x
  def trit_mul(0, _x), do: 0

  @doc """
  Dot product of two lists using ternary weights and activations.
  """
  @spec dot_product([trit()], [activation()]) :: activation()
  def dot_product(weights, activations) do
    weights
    |> Enum.zip(activations)
    |> Enum.reduce(0, fn
      {1, x}, acc -> acc + x
      {-1, x}, acc -> acc - x
      {0, _}, acc -> acc
    end)
  end
end
```

### 1.1.1 Why Not Ternary Activations?

A natural question arises: if ternary weights save so much, why not also ternarize activations? The answer is that **ternary activations compound quantization errors across layers**, while keeping activations at INT8/INT4 preserves gradient flow and numerical stability.

Each layer's output becomes the next layer's input. If both weights and activations are ternarized, the quantization error from layer *n* is fed directly into layer *n+1*, where it is amplified by the next ternarization step. After *L* layers, the accumulated error grows roughly as `O(L × ε)` where `ε` is the per-layer quantization noise — for deep networks (e.g., 96-layer LLMs), this becomes catastrophic.

Keeping activations at INT8 or INT4 provides two key benefits:

1. **Gradient flow during training**: Backpropagation requires smooth, differentiable activations. Ternary activations create a staircase function with zero gradient almost everywhere, making SGD ineffective. INT8 retains 256 levels — enough for gradient signals to propagate.
2. **Numerical stability in attention**: Softmax and LayerNorm are sensitive to input precision. Ternary activations would collapse the attention distribution to 3 values, destroying the model's ability to focus selectively.

| Aspect                | Ternary Activations | INT8 Activations |
|-----------------------|---------------------|------------------|
| Levels                | 3                   | 256              |
| Per-layer error       | ~5-15%              | ~0.1-0.5%        |
| Error accumulation    | Catastrophic (×L)   | Manageable       |
| Gradient flow         | Broken              | Preserved        |
| Softmax quality       | Severely degraded   | Near FP16        |
| Training convergence  | Fails beyond ~4 layers | Converges normally |
| Hardware cost         | Lowest              | Low (INT8 MAC)   |
| Recommended           | ✗ No                | ✓ Yes             |

**Bottom line**: Ternary is ideal for weights (static, trained to compensate) but harmful for activations (dynamic, error-amplifying). The hybrid approach — ternary weights with INT8/INT4 activations — captures 90% of the benefit with none of the stability problems.

---

## 1.2 Hybrid Precision Model

A practical neural network cannot run entirely in ternary. A realistic design uses **hybrid precision**:

| Component              | Precision          | Why                                        |
|------------------------|--------------------|--------------------------------------------|
| Weight matrices        | Ternary {-1,0,+1}  | Largest storage savings                    |
| Scale factors          | FP16/BF16          | Per-channel correction for quantization    |
| Activations            | INT4 / INT8        | Cheaper than FP, good accuracy             |
| Attention scores       | FP16               | Softmax requires exponential precision     |
| Softmax                | FP16               | Numerical stability                        |
| LayerNorm              | FP16/BF16          | Small tensors, high sensitivity            |
| Residual connections   | FP16/BF16          | Accumulation of many layers                |
| KV cache               | INT4 / ternary     | Large memory consumer in LLMs              |
| Embedding tables       | INT8 / FP16        | Sensitive to quantization                  |
| Gradient accumulation  | FP32               | Training only; avoids precision loss over millions of small updates |
| Router weights (MoE)   | FP16               | Sensitive to quantization; routing decisions require fine-grained precision |

---

## 1.3 Layer Types and Their Ternary Mapping

### Dense / Linear Layer

```
Input:  x ∈ ℝⁿ
Weight: W ∈ {-1, 0, +1}ᵐˣⁿ
Scale:  α ∈ ℝᵐ  (per-output-channel)
Bias:   b ∈ ℝᵐ

Output: yⱼ = αⱼ × (Σᵢ Wⱼᵢ × xᵢ) + bⱼ
```

The inner sum `Σᵢ Wⱼᵢ × xᵢ` is computed via add/subtract of input activations. The scalar `αⱼ` is applied after accumulation.

### Convolution Layer

```
Input:  x ∈ ℝᶜˣʰˣʷ
Kernel: K ∈ {-1, 0, +1}ᶜᵒᵘᵗ ˣ ᶜⁱⁿ ˣ ᵏʰ ˣ ᵏʷ
Scale:  α ∈ ℝᶜᵒᵘᵗ
```

Each convolution window becomes a sum/subtract of selected input pixels. The `0` kernel values skip inputs entirely, providing natural sparsity.

### Attention Projection (Q/K/V/O)

Same as dense layers. The Q, K, V, and output projections are all ternary-weighted.

### Attention Score Computation

```
Score = Q × Kᵀ
```

This is a matrix multiply. If both Q and K are FP16/BF16 (from ternary projections), the score computation uses standard FP16 math.

### MLP Layers

Typically:

```
Gate projection:  ternary weight
Up projection:    ternary weight
Down projection:  ternary weight
```

All three projections can be ternary, giving 3× compression over FP32.

---

## 1.4 Data Flow Diagram

```
                 ┌─────────────────────┐
                 │ Input Embedding     │
                 │ (INT8 / FP16)       │
                 └──────────┬──────────┘
                            ▼
                 ┌─────────────────────┐
                 │ Ternary Q Projection │
                 │  add/sub/skip       │
                 └──────────┬──────────┘
                            ▼
    ┌──────────────────────────────────────┐
    │ ┌────────┐    ┌────────┐             │
    │ │Ternary  │    │Ternary  │            │
    │ │K Proj   │    │V Proj   │            │
    │ └───┬────┘    └───┬────┘             │
    │     ▼             ▼                  │
    │ ┌────────┐    ┌────────┐             │
    │ │KV Cache│    │KV Cache│             │
    │ │(INT4/  │    │(INT4/  │             │
    │ │Ternary)│    │Ternary)│             │
    │ └───┬────┘    └───┬────┘             │
    │     ▼             ▼                  │
    │ ┌────────────────────────┐           │
    │ │ FP16 Attention Score   │           │
    │ └──────────┬─────────────┘           │
    │            ▼                         │
    │ ┌────────────────────────┐           │
    │ │ FP16 Softmax           │           │
    │ └──────────┬─────────────┘           │
    │            ▼                         │
    │ ┌────────────────────────┐           │
    │ │ FP16 Attention Apply   │           │
    │ └──────────┬─────────────┘           │
    └────────────┼─────────────────────────┘
                 ▼
    ┌─────────────────────┐
    │ Ternary O Projection│
    └──────────┬──────────┘
                 ▼
    ┌─────────────────────┐
    │ FP16 Residual Add   │
    └──────────┬──────────┘
                 ▼
    ┌─────────────────────┐
    │ FP16 LayerNorm      │
    └──────────┬──────────┘
                 ▼
    ┌─────────────────────┐
    │ Ternary Gate + Up   │
    │ Projections (MLP)   │
    └──────────┬──────────┘
                 ▼
    ┌─────────────────────┐
    │ FP16 Activation     │
    └──────────┬──────────┘
                 ▼
    ┌─────────────────────┐
    │ Ternary Down Proj   │
    └──────────┬──────────┘
                 ▼
    ┌─────────────────────┐
    │ FP16 Residual Add   │
    └─────────────────────┘
```

### 1.4.1 Pipeline Stages per Transformer Layer

A well-designed ternary accelerator overlaps computation and memory transfers to hide latency. Below is a timing breakdown for a single transformer layer with hidden dimension 4096, sequence length 2048, running at 1 GHz.

```
Stage                     | Description                    | Est. Cycles | Overlaps With
--------------------------|--------------------------------|-------------|--------------
1. Weight fetch (HBM→SRAM)| Load packed ternary weights    | 2,000       | —
2. Ternary GEMM (Q/K/V)   | Add/skip/subtract accumulation | 8,000       | Act fetch
3. Activation fetch       | Load INT8 activations to PE    | 1,500       | GEMM
4. FP16 Attention Score   | Q × Kᵀ in FP16                 | 4,000       | KV cache load
5. FP16 Softmax           | Exponent + normalize           | 1,000       | —
6. FP16 Attention Apply   | Score × V in FP16              | 2,000       | —
7. Ternary GEMM (O proj)  | Output projection              | 2,000       | —
8. FP16 Residual + LN     | Skip connection + normalize    | 500         | —
9. Ternary GEMM (MLP up)  | Gate + up projections          | 4,000       | —
10. FP16 Activation       | SiLU/GELU in FP16              | 500         | —
11. Ternary GEMM (MLP dn) | Down projection                | 2,000       | —
12. FP16 Residual         | Final skip connection          | 200         | Weight fetch (next layer)
--------------------------|--------------------------------|-------------|--------------
Total per layer           |                                | ~27,700     |
Effective (with overlap)  |                                | ~18,000     |
```

```
Timeline (cycles, not to scale):
  0    2k   4k   6k   8k   10k  12k  14k  16k  18k
  ├────┼────┼────┼────┼────┼────┼────┼────┼────┤
  [Wt]                                         [Wt+1]
  [====GEMM QKV===]
       [Act]  [Attn]  [SM] [Attn] [O] [LN] [MLP] [Act] [MLP] [Res]
  ◄─────────────── Layer N ──────────────────────►◄── N+1 ──►
```

Key observations:
- **Ternary GEMM dominates** (~65% of non-overlapped time) but benefits most from the add/skip/subtract simplification — it runs 3-4× faster than an equivalent INT8 GEMM.
- **FP16 attention** is the second bottleneck; overlapping it with weight fetches for the next layer hides ~20% of its latency.
- **Memory transfers** (weight fetch, activation fetch) are hidden behind computation in steady-state execution, reducing effective cycles from ~27,700 to ~18,000 per layer.
- **LayerNorm and residuals** are negligible (<5% of cycles) but must remain in FP16 for numerical stability.

---

## 1.5 Sparse Ternary Representation

The zero trit provides **two advantages at once**:

1. **Quantization**: near-zero weights are forced to 0, reducing noise
2. **Pruning**: zero weights skip computation entirely

A typical trained ternary model after sparsity-regularized training might have:

```
Sparsity level:  50-80% zeros
Positive:         10-25% +1
Negative:         10-25% -1
```

At 75% sparsity, the effective computation is 4× less than a dense ternary model.

---

## 1.6 Scaling Factor Strategy

Pure ternary weights are too lossy for many models. The fix is **per-channel scaling**:

```
W_effective[j, :] = αⱼ × W_ternary[j, :]
```

Where `αⱼ` is learned or calibrated per output channel.

Training approaches:

| Method | Description | Accuracy |
|--------|-------------|----------|
| Per-tensor scale | Single α for whole weight matrix | Poor |
| Per-channel scale (vector) | One α per output channel | Good |
| Per-channel scale (vector + bias) | α + β offset per channel | Better |
| Group-wise scale | One α per group of K channels | Good tradeoff |
| Learnable scale | α is trained with SGD | Best |

Recommended default:

```
Per-channel FP16 scale factor
```

Storage overhead of scales:

```
1B parameter model
  → 1B ternary weights × 1.585 bits = ~200 MB
  → If model has 4096 output channels per layer, 512 layers
  → 512 × 4096 × 16 bits = ~4 MB of scale factors
  → Overhead: ~2%
```

> **Scale factor initialization**: Best practice is to initialize each channel's scale factor `αⱼ` from the L2 norm of that channel's full-precision weight vector *before* ternarization. Specifically, `αⱼ = ||W_fp[j, :]||₂ / sqrt(n)` where `n` is the channel width. This ensures the ternary weight matrix starts with the same per-channel energy as the original FP model, minimizing the initial accuracy drop. The scale factors are then fine-tuned during quantization-aware training. Models initialized this way typically recover within 0.5% of FP32 accuracy after 1-2 epochs of fine-tuning, compared to 2-3% loss with naive uniform initialization.

---

## 1.7 Edge vs Server Deployment

### Edge Device (Smartphone / IoT)

```
Memory:     4-8 GB shared
Model:      100M-1B parameters
Ternary:    20-200 MB weights
Activations:INT4/INT8
FP16 units: yes, small
Goal:       on-device LLM, real-time inference
Power:      <5W
Advantage:  ternary fits entirely in on-chip SRAM
```

### Server / Data Center

```
Memory:     80 GB+ HBM
Model:      7B-70B parameters
Ternary:    1.4-14 GB weights
Activations:INT8/BF16 with FP16 attention
FP16 units: large tensor cores
Goal:       efficient batch inference
Power:      100-500W
Advantage:  ternary reduces HBM bandwidth bottleneck
```

---

## 1.8 Comparison Summary

| Property              | FP32 Baseline   | INT8 Baseline    | Ternary (this design) |
|-----------------------|-----------------|------------------|-----------------------|
| Weight size           | 32 bits         | 8 bits           | ~1.585 bits           |
| Model size (1B params)| 4 GB            | 1 GB             | ~200 MB               |
| Multiplication        | Full FP32       | INT8 multiply    | Add/sub/skip          |
| Sparsity support      | No              | No               | Natural (zero weight) |
| Accuracy (typical)    | Reference       | Near-lossless    | Slight loss           |
| Hardware complexity   | High            | Medium           | Low                   |
| Memory bandwidth need | Highest         | High             | Lowest                |
| Suitable for          | Anything        | Inference        | Edge inference        |

---

## 1.9 Key Design Principles

1. **Ternary weights, not ternary everything** — keep activations and control math higher precision
2. **Per-channel scaling** — compensates for quantization loss
3. **Sparsity is free** — zero trits skip compute and storage naturally
4. **Memory bandwidth is the bottleneck** — ternary attacks it directly
5. **Approximate computation is acceptable** — neural networks are naturally error-tolerant
6. **Quantization-aware training** — required for good accuracy
7. **Hybrid precision** — use the right tool for each operation

---

## 1.10 Software Stack Overview

Deploying a ternary model requires a full compiler pipeline that transforms a standard PyTorch model into packed binary code for the accelerator:

```
┌──────────────┐     ┌──────────────┐     ┌──────────────────┐
│  PyTorch      │     │   ONNX       │     │  Ternary          │
│  Model        │────▶│   Export     │────▶│  Quantizer        │
│  (FP32/BF16)  │     │   (.onnx)    │     │  (calibration +   │
│               │     │              │     │   ternarization)  │
└──────────────┘     └──────────────┘     └────────┬─────────┘
                                                    │
                                                    ▼
┌──────────────┐     ┌──────────────┐     ┌──────────────────┐
│  Accelerator │     │  Accelerator │     │  Packed Binary    │
│  Runtime     │◀────│  Driver      │◀────│  (.tbin)          │
│  (inference) │     │  (load +     │     │  (2-bit trits +   │
│              │     │   schedule)  │     │   FP16 scales)    │
└──────────────┘     └──────────────┘     └──────────────────┘
```

### Stage 1: PyTorch Model
The starting point is a standard PyTorch model in FP32 or BF16. No special annotations are required — the quantizer handles conversion automatically.

### Stage 2: ONNX Export
The model is exported to ONNX format, producing a portable graph representation. This decouples the quantization toolchain from the training framework and allows the same quantizer to serve models from JAX, TensorFlow, or other frontends.

### Stage 3: Ternary Quantizer
The quantizer performs three operations:
1. **Calibration**: Runs a small representative dataset (typically 100-500 samples) through the ONNX graph to collect activation statistics (min/max, percentiles) and determine optimal per-channel scale factors.
2. **Ternarization**: Converts each weight tensor to `{-1, 0, +1}` using the learned threshold `Δⱼ = αⱼ × threshold_factor`. Weights below the threshold become 0 (sparsity), others become ±1.
3. **Scale factor initialization**: Computes `αⱼ = ||W_fp[j, :]||₂ / sqrt(n)` per channel (see §1.6) and stores them as FP16 alongside the ternary weights.

### Stage 4: Packed Binary (`.tbin`)
The ternary weights are packed into a dense binary format:
- **Ternary data**: 2 bits per trit (encoding `-1 → 0b00`, `0 → 0b01`, `+1 → 0b10`; `0b11` unused), packed into 64-bit words (32 trits per word).
- **Scale factors**: FP16 array, one per output channel, stored in a separate header section.
- **Metadata**: Layer dimensions, sparsity masks, pipeline scheduling hints.

For a 1B-parameter model, the `.tbin` file is ~200 MB (weights) + ~4 MB (scales) + negligible metadata.

### Stage 5: Accelerator Runtime
The runtime driver loads the `.tbin` into accelerator memory, configures the PE array, and executes the inference schedule:
1. Fetches packed weights from HBM into on-chip SRAM.
2. Unpacks trits on-the-fly in the PE array.
3. Executes the ternary GEMM pipeline (§1.4.1) with overlapping FP16 attention stages.
4. Streams results back to host memory.

The runtime also handles dynamic shapes (variable sequence lengths) by adjusting the GEMM tiling parameters without recompilation.
