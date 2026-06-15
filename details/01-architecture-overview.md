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

---

## 1.2 Hybrid Precision Model

A practical neural network cannot run entirely in ternary. A realistic design uses **hybrid precision**:

| Component           | Precision          | Why                                     |
|---------------------|--------------------|-----------------------------------------|
| Weight matrices     | Ternary {-1,0,+1}  | Largest storage savings                 |
| Scale factors       | FP16/BF16          | Per-channel correction for quantization |
| Activations         | INT4 / INT8        | Cheaper than FP, good accuracy          |
| Attention scores    | FP16               | Softmax requires exponential precision  |
| Softmax             | FP16               | Numerical stability                     |
| LayerNorm           | FP16/BF16          | Small tensors, high sensitivity         |
| Residual connections| FP16/BF16          | Accumulation of many layers             |
| KV cache            | INT4 / ternary     | Large memory consumer in LLMs           |
| Embedding tables    | INT8 / FP16        | Sensitive to quantization               |

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
