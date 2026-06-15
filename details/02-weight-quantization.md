# 2. Weight Quantization Strategy

## 2.1 The Basic Ternary Quantizer

The simplest mapping from a full-precision weight `w` to a ternary value:

```
Q(w) = +1   if   w >  Δ
Q(w) =  0   if  -Δ ≤ w ≤ Δ
Q(w) = -1   if   w < -Δ
```

Where `Δ` is a threshold.

### Elixir: Basic Ternary Quantizer

```elixir
defmodule TernaryQuantizer do
  @type weight :: float()
  @type trit :: -1 | 0 | 1

  @doc """
  Quantize a single weight to a ternary value {-1, 0, +1}.
  Uses delta threshold to determine the zero region.
  """
  @spec ternarize(weight(), weight()) :: trit()
  def ternarize(w, delta) when is_number(w) and is_number(delta) do
    cond do
      w > delta  -> 1
      w < -delta -> -1
      true       -> 0
    end
  end

  @doc """
  Quantize a whole layer of weights (list of lists).
  Each inner list is one output channel.
  """
  @spec ternarize_layer([[weight()]], weight()) :: [[trit()]]
  def ternarize_layer(weights, delta) do
    Enum.map(weights, fn channel ->
      Enum.map(channel, &ternarize(&1, delta))
    end)
  end

  @doc """
  Compute per-channel MSE-optimal scale factor.
  scale_j = (W_j · T_j) / (T_j · T_j)
  """
  @spec compute_scale([weight()], [trit()]) :: float()
  def compute_scale(weights, trits) do
    dot = Enum.zip(weights, trits)
          |> Enum.reduce(0, fn {w, t}, acc -> acc + w * t end)

    norm_sq = Enum.reduce(trits, 0, fn t, acc -> acc + t * t end)

    if norm_sq == 0, do: 1.0, else: dot / norm_sq
  end
end
```

---

## 2.2 Threshold Selection

The choice of `Δ` significantly affects accuracy.

### Method A: Statistical Threshold

```
Δ = t × σ
```

Where `σ` is the standard deviation of weights in a layer, and `t` is a hyperparameter (typically 0.5–1.5).

### Method B: Balanced Distribution Threshold

Choose `Δ` so that approximately equal numbers of weights fall into +1 and −1:

```
∑ 𝟙(wᵢ > Δ) ≈ ∑ 𝟙(wᵢ < -Δ)
```

### Method C: Grid Search

Try a range of `Δ` values on a calibration dataset and pick the one with lowest task loss.

### Method D: Trained Threshold

Make `Δ` a learnable parameter during quantization-aware training.

---

## 2.3 Scaling Factor Methods

Pure ternary has limited expressiveness. **Scaling** rescues accuracy.

### Per-Tensor Scale

```
W_effective = α × T
```

One scale for the entire weight matrix. Simple but usually inaccurate.

### Per-Channel Scale (Recommended)

```
W_effective[j, :] = αⱼ × T[j, :]
```

One `αⱼ` per output channel. Good accuracy, low overhead.

### Per-Channel Scale + Bias

```
W_effective[j, :] = αⱼ × (T[j, :] + βⱼ)
```

A per-channel offset `βⱼ` adds another degree of freedom.

### Group-Wise Scale

```
group_size = 32 or 64
W_effective[g, :] = α_g × T[g, :]
```

Groups multiple output channels sharing one scale. Good tradeoff.

---

## 2.4 Scale Calibration

Scales can be determined in several ways:

### L2-Norm Based

```
αⱼ = ‖W_j‖₂ / ‖T_j‖₂
```

Minimizes the L2 error of the approximation.

### MSE-Based

```
αⱼ = argmin ‖W_j − α × T_j‖₂²
```

Closed-form solution:

```
αⱼ = (W_j · T_j) / (T_j · T_j)
```

### Learned

Fine-tune `αⱼ` via gradient descent after ternarizing T.

---

## 2.5 Full Quantization Pipeline

### Step 1: Train Baseline

```
FP32 model → high accuracy reference
```

### Step 2: Scale Calibration

```
For each layer:
    compute scale αⱼ per channel
    using MSE or L2 method
```

### Step 3: Ternarize

```
For each layer:
    T[j, i] = ternarize(W[j, i] / αⱼ, Δ)
```

### Step 4: Fine-Tune

```
Freeze T
Train only α and β
Optionally: re-train T with STE
```

### Step 5: Validate

```
Check accuracy on validation set
If too low:
    - try per-channel bias
    - increase group size
    - leave some layers in INT8
```

---

## 2.6 Quantization-Aware Training (QAT)

During training, weights are stored in full precision but ternarized on-the-fly in the forward pass:

```
Forward:  y = (α × T) · x
          T = ternarize(W_shadow / α)

Backward: ∂L/∂W_shadow ≈ ∂L/∂W_effective
          (straight-through estimator)

Update:   W_shadow ← W_shadow − lr × ∂L/∂W_shadow
          α ← α − lr × ∂L/∂α
```

The straight-through estimator (STE) bypasses the derivative of the ternarization step, which is zero almost everywhere:

```
∂T/∂W_shadow ≈ 1  (STE approximation)
```

This allows gradient flow through the quantization step.

---

## 2.7 Weight Clipping

To control outliers that degrade ternary quantization:

```
W_clipped = clamp(W, -c, +c)
```

Where `c` is a clipping threshold (e.g., 3× standard deviation).

Clipping can be applied before ternarization:

```
T = ternarize(W_clipped / α)
```

This prevents a few extreme values from dominating scale selection.

---

## 2.8 Sensitivity Analysis

Not all layers tolerate ternary equally well.

Typical sensitivity ranking (from most to least sensitive):

```
1. Embedding layer          (most sensitive)
2. Final classifier layer
3. Early attention layers
4. Late MLP projections
5. Middle MLP projections   (least sensitive)
```

A pragmatic approach:

```
- Ternarize all MLP projections       (largest matrices)
- Ternarize attention projections     (Q/K/V/O)
- Keep embedding in INT8 or FP16
- Keep final layer in INT8 or FP16
```

---

## 2.9 Mixed-Precision Quantization Decision Flow

```
For each layer:
    ┌─────────────────────┐
    │ Is layer an         │
    │ embedding table?    │──── YES ──► keep FP16
    └──────────┬──────────┘
               │ no
               ▼
    ┌─────────────────────┐
    │ Is this the final   │
    │ classifier head?    │──── YES ──► keep FP16
    └──────────┬──────────┘
               │ no
               ▼
    ┌─────────────────────┐
    │ Is it an attention  │
    │ score or softmax?   │──── YES ──► keep FP16
    └──────────┬──────────┘
               │ no
               ▼
    ┌─────────────────────┐
    │ Apply ternary       │
    │ with per-channel    │
    │ scale factors       │
    └─────────────────────┘
```

---

## 2.10 Example: Quantizing an MLP Layer

```
Original FP32 weights shape: [4096, 11008]

Step 1: Compute per-channel scales
  αⱼ = ‖W[j,:]‖₂ / sqrt(11008)
  Shape: [4096]

Step 2: Normalize weights
  W_norm[j,:] = W[j,:] / αⱼ

Step 3: Ternarize with Δ = 0.5
  T[j,i] = +1 if W_norm[j,i] > 0.5
  T[j,i] = -1 if W_norm[j,i] < -0.5
  T[j,i] =  0 otherwise

Step 4: Store
  T: 4096 × 11008 × 1.585 bits = ~89 Mb
  α: 4096 × 16 bits = ~65 Kb

Step 5: Effective weight
  W_effective[j,i] = αⱼ × T[j,i]
```

Result for this layer:

```
Original:  4096 × 11008 × 32 = 1.44 Gb
Ternary:   89 Mb + 65 Kb     = ~89 Mb
Compression: ~16×
```