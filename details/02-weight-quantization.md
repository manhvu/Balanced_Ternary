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

### 2.2.1 Weight Distribution Analysis

In practice, most neural network weights follow a bell-shaped distribution that is well-approximated by a Gaussian (normal) distribution with mean ≈ 0. This holds across architectures (MLPs, convolutions, attention projections) and training regimes.

For a Gaussian distribution with standard deviation `σ`, the probability density is:

```
p(w) = (1 / √(2πσ²)) × exp(−w² / (2σ²))
```

The optimal threshold `Δ` typically falls at the **inflection point of the cumulative distribution function (CDF)** — the point where the CDF transitions from concave to convex, which for a Gaussian occurs at `w = ±σ`. This is the region where the density begins to fall off most rapidly, meaning the quantization boundaries naturally separate the dense central mass (mapped to 0) from the tails (mapped to ±1).

A text-based visualization of the weight distribution with ternary regions marked:

```
Weight Distribution (Gaussian, σ = 1.0)
═══════════════════════════════════════════════════

  0.4 ┤              ████
      │            ████████
  0.3 ┤          ████████████
      │        ████████████████
  0.2 ┤      ████████████████████
      │    ████████████████████████
  0.1 ┤  ████████████████████████████
      │████████████████████████████████
  0.0 ┼──────────┼──────────┼──────────┼────
     -3σ    -Δ  -σ   0   +σ   +Δ     +3σ

  Legend:
  │←── −1 region ──→│←── 0 region ──→│←── +1 region ──→│
     (w < -Δ)          (-Δ ≤ w ≤ Δ)       (w > +Δ)

  For Δ ≈ 0.67σ (optimal for Gaussian):
    P(w < -Δ) ≈ 25%     P(-Δ ≤ w ≤ Δ) ≈ 50%     P(w > +Δ) ≈ 25%
```

The 50/25/25 split at `Δ ≈ 0.6745σ` (the interquartile point) minimizes MSE for a unit-variance Gaussian. In practice, values in the range `Δ ∈ [0.5σ, 1.0σ]` work well, with the exact optimum depending on the scale calibration method used.

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

> **Note on Outlier Channels:** Some output channels have much larger weight magnitudes than others — often 5–10× the median channel norm. These outlier channels distort the ternarization of neighboring channels when a shared threshold is used, and they inflate the MSE-based scale factor `αⱼ` for their own channel. The recommended approach is **clip-and-scale**: before computing `αⱼ`, clip each channel's weights to `±3σⱼ` (where `σⱼ` is that channel's own standard deviation), then compute the scale factor on the clipped weights. This prevents a handful of extreme values from dominating the calibration while preserving the bulk of the distribution. In practice, this simple preprocessing step can recover 0.2–0.5% accuracy on downstream tasks.

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

QAT stores weights in full precision and ternarizes them on-the-fly during the forward pass. The key challenge is that ternarization is a discontinuous operation whose gradient is zero almost everywhere; this is solved with the **straight-through estimator (STE)**, which approximates the backward gradient as if the quantization step were the identity function, enabling gradient-based optimization of the shadow weights and scale factors.

> For detailed QAT implementation with Python and Elixir code, see §3.3 in [03-training.md](03-training.md).

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

> For a detailed accuracy impact analysis of these choices, see [07-accuracy-analysis.md](07-accuracy-analysis.md).

### 2.8.1 Automatic Mixed-Precision Selection

Rather than manually deciding which layers to ternarize, an automatic algorithm can find the optimal mixed-precision assignment:

**Algorithm: Greedy Sensitivity-Promotion**

```
Input:
    L = {l₁, l₂, ..., lₙ}   (all quantizable layers)
    accuracy_target           (minimum acceptable accuracy)

Output:
    precision[l] for each layer l ∈ L  (one of {ternary, INT8})

Algorithm:
    1. Start: set precision[l] = ternary for all l ∈ L

    2. Evaluate model accuracy on calibration set
       If accuracy ≥ target → return current assignment

    3. For each layer l still ternary:
         a. Temporarily promote l to INT8
         b. Measure accuracy recovery: δₗ = acc_INT8(l) − acc_current
         c. Revert l to ternary

    4. Promote the layer l* with highest δₗ to INT8 (permanently)

    5. Re-evaluate accuracy
       If accuracy ≥ target → return current assignment
       Else → go to step 3

Pseudocode:

    function auto_mixed_precision(layers, model, data, target_acc):
        assignment = {l: TERNARY for l in layers}
        current_acc = evaluate(model, data, assignment)

        while current_acc < target_acc:
            best_layer = None
            best_recovery = 0

            for l in layers where assignment[l] == TERNARY:
                assignment[l] = INT8
                acc = evaluate(model, data, assignment)
                recovery = acc - current_acc

                if recovery > best_recovery:
                    best_recovery = recovery
                    best_layer = l

                assignment[l] = TERNARY

            if best_layer is None:
                break  # no single promotion helps enough

            assignment[best_layer] = INT8
            current_acc = current_acc + best_recovery

            print(f"Promoted {best_layer}: "
                  f"acc = {current_acc:.4f}, "
                  f"recovery = +{best_recovery:.4f}")

        return assignment
```

The key insight is that each evaluation in step 3 only requires re-running the affected layer in INT8 while keeping all others ternary, so the per-iteration cost is small. In practice, this converges in 5–15 promotions for typical LLM architectures, leaving 70–90% of layers in ternary.

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

> **Scale Factor Storage:** For a typical LLaMA-style model with 4096 output channels and 32 layers, total scale storage is 32 × 4096 × 2 bytes = 256 KB — negligible compared to 200 MB of ternary weights.

---

## 2.11 Quantization Error Analysis

We derive the expected mean squared error (MSE) for ternary quantization of a Gaussian weight distribution. This provides a theoretical foundation for choosing `Δ` and predicting accuracy loss.

**Note**: This analysis applies to both CNN and Transformer architectures. Recent work (BitNet a4.58, 2024) has shown that the optimal threshold `Δ*` is architecture-dependent, with Transformers preferring slightly larger thresholds than CNNs due to their different weight distributions.

**Setup:** Let `w ~ N(0, σ²)` be a weight drawn from a zero-mean Gaussian. The ternary quantizer with threshold `Δ` and scale `α` produces:

```
Q(w) = α × T, where
  T = +1  if  w > Δ
  T =  0  if  -Δ ≤ w ≤ Δ
  T = -1  if  w < -Δ
```

**Step 1: Define the MSE.** The per-weight MSE is:

```
MSE(α, Δ) = E[(w − αT)²]
           = ∫₋∞⁻ᵟ (w + α)² p(w) dw
           + ∫₋ᵟ⁺ᵟ w² p(w) dw
           + ∫ᵟ⁺∞ (w − α)² p(w) dw
```

where `p(w) = (1/√(2πσ²)) exp(−w²/(2σ²))`.

**Step 2: Optimal scale α* for a given Δ.** Setting `∂MSE/∂α = 0`:

```
α*(Δ) = E[|w| · 𝟙(|w| > Δ)] / P(|w| > Δ)
```

For a Gaussian, this evaluates to:

```
α*(Δ) = σ × √(2/π) × exp(−Δ²/(2σ²)) / erfc(Δ/(σ√2))
```

where `erfc` is the complementary error function.

**Step 3: Minimum MSE at optimal α.** Substituting `α*` back:

```
MSE_min(Δ) = σ² − α*(Δ)² × P(|w| > Δ)
           = σ² − σ² × (2/π) × exp(−Δ²/σ²) / erfc²(Δ/(σ√2))
```

**Step 4: Optimal threshold Δ*.** Minimizing `MSE_min(Δ)` numerically yields:

```
Δ* ≈ 0.6745 σ
```

This is the interquartile point of the Gaussian — consistent with the distribution analysis in §2.2.1.

**Step 5: Resulting MSE at the optimum.** Substituting `Δ*`:

```
MSE_min(Δ*) ≈ 0.3634 σ²
```

The **signal-to-quantization-noise ratio (SQNR)** is:

```
SQNR = σ² / MSE = 1 / 0.3634 ≈ 2.75  (or 4.4 dB)
```

**Summary table:**

```
┌──────────────────────────┬────────────────────┐
│ Quantity                 │ Value              │
├──────────────────────────┼────────────────────┤
│ Optimal threshold Δ*     │ 0.6745 σ           │
│ Optimal scale α*         │ 1.0540 σ           │
│ Minimum MSE              │ 0.3634 σ²          │
│ Fraction quantized to 0  │ 50.0%              │
│ Fraction quantized to ±1 │ 25.0% each         │
│ SQNR                     │ 4.4 dB             │
└──────────────────────────┴────────────────────┘
```

This means ternary quantization inherently discards about 63.7% of the signal energy (retaining only 36.3%). The per-channel scale factor `α` recovers some of this by stretching the ±1 values, but the fundamental information loss is significant. This is why mixed-precision (keeping sensitive layers in INT8) and fine-tuning are essential for maintaining accuracy.
