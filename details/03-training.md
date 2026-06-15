# 3. Training a Balanced-Ternary Neural Network

## 3.1 Training Pipeline Overview

```
Phase 1: Pre-train or load FP32 model
Phase 2: Quantization-aware fine-tuning
Phase 3: Sparsity regularization
Phase 4: Post-training calibration
Phase 5: Final ternary model export
```

---

## 3.2 Phase 1: Baseline Model

Start with a standard FP32 model. Options:

| Option | When to Choose |
|--------|---------------|
| Train from scratch | Novel architecture |
| Load pretrained model | Using existing model (LLaMA, GPT, etc.) |
| Knowledge distillation | Student gets teacher guidance |
| Continue pre-training | Domain adaptation |

The baseline should be as good as possible. Any existing accuracy gap will worsen after ternarization.

---

## 3.3 Phase 2: Quantization-Aware Training (QAT)

### Forward Pass (Inference)

```
1. Retrieve shadow weight W_shadow (FP32)
2. Divide by per-channel scale α: W_norm = W_shadow / α
3. Ternarize: T = ternarize(W_norm, Δ)
4. Reconstruct: W_effective = α × T
5. Compute layer output: y = W_effective @ x + b
```

### Backward Pass

```
Gradient of ternarization is zero almost everywhere.
Solution: Straight-Through Estimator (STE).

∂L / ∂W_norm ≈ ∂L / ∂T   (identity mapping for gradient)

∂L / ∂W_shadow = ∂L / ∂T × (1/α)
∂L / ∂α = ∂L / ∂T × T
```

### Pseudocode

```python
def forward(self, x):
    W_norm = self.W_shadow / self.scale            # [out_dim, in_dim]
    T = self.ternarize(W_norm, self.delta)         # {-1,0,+1}
    W_effective = self.scale * T                    # [out_dim, in_dim]
    return F.linear(x, W_effective, self.bias)

def ternarize(W, delta):
    return torch.where(W > delta, 1.0,
           torch.where(W < -delta, -1.0, 0.0))

def backward_hook(grad):
    # STE: pass gradient through unchanged
    return grad
```

### Elixir (Nx): Ternary Linear Layer with QAT

```elixir
defmodule TernaryLayer do
  import Nx.Defn

  @doc """
  Ternarize a tensor: {-1, 0, +1} with threshold delta.
  Uses Nx for hardware-accelerated tensor ops.
  """
  defn ternarize(w, delta) do
    Nx.select(w > delta, 1.0,
      Nx.select(w < -delta, -1.0, 0.0))
  end

  @doc """
  Forward pass with Straight-Through Estimator.
  W_shadow: full-precision shadow weights (FP32)
  scale: per-channel scale factors
  x: input tensor
  """
  defn forward(w_shadow, scale, x, delta) do
    # Normalize weights
    w_norm = w_shadow / Nx.new_axis(scale, 1)

    # Ternarize
    t = ternarize(w_norm, delta)

    # Reconstruct effective weights
    w_effective = Nx.new_axis(scale, 1) * t

    # STE: pass gradient through unchanged
    w_ste = Nx.as_type(w_shadow, Nx.type(w_shadow)) +
            (w_effective - Nx.as_type(w_shadow, Nx.type(w_shadow)))
            |> Nx.as_type(Nx.type(w_shadow))

    Nx.dot(x, w_ste)
  end

  @doc """
  Sparsity regularization loss.
  Encourages weights toward zero by penalizing non-zero entries.
  """
  defn sparsity_loss(w, delta, lambda) do
    t = ternarize(w, delta)
    density = Nx.mean(Nx.abs(t))
    lambda * density
  end
end
```

---

## 3.4 Straight-Through Estimator Variants

| Variant | Description | Stability |
|---------|------------|-----------|
| Identity STE | ∂T/∂W = 1 in backward | Medium |
| Clipped STE | ∂T/∂W = 1 for |W_norm| ≤ 1, else 0 | Good |
| Hard sigmoid STE | ∂T/∂W = max(0, 1 − |W_norm|) | Best |

Recommended: **Clipped STE** for most applications.

---

## 3.5 Phase 3: Sparsity Regularization

Encourage more weights to become 0 during training.

### L1 Regularization

```
L_sparsity = λ × |W_shadow / α|
```

This pushes small weights toward 0.

### Hard Threshold Decay

```
At epoch E:
    Δ = Δ_initial × (1 − E / total_epochs)

Or step schedule:
    Δ = step(E, [0, 10, 20], [0.3, 0.6, 1.0])
```

Gradually increasing `Δ` forces more weights to zero.

### Soft Sparsity Target

```
Target sparsity: S_target (e.g., 0.5 = 50%)

L_sparsity = λ × |sparsity(W_ternary) − S_target|
```

Train toward a specific sparsity level.

---

## 3.6 Phase 4: Post-Training Calibration

After QAT, freeze all weights (including T) and calibrate scales.

### Calibration Dataset

Use a small representative dataset (e.g., 512–4096 samples).

### Scale Optimization

For each layer:

```
Given:    T_ternary (fixed)
Given:    validation samples {x_i, y_i}
Optimize: α to minimize task loss

Method: grid search or gradient descent on α
```

### Bias Correction

After ternarization, the expected output distribution shifts:

```
E[FP32 output] — E[ternary output] ≠ 0
```

Fix: subtract the mean error per channel:

```
b_corrected = b + E[FP32 output] − E[ternary output]
```

---

## 3.7 Knowledge Distillation for Ternary Models

Training a ternary model from scratch is hard. Distillation helps significantly.

### Set-up

```
Teacher: FP32 model (or high-precision model)
Student: Ternary model (weights = {-1,0,+1})
```

### Loss Function

```
L = α × L_task(y_pred, y_true)
  + β × L_distill(y_pred, y_teacher)
  + γ × L_sparsity(W_ternary)
  + δ × L_scale_decay(α)
```

Where:

- `L_task`: standard task loss (cross-entropy, MSE, etc.)
- `L_distill`: KL divergence or MSE between student and teacher logits
- `L_sparsity`: regularization pushing weights toward 0
- `L_scale_decay`: weight decay on scale factors

Typical ratio: `α:β:γ:δ = 0.5:0.5:0.01:0.001`

---

## 3.8 Training Schedule Example

### For a 1B parameter LLM

```
Phase 1 (Baseline):
    Steps: 100K
    Learning rate: 3e-4
    Precision: FP32/BF16
    Hardware: 8× GPU

Phase 2 (QAT):
    Steps: 20K
    Learning rate: 1e-5 (lower)
    Schedule: cosine decay
    Tricks: 
        - Clipped STE
        - Gradual Δ increase
        - Warm-up for 500 steps

Phase 3 (Sparsity):
    Steps: 10K
    Learning rate: 1e-5
    λ_sparsity: 0.01 → 0.1 (increase slowly)

Phase 4 (Calibration):
    Steps: 1K (forward only)
    Dataset: 1024 validation examples
    Optimize: per-channel α
```

---

## 3.9 When Ternary Training Fails

Common failure modes and fixes:

| Symptom | Likely Cause | Fix |
|---------|-------------|-----|
| Accuracy drop > 10% | Δ too large | Reduce Δ, use per-channel scale |
| Model outputs NaN | Gradient explosion | Gradient clipping, lower lr |
| All weights become 0 | Δ too large, or lr too high | Lower Δ, reduce sparsity penalty |
| Only +1 and −1, no zeros | Δ too small, no sparsity | Increase Δ, add sparsity regularization |
| Training diverges | STE instability | Use clipped STE, reduce lr 10× |

---

## 3.10 Hardware-Aware Training

Simulate hardware constraints during training:

```
1. Pack ternary weights into 16-bit words
2. Simulate add/sub/skip compute
3. Model memory bandwidth savings
4. Account for zero-skip speedup

This gives realistic performance estimates.
```

### Training-Time Speed Simulation

Track:

```
Density: fraction of non-zero weights
Effective MACs: density × total_params
Skipped operations: 1 − density
```

End-to-end latency estimate:

```
Latency ≈ (effective_MACs × compute_time)
         + (non-zero_weight_transfer × bandwidth_time)
         + (activations × compute_time)
```

---

## 3.11 Recommended Training Framework

```
Software stack:

PyTorch          → training framework
TorchDistrib     → distributed training
Custom QAT       → ternary quantization layers
Triton kernels   → ternary GEMM simulation
HuggingFace      → model loading and tokenizer
```

### Minimum Viable Training Script

```python
class TernaryLinear(nn.Module):
    def __init__(self, in_features, out_features):
        super().__init__()
        self.W_shadow = nn.Parameter(torch.randn(out_features, in_features) * 0.01)
        self.scale = nn.Parameter(torch.ones(out_features))
        self.delta = 0.5

    def forward(self, x):
        W_norm = self.W_shadow / self.scale.unsqueeze(1)
        T = torch.where(W_norm > self.delta, 1.0,
            torch.where(W_norm < -self.delta, -1.0, 0.0))

        # Apply STE
        W_effective = self.scale.unsqueeze(1) * T
        W_effective = self.W_shadow + (W_effective - self.W_shadow).detach()

        return F.linear(x, W_effective, self.bias)
```