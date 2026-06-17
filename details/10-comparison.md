# 10. Comparison with Existing Approaches

## 10.1 Why Balanced Ternary vs Binary Neural Networks

Binary neural networks (BNN) constrain weights to `{-1, +1}`.

| Property | Binary | Balanced Ternary |
|----------|--------|-----------------|
| Bits per weight | 1 | ~1.585 |
| States | 2 | 3 |
| Zero support | No | Yes |
| Sparsity encoding | Impossible | Natural |
| XNOR-popcount hardware | Yes | Add/sub/skip |
| Typical accuracy vs FP32 | -5 to -10% | -1 to -3% |
| Model size vs FP32 | 32× smaller | 20× smaller |

**Ternary wins because of the zero state.** It allows sparsity and better accuracy.

### 10.1.1 Ternary vs Binary with Ternary Activations

Some recent work goes beyond ternary weights and also quantizes *activations* to ternary values `{-1, 0, +1}`. While this further simplifies the datapath (no multiplier needed at all), it compounds quantization errors: the output of each layer is already noisy, and feeding that noisy signal into the next ternary layer amplifies the degradation. Training such networks requires careful straight-through estimator tuning, layer-wise calibration, and often knowledge distillation from a full-precision teacher. In practice, ternary-weights-only models are a safer bet for most deployments, with ternary activations reserved for very small models or highly noise-tolerant tasks.

### 10.1.2 Pros and Cons: Balanced Ternary vs Binary

**Balanced Ternary Pros over Binary:**
- **Zero state enables sparsity**: Ternary's `0` weight means "skip this computation entirely." Binary has no such option — every weight must participate. At 50-75% sparsity, ternary does 2-4× fewer operations than binary for the same model shape.
- **Better accuracy**: The zero state acts as a built-in pruning mechanism. Near-zero weights (which are common in trained networks) map to `0` instead of being forced to `±1`. Typical accuracy loss is 1-3% vs FP32, compared to 5-10% for binary.
- **Natural representation of "unimportant"**: Many trained networks have a large fraction of near-zero weights after regularization. Ternary captures this naturally; binary cannot.
- **No sign bit needed**: Balanced ternary represents negative values inherently (`T = -1`), while binary requires a separate sign bit or two's complement.
- **Richer expressiveness per weight**: 3 states vs 2 means fewer weights are needed for the same representational capacity. A ternary layer with N weights has 3^N possible states vs 2^N for binary.

**Balanced Ternary Cons vs Binary:**
- **Higher storage**: 1.585 bits/weight vs 1 bit/weight (58.5% more storage).
- **More complex compute**: Ternary needs add/sub/skip logic vs binary's simple XNOR-popcount. The ternary PE is ~3× larger than a binary PE.
- **Less mature hardware**: Binary neural network accelerators exist (e.g., Binarized Neural Network FPGA implementations). Ternary-specific hardware is still research-stage.
- **Training complexity**: Ternary QAT requires careful delta threshold scheduling and scale factor learning, while binary networks can use simpler sign functions.
- **Diminishing returns at small models**: For very small models (<10M params), the accuracy advantage of ternary over binary shrinks, making binary's simplicity more attractive.

### Elixir: Format Comparison Benchmark

```elixir
defmodule FormatComparison do
  @type format :: :ternary | :binary | :int8 | :fp32
  @type metric :: %{format() => %{
                   accuracy: float(),
                   model_size_mb: float(),
                   mac_cost_per_op: integer()
                 }}

  @doc """
  Compare all formats head-to-head on the same dataset.
  """
  @spec compare_all(
          [{weights :: [float()], activations :: [float()]}],
          [integer()],
          float()
        ) :: metric()
  def compare_all(dataset, ground_truth, delta) do
    %{
      fp32:   evaluate(:fp32, dataset, ground_truth),
      int8:   evaluate(:int8, dataset, ground_truth),
      binary: evaluate(:binary, dataset, ground_truth),
      ternary: evaluate(:ternary, dataset, ground_truth, delta)
    }
  end

  @doc """
  Evaluate a single format.
  """
  def evaluate(:ternary, dataset, truth, delta) do
    correct = dataset
              |> Enum.zip(truth)
              |> Enum.count(fn {{w, x}, t} ->
                tern = Enum.map(w, &TernaryQuantizer.ternarize(&1, delta))
                dot = TernaryMAC.dot_product(tern, x)
                pred = if dot > 0, do: 1, else: 0
                pred == t
              end)

    %{
      accuracy: correct / length(truth),
      model_size_mb: model_size(:ternary, length(hd(Enum.map(dataset, fn {w, _} -> w end)))),
      mac_cost_per_op: 1  # add/sub/skip
    }
  end

  def evaluate(:binary, dataset, truth) do
    correct = dataset
              |> Enum.zip(truth)
              |> Enum.count(fn {{w, x}, t} ->
                bin = Enum.map(w, fn w -> if w >= 0, do: 1, else: -1 end)
                dot = TernaryMAC.dot_product(bin, x)
                pred = if dot > 0, do: 1, else: 0
                pred == t
              end)

    %{
      accuracy: correct / length(truth),
      model_size_mb: model_size(:binary, length(hd(Enum.map(dataset, fn {w, _} -> w end)))),
      mac_cost_per_op: 1  # XNOR-popcount
    }
  end

  def evaluate(:int8, dataset, truth) do
    correct = dataset
              |> Enum.zip(truth)
              |> Enum.count(fn {{w, x}, t} ->
                dot = Enum.zip(w, x)
                      |> Enum.reduce(0, fn {w_i, x_i}, acc ->
                        w8 = trunc(w_i * 127) |> max(-128) |> min(127)
                        x8 = trunc(x_i) |> max(-128) |> min(127)
                        acc + w8 * x8
                      end)
                pred = if dot > 0, do: 1, else: 0
                pred == t
              end)

    %{
      accuracy: correct / length(truth),
      model_size_mb: model_size(:int8, length(hd(Enum.map(dataset, fn {w, _} -> w end)))),
      mac_cost_per_op: 8
    }
  end

  def evaluate(:fp32, dataset, truth) do
    correct = dataset
              |> Enum.zip(truth)
              |> Enum.count(fn {{w, x}, t} ->
                dot = Enum.zip(w, x)
                      |> Enum.reduce(0, fn {w_i, x_i}, acc -> acc + w_i * x_i end)
                pred = if dot > 0, do: 1, else: 0
                pred == t
              end)

    %{
      accuracy: correct / length(truth),
      model_size_mb: model_size(:fp32, length(hd(Enum.map(dataset, fn {w, _} -> w end)))),
      mac_cost_per_op: 32
    }
  end

  @doc """
  Estimate model size in MB for 1B parameters.
  """
  defp model_size(:ternary, _n_params) do
    1_000_000_000 * 1.585 / 8 / 1024 / 1024 |> Float.round(1)
  end
  defp model_size(:binary, _n_params), do: 1_000_000_000 * 1 / 8 / 1024 / 1024 |> Float.round(1)
  defp model_size(:int8, _n_params), do:  1_000_000_000 * 8 / 8 / 1024 / 1024 |> Float.round(1)
  defp model_size(:fp32, _n_params), do: 1_000_000_000 * 32 / 8 / 1024 / 1024 |> Float.round(1)

  @doc """
  Print summary comparison table.
  """
  @spec print_summary(metric()) :: :ok
  def print_summary(metrics) do
    IO.puts("Format   | Accuracy | Size (1B) | MAC cost")
    IO.puts("---------|----------|-----------|---------")

    Enum.each([:fp32, :int8, :binary, :ternary], fn fmt ->
      m = metrics[fmt]
      IO.puts("#{pad(fmt, 7)} | " <>
              "#{pad(Float.round(m.accuracy, 4), 7)} | " <>
              "#{pad(m.model_size_mb, 8)} MB | " <>
              "#{m.mac_cost_per_op}")
    end)
  end

  defp pad(val, width) when is_atom(val), do: pad(Atom.to_string(val), width)
  defp pad(val, width) when is_float(val), do: pad(Float.to_string(val), width)
  defp pad(val, width) when is_integer(val), do: pad(Integer.to_string(val), width)
  defp pad(str, width) when is_binary(str) do
    str <> String.duplicate(" ", max(0, width - String.length(str)))
  end
end
```

---

## 10.2 Why Balanced Ternary vs INT8

INT8 is the current industry standard for inference.

| Property | INT8 | Balanced Ternary |
|----------|------|-----------------|
| Bits per weight | 8 | ~1.585 |
| Model size (1B) | 1 GB | ~200 MB |
| Multiply cost | INT8 multiplies | Add/sub/skip |
| Accuracy loss vs FP32 | ~0-0.5% | ~1-3% |
| Hardware support | GPU tensor cores | Requires custom HW |
| Ease of deployment | Everywhere | Specialized |
| Memory bandwidth need | Low | Lowest |

**Ternary wins on memory bandwidth but loses on deployment ease.** The bet is that memory bandwidth is the bigger bottleneck on edge devices.

### 10.2.1 Pros and Cons: Balanced Ternary vs INT8

**Balanced Ternary Pros over INT8:**
- **5× smaller model size**: 1.585 bits/weight vs 8 bits/weight. A 1B parameter model drops from 1 GB to ~200 MB, fitting entirely in on-chip SRAM.
- **No multiplier needed**: Ternary MAC becomes add/sub/skip — a single INT8 adder vs a full INT8 multiplier. This saves ~80% of MAC unit area and power.
- **Natural sparsity**: Zero weights skip computation entirely. INT8 has no zero-weight skip unless explicitly structured (2:4 sparsity).
- **Lower memory bandwidth**: 5× less weight data movement directly translates to 5× lower energy from memory access (often the dominant cost in edge inference).
- **Better for memory-bound workloads**: LLM decode is typically memory-bandwidth-bound, not compute-bound. Ternary's bandwidth advantage directly improves throughput.

**Balanced Ternary Cons vs INT8:**
- **Lower accuracy**: ~1-3% loss vs ~0-0.5% for INT8. The 3-state quantization is inherently more lossy than 256-state.
- **No existing hardware**: INT8 runs on every GPU, NPU, and DSP. Ternary requires custom accelerators or FPGA emulation.
- **Training overhead**: Ternary requires QAT with STE, delta scheduling, and scale factor calibration. INT8 can often use simple post-training quantization.
- **Software ecosystem**: INT8 has mature toolchains (TensorRT, ONNX Runtime, Core ML). Ternary toolchains are research-grade.
- **Numerical range**: INT8 can represent 256 distinct values per weight. Ternary can only represent 3, requiring per-channel scale factors to compensate.

> **Note on INT8 with sparsity:** Some modern accelerators (e.g., NVIDIA sparse tensor cores, Qualcomm AI engines) support *sparse INT8*, which skips zero-valued weights during compute and memory transfers. This achieves memory savings comparable to ternary packing while retaining INT8's superior per-weight accuracy. The trade-off is that sparsity must be explicitly structured (e.g., 2:4 patterns) to match hardware constraints, whereas ternary zeros are a natural byproduct of quantization.

---

## 10.3 Why Balanced Ternary vs INT4

INT4 is gaining traction for edge LLM inference.

| Property | INT4 | Balanced Ternary |
|----------|------|-----------------|
| Bits per weight | 4 | ~1.585 |
| Model size (1B) | 500 MB | ~200 MB |
| Multiply cost | INT4 multiply | Add/sub/skip |
| Accuracy vs FP32 | ~0.5-1% | ~1-3% |
| Hardware | GPU tensor cores (some) | Custom HW |

**INT4 is the nearest competitor.** Ternary has ~2.5× smaller storage but potentially higher accuracy loss. The choice depends on whether sparsity can be exploited.

### 10.3.1 Pros and Cons: Balanced Ternary vs INT4

**Balanced Ternary Pros over INT4:**

- **Simpler compute**: Add/sub/skip vs INT4 multiply-accumulate. No multiplier means smaller PE area and lower power.
- **Natural sparsity**: Zero weights are free — they skip both compute and memory access. INT4 zeros still require a multiply (by zero).
- **No grouping/scaling overhead**: INT4 typically requires per-group scale factors and zero-points, adding metadata overhead. Ternary only needs per-channel FP16 scales (~2% overhead).
- **10-100× energy advantage**: For edge devices, ternary's multiplier-free compute is dramatically more power-efficient than INT4.

**Balanced Ternary Cons vs INT4:**

- **Lower accuracy**: Ternary's 3 states are more lossy than INT4's 16 states. INT4 typically loses 0.5-1% vs FP32, ternary loses 1-3%.
- **Less hardware support**: INT4 is supported on modern GPUs (NVIDIA Hopper/Blackwell, AMD RDNA4). Ternary has no commercial hardware.
- **Training complexity**: Both require QAT, but INT4 QAT is more mature (GPTQ, AWQ). Ternary QAT with delta scheduling is less explored, though BitNet a4.58 (2024) is closing this gap.
- **Numerical precision**: INT4 can represent 16 distinct values, capturing more weight distribution detail than ternary's 3.

---

## 10.4 Why Balanced Ternary vs Structured Pruning

Structured pruning removes entire channels or heads.

| Property | Structured Pruning | Balanced Ternary Sparsity |
|----------|-------------------|--------------------------|
| Granularity | Channel/head | Element-wise |
| Sparsity pattern | Regular | Irregular |
| Hardware support | Good (dense after removal) | Poor (irregular) |
| Accuracy impact | High per removed channel | Lower (per-element) |
| Compiler support | Mature | Experimental |
| Storage saving | 30-50% | 75-90% (with packing) |

**Ternary sparsity is finer-grained but harder to exploit in hardware.** Block sparsity (32×32 blocks) bridges this gap.

### 10.4.1 Pros and Cons: Balanced Ternary vs Structured Pruning

**Balanced Ternary Pros over Structured Pruning:**
- **Higher compression**: 75-90% storage reduction with packing vs 30-50% for structured pruning.
- **Finer granularity**: Element-wise zeros vs channel-level removal. Ternary can zero out individual weights while keeping the channel structure intact.
- **No accuracy cliff**: Structured pruning can cause sudden accuracy drops when important channels are removed. Ternary's per-element quantization degrades more gracefully.
- **No retraining of architecture**: Structured pruning changes the model shape, requiring architecture-aware retraining. Ternary preserves the original model shape.
- **Composable**: Ternary can be applied on top of structured pruning for multiplicative compression.

**Balanced Ternary Cons vs Structured Pruning:**
- **Irregular sparsity**: Element-wise zeros are hard to exploit in hardware without custom sparse accelerators. Structured pruning produces regular dense matrices.
- **Compiler maturity**: Structured pruning has mature compiler support (TVM, MLIR). Ternary sparse execution requires custom runtimes.
- **Reduced model capacity**: Ternary quantization reduces the representational capacity of each weight, while structured pruning preserves full precision for remaining weights.

---

## 10.5 Why Balanced Ternary vs Unstructured Pruning

Unstructured pruning removes individual weights.

| Property | Unstructured Pruning (FP32) | Unstructured Pruning (Ternary) |
|----------|----------------------------|-------------------------------|
| Base precision | FP32 (32b) | Ternary (1.585b) |
| Storage | 32b + indices | 1.585b + indices (after packing) |
| Accuracy preservation | High | Moderate |
| Hardware compatibility | Poor (sparse GEMM) | Same (but cheaper compute) |

**Ternary + pruning = double compression.** The zero trit embeds pruning into quantization.

### 10.5.1 Ternary + Structured Pruning

Ternary quantization and structured pruning are complementary rather than competing techniques. Ternary handles *element-wise* quantization (reducing each weight from 32 bits to ~1.585 bits), while structured pruning removes *entire channels, heads, or layers* to reduce the model's shape. Applying both in sequence yields multiplicative compression: a model that is first pruned from 1B to 600M parameters and then ternarized goes from 4 GB to roughly 120 MB — a ~33× reduction. The key insight is that pruning decisions are best made in full precision (where importance scores are most reliable), after which the remaining dense weights are ternarized for deployment. This two-stage pipeline is the recommended approach for maximizing edge model density.

---

## 10.6 Why Balanced Ternary vs FP8 / Emerging Formats

FP8 (E4M3/E5M2) is the latest frontier in low-precision inference, supported by NVIDIA Hopper and later GPUs.

| Property | FP8 (E4M3) | Balanced Ternary |
|----------|------------|-----------------|
| Bits per weight | 8 | ~1.585 |
| Exponent bits | 4 | 0 |
| Mantissa bits | 3 | 0 |
| Dynamic range | ~10^±4 | Fixed ±1 (with scale) |
| Accuracy vs FP32 | ~0.1-0.5% | ~1-3% |
| Hardware | NVIDIA Hopper+ | Custom HW |
| Training support | Yes (FP8 training) | QAT only |

### 10.6.1 Pros and Cons: Balanced Ternary vs FP8

**Balanced Ternary Pros over FP8:**
- **5× smaller**: 1.585 bits vs 8 bits per weight.
- **No multiplier**: Ternary eliminates multipliers entirely; FP8 still requires FP8 MAC units (simpler than FP16 but still more complex than add/sub).
- **Natural sparsity**: Ternary zeros skip compute; FP8 zeros still participate in MAC.
- **Deterministic precision**: Ternary has uniform precision across all weights. FP8's floating-point format gives more precision to small values and less to large ones, which can be problematic for outlier weights.

**Balanced Ternary Cons vs FP8:**
- **Much lower accuracy**: FP8 loses ~0.1-0.5% vs FP32; ternary loses 1-3%. FP8's dynamic range and mantissa preserve far more information.
- **No native training**: FP8 supports native training (forward + backward). Ternary requires QAT with STE, which is less effective for training from scratch.
- **Industry momentum**: FP8 has massive industry backing (NVIDIA, AMD, Intel). Ternary has no major hardware vendor support.
- **Dynamic range**: FP8 can represent values from ~10^-4 to ~10^4. Ternary with per-channel scaling can approximate this but loses relative precision.

---

## 10.7 Comprehensive Pros and Cons Summary

### Balanced Ternary: All Pros

| # | Advantage | Impact |
|---|-----------|--------|
| 1 | **20× model compression** | Fits 1B param models in 200 MB — enables on-chip SRAM inference |
| 2 | **No multiplier in MAC** | ~80% MAC unit area/power savings vs INT8 |
| 3 | **Natural sparsity** | Zero weights skip compute and memory access for free |
| 4 | **Inherent sign representation** | No sign bit needed; T = -1 is a native state |
| 5 | **Differential encoding** | Wire-efficient 2-wire representation with noise immunity |
| 6 | **Hybrid precision compatible** | Ternary weights + FP16 activations = best of both worlds |
| 7 | **Composable with pruning** | Ternary + structured pruning = 33×+ compression |
| 8 | **Lower memory bandwidth** | 5× less data movement than INT8, 20× less than FP32 |
| 9 | **Fanless operation** | Lower TDP enables passive cooling for edge devices |
| 10 | **Deterministic compute** | No floating-point rounding variability |

### Balanced Ternary: All Cons

| # | Disadvantage | Mitigation |
|---|-------------|------------|
| 1 | **1-3% accuracy loss vs FP32** | QAT + knowledge distillation + mixed precision |
| 2 | **No commercial hardware** | FPGA prototyping path; ASIC at volume |
| 3 | **Immature software toolchain** | Extend MLIR/TVM for ternary ops |
| 4 | **Training complexity** | STE variants, delta scheduling, scale factor learning |
| 5 | **Not suitable for training** | Use FP16/BF16 for training, convert to ternary for inference |
| 6 | **Outlier sensitivity** | Per-channel scaling + weight clipping |
| 7 | **Irregular sparsity** | Block sparse formats (32×32) for hardware efficiency |
| 8 | **Small model penalty** | Accuracy loss proportionally larger for <10M param models |
| 9 | **No dynamic range** | Per-channel FP16 scales compensate |
| 10 | **Deployment risk** | Custom HW means longer time-to-market vs INT8/INT4 |

---

## 10.8 Industry Product Comparison

| Product | Weight Format | MAC Unit | Target Power | Typical Model Size |
|---------|-------------|----------|-------------|-------------------|
| Google EdgeTPU | INT8 | INT8 MAC | 2W | 100-500 MB |
| Apple Neural Engine | INT8/FP16 | Mixed | 1-5W | 200 MB-2 GB |
| Qualcomm Hexagon | INT8 | INT8 MAC | 1-3W | 100 MB-1 GB |
| NVIDIA Jetson | FP16/INT8 | Tensor core | 10-30W | 1-5 GB |
| This design | Ternary | Add/sub/skip | 2-10W | 20-200 MB |

**Ternary accelerator targets a niche: sub-10W devices running 100M-1B parameter models entirely in on-chip SRAM.**

---

## 10.9 Related Research

| Paper | Year | Key Idea | Difference from This |
|-------|------|----------|---------------------|
| Ternary Weight Networks | 2016 | Weights ∈ {-1,0,+1}, thresholds | No HW architecture |
| Trained Ternary Quantization | 2016 | Learned scale per layer | No packing or HW |
| LUT-GEMM | 2023 | Ternary LLMs with lookup tables | Software only |
| BitNet | 2024 | 1.58-bit weights, INT8 activations | No HW accelerator |
| EdgeTPU | 2018 | INT8 systolic array | Binary, not ternary |
| NVIDIA Hopper Transformer Engine | 2022 | FP8/FP16 hybrid | Binary floating point |
| SpAtten | 2021 | Sparse attention with token/head pruning | Algorithm-level sparsity, no HW |
| SparseGPT | 2023 | One-shot unstructured pruning for large language models | Pruning only, no quantization |
| AQLM | 2024 | Extreme quantization (down to 2-bit) with additive codebooks | Codebook-based, not ternary |

**This work's novel contributions:**
1. Combined ternary quantization + dense packing + sparse encoding under one unified memory format
2. Proposed differential encoding for wire-efficient ternary computation
3. Designed a complete edge LLM accelerator around ternary weights
4. Mapped all transformer operations to the ternary + FP16 hybrid model
5. Quantified the real memory bandwidth advantage for on-device LLM

---

## 10.10 Where Ternary Fails

Ternary quantization is not suitable for:

| Scenario | Why it fails |
|----------|-------------|
| Training from scratch | Too lossy as weight representation during training |
| High-precision math (softmax, layernorm) | Ternary has no exponent or mantissa |
| Very small models (<10M params) | Accuracy loss is proportionally larger |
| Models with extreme weight outliers | Outliers dominate scale, rest gets compressed poorly |
| General-purpose computing | Cannot represent arbitrary numbers |
| Arithmetic > 3 states (e.g., 4-bit) | Ternary is a strict subset |
| Low-latency batch inference (server) | Server cares more about raw TOPS than memory bandwidth |
| Audio/spectral processing | Complex numbers impossible in ternary |

This is why the proposed design is **not a ternary CPU** but a **ternary weight LLM accelerator** — it only targets the area where ternary is strong.

---

## 10.11 Decision Matrix

```
Select ternary if:
    ✓ Edge device (smartphone, IoT, drone, robot)
    ✓ Memory bandwidth is the bottleneck
    ✓ Model fits in on-chip SRAM as ternary
    ✓ Sparsity can be trained into the model
    ✓ Accuracy loss of 1-3% is acceptable
    ✓ Custom hardware is feasible

Select INT8 if:
    ✓ Server inference (HBM has plenty bandwidth)
    ✓ Accuracy loss of <0.5% required
    ✓ Need to use existing GPU/NPU hardware
    ✓ No hardware design resources
    ✓ Model is already deployed

Select INT4 if:
    ✓ Need balance of accuracy and compression
    ✓ Hardware supports INT4 tensor operations
    ✓ Model > 1B parameters, edge deployment
    ✓ Cannot tolerate 3% accuracy loss

Select binary if:
    ✓ Ultra-low power (<1W)
    ✓ Very small model (<50M params)
    ✓ Accuracy loss of 5-10% is acceptable
    ✓ Maximum hardware simplicity needed

Select FP8 if:
    ✓ Training from scratch with low precision
    ✓ Need dynamic range for outlier weights
    ✓ Have Hopper-class or newer GPU hardware
    ✓ Accuracy loss must be <0.5%
```

---

## 10.12 Total Cost of Ownership Analysis

Beyond raw performance, the economic viability of a ternary accelerator depends on the full cost stack over a product's lifetime. The table below compares a hypothetical ternary ASIC against a mid-range GPU (e.g., NVIDIA Jetson-class) for edge deployment over a 5-year product lifecycle.

| Cost Factor | Ternary ASIC (Edge) | GPU (Jetson-class) | Notes |
|-------------|--------------------|--------------------|-------|
| **Chip unit cost** | $8–15 (at 100K units) | $150–400 | ASIC benefits from smaller die (no INT8 multiplier array) |
| **Power consumption** | 2–5 W | 15–30 W | Ternary add/sub/skip uses far fewer transistors switching |
| **Power cost (5 yr)** | $2–5 | $15–30 | At $0.10/kWh, 24/7 operation |
| **Cooling** | Passive (heatsink) | Active (fan) or large heatsink | Lower TDP eliminates fan → higher reliability |
| **Board area** | 25–50 mm² die | 200–400 mm² die + DRAM | Ternary's smaller model fits in less off-chip memory |
| **DRAM requirement** | 200–500 MB (on-chip SRAM + LPDDR4) | 2–8 GB GDDR6 | Ternary model is 4–10× smaller |
| **Software development** | High initial (custom toolchain, compiler, kernels) | Low (CUDA, TensorRT, mature ecosystem) | One-time NRE cost amortized over volume |
| **NRE (one-time)** | $200K–1M (design, verification, tape-out) | $0 (off-the-shelf) | Break-even at ~50K–200K units |
| **Per-unit BOM** | $15–25 | $200–500 | Includes chip, DRAM, passives, PCB |
| **Total 5-yr cost @ 10K units** | $35–60 per device | $230–560 per device | Dominated by BOM + power |
| **Total 5-yr cost @ 100K units** | $25–40 per device | $215–530 per device | NRE amortized; ASIC advantage grows |

**Key takeaways:**
- **Below ~10K units**, a GPU is cheaper (no NRE, mature software).
- **Above ~50K units**, the ternary ASIC's lower BOM and power costs dominate, yielding 4–8× lower total cost.
- **The software gap is closing:** compiler stacks like MLIR and TVM increasingly support custom datatypes, reducing the NRE burden for ternary.
- **Reliability matters:** fanless ternary designs suit automotive, medical, and industrial environments where a GPU's cooling solution is a liability.
