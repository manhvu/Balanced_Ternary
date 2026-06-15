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
    # Evaluate each format
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

---

## 10.6 Industry Product Comparison

| Product | Weight Format | MAC Unit | Target Power | Typical Model Size |
|---------|-------------|----------|-------------|-------------------|
| Google EdgeTPU | INT8 | INT8 MAC | 2W | 100-500 MB |
| Apple Neural Engine | INT8/FP16 | Mixed | 1-5W | 200 MB-2 GB |
| Qualcomm Hexagon | INT8 | INT8 MAC | 1-3W | 100 MB-1 GB |
| NVIDIA Jetson | FP16/INT8 | Tensor core | 10-30W | 1-5 GB |
| This design | Ternary | Add/sub/skip | 2-10W | 20-200 MB |

**Ternary accelerator targets a niche: sub-10W devices running 100M-1B parameter models entirely in on-chip SRAM.**

---

## 10.7 Related Research

| Paper | Year | Key Idea | Difference from This |
|-------|------|----------|---------------------|
| Ternary Weight Networks | 2016 | Weights ∈ {-1,0,+1}, thresholds | No HW architecture |
| Trained Ternary Quantization | 2016 | Learned scale per layer | No packing or HW |
| LUT-GEMM | 2023 | Ternary LLMs with lookup tables | Software only |
| BitNet | 2024 | 1.58-bit weights, INT8 activations | No HW accelerator |
| EdgeTPU | 2018 | INT8 systolic array | Binary, not ternary |
| NVIDIA Hopper Transformer Engine | 2022 | FP8/FP16 hybrid | Binary floating point |

**This work's novel contributions:**
1. Combined ternary quantization + dense packing + sparse encoding under one unified memory format
2. Proposed differential encoding for wire-efficient ternary computation
3. Designed a complete edge LLM accelerator around ternary weights
4. Mapped all transformer operations to the ternary + FP16 hybrid model
5. Quantified the real memory bandwidth advantage for on-device LLM

---

## 10.8 Where Ternary Fails

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

## 10.9 Decision Matrix

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
```