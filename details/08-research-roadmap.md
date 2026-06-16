# 8. Research Roadmap

## 8.1 Phase 0: Foundation (Weeks 1-4)

### Goal
Understand balanced ternary fundamentals and build simulation infrastructure.

### Tasks

```
Week 1-2: Literature Review
    [ ] Review TTQ, TWN, DoReFa-Net papers
    [ ] Study existing ternary quantization methods
    [ ] Understand STE variants (Identity, Clipped, Hard Sigmoid)
    [ ] Review existing edge AI accelerators (EdgeTPU, NNA, NPU)

Week 3-4: Software Infrastructure
    [ ] Implement TernaryLinear PyTorch module
    [ ] Implement 5→8 packed storage format
    [ ] Build basic ternary GEMM simulation (software)
    [ ] Implement per-channel scale calibration
    [ ] Validate on MLP (MNIST test)
```

### Deliverables

```
- Working PyTorch ternary layer
- Packing/unpacking code for 5→8 and 10→16
- Accuracy benchmark on small model
- Baseline measurements (PPL, latency estimate)
```

### 8.1.1 Key Papers to Read

The following 10 papers form the essential reading list for this project:

| # | Paper | Year | Key Contribution |
|---|-------|------|------------------|
| 1 | **TTQ** (Ternary Weight Networks) | 2016 | Introduces learnable per-layer scale factors for ternary weights; foundational for all subsequent ternary quantization work. |
| 2 | **TWN** (Ternary Weight Networks, Ma et al.) | 2016 | Proposes optimal threshold Δ and per-layer scaling; shows near-ImageNet-accuracy with ternary ResNet. |
| 3 | **DoReFa-Net** | 2016 | General framework for quantizing weights, activations, and gradients to low bit-widths; introduces bit-serial GEMM concept. |
| 4 | **LUT-GEMM** | 2023 | Replaces arithmetic MAC with table lookups for ultra-low-bit inference; directly relevant to ternary weight decoding and packing. |
| 5 | **BitNet** (Microsoft) | 2024 | Demonstrates 1-bit LLM inference at scale; BitNet b1.58 uses ternary {-1,0,+1} weights, proving viability for billion-parameter models. |
| 6 | **QBERT / Q8BERT** | 2019-2020 | Shows INT8 quantization of BERT with minimal accuracy loss; establishes the mixed-precision recipe (ternary weights + higher-precision activations). |
| 7 | **SparseBERT** | 2021 | Combines pruning and quantization for BERT; demonstrates that structured sparsity + low-bit weights compound memory savings. |
| 8 | **GOBO** (Google) | 2020 | Compresses BERT weights to 3-4 bits with error compensation; practical insights on weight clustering for ternary-like formats. |
| 9 | **BinaryBERT** | 2020 | Distills BERT to binary/ternary weights; shows knowledge distillation is critical for low-bit language models. |
| 10 | **BitBlade** | 2023 | Hardware accelerator for low-bit LLM inference using bit-serial PEs; architectural template for the ternary PE array in this project. |

---

### Elixir: Phase 0 Foundation Module

```elixir
defmodule TernaryResearch do
  @moduledoc """
  Foundation module for the balanced ternary research project.
  Provides implementations for Phase 0-1: basic building blocks
  that are later composed into training and inference pipelines.
  """

  @doc """
  Simulate training loop for ternary quantization research.
  Track: accuracy, sparsity, and reconstruction error per epoch.
  """
  @spec simulate_training(
          keyword(),
          [{keyword(), [float()], [float()]}])
        :: %{String.t() => [float()]}
  def simulate_training(config, dataset) do
    learning_rate = Keyword.get(config, :learning_rate, 0.01)
    delta = Keyword.get(config, :delta, 0.5)
    n_epochs = Keyword.get(config, :epochs, 100)

    # Track metrics over time
    %{
      "accuracy" => [],
      "sparsity" => [],
      "reconstruction_error" => []
    }
  end

  @doc """
  Generate a synthetic dataset for testing ternary quantization.
  Creates random weight-activity pairs with known sparsity.
  """
  @spec synthetic_dataset(pos_integer(), pos_integer(), float()) ::
          [{weights :: [float()], activations :: [float()]}]
  def synthetic_dataset(n_samples, dim, sparsity_target) do
    Enum.map(1..n_samples, fn _ ->
      # Random weights with some zeros
      weights = Enum.map(1..dim, fn _ ->
        if :rand.uniform() < sparsity_target,
          do: 0.0,
          else: (:rand.uniform() * 2 - 1) * 0.5
      end)
      activations = Enum.map(1..dim, fn _ ->
        :rand.uniform(100) - 50
      end)

      {weights, activations}
    end)
  end

  @doc """
  Compare ternary, INT8, and FP32 across multiple metrics.
  """
  @spec benchmark_formats(
          [{weights :: [float()], activations :: [float()]}],
          float())
        :: %{String.t() => float()}
  def benchmark_formats(dataset, delta) do
    results = dataset
      |> Enum.map(fn {w, a} ->
        # FP32 reference
        fp32_dot = Enum.zip(w, a)
                  |> Enum.reduce(0, fn {w_i, a_i}, acc -> acc + w_i * a_i end)

        # Ternary
        t = Enum.map(w, &TernaryQuantizer.ternarize(&1, delta))
        ternary_dot = TernaryMAC.dot_product(t, a)

        # INT8 simulation (truncate to 8-bit range)
        int8_dot = Enum.zip(w, a)
                  |> Enum.reduce(0, fn {w_i, a_i}, acc ->
                    w8 = trunc(w_i * 127) |> max(-128) |> min(127)
                    a8 = trunc(a_i) |> max(-128) |> min(127)
                    acc + w8 * a8
                   end)

        %{
          fp32: fp32_dot,
          ternary: ternary_dot,
          int8: int8_dot
        }
      end)

    # Compute mean absolute error vs FP32
    n = length(results)
    mae_ternary = results |> Enum.map(fn %{fp32: r, ternary: t} -> abs(r - t) end) |> Enum.sum() |> Kernel./(n)
    mae_int8 = results |> Enum.map(fn %{fp32: r, int8: i} -> abs(r - i) end) |> Enum.sum() |> Kernel./(n)

    %{
      "ternary_vs_fp32_mae" => mae_ternary,
      "int8_vs_fp32_mae" => mae_int8
    }
  end
end
```

---

## 8.2 Phase 1: Small Model Validation (Weeks 5-8)

### Goal
Validate ternary quantization on small CV models.

### Tasks

```
Week 5-6: ResNet-18 / ResNet-50
    [ ] Apply ternary to all Conv layers
    [ ] Apply per-channel scaling
    [ ] Train with STE + QAT
    [ ] Compare: FP32, INT8, binary, ternary
    [ ] Measure: accuracy, sparsity, scale values

Week 7-8: Ablation Studies
    [ ] Vary Δ threshold (0.1 → 2.0)
    [ ] Vary sparsity regularization λ
    [ ] Test different STE variants
    [ ] Test per-tensor vs per-channel scale
    [ ] Report optimal configurations
```

### Deliverables

```
- ImageNet accuracy table (FP32 vs INT8 vs ternary)
- Ablation tables for each hyperparameter
- Optimal training recipe for CV ternary models
- Visualization of weight distribution before/after ternarization
```

---

## 8.3 Phase 2: Transformer Validation (Weeks 9-14)

### Goal
Apply ternary quantization to small-to-medium LLMs.

### Tasks

```
Week 9-10: BERT-base / BERT-small
    [ ] Apply ternary to all linear layers
    [ ] Keep embedding, classifier, layernorm in FP16
    [ ] Fine-tune on GLUE tasks
    [ ] Compare: FP16 vs ternary GLUE scores

Week 11-12: GPT-2 (124M)
    [ ] Apply ternary to all linear layers
    [ ] Evaluate perplexity on WikiText
    [ ] Implement KV cache (INT4)
    [ ] Measure prefill and decode throughput
    [ ] Profile per-layer sensitivity

Week 13-14: LLaMA-style 1B model
    [ ] Load pre-trained 1B model
    [ ] Apply ternary + per-channel scale
    [ ] Quantization-aware training
    [ ] Knowledge distillation from FP16 teacher
    [ ] Evaluate: PPL, zero-shot tasks
```

### Deliverables

```
- GLUE scores for ternary BERT
- Perplexity table for GPT-2 (ternary)
- Per-layer sensitivity heatmap
- Optimal mixed-precision configuration
- Working 1B ternary model
```

---

## 8.4 Phase 3: Sparsity and Efficiency (Weeks 15-18)

### Goal
Maximize sparsity and compute efficiency.

### Tasks

```
Week 15-16: Sparsity Training
    [ ] Implement sparsity regularization losses
    [ ] Train models to 50%, 75%, 85% sparsity
    [ ] Evaluate accuracy at each sparsity level
    [ ] Measure: effective MAC count vs accuracy

Week 17-18: Sparse Inference
    [ ] Implement index+sign sparse format
    [ ] Build sparse GEMM kernel (PyTorch custom op)
    [ ] Profile: latency vs sparsity
    [ ] Determine optimal per-layer sparsity target
```

### Hardware-in-the-Loop Training

During Phase 3, consider **hardware-in-the-loop training**: integrate the cycle-accurate simulator (from Phase 4, started early) directly into the training loop so that each forward/backward pass reports actual simulated latency rather than proxy metrics (FLOP count, sparsity %). This gives the optimizer a direct signal to minimize real latency, not just theoretical compute. The trade-off is ~10-100× slower training iterations, so use it for fine-tuning after initial convergence with proxy metrics.

### Deliverables

```
- Accuracy vs sparsity curve
- Sparse GEMM kernel (CUDA or CPU)
- Latency vs sparsity measurements
- Per-layer sparsity allocation strategy
```

---

## 8.5 Phase 4: Hardware Simulation (Weeks 19-24)

### Goal
Build cycle-accurate simulator for the ternary accelerator.

### Tasks

```
Week 19-20: Simulator
    [ ] Implement ternary PE array model (Verilator/Python)
    [ ] Implement decoder (5→8, 10→16)
    [ ] Implement systolic dataflow
    [ ] Implement zero-skip clock gating

Week 21-22: Memory Model
    [ ] Model weight SRAM (capacity, bandwidth)
    [ ] Model scratchpad SRAM
    [ ] Model DRAM controller
    [ ] Simulate: prefill, decode phases

Week 23-24: Power Model
    [ ] Estimate per-operation energy
    [ ] Model SRAM read/write energy
    [ ] Estimate total power per token
    [ ] Compare: vs FPGA, GPU, ASIC baselines
```

### Deliverables

```
- Cycle-accurate simulator (open-source)
- Simulated latency: prefill, decode
- Simulated power: peak and average
- RTL-level power estimation file
```

---

## 8.6 Phase 5: Hardware Prototype (Weeks 25-36)

### Goal
FPGA prototype of the ternary accelerator.

### Tasks

```
Week 25-28: RTL Design
    [ ] Write Verilog/VHDL for ternary PE
    [ ] Write decoder module
    [ ] Write accumulator + scale unit
    [ ] Write weight SRAM controller

Week 29-32: Integration
    [ ] Integrate all modules
    [ ] Interface with host via PCIe/AXI
    [ ] Build software driver
    [ ] Write compiled model loader

Week 33-36: FPGA Test
    [ ] Deploy on FPGA board (Xilinx/Altera)
    [ ] Run small ternary model (ResNet-18)
    [ ] Measure: accuracy, latency, power
    [ ] Iterate on design
```

### Deliverables

```
- RTL codebase (Verilog)
- FPGA bitstream
- FPGA measurements (accuracy, latency, power)
- Driver software
- Hardware-in-the-loop demo
```

---

## 8.7 Phase 6: ASIC Pathfinding (Weeks 37-48)

### Goal
Evaluate ASIC feasibility and produce tape-out-ready specifications.

### Tasks

```
Week 37-40: Floorplan & Area
    [ ] Estimate PE array area
    [ ] Estimate SRAM area
    [ ] Estimate decoder area
    [ ] Total die area estimate (12nm, 7nm)

Week 41-44: Synthesis
    [ ] Synthesize RTL to target process
    [ ] Report: max clock frequency, power
    [ ] Report: area breakdown
    [ ] Optimize critical paths

Week 45-48: Tape-Out Package
    [ ] Final specification document
    [ ] Power-performance-area (PPA) numbers
    [ ] Software stack (compiler, runtime)
    [ ] Test plan
    [ ] Die photo mockup
```

### Deliverables

```
- ASIC specification
- PPA table (performance, power, area)
- Floorplan diagram
- Software compiler and runtime
- Tape-out-ready RTL
```

---

### 8.7.1 Alternative: FPGA Product

If ASIC tape-out cost ($2-5M at 7nm) is prohibitive, a standalone **FPGA product** is a viable alternative:

- **Xilinx Versal AI Core (VC1902)** or **Intel Agilex 7 (AGF014)** can host a ternary PE array of 256-512 PEs.
- The PE array maps naturally to FPGA LUTs + DSP slices; the decoder (5→8, 10→16) is pure combinational logic.
- **Main limitation: SRAM capacity.** On-chip BRAM on a large FPGA is ~30-50 MB, enough for a ~1B parameter ternary model (1 bit per weight + scale factors) but not for 7B+ without external HBM.
- External HBM2e (available on Versal HBM variants) removes this ceiling, supporting up to 32 GB.
- An FPGA product can ship within 12-18 months of a working RTL prototype, vs. 24-36 months for ASIC.

---

## 8.8 Resource Requirements

| Phase | Compute | Memory | People | Duration |
|-------|---------|--------|--------|----------|
| 0: Foundation | 1 GPU | 32 GB | 1-2 engineers | 4 weeks |
| 1: Small Model | 4 GPUs | 64 GB | 2 engineers | 4 weeks |
| 2: Transformer | 8 GPUs | 128 GB | 3-4 engineers | 6 weeks |
| 3: Sparsity | 8 GPUs | 128 GB | 2 engineers | 4 weeks |
| 4: Simulator | CPU cluster | 64 GB | 2 HW engineers | 6 weeks |
| 5: FPGA | FPGA board | - | 2-3 HW engineers | 12 weeks |
| 6: ASIC | EDA tools | - | 3-4 ASIC engineers | 12 weeks |

---

## 8.9 Risk Factors

| Risk | Likelihood | Impact | Mitigation |
|------|-----------|--------|------------|
| Accuracy loss > 5% | Medium | High | Mixed-precision fallback, larger teacher |
| Training instability | Low | Medium | Clipped STE, lower LR, gradient clipping |
| FPGA resource exhaustion | Medium | Medium | Reduce PE count, use 5→8 packing |
| Power > 10W target | Medium | Medium | Aggressive clock gating, voltage scaling |
| No significant accuracy improvement over INT8 | Medium | Low | Focus on memory bandwidth advantage |
| Hardware bug in PE array | Low | High | Extensive simulation before tape-out |
| Quantization-aware training too slow | Medium | Medium | Use fewer calibration steps, freeze non-sensitive layers early |
| Sparsity not hardware-exploitable | Medium | High | Use block sparsity (e.g., 4×4 or 8×8 blocks) to align with PE array dataflow |
| Compiler toolchain complexity | Medium | Medium | Extend TVM/MLIR with ternary dialect rather than building from scratch |

---

## 8.10 Success Criteria

### Minimum Viable Product (Phase 2)

```
- Model  : 1B parameter LLM
- Accuracy: PPL within +3 of FP16
- Sparsity: ≥50%
- Inference: functional (software)
```

### Target Product (Phase 5-6)

```
- Model    : 1B parameter LLM
- Accuracy : PPL within +2 of FP16
- Sparsity : ≥75%
- Latency  : ≤100 µs/token (decode)
- Power    : ≤5W (chip)
- Hardware : FPGA or ASIC prototype
```

### Stretch Goal

```
- Model    : 7B parameter LLM
- Accuracy : PPL within +1 of FP16
- Sparsity : ≥80%
- Latency  : ≤50 µs/token (decode)
- Power    : ≤10W
- Silicon  : Tape-out
```

---

## 8.11 Publication Strategy

Target the following venues based on the type of results available:

| Venue | Type | What to Target |
|-------|------|----------------|
| **ISCA** (International Symposium on Computer Architecture) | Architecture | Full ternary accelerator architecture: PE array, decoder, dataflow, and ASIC PPA numbers. Requires hardware results (FPGA or ASIC). |
| **MICRO** (International Symposium on Microarchitecture) | Architecture | Microarchitectural innovations: zero-skip clock gating, 5→8 packing, sparse GEMM kernel. Requires cycle-accurate simulation + RTL. |
| **NeurIPS** (Conference on Neural Information Processing Systems) | ML | Ternary training algorithm: novel STE variant, sparsity regularization, or QAT recipe that achieves SOTA accuracy on ImageNet/LLM benchmarks. |
| **ICML** (International Conference on Machine Learning) | ML | Theoretical contributions: convergence analysis of ternary optimization, information-theoretic bounds on ternary weight capacity. |
| **ISSCC / VLSI Symposium** | Circuit | Circuit-level ternary PE design: low-power adder tree, decoder circuit, SRAM macros. Requires silicon or detailed post-layout results. |
| **FPGA** (ACM/SIGDA International Symposium on FPGA) | Demo | Working FPGA prototype demo: live inference of ternary LLM on FPGA with latency/power measurements. |

**Recommended publication sequence:**
1. **Year 1:** Submit ternary training method to NeurIPS or ICML (Phases 1-2 results).
2. **Year 2:** Submit accelerator architecture to ISCA or MICRO (Phases 4-5 results).
3. **Year 2-3:** Submit FPGA demo paper to FPGA symposium (Phase 5 results).
4. **Year 3:** Submit ASIC PPA to ISSCC/VLSI (Phase 6 results).