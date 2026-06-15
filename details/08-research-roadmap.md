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