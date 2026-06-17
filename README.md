# Balanced Ternary for Neural Network Inference

A comprehensive exploration of balanced ternary (`{-1, 0, +1}`) as a weight representation for neural network accelerators — spanning quantization theory, training methodology, memory packing, hardware architecture, LLM inference engine design, model conversion tools, and a 48-week research roadmap.

## Why Balanced Ternary

| Property | FP32 | INT8 | Ternary |
|----------|------|------|---------|
| Bits per weight | 32 | 8 | ~1.585 |
| Model size (1B params) | 4 GB | 1 GB | ~200 MB |
| Multiply cost | Full FP32 MUL | INT8 MUL | Add/sub/skip |
| Sparsity support | No | No | Natural (zero weight) |

The zero trit is the key advantage: it encodes sparsity directly, turning near-zero weights into skipped computation.

## Contents

### Concept & Architecture

| File | Topic |
|------|-------|
| [`concept.md`](concept.md) | High-level overview, storage efficiency, packing, hybrid architecture, LLM accelerator vision |
| [`details/01-architecture-overview.md`](details/01-architecture-overview.md) | Hybrid precision model, layer mapping, data flow, scaling strategies, software stack |
| [`details/05-hardware-architecture.md`](details/05-hardware-architecture.md) | Systolic PE array, decoder, memory hierarchy, power estimates, ISA, verification |
| [`details/06-llm-inference-engine.md`](details/06-llm-inference-engine.md) | Prefill/decode, KV cache, throughput estimation, batch processing, energy analysis |

### Quantization & Training

| File | Topic |
|------|-------|
| [`details/02-weight-quantization.md`](details/02-weight-quantization.md) | Quantizer design, threshold selection, per-channel scaling, QAT, error analysis |
| [`details/03-training.md`](details/03-training.md) | Training pipeline, STE variants, sparsity regularization, distillation, distributed training |
| [`details/07-accuracy-analysis.md`](details/07-accuracy-analysis.md) | Benchmarks, sensitivity analysis, ablation studies, reproducibility checklist |

### Storage & Encoding

| File | Topic |
|------|-------|
| [`details/04-storage-format.md`](details/04-storage-format.md) | 5→8 / 10→16 packing, sparse encoding, hybrid storage, bandwidth calculation |
| [`details/09-differential-encoding.md`](details/09-differential-encoding.md) | Two-wire differential trit encoding, wire-swap negation, CMOS compatibility, DFT |

### Hardware & Deployment

| File | Topic |
|------|-------|
| [`details/11-current-hardware-gpu-cpu-npu.md`](details/11-current-hardware-gpu-cpu-npu.md) | Running ternary models on GPU, CPU, NPU, FPGA today — CUDA kernels, AMX, HVX, Vulkan |
| [`details/12-custom-ternary-accelerator-design.md`](details/12-custom-ternary-accelerator-design.md) | Full ASIC architecture spec: PE design, memory subsystem, ISA, compiler, cost analysis |

### Conversion & Tools

| File | Topic |
|------|-------|
| [`details/13-model-conversion-guide.md`](details/13-model-conversion-guide.md) | PTQ/QAT pipelines, model-specific recipes, validation suite, Elixir conversion toolchain |
| [`tools/ternary_converter/`](tools/ternary_converter/) | Full Elixir application: convert, demo, info, validate CLI + library API |

### Feasibility Analysis

| File | Topic |
|------|-------|
| [`details/14-ternary-llm-feasibility.md`](details/14-ternary-llm-feasibility.md) | Ternary for trillion-parameter LLMs: scaling laws, MoE, cost analysis |
| [`details/15-ternary-vision-computing.md`](details/15-ternary-vision-computing.md) | Ternary for vision: CNNs, ViT, detection, segmentation, edge deployment |
| [`details/16-fpga-experiment-guide.md`](details/16-fpga-experiment-guide.md) | FPGA experiment guide: board selection, RTL design, benchmarking |
| [`details/17-asic-implementation-guide.md`](details/17-asic-implementation-guide.md) | ASIC implementation guide: RTL design, physical design, DFT, manufacturing |

### Comparison & Roadmap

| File | Topic |
|------|-------|
| [`details/10-comparison.md`](details/10-comparison.md) | Detailed comparison with BNN, INT8, INT4, FP8, pruning — pros/cons, decision matrix, TCO |
| [`details/08-research-roadmap.md`](details/08-research-roadmap.md) | 48-week roadmap from software simulation to ASIC tape-out, publication strategy |

### Examples

| Directory | Content |
|-----------|---------|
| [`examples/ternary_pure/`](examples/ternary_pure/) | Pure Elixir demo: quantize → pack → forward pass → differential encoding |
| [`examples/ternary_nx/`](examples/ternary_nx/) | Nx-based demo: tensor quantization → GEMM → sparsity analysis → packing |

## Target Application

A **balanced-ternary transformer inference accelerator** for edge devices — 2–10 W, 100M–1B parameter models fitting entirely in on-chip SRAM, with throughput of ~50K tokens/s decode.

## Quick Start

### Run the Elixir conversion tool

```bash
cd tools/ternary_converter
mix deps.get

# Run a full demo with synthetic weights
mix run -e "TernaryConverter.CLI.main([\"demo\"])"

# Convert to .tbin format
mix run -e "TernaryConverter.CLI.main([\"convert\", \"--delta\", \"0.5\", \"--output\", \"model.tbin\"])"

# Inspect a .tbin model
mix run -e "TernaryConverter.CLI.main([\"info\", \"--model\", \"model.tbin\"])"
```

### Run the pure Elixir example

```bash
cd examples/ternary_pure
mix deps.get
mix run -e "TernaryPure.run_demo()"
```

### Run the Nx-based example

```bash
cd examples/ternary_nx
mix deps.get
mix run -e "TernaryNx.run_demo()"
```

## License

```
Copyright 2026 Balanced Ternary Project

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
```
