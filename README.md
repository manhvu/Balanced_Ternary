# Balanced Ternary for Neural Network Inference

A comprehensive exploration of balanced ternary (`{-1, 0, +1}`) as a weight representation for neural network accelerators — spanning quantization theory, training methodology, memory packing, hardware architecture, LLM inference engine design, and a 48-week research roadmap.

## Why Balanced Ternary

| Property | FP32 | INT8 | Ternary |
|----------|------|------|---------|
| Bits per weight | 32 | 8 | ~1.585 |
| Model size (1B params) | 4 GB | 1 GB | ~200 MB |
| Multiply cost | Full FP32 MUL | INT8 MUL | Add/sub/skip |
| Sparsity support | No | No | Natural (zero weight) |

The zero trit is the key advantage: it encodes sparsity directly, turning near-zero weights into skipped computation.

## Contents

| File | Topic |
|------|-------|
| [`details/01-architecture-overview.md`](details/01-architecture-overview.md) | Hybrid precision model, layer mapping, data flow |
| [`details/02-weight-quantization.md`](details/02-weight-quantization.md) | Quantizer design, threshold selection, per-channel scaling |
| [`details/03-training.md`](details/03-training.md) | Training pipeline, STE, sparsity regularization, distillation |
| [`details/04-storage-format.md`](details/04-storage-format.md) | 5→8 / 10→16 packing, sparse encoding, hybrid storage |
| [`details/05-hardware-architecture.md`](details/05-hardware-architecture.md) | Systolic PE array, decoder, memory hierarchy, power estimates |
| [`details/06-llm-inference-engine.md`](details/06-llm-inference-engine.md) | Prefill/decode, KV cache, throughput estimation |
| [`details/07-accuracy-analysis.md`](details/07-accuracy-analysis.md) | Accuracy benchmarks, sensitivity analysis, ablation studies |
| [`details/08-research-roadmap.md`](details/08-research-roadmap.md) | 48-week roadmap from software to ASIC tape-out |
| [`details/09-differential-encoding.md`](details/09-differential-encoding.md) | Two-wire differential trit encoding, wire-swap negation |
| [`details/10-comparison.md`](details/10-comparison.md) | Comparison with BNN, INT8, INT4, pruning; decision matrix |

## Target Application

A **balanced-ternary transformer inference accelerator** for edge devices — 2–10 W, 100M–1B parameter models fitting entirely in on-chip SRAM, with throughput of ~20K tokens/s decode.

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
