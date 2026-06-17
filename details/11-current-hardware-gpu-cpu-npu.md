# 11. Running Ternary Models on Current Hardware (GPU, CPU, NPU, FPGA)

## 11.1 Overview

Purpose-built ternary accelerators do not yet exist as production hardware. This guide covers how to deploy and accelerate ternary neural networks on commercially available hardware today — GPUs, CPUs, NPUs, and FPGAs — while outlining the gaps that a custom ternary ASIC would fill.

The fundamental challenge: **current hardware is designed for binary arithmetic** (INT8/FP16/FP32), so ternary's core advantage (add/sub/skip instead of multiply) is not natively supported. However, ternary's storage and bandwidth advantages can still be exploited through software and clever packing.

---

## 11.2 GPU Deployment

### 11.2.1 NVIDIA GPUs (Ampere → Hopper → Blackwell)

NVIDIA GPUs are the most capable platform for ternary inference today, thanks to their mature software stack and high memory bandwidth.

**Hardware Evolution:**

| GPU | Year | INT8 TOPS | HBM BW | Notable |
|-----|------|-----------|--------|--------|
| A100 (Ampere) | 2020 | 624 | 2.0 TB/s | First-gen tensor core INT8 |
| H100 (Hopper) | 2022 | 3952 | 3.35 TB/s | FP8 transformer engine |
| H200 (Hopper) | 2024 | 3952 | 4.8 TB/s | 141 GB HBM3e |
| B100/B200 (Blackwell) | 2024 | 9000+ | 8.0 TB/s | FP4 support, 192 GB HBM3e |

**Approach: Ternary Weights + INT8 Activations via cuBLAS/TensorRT**

The most practical path is to store weights in packed ternary format (10 trits per 16-bit word) in GPU memory, then unpack to INT8 at runtime and use INT8 tensor cores for the GEMM:

```
Packed ternary weights (200 MB for 1B params)
    ↓  Unpack kernel (custom CUDA)
INT8 weight matrix (600 MB for 1B params, with scale factors)
    ↓  INT8 GEMM (cuBLAS / TensorRT)
Output activations
```

**Key insight:** Unpacking 10 trits from 16 bits to 10 INT8 values is a memory-bandwidth-bound operation. On an NVIDIA A100 (2 TB/s HBM bandwidth), unpacking 200 MB of packed weights takes ~0.1 ms — negligible compared to the GEMM itself.

**Custom CUDA Kernel for Ternary Unpack + GEMM:**

```cuda
// Each thread block handles one output channel
// Unpack 10-trit words on-the-fly from shared memory
__global__ void ternary_gemm_kernel(
    const uint16_t* __restrict__ packed_weights,  // 10 trits per 16-bit word
    const int8_t*   __restrict__ activations,
    const float*    __restrict__ scales,           // per-channel FP16 scales
    float*          __restrict__ output,
    int out_channels, int in_channels
) {
    int oc = blockIdx.x;
    float sum = 0.0f;
    float alpha = __half2float(scales[oc]);

    for (int i = 0; i < in_channels; i += 10) {
        uint16_t word = packed_weights[oc * (in_channels / 10) + i / 10];

        // Unpack 10 trits (base-3 extraction)
        #pragma unroll
        for (int k = 0; k < 10 && (i + k) < in_channels; k++) {
            int digit = word % 3;
            word /= 3;
            int trit = digit - 1;  // 0→-1, 1→0, 2→+1

            if (trit != 0) {
                int8_t x = activations[i + k];
                sum += (trit == 1) ? (float)x : (float)(-x);
            }
        }
    }

    output[oc] = alpha * sum;
}
```

**Performance Estimates (1B parameter model, decode latency):**

| GPU | Weight Unpack | INT8 GEMM | Total | Power | Efficiency |
|-----|---------------|-----------|-------|-------|------------|
| A100 | ~0.1 ms | ~1.2 ms | ~45 ms | 300W | 1.5 ms/J |
| H100 | ~0.06 ms | ~0.4 ms | ~15 ms | 350W | 4.3 ms/J |
| H200 | ~0.04 ms | ~0.3 ms | ~10 ms | 400W | 5.0 ms/J |
| B200 | ~0.03 ms | ~0.15 ms | ~5 ms | 500W | 10 ms/J |

Compare with FP16: A100 ~30 ms, H100 ~10 ms. The ternary unpack overhead adds ~50% latency on A100 but only ~10% on H100/B200 due to faster memory.

**Triton kernel for fused unpack + GEMM (PyTorch 2.0+):**

```python
import triton
import triton.language as tl

@triton.jit
def ternary_gemm_kernel(
    packed_weights, activations, scales, output,
    M, N, BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr
):
    pid = tl.program_id(0)
    offs_m = pid * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = tl.arange(0, BLOCK_N)
    
    # Load packed words (10 trits each)
    packed_ptrs = packed_weights + offs_m * (N // 10) + offs_n // 10
    word = tl.load(packed_ptrs)
    
    # Extract trit via base-3 digit
    trit_idx = offs_n % 10
    digit = (word // (3 ** trit_idx)) % 3
    w = digit - 1  # 0→-1, 1→0, 2→+1
    
    # Load activation and compute add/sub/skip
    act = tl.load(activations + offs_n)
    acc = tl.sum(w * act, axis=0)
    
    # Apply scale
    alpha = tl.load(scales + offs_m)
    tl.store(output + offs_m, alpha * acc)
```

### 11.2.2 AMD GPUs (RDNA 3/4, CDNA 3)

AMD GPUs have improved INT8 support across generations:

| GPU | Year | INT8 TOPS | HBM BW | Notable |
|-----|------|-----------|--------|--------|
| MI250X (CDNA 2) | 2021 | 383 | 3.2 TB/s | Matrix cores, ROCm |
| MI300X (CDNA 3) | 2023 | 2610 | 5.3 TB/s | 192 GB HBM3, FP8 |
| MI350 (CDNA 4) | 2025 | ~5000+ | 6.0 TB/s | FP4 support expected |
| RX 7900 XTX (RDNA 3) | 2022 | 1230 | 960 GB/s | Consumer, limited INT8 |
| RX 9070 XT (RDNA 4) | 2025 | ~2000 | 1.5 TB/s | Improved AI accelerators |

**Approach:** Same unpack-then-GEMM strategy, using ROCm's `rocBLAS` INT8 GEMM or HIP kernels. The MI300X is competitive with H100 for ternary inference due to its 192 GB HBM3 capacity.

**Key advantage for MI300X:** 192 GB HBM3 can hold a 7B ternary model entirely in memory (11 GB weights + scales), enabling single-device inference without model parallelism.

### 11.2.3 Mobile GPUs (ARM Mali, Qualcomm Adreno, Apple GPU)

Mobile GPUs are memory-bandwidth-constrained (shared LPDDR), making ternary's bandwidth advantage highly relevant.

| GPU | INT8 Support | BW | Best For |
|-----|-------------|-----|----------|
| Apple A17 Pro | Limited (ANE preferred) | 100 GB/s | iOS deployment |
| Adreno 750 (SD 8 Gen 3) | Yes (via HVX) | 77 GB/s | Android flagship |
| Mali-G720 (Dimensity 9300) | Partial | 68 GB/s | MediaTek devices |

**Approach:** Use GPU compute shaders (Vulkan/OpenCL) to implement the ternary add/sub/skip directly. The 20× weight reduction means fewer memory accesses, which is the dominant cost on mobile.

**Example (Vulkan compute shader):**

```glsl
#version 450
layout(local_size_x = 256) in;

layout(std430, binding = 0) readonly buffer PackedWeights {
    uint packed[];  // 10 trits per 16-bit word (packed in uint16)
};

layout(std430, binding = 1) readonly buffer Activations {
    int activations[];
};

layout(std430, binding = 2) readonly buffer Scales {
    float scales[];
};

layout(std430, binding = 3) writeonly buffer Output {
    float output[];
};

void main() {
    uint oc = gl_GlobalInvocationID.x;
    float sum = 0.0;
    float alpha = scales[oc];
    uint n_words = uint(activations.length()) / 10u;

    for (uint w = 0u; w < n_words; w++) {
        uint word = packed[oc * n_words + w];
        uint base_idx = w * 10u;

        for (uint k = 0u; k < 10u; k++) {
            uint digit = word % 3u;
            word /= 3u;
            if (digit == 2u) sum += float(activations[base_idx + k]);
            else if (digit == 0u) sum -= float(activations[base_idx + k]);
            // digit == 1 → skip (zero)
        }
    }

    output[oc] = alpha * sum;
}
```

---

## 11.3 CPU Deployment

### 11.3.1 x86-64 (Intel/AMD Server CPUs)

Modern x86 CPUs with AMX (Advanced Matrix Extensions) can run ternary models efficiently by unpacking to INT8.

| CPU | Year | AMX INT8 TOPS | TDP | Notable |
|-----|------|---------------|-----|--------|
| Intel Xeon w9-3595X | 2024 | ~2.0 | 350W | 60 cores, AMX |
| Intel Xeon 6 (Granite Rapids) | 2024 | ~3.0 | 350W | Improved AMX |
| AMD EPYC 9755 (Turin) | 2024 | ~1.5 | 500W | 192 cores, AVX-512 |

**Approach: Packed Ternary → AMX INT8 GEMM**

Intel Sapphire Rapids+ support AMX with INT8 throughput of up to 2048 INT8 ops/cycle per tile. The key is using `_tile_dpbusd` (dot product of uint8 × int8 → int32) after unpacking ternary weights to INT8.

**Performance Estimates (1B parameter model, decode latency):**

| CPU | Ternary Unpack | AMX GEMM | Total | Power | Efficiency |
|-----|---------------|----------|-------|-------|------------|
| Xeon w9-3595X | ~0.1 ms | ~3 ms | ~100 ms | 350W | 0.29 ms/J |
| EPYC 9755 | ~0.15 ms | ~4 ms | ~130 ms | 500W | 0.26 ms/J |

**Key insight:** CPUs are 3-5× slower than GPUs for ternary inference but use 2-3× less power. For edge servers with power constraints, CPUs may be the better choice.

### 11.3.2 ARM CPUs (Cortex-A, Apple M-series, AWS Graviton)

ARM NEON and SVE2 provide SIMD dot-product instructions that can accelerate the unpacked ternary GEMM.

**Apple M-series (M2/M3/M4/M5):**
- M2/M3: NEON 128-bit SIMD, 16 INT8 ops/cycle
- M4/M5: Enhanced NEON with 2× INT8 throughput, 48 GB/s unified memory
- Unified memory architecture means no CPU-GPU transfer overhead
- Practical for models up to ~2B parameters (fits in 24-32 GB unified memory)

| Chip | INT8 Ops/Cycle | Memory BW | Max Ternary Model |
|------|---------------|-----------|-------------------|
| M2 | 16 | 100 GB/s | ~500M |
| M3 | 16 | 100 GB/s | ~700M |
| M4 | 32 | 120 GB/s | ~1.5B |
| M5 (projected) | 48 | 150 GB/s | ~2B |

```c
// ARM NEON ternary unpack + accumulate
int32x4_t ternary_dot_product_neon(
    uint16_t* packed_row,
    int8_t* activations,
    int n_trits
) {
    int32x4_t sum = vdupq_n_s32(0);

    for (int i = 0; i < n_trits; i += 10) {
        uint16_t word = packed_row[i / 10];

        // Extract 10 trits
        for (int k = 0; k < 10; k++) {
            int trit = (word % 3) - 1;
            word /= 3;

            if (trit != 0) {
                int8x16_t x = vld1q_s8(activations + i + k);
                // Broadcast single activation, multiply by ±1
                int8_t val = activations[i + k];
                int16x8_t product = vmulq_n_s16(vmovl_s8(vdup_n_s8(val)), trit);
                sum = vpadalq_s16(sum, product);
            }
        }
    }

    return sum;
}
```

**Performance Estimates (1B parameter model, decode latency):**

| Chip | Decode Latency | Power | Tokens/Joule |
|------|---------------|-------|-------------|
| M2 | ~300-400 ms | 8-12W | ~2-3 |
| M3 | ~250-350 ms | 8-12W | ~3-4 |
| M4 | ~150-200 ms | 10-15W | ~5-7 |
| M5 (projected) | ~100-150 ms | 10-15W | ~7-10 |

### 11.3.3 RISC-V (SiFive, Alibaba Xuantie)

RISC-V is relevant because the proposed ternary accelerator uses a RISC-V host CPU. The RISC-V Vector extension (V-extension) can accelerate ternary unpack + INT8 dot product similarly to ARM NEON.

**Key advantage:** RISC-V cores are small and power-efficient, leaving more die area for the ternary GEMM array in a custom SoC.

---

## 11.4 NPU Deployment

### 11.4.1 Qualcomm Hexagon (Snapdragon 8 Gen 3/4)

Qualcomm's Hexagon NPU has evolved significantly:

| NPU | Year | INT8 TOPS | Notable |
|-----|------|-----------|--------|
| Hexagon (SD 8 Gen 2) | 2022 | 26 | First with INT8 matrix |
| Hexagon (SD 8 Gen 3) | 2023 | 45 | Improved HVX |
| Hexagon (SD 8 Elite) | 2024 | 75 | 2× INT8 throughput |

**Architecture:** Hexagon uses a scalar + HVX (Hexagon Vector eXtensions) architecture. HVX can do 128 INT8 multiplies/cycle.

**Performance Estimates (1B parameter model):**

| NPU | Decode Latency | Power | Model Fit |
|-----|---------------|-------|----------|
| SD 8 Gen 3 | ~150-250 ms | 3-5W | Needs off-chip DRAM |
| SD 8 Elite | ~80-120 ms | 4-6W | Partially in SRAM |

**Key advantage:** On-device LLM inference without cloud connectivity. Ternary's 20× weight reduction enables 1B models to run on flagship smartphones.

### 11.4.2 Apple Neural Engine (ANE)

The Apple Neural Engine is a proprietary matrix unit supporting INT8 and FP16.

| ANE | Year | INT8 TOPS | Notable |
|-----|------|-----------|--------|
| ANE 16-core (A16) | 2022 | 17 | iPhone 14 Pro |
| ANE 16-core (A17 Pro) | 2023 | 35 | iPhone 15 Pro |
| ANE 16-core (M4) | 2024 | 38 | iPad Pro, MacBook |

**Approach:** Convert ternary model to INT8 in Core ML format. The ANE will run INT8 GEMM natively. The ternary advantage comes from the smaller model size (200 MB vs 1 GB for INT8), reducing memory bandwidth.

**Limitation:** The ANE's internal programming model is opaque. Custom ternary operations are not possible; you get whatever Core ML's quantization passes produce.

**Practical path:** Use `coremltools` to convert a ternary model (with weights dequantized to INT8) to Core ML format. The ANE handles the INT8 GEMM natively, while the CPU/GPU handles attention and other operations.

### 11.4.3 Google EdgeTPU and Axion

Google has expanded its AI accelerator portfolio:

| Accelerator | Year | INT8 TOPS | Power | Target |
|-------------|------|-----------|-------|--------|
| EdgeTPU (Coral) | 2019 | 4 | 2W | IoT/Edge |
| EdgeTPU (2nd gen) | 2023 | 13 | 3W | Edge servers |
| Google Axion (Arm-based) | 2024 | ~100 | 50W | Cloud/Edge |

**Approach:** Same as ANE — convert ternary to INT8, let the accelerator handle the INT8 GEMM. The ternary packing is only used for storage/compute-offload reduction.

**Performance (1B parameter model):**

| Accelerator | Decode Latency | Power |
|-------------|---------------|-------|
| EdgeTPU (2nd gen) | ~400-600 ms | 3W |
| Axion | ~50-80 ms | 50W |

**Note:** The EdgeTPU's 4 MB on-chip SRAM can only hold ~20M ternary parameters, requiring aggressive tiling for larger models. The Axion's higher memory capacity makes it more practical for 1B+ models.

### 11.4.4 Intel NPU / AMD XDNA / MediaTek APU

| NPU | Year | INT8 TOPS | Platform | Notable |
|-----|------|-----------|----------|--------|
| Intel Meteor Lake NPU | 2023 | 10 | Laptop | First Intel NPU |
| Intel Lunar Lake NPU | 2024 | 48 | Laptop | 4× improvement |
| AMD XDNA (Ryzen AI) | 2023 | 10 | Laptop | Ryzen 7040 series |
| AMD XDNA 2 | 2024 | 50 | Laptop | Ryzen AI 300 series |
| MediaTek Dimensity 9300 APU | 2024 | 46 | Mobile | On-device LLM |

All follow the same pattern: convert ternary to INT8, deploy via ONNX Runtime or vendor SDK.

**Key insight:** These NPUs are designed for low-power on-device inference. Ternary's 20× weight reduction is particularly valuable here, as it can reduce memory bandwidth requirements enough to enable on-device LLMs that would otherwise require too much DRAM access.

---

## 11.5 FPGA Deployment

FPGAs are the most flexible platform for ternary acceleration because they allow custom datapaths. This is the recommended path for prototyping a ternary accelerator before ASIC tape-out.

### 11.5.1 Why FPGA for Ternary?

| Feature | GPU/NPU | FPGA | Ternary ASIC |
|---------|---------|------|-------------|
| Custom datapath | No (fixed function) | Yes | Yes |
| Ternary add/sub/skip | Emulated | Native | Native |
| Packed ternary decode | Software | Hardware | Hardware |
| Power efficiency | Medium | Good | Best |
| Development time | Hours | Weeks | Months |
| Unit cost | Low | High | Low (volume) |

### 11.5.2 Recommended FPGA Boards

> For a comprehensive board comparison with resource estimates, selection criteria, and pricing, see §16.3 of [16-fpga-experiment-guide.md](16-fpga-experiment-guide.md).

### 11.5.3 FPGA Architecture for Ternary Acceleration

```
┌─────────────────────────────────────────────────┐
│ FPGA Ternary Accelerator                         │
│                                                  │
│  ┌──────────┐    ┌──────────────────────────┐   │
│  │ ARM Host │    │ Ternary Compute Pipeline  │   │
│  │ (Zynq PS)│───►│                          │   │
│  │          │    │  ┌────────┐  ┌─────────┐ │   │
│  │ Model    │    │  │Trit    │  │ PE Array │ │   │
│  │ Loader   │    │  │Decoder │──►│(add/sub/ │ │   │
│  │ Scheduler│    │  │(10→16) │  │ skip)    │ │   │
│  └──────────┘    │  └────────┘  └────┬────┘ │   │
│       │          │                   │      │   │
│       │          │  ┌────────────────▼────┐ │   │
│       │          │  │ Accumulator + Scale │ │   │
│       │          │  │ (DSP slices)        │ │   │
│       │          │  └────────────────┬────┘ │   │
│       │          └───────────────────┼──────┘   │
│       │                              │          │
│       ▼                              ▼          │
│  ┌──────────────────────────────────────────┐  │
│  │ DDR4 / HBM2 Memory Controller            │  │
│  │ (packed ternary weights + activations)   │  │
│  └──────────────────────────────────────────┘  │
└─────────────────────────────────────────────────┘
```

### 11.5.4 Trit Decoder in FPGA Logic

The 10→16 decoder maps naturally to FPGA LUTs. Each 16-bit word produces 10 trits (20 bits). This can be implemented as a combinational lookup:

> For complete Verilog implementations of both 5→8 and 10→16 decoders with resource estimates, see §16.5 of [16-fpga-experiment-guide.md](16-fpga-experiment-guide.md).

### 11.5.5 PE Design in FPGA

> For the complete Verilog PE design with timing analysis, see §12.3 of [12-custom-ternary-accelerator-design.md](12-custom-ternary-accelerator-design.md).

### 11.5.6 Performance Estimate (ZCU104, 64×64 PE array @ 200 MHz)

| Metric | Value |
|--------|-------|
| PE array throughput | 64×64 × 200M = 819 GOPS |
| Decode (1B model, 75% sparse) | ~5-8 ms/total |
| Power | ~3-5W |
| vs. GPU (A100) | ~10× slower, ~100× less power |
| vs. CPU (M4) | ~2× faster, similar power |

---

## 11.6 Software Stack for Current Hardware

### 11.6.1 PyTorch Custom Autograd Function

The simplest way to deploy ternary models on existing hardware is a custom PyTorch autograd function that handles packing/unpacking transparently:

```python
import torch
import torch.nn as nn

class TernaryLinear(nn.Module):
    """Ternary linear layer that works on any PyTorch-supported hardware."""

    def __init__(self, in_features, out_features, bias=True):
        super().__init__()
        self.in_features = in_features
        self.out_features = out_features

        # Full-precision shadow weights (updated during training)
        self.weight_shadow = nn.Parameter(
            torch.randn(out_features, in_features) * 0.01
        )
        # Per-channel scale factors
        self.scale = nn.Parameter(torch.ones(out_features))
        if bias:
            self.bias = nn.Parameter(torch.zeros(out_features))
        else:
            self.register_parameter('bias', None)

        self.delta = 0.5  # Ternarization threshold

    def quantize_weight(self):
        """Ternarize weights with per-channel scaling."""
        # Normalize by scale
        w_norm = self.weight_shadow / self.scale.unsqueeze(1)

        # Ternarize
        w_ternary = torch.where(
            w_norm > self.delta, torch.ones_like(w_norm),
            torch.where(w_norm < -self.delta, -torch.ones_like(w_norm),
                        torch.zeros_like(w_norm))
        )
        return w_ternary * self.scale.unsqueeze(1)

    def forward(self, x):
        w_q = self.quantize_weight()
        # Straight-through estimator for training
        w_ste = self.weight_shadow + (w_q - self.weight_shadow).detach()
        return torch.nn.functional.linear(x, w_ste, self.bias)


class PackedTernaryLinear(nn.Module):
    """Ternary layer with packed storage for memory-constrained devices."""

    def __init__(self, in_features, out_features):
        super().__init__()
        # Pack 10 trits per 16-bit word
        n_words = (in_features + 9) // 10 * out_features
        self.register_buffer('packed_weight', torch.zeros(n_words, dtype=torch.int16))
        self.scale = nn.Parameter(torch.ones(out_features))
        self.bias = nn.Parameter(torch.zeros(out_features))
        self.in_features = in_features
        self.out_features = out_features

    def pack_weights(self, ternary_weight):
        """Pack ternary weights into 10-trit-per-word format.

        10→16 packing encoding (base-3, -1→0, 0→1, +1→2).
        See §4.2 of 04-storage-format.md for the canonical encoding spec.
        """
        # Implementation: base-3 encoding of each 10-trit group
        w = ternary_weight.cpu().numpy()
        packed = []
        for row in w:
            for i in range(0, len(row), 10):
                chunk = row[i:i+10]
                # Pad to 10
                if len(chunk) < 10:
                    chunk = np.concatenate([chunk, np.zeros(10 - len(chunk))])
                # Encode: -1→0, 0→1, +1→2
                val = sum((int(t) + 1) * (3 ** k) for k, t in enumerate(chunk))
                packed.append(val)
        self.packed_weight.copy_(torch.tensor(packed, dtype=torch.int16))

    def forward(self, x):
        # Unpack on-the-fly (can be cached for repeated calls)
        w = self.unpack_weights()
        return torch.nn.functional.linear(x, w, self.bias)

    def unpack_weights(self):
        """Unpack 10-trit words to INT8 weight matrix."""
        packed = self.packed_weight.cpu().numpy()
        rows = []
        words_per_row = (self.in_features + 9) // 10
        for oc in range(self.out_features):
            row = []
            for w in range(words_per_row):
                val = packed[oc * words_per_row + w]
                for k in range(10):
                    digit = val % 3
                    val //= 3
                    row.append(digit - 1)  # 0→-1, 1→0, 2→+1
            rows.append(row[:self.in_features])
        return torch.tensor(rows, dtype=torch.float32, device=x.device)
```

### 11.6.2 ONNX Export Path

For deployment on NPUs and other accelerators that accept ONNX:

```
PyTorch Model (ternary weights)
    ↓ torch.onnx.export (with custom QuantizeLinear nodes)
ONNX Model (with custom "TernaryLinear" op)
    ↓ ONNX Runtime / vendor SDK
Target hardware (NPU, DSP, etc.)
```

**Challenge:** ONNX does not have a standard "TernaryLinear" operator. Workarounds:
1. Decompose into standard ops: `Where(GT(w, δ), 1, Where(LT(w, -δ), -1, 0)) × scale × x`
2. Register a custom operator in ONNX Runtime
3. Use ONNX's quantization support (maps to INT8 on most targets)

---

## 11.7 Summary: Current Hardware Capabilities

### Desktop/Server

| Platform | Ternary Native? | INT8 GEMM | Power | 1B Model Decode | 7B Model Decode |
|----------|----------------|-----------|-------|-----------------|-----------------|
| NVIDIA B200 | No (emulated) | Yes (tensor cores) | 500W | ~5 ms | ~35 ms |
| NVIDIA H100 | No (emulated) | Yes (tensor cores) | 350W | ~15 ms | ~100 ms |
| NVIDIA A100 | No (emulated) | Yes (tensor cores) | 300W | ~45 ms | ~300 ms |
| AMD MI300X | No (emulated) | Yes (matrix cores) | 750W | ~20 ms | ~140 ms |
| Intel Xeon 6 (AMX) | No (emulated) | Yes (AMX INT8) | 350W | ~100 ms | ~700 ms |
| Apple M4 | No (emulated) | Partial (NEON) | 15W | ~200 ms | ~1.4s |

### Mobile/Edge

| Platform | Ternary Native? | INT8 GEMM | Power | 1B Model Decode |
|----------|----------------|-----------|-------|-----------------|
| Snapdragon 8 Elite NPU | No (emulated) | Yes (HVX INT8) | 5W | ~100 ms |
| Apple A17 Pro ANE | No (emulated) | Yes (native INT8) | 3W | ~150 ms |
| Google EdgeTPU (2nd gen) | No (must use INT8) | Yes (native INT8) | 3W | ~500 ms |
| Intel Lunar Lake NPU | No (emulated) | Yes (native INT8) | 5W | ~120 ms |
| AMD XDNA 2 | No (emulated) | Yes (native INT8) | 5W | ~100 ms |

### Specialized

| Platform | Ternary Native? | INT8 GEMM | Power | 1B Model Decode |
|----------|----------------|-----------|-------|-----------------|
| FPGA (ZCU104) | **Yes (custom)** | N/A (ternary native) | 5W | ~50 ms |
| FPGA (Alveo U250) | **Yes (custom)** | N/A (ternary native) | 25W | ~10 ms |
| **Ternary ASIC (projected)** | **Yes (native)** | N/A (ternary native) | **5W** | **~50 µs** |

**Key takeaways:**
1. **FPGAs are the only current platform** that can implement the ternary add/sub/skip datapath natively
2. **All other platforms emulate ternary** via unpacking to INT8/FP16, recovering storage/bandwidth but losing compute simplification
3. **Blackwell B200** nearly closes the gap with custom ternary hardware for 1B models (~5 ms vs ~50 µs)
4. **Mobile NPUs** (Snapdragon 8 Elite, Apple ANE) enable on-device 1B model inference thanks to ternary's 20× weight reduction
5. **The custom ternary ASIC advantage** is most pronounced for edge devices (5W power envelope) where every operation counts
