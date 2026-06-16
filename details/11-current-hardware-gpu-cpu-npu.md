# 11. Running Ternary Models on Current Hardware (GPU, CPU, NPU, FPGA)

## 11.1 Overview

Purpose-built ternary accelerators do not yet exist as production hardware. This guide covers how to deploy and accelerate ternary neural networks on commercially available hardware today — GPUs, CPUs, NPUs, and FPGAs — while outlining the gaps that a custom ternary ASIC would fill.

The fundamental challenge: **current hardware is designed for binary arithmetic** (INT8/FP16/FP32), so ternary's core advantage (add/sub/skip instead of multiply) is not natively supported. However, ternary's storage and bandwidth advantages can still be exploited through software and clever packing.

---

## 11.2 GPU Deployment

### 11.2.1 NVIDIA GPUs (Ampere, Ada Lovelace, Hopper)

NVIDIA GPUs are the most capable platform for ternary inference today, thanks to their mature software stack and high memory bandwidth.

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

**Performance Estimate (NVIDIA A100, 1B parameter model):**

| Operation | Time | Notes |
|-----------|------|-------|
| Weight unpack (200 MB) | ~0.1 ms | Memory-bound |
| INT8 GEMM (600 MB weights) | ~1.2 ms | Tensor cores @ 312 TOPS |
| Scale + bias apply | ~0.05 ms | Element-wise |
| **Total per layer** | **~1.4 ms** | |
| **Total 32 layers** | **~45 ms** | |

Compare with FP16 on same hardware: ~30 ms (tensor cores @ 624 TOPS for FP16). The ternary unpack overhead adds ~50% latency but reduces memory footprint by 4×.

### 11.2.2 AMD GPUs (RDNA 3, CDNA 2)

AMD GPUs support INT8 via ROCm/MIOpen but lack dedicated tensor cores for INT8 in consumer (RDNA) parts. CDNA 2 (Instinct MI200 series) has matrix cores similar to NVIDIA's.

**Approach:** Same unpack-then-GEMM strategy, but using ROCm's `rocBLAS` INT8 support. The unpack kernel would use HIP instead of CUDA.

**Limitation:** AMD's INT8 throughput is typically 2-4× lower than NVIDIA's for the same power budget, making the ternary advantage less pronounced.

### 11.2.3 Mobile GPUs (ARM Mali, Qualcomm Adreno, Apple GPU)

Mobile GPUs are memory-bandwidth-constrained (shared LPDDR), making ternary's bandwidth advantage highly relevant. However, they lack INT8 tensor cores in most cases.

**Approach:** Use GPU compute shaders (Vulkan/OpenCL) to implement the ternary add/sub/skip directly. Since mobile GPUs are scalar/SIMD rather than systolic, the multiplier elimination advantage is less impactful, but the 20× weight reduction means fewer memory accesses.

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

Modern x86 CPUs with AVX-512 or AMX (Advanced Matrix Extensions) can run ternary models efficiently by unpacking to INT8 and using INT8 dot-product instructions.

**Approach: Packed Ternary → AMX INT8 GEMM**

Intel Sapphire Rapids (4th gen Xeon) and Emerald Rapids (5th gen Xeon) support AMX with INT8 throughput of up to 2048 INT8 ops/cycle per tile.

```c
// Pseudocode for AMX-based ternary GEMM
void ternary_gemm_amx(
    uint16_t* packed_weights,  // 10 trits per word
    int8_t* activations,
    float* scales,
    float* output,
    int M, int N
) {
    for (int m = 0; m < M; m++) {
        // Unpack one row of ternary weights to INT8 buffer
        int8_t unpacked[11008];  // Max row size
        unpack_ternary_row(packed_weights + m * (N/10), unpacked, N);

        // Use AMX tile for INT8 dot product
        // AMX _tile_dpbusd: dot product of uint8 × int8 → int32
        __tile_dpbusd(&tile_cfg, unpacked, activations);

        // Apply scale factor
        output[m] = scales[m] + (float)tile_accumulator[m];
    }
}
```

**Performance Estimate (Intel Xeon w9-3595X, 64 cores, AMX):**

| Operation | Throughput |
|-----------|-----------|
| Ternary unpack (per core) | ~2 GB/s |
| AMX INT8 GEMM | ~2 TOP/s per socket |
| 1B param model, decode | ~80-120 ms/token |

This is ~3-4× slower than an A100 GPU but uses ~10× less power (350W vs 400W, but the GPU is doing more work per watt for dense models).

### 11.3.2 ARM CPUs (Cortex-A, Apple M-series, AWS Graviton)

ARM NEON and SVE2 provide SIMD dot-product instructions that can accelerate the unpacked ternary GEMM.

**Apple M-series (M2/M3/M4):**
- No dedicated INT8 matrix unit, but NEON can do 16 INT8 multiplies/cycle
- Unified memory architecture means no CPU-GPU transfer overhead
- Practical for models up to ~500M parameters (fits in unified memory)

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

**Performance Estimate (Apple M4, 1B parameter model):**

| Metric | Value |
|--------|-------|
| Decode latency | ~200-300 ms/token |
| Power | ~8-12W |
| Tokens/joule | ~3-5 |

### 11.3.3 RISC-V (SiFive, Alibaba Xuantie)

RISC-V is relevant because the proposed ternary accelerator uses a RISC-V host CPU. The RISC-V Vector extension (V-extension) can accelerate ternary unpack + INT8 dot product similarly to ARM NEON.

**Key advantage:** RISC-V cores are small and power-efficient, leaving more die area for the ternary GEMM array in a custom SoC.

---

## 11.4 NPU Deployment

### 11.4.1 Qualcomm Hexagon (Snapdragon 8 Gen 3)

Qualcomm's Hexagon NPU supports INT8 and INT16 operations with dedicated matrix units. It does not natively support ternary, but the same unpack-then-GEMM approach works.

**Architecture:** Hexagon uses a scalar + HVX (Hexagon Vector eXtensions) architecture. HVX can do 128 INT8 multiplies/cycle.

**Performance Estimate (Snapdragon 8 Gen 3):**

| Metric | Value |
|--------|-------|
| INT8 TOPS | ~45 TOP/s |
| Decode (1B model) | ~150-250 ms/token |
| Power | ~3-5W |
| Model fit | Needs off-chip DRAM for 1B params |

### 11.4.2 Apple Neural Engine (ANE)

The Apple Neural Engine is a proprietary matrix unit supporting INT8 and FP16. It is not programmable at a low level — models must be compiled via Core ML.

**Approach:** Convert ternary model to INT8 in Core ML format. The ANE will run INT8 GEMM natively. The ternary advantage comes from the smaller model size (200 MB vs 1 GB for INT8), which reduces memory bandwidth and may allow the model to fit in the ANE's internal SRAM.

**Limitation:** The ANE's internal programming model is opaque. Custom ternary operations are not possible; you get whatever Core ML's quantization passes produce.

### 11.4.3 Google EdgeTPU

The EdgeTPU is a purpose-built INT8 inference accelerator. It supports models in TFLite format with INT8 weights.

**Approach:** Same as ANE — convert ternary to INT8, let the EdgeTPU handle the INT8 GEMM. The ternary packing is only used for storage/compute-offload reduction.

**Performance (EdgeTPU, 1B INT8 model):**

| Metric | Value |
|--------|-------|
| INT8 TOPS | 4 TOP/s |
| Decode (1B model) | ~500 ms/token |
| Power | ~2W |

### 11.4.4 Intel Neural Compute Engine (NCE) / AMD XDNA

Intel's NCE (in Meteor Lake and Lunar Lake) and AMD's XDNA (in Ryzeon AI) are NPUs designed for INT8/INT16 inference. Both follow the same pattern: convert ternary to INT8, deploy via ONNX Runtime or OpenVINO.

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

| Board | FPGA | LUTs | BRAM | Cost | Best For |
|-------|------|------|------|------|----------|
| Xilinx ZCU104 | Zynq UltraScale+ | 274K | 32.1 Mb | $350 | Prototyping, ARM host |
| Xilinx Alveo U250 | Virtex UltraScale+ | 1.7M | 265 Mb | $3K | Large models, HBM |
| Intel Agilex 7 | AGI 027 | 2.7M | 128 Mb | $5K | High performance |
| Lattice Certus-NX | Certus-NX | 19K | 3.8 Mb | $50 | Ultra-low power |

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

```verilog
// 10-to-16 Trit Decoder (combinational)
module trit_decoder_10to16 (
    input  wire [15:0] packed_in,
    output wire [19:0] trits_out   // 10 trits × 2 bits each
);

    // Base-3 digit extraction via division
    wire [15:0] v0, v1, v2, v3, v4, v5, v6, v7, v8;

    assign v0 = packed_in;
    assign trits_out[1:0]  = v0 % 3;  // trit 0
    assign v1 = v0 / 3;
    assign trits_out[3:2]  = v1 % 3;  // trit 1
    assign v2 = v1 / 3;
    assign trits_out[5:4]  = v2 % 3;  // trit 2
    // ... repeat for all 10 trits

    // Note: Synthesis tool optimizes /3 and %3 into
    // constant-division circuits (multiplication by reciprocal)
    // For 16-bit / 3: approx 48 LUTs per divider
    // Total decoder: ~500 LUTs for 10 trits

endmodule
```

**Resource estimate:** ~500 LUTs per decoder. A 128-column array needs 128 decoders = ~64K LUTs (~23% of ZCU104).

### 11.5.5 PE Design in FPGA

Each ternary PE needs:
- 1 INT8 adder/subtractor: ~8 LUTs + 1 DSP slice
- 1 zero-skip mux: ~4 LUTs
- 1 accumulator register (32-bit): 32 flip-flops

**Total per PE:** ~12 LUTs + 1 DSP + 32 FFs

For a 64×64 PE array:
- 4096 PEs
- ~49K LUTs + 4096 DSP slices + 131K FFs

This fits comfortably in a Zynq UltraScale+ but would need a larger FPGA (Alveo U250) for a 128×128 array.

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
        """Pack ternary weights into 10-trit-per-word format."""
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

| Platform | Ternary Native? | Packed Storage? | Effective INT8 GEMM? | Power | 1B Model Decode |
|----------|----------------|-----------------|---------------------|-------|-----------------|
| NVIDIA GPU | No (emulated) | Yes (software) | Yes (tensor cores) | 200-400W | ~45 ms |
| AMD GPU | No (emulated) | Yes (software) | Yes (matrix cores) | 200-350W | ~80 ms |
| Intel CPU (AMX) | No (emulated) | Yes (software) | Yes (AMX INT8) | 200-350W | ~100 ms |
| Apple M4 | No (emulated) | Yes (software) | Partial (NEON) | 8-12W | ~250 ms |
| Qualcomm Hexagon | No (emulated) | Yes (software) | Yes (HVX INT8) | 3-5W | ~200 ms |
| Google EdgeTPU | No (must use INT8) | N/A | Yes (native INT8) | 2W | ~500 ms |
| FPGA (ZCU104) | **Yes (custom)** | **Yes (hardware)** | N/A (ternary native) | 3-5W | ~50 ms |
| **Ternary ASIC** | **Yes (native)** | **Yes (hardware)** | N/A (ternary native) | **2-10W** | **~50 µs** |

**Key takeaway:** FPGAs are the only current platform that can implement the ternary add/sub/skip datapath natively. All other platforms must emulate ternary via unpacking to INT8/FP16, which recovers the storage/bandwidth advantage but loses the compute simplification advantage. This is exactly the gap a purpose-built ternary ASIC would fill.
