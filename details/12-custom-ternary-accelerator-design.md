# 12. Designing a Custom Ternary Accelerator

## 12.1 Design Philosophy

A custom ternary accelerator is not a general-purpose processor. It is a **domain-specific architecture** optimized for one workload: transformer inference with ternary weights. Every design decision should be evaluated against a single question: *does this improve tokens-per-watt for ternary transformer inference?*

### Design Principles

1. **Ternary-native datapath** — add/sub/skip from the ground up, not emulated
2. **Memory-centric architecture** — minimize data movement at every level
3. **Sparsity-exploiting** — zero weights must skip compute, not just multiply by zero
4. **Hybrid precision** — ternary weights + FP16 control math, not ternary everything
5. **Compiler-driven** — hardware exposes capabilities, software schedules them

---

## 12.2 Architecture Specification

### 12.2.1 Target Workload

| Parameter | Edge Target | Server Target |
|-----------|-------------|---------------|
| Model size | 100M – 1B params | 1B – 70B params |
| Sequence length | 512 – 4096 | 2048 – 32768 |
| Batch size | 1 – 4 | 8 – 64 |
| Latency budget | < 100 ms/token | < 20 ms/token |
| Power budget | < 5W | < 100W |
| Weight format | Ternary {-1,0,+1} | Ternary {-1,0,+1} |
| Activation format | INT8 | INT8/BF16 |

### 12.2.2 Top-Level Architecture

```
┌──────────────────────────────────────────────────────────────────┐
│                    Ternary Transformer Accelerator                │
│                                                                    │
│  ┌─────────────────────────────────────────────────────────────┐  │
│  │ Control Processor (RISC-V RV64GC)                            │  │
│  │  • Instruction fetch/decode                                   │  │
│  │  • Layer scheduling                                           │  │
│  │  • DMA descriptor generation                                  │  │
│  │  • Interrupt handling                                         │  │
│  │  • 512 KB instruction SRAM                                   │  │
│  └──────────────────────────┬──────────────────────────────────┘  │
│                             │ AXI4 master                         │
│  ┌──────────────────────────▼──────────────────────────────────┐  │
│  │ Interconnect (AXI4 crossbar, 64-bit @ 1 GHz)                 │  │
│  └───┬──────────┬──────────┬──────────┬──────────┬─────────────┘  │
│      │          │          │          │          │                 │
│      ▼          ▼          ▼          ▼          ▼                 │
│  ┌────────┐┌────────┐┌────────┐┌────────┐┌────────┐             │
│  │ Weight ││Ternary ││ FP16   ││ DMA    ││ PCIe   │             │
│  │ SRAM   ││ GEMM   ││ Compute││ Engine ││/AXI    │             │
│  │ (8 MB) ││ Array  ││ Unit   ││ (8 ch) ││ Host   │             │
│  │        ││(128×128)││       ││        ││ I/F    │             │
│  └───┬────┘└───┬────┘└───┬────┘└───┬────┘└───┬────┘             │
│      │         │         │         │         │                   │
│      ▼         ▼         ▼         ▼         ▼                   │
│  ┌────────────────────────────────────────────────────────────┐  │
│  │ Scratchpad SRAM (4 MB, multi-banked)                       │  │
│  │  • Bank 0-1: Activation tensors (INT8/FP16)                │  │
│  │  • Bank 2:   KV cache (INT4/INT8)                          │  │
│  │  • Bank 3:   Intermediate results, scale factors           │  │
│  └────────────────────────────────────────────────────────────┘  │
│                                                                    │
└──────────────────────────────────────────────────────────────────┘
```

### 12.2.3 Die Floorplan (7nm)

```
┌─────────────────────────────────────────────┐
│ Total die: 5.0 × 5.0 mm = 25 mm²            │
│                                              │
│  ┌──────────────┐  ┌──────────────────────┐ │
│  │ Weight SRAM  │  │  Ternary GEMM Array  │ │
│  │   8 MB       │  │     128 × 128 PEs    │ │
│  │  3.0 × 2.5mm │  │  3.0 × 3.0 mm        │ │
│  │  ~15 mm²     │  │  ~9 mm²              │ │
│  └──────────────┘  └──────────────────────┘ │
│                                              │
│  ┌──────────────┐  ┌──────────────────────┐ │
│  │ Scratchpad   │  │  FP16 Compute Unit   │ │
│  │ SRAM 4 MB    │  │  + Accumulator       │ │
│  │ 2.0 × 2.0 mm │  │  2.0 × 1.5 mm        │ │
│  │  ~4 mm²      │  │  ~3 mm²              │ │
│  └──────────────┘  └──────────────────────┘ │
│                                              │
│  ┌──────────────────────────────────────────┐│
│  │  RISC-V Control + Interconnect + I/O     ││
│  │  ~4 mm²                                  ││
│  └──────────────────────────────────────────┘│
│                                              │
│  Area breakdown:                             │
│  SRAM:    19 mm² (76%)                       │
│  Logic:    6 mm² (24%)                       │
│  Total:   25 mm²                             │
└─────────────────────────────────────────────┘
```

---

## 12.3 Ternary GEMM Array — Detailed Design

### 12.3.1 Processing Element (PE)

The PE is the fundamental compute unit. It performs one ternary MAC operation per cycle:

```
┌─────────────────────────────────────────┐
│ Processing Element (PE)                  │
│                                          │
│  Inputs:                                 │
│    activation_x  ──► [8-bit INT]        │
│    weight_trit   ──► [2-bit: T/0/1]     │
│    partial_sum   ──► [32-bit INT]       │
│                                          │
│  ┌─────────────┐                         │
│  │ Zero-Skip   │──► clock_gate_enable   │
│  │ Detector    │                        │
│  └──────┬──────┘                         │
│         │                                │
│  ┌──────▼──────┐                         │
│  │ Negation    │                         │
│  │ Mux         │──► if T: -x, if 1: +x  │
│  └──────┬──────┘                         │
│         │                                │
│  ┌──────▼──────┐                         │
│  │ 32-bit      │                         │
│  │ Adder       │──► sum ± x              │
│  └──────┬──────┘                         │
│         │                                │
│  ┌──────▼──────┐                         │
│  │ 32-bit      │                         │
│  │ Accumulator │──► partial_sum_out      │
│  │ Register    │                        │
│  └─────────────┘                         │
│                                          │
│  Outputs:                                │
│    partial_sum_out ──► to PE below       │
│    activation_x_out ──► to PE right      │
│                                          │
│  Gate count: ~720 gates                  │
│  Area: ~140 µm² @ 7nm                    │
│  Critical path: 0.6 ns (1.6 GHz max)    │
└─────────────────────────────────────────┘
```

**Verilog Implementation:**

```verilog
module ternary_pe (
    input  wire        clk,
    input  wire        rst_n,
    input  wire        enable,           // clock gating
    input  wire [7:0]  activation_x,     // INT8 activation
    input  wire [1:0]  weight_trit,      // 2'b00=T, 2'b01=0, 2'b10=1
    input  wire [31:0] partial_sum_in,   // from PE above
    output reg  [31:0] partial_sum_out,  // to PE below
    output wire [7:0]  activation_x_out // to PE right
);

    // Trit decoding
    wire is_zero   = (weight_trit == 2'b01);
    wire is_pos    = (weight_trit == 2'b10);
    wire is_neg    = (weight_trit == 2'b00);

    // Sign extension for activation
    wire signed [8:0] x_signed = {activation_x[7], activation_x};
    wire signed [8:0] x_negated = -x_signed;

    // Conditional add/sub/skip
    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            partial_sum_out <= 32'd0;
        end else if (enable) begin
            if (is_zero) begin
                partial_sum_out <= partial_sum_in;       // skip
            end else if (is_pos) begin
                partial_sum_out <= partial_sum_in + {{23{x_signed[8]}}, x_signed};  // add
            end else if (is_neg) begin
                partial_sum_out <= partial_sum_in + {{23{x_negated[8]}}, x_negated}; // sub
            end
        end
    end

    // Pass activation to right neighbor
    assign activation_x_out = activation_x;

endmodule
```

### 12.3.2 Systolic Array Dataflow

The 128×128 PE array uses a weight-stationary systolic dataflow:

```
Cycle 0:   Activations injected into left column
           Weights pre-loaded into all PEs

Cycle 1:   Row 0 activations reach column 1
           Row 0 partial sums propagate down 1 row

Cycle N:   Row 0 activations reach column N
           Row 0 partial sums at row N

Cycle 127: All activations have traversed the array
           Partial sums at bottom row = final results

Cycle 128-130: Scale + bias + writeback (3 cycles)
```

**Latency for one tile (128×128):**
- Weight preload: 1 cycle (from decoder)
- Activation traversal: 128 cycles
- Accumulation: 128 cycles (pipelined with traversal)
- Scale + bias: 3 cycles
- **Total: ~260 cycles = 260 ns @ 1 GHz**

### 12.3.3 Zero-Skip Optimization

When a PE's weight is 0, it can skip the add operation. This saves power and can improve throughput:

**Clock gating approach:**
```
if (weight_trit == 2'b01) then
    disable PE clock → save ~0.01 pJ per skip
```

**Dataflow approach (more aggressive):**
```
if (weight_trit == 2'b01) then
    forward partial_sum directly to PE below
    forward activation directly to PE right
    → effectively bypass this PE in 0 cycles
```

With 75% sparsity, the effective throughput is 4× higher than the raw array throughput:
```
Raw: 128 × 128 × 1 GHz = 16.4 TOPS
Effective (75% sparse): 4.1 TOPS actual adds
                          + 12.3 TOPS worth of skips (free)
```

---

## 12.4 Memory Subsystem

### 12.4.1 Weight SRAM

| Parameter | Value |
|-----------|-------|
| Capacity | 8 MB |
| Organization | 128 banks × 64 KB |
| Word width | 16 bits (one packed word = 10 trits) |
| Bandwidth | 128 words/cycle = 256 bytes/cycle = 256 GB/s @ 1 GHz |
| Technology | 7nm SRAM, single-port |
| Area | ~15 mm² |

**Bank mapping:** Each bank feeds one column of the PE array. Bank `i` stores the packed weights for output channel `i` of the current tile.

### 12.4.2 Scratchpad SRAM

| Parameter | Value |
|-----------|-------|
| Capacity | 4 MB |
| Organization | 4 banks × 1 MB (true dual-port) |
| Word width | 128 bits (16 INT8 values or 8 FP16 values) |
| Bandwidth | 128 bits × 2 ports = 32 bytes/cycle = 32 GB/s per bank |
| Area | ~4 mm² |

**Bank assignment:**
- Bank 0: Current layer input activations (read)
- Bank 1: Current layer output activations (write)
- Bank 2: KV cache (read/write, growing with sequence)
- Bank 3: Scale factors, biases, intermediate results

### 12.4.3 DMA Engine

The DMA engine handles data movement between host DRAM and on-chip SRAM:

| Channel | Direction | Purpose |
|---------|-----------|---------|
| 0 | Host → Weight SRAM | Load packed ternary weights |
| 1 | Host → Scratchpad | Load input embeddings |
| 2 | Scratchpad → Host | Write output tokens |
| 3 | Host → Scratchpad | Load KV cache (prefill) |
| 4 | Scratchpad → Scratchpad | KV cache append (decode) |
| 5 | Host → Scratchpad | Load scale factors |
| 6 | Scratchpad → FP16 unit | Feed attention scores |
| 7 | Scratchpad → Host | Debug/trace |

**DMA bandwidth:** 8 channels × 16 bytes/cycle = 128 GB/s aggregate.

---

## 12.5 FP16 Compute Unit

### 12.5.1 Operations Supported

| Unit | Operation | Latency | Throughput |
|------|-----------|---------|------------|
| Softmax engine | exp, sum, divide | 10 cycles/element | 1 element/cycle |
| LayerNorm engine | mean, variance, normalize | 8 cycles/element | 1 element/cycle |
| FP16 MAC array | Q×Kᵀ dot product | Systolic | 32×32 @ 1 GHz |
| Activation unit | SiLU, GELU, ReLU | 3 cycles/element | 1 element/cycle |
| Residual add | Element-wise FP16 add | 1 cycle/element | 16 elements/cycle |

### 12.5.2 Softmax Engine

The softmax unit computes `exp(x_i) / Σ exp(x_j)` using a 3-stage pipeline:

```
Stage 1: Subtract max(x) for numerical stability
Stage 2: exp(x_i - max) via 256-entry LUT + linear interpolation
Stage 3: Accumulate sum, then divide each exp by sum
```

**LUT-based exp approximation:**
- 256 entries cover range [-10, 0] (sufficient after max subtraction)
- Linear interpolation between entries: < 0.05% error
- 2 cycles per element (1 for LUT lookup, 1 for interpolation)

### 12.5.3 FP16 Systolic Array (32×32)

A small FP16 systolic array handles attention score computation:

```
Q × Kᵀ where Q: [n_seq, d_head], K: [n_seq, d_head]
Result: [n_seq, n_seq] attention scores

For d_head = 128, n_seq = 2048:
  2048 × 2048 × 128 = 537M FP16 MACs
  @ 32×32 × 1 GHz = 1024 MACs/cycle
  = 524K cycles = 0.52 ms
```

---

## 12.6 Instruction Set Architecture

### 12.6.1 Instruction Format

All instructions are 64 bits:

```
[63:56]  Opcode
[55:48]  Source/destination select
[47:32]  Address/offset
[31:16]  Size/count
[15:0]   Flags/parameters
```

### 12.6.2 Instruction Set

| Opcode | Mnemonic | Description | Latency |
|--------|----------|-------------|---------|
| 0x01 | `LOAD_WEIGHT` | DMA packed weights to weight SRAM | Variable |
| 0x02 | `LOAD_ACT` | DMA activations to scratchpad | Variable |
| 0x03 | `STORE` | DMA results to host | Variable |
| 0x10 | `GEMM_TILE` | Execute ternary GEMM tile | ~260 cycles |
| 0x11 | `GEMM_SPARSE` | Execute sparse ternary GEMM | Variable |
| 0x20 | `ATTN_SCORE` | FP16 Q×Kᵀ via systolic array | Variable |
| 0x21 | `SOFTMAX` | Compute softmax over vector | 10×N cycles |
| 0x22 | `LAYER_NORM` | Compute layer normalization | 8×N cycles |
| 0x23 | `ACTIVATE` | Apply SiLU/GELU/ReLU | 3×N cycles |
| 0x24 | `RESIDUAL_ADD` | Element-wise FP16 add | N cycles |
| 0x30 | `APPLY_SCALE` | Multiply accumulator by α, add bias | 3 cycles |
| 0x31 | `QUANT_ACT` | Quantize FP16 → INT8 | 2×N cycles |
| 0x40 | `KV_CACHE_APPEND` | Append new K,V to cache | Variable |
| 0x41 | `KV_CACHE_READ` | Read KV cache for attention | Variable |
| 0x50 | `SYNC` | Wait for all units to complete | 1 cycle |
| 0x51 | `BARRIER` | Synchronization barrier | 1 cycle |
| 0xFF | `NOP` | No operation | 1 cycle |

### 12.6.3 Layer Compilation Example

A single transformer layer compiles to ~25 instructions:

```asm
; === Attention Block ===
LOAD_WEIGHT  W_Q, scale_Q        ; Load Q projection weights
GEMM_TILE    x, W_Q, tile_0      ; Q = ternary_GEMM(W_Q, x)
APPLY_SCALE  Q, scale_Q, bias_Q  ; Apply per-channel scale

LOAD_WEIGHT  W_K, scale_K        ; Load K projection weights
GEMM_TILE    x, W_K, tile_0      ; K = ternary_GEMM(W_K, x)
APPLY_SCALE  K, scale_K, bias_K

KV_CACHE_APPEND K, position      ; Append K to cache

LOAD_WEIGHT  W_V, scale_V        ; Load V projection weights
GEMM_TILE    x, W_V, tile_0      ; V = ternary_GEMM(W_V, x)
APPLY_SCALE  V, scale_V, bias_V

KV_CACHE_APPEND V, position      ; Append V to cache

KV_CACHE_READ  K_cache, V_cache  ; Read full KV cache
ATTN_SCORE     Q, K_cache         ; FP16 attention scores
SOFTMAX        scores             ; Softmax normalization
ATTN_APPLY     scores, V_cache    ; Apply attention to values

LOAD_WEIGHT  W_O, scale_O        ; Load output projection
GEMM_TILE    attn_out, W_O, tile_0
APPLY_SCALE  O, scale_O, bias_O

RESIDUAL_ADD x, O                ; x = x + O
LAYER_NORM   x, gamma, beta      ; LayerNorm

; === MLP Block ===
LOAD_WEIGHT  W_gate, scale_gate
GEMM_TILE    x, W_gate, tile_0
APPLY_SCALE  gate, scale_gate, bias_gate

LOAD_WEIGHT  W_up, scale_up
GEMM_TILE    x, W_up, tile_0
APPLY_SCALE  up, scale_up, bias_up

ACTIVATE     gate, SiLU          ; gate = SiLU(gate)
MUL_ELEMENT  gate, up            ; gate = gate * up

LOAD_WEIGHT  W_down, scale_down
GEMM_TILE    gate, W_down, tile_0
APPLY_SCALE  down, scale_down, bias_down

RESIDUAL_ADD x, down             ; x = x + down
LAYER_NORM   x, gamma, beta      ; LayerNorm

SYNC                              ; Wait for all operations
```

---

## 12.7 Compiler Design

### 12.7.1 Compilation Pipeline

```
┌──────────────────────────────────────────────────────────────┐
│                    Compilation Pipeline                        │
│                                                                │
│  ┌──────────┐    ┌──────────┐    ┌──────────┐    ┌────────┐ │
│  │ PyTorch  │    │ Ternary  │    │ Kernel   │    │ Binary │ │
│  │ Model    │───►│ Quantizer│───►│ Scheduler│───►│ .tbin  │ │
│  │ (.pth)   │    │          │    │          │    │        │ │
│  └──────────┘    └──────────┘    └──────────┘    └────────┘ │
│       │               │               │               │      │
│       ▼               ▼               ▼               ▼      │
│  FP32 weights    Ternary weights  Instruction     Packed     │
│  + architecture  + scale factors  sequence +      binary     │
│                                   memory plan     for DMA    │
└──────────────────────────────────────────────────────────────┘
```

### 12.7.2 Ternary Quantizer Pass

The quantizer converts FP32 weights to ternary + scales:

```python
class TernaryQuantizer:
    def quantize_model(self, model: nn.Module) -> TernaryModel:
        ternary_model = TernaryModel(model.config)

        for name, param in model.named_parameters():
            if 'weight' in name and param.dim() >= 2:
                # Per-channel ternarization
                scale = self.compute_optimal_scale(param)
                ternary_weight = self.ternarize(param, scale)
                packed = self.pack_ternary(ternary_weight)

                ternary_model.set_packed_weight(name, packed)
                ternary_model.set_scale(name, scale)
            else:
                # Keep biases and norms in FP16
                ternary_model.set_param(name, param.half())

        return ternary_model

    def compute_optimal_scale(self, weight):
        """MSE-optimal per-channel scale."""
        # α_j = (W_j · T_j) / (T_j · T_j)
        w = weight.float()
        t = self.ternarize_simple(w)
        alpha = (w * t).sum(dim=1) / (t * t).sum(dim=1).clamp(min=1)
        return alpha

    def pack_ternary(self, ternary_weight):
        """Pack ternary weights: 10 trits per 16-bit word."""
        # Base-3 encoding: -1→0, 0→1, +1→2
        shifted = (ternary_weight + 1).long()  # {0, 1, 2}
        packed = []
        for row in shifted:
            for i in range(0, len(row), 10):
                chunk = row[i:i+10]
                if len(chunk) < 10:
                    chunk = torch.cat([chunk, torch.ones(10 - len(chunk))])
                val = sum(chunk[k].item() * (3 ** k) for k in range(10))
                packed.append(int(val))
        return packed
```

### 12.7.3 Kernel Scheduler

The scheduler maps the computation graph to accelerator instructions:

```
Input: Computation graph (ternary GEMMs, FP16 ops, memory ops)
Output: Instruction sequence + memory allocation plan

Algorithm:
1. Topological sort of operations
2. Memory allocation: assign tensors to scratchpad banks
3. Tile large GEMMs to fit in weight SRAM
4. Insert DMA transfers for data movement
5. Schedule independent operations in parallel
6. Insert SYNC barriers at data dependencies
```

**Tiling strategy:**
```
For a GEMM [4096, 11008] × [11008, 4096]:
  Weight tile must fit in 8 MB weight SRAM
  Max tile: 128 output channels × 11008 input channels
           = 128 × 11008 × 1.585 bits = 2.8 MB ✓

  Number of tiles: 4096 / 128 = 32 tiles
  Each tile: ~260 cycles compute + ~50 cycles DMA
  Total: 32 × 310 = ~10K cycles = 10 µs
```

---

## 12.8 Physical Design

### 12.8.1 Clocking

| Domain | Frequency | Purpose |
|--------|-----------|---------|
| Core | 1.0 GHz | PE array, FP16 unit, control |
| Memory | 1.0 GHz | SRAM interfaces (synchronous) |
| DMA | 500 MHz | DMA engine, interconnect |
| PCIe | 250 MHz | Host interface (PCIe 4.0 ×4) |

### 12.8.2 Power Estimation

| Component | Dynamic Power | Static Power | Total |
|-----------|--------------|-------------|-------|
| Ternary GEMM array (128×128) | 1.2W | 0.3W | 1.5W |
| FP16 compute unit | 0.5W | 0.1W | 0.6W |
| Weight SRAM (8 MB) | 0.8W | 0.2W | 1.0W |
| Scratchpad SRAM (4 MB) | 0.4W | 0.1W | 0.5W |
| RISC-V control | 0.1W | 0.05W | 0.15W |
| Interconnect + I/O | 0.2W | 0.05W | 0.25W |
| **Total** | **3.2W** | **0.8W** | **4.0W** |

### 12.8.3 Performance Summary

| Metric | Value |
|--------|-------|
| Process | 7nm |
| Die area | 25 mm² |
| Clock | 1 GHz |
| Ternary TOPS (raw) | 16.4 TOPS |
| Ternary TOPS (75% sparse effective) | 4.1 TOPS |
| FP16 TOPS | 1.0 TOPS |
| Weight SRAM bandwidth | 256 GB/s |
| Decode throughput (1B model) | ~20K tokens/s |
| Prefill throughput (1B, 2048 seq) | ~120K tokens/s |
| Power | ~4W |
| Tokens/joule (decode) | ~5,000 |

---

## 12.9 Verification Strategy

### 12.9.1 Verification Hierarchy

```
Level 1: Unit Tests (Verilator simulation)
  ├── PE: all 3 trit values × edge cases
  ├── Decoder: all 243 input values (5-trit)
  ├── Accumulator: overflow, underflow
  └── SRAM controller: read/write, bank conflicts

Level 2: Subsystem Tests (Verilator + SystemVerilog)
  ├── 4×4 PE array: known matrix inputs
  ├── Full decoder array: random packed inputs
  ├── DMA engine: transfer correctness
  └── FP16 unit: accuracy vs. golden model

Level 3: System Tests (FPGA emulation)
  ├── Full 128×128 array: GEMM correctness
  ├── End-to-layer: one transformer layer
  ├── End-to-end: full model inference
  └── Performance: cycle count validation

Level 4: Silicon Validation (post tape-out)
  ├── At-speed testing
  ├── Power/thermal characterization
  └── Model accuracy validation
```

### 12.9.2 Co-simulation with PyTorch

The primary verification method is co-simulation: run the same inputs through PyTorch (golden model) and the RTL simulator, and compare outputs:

```python
# Verification script
def verify_gemm(packed_weights, activations, scale, rtl_sim):
    # PyTorch golden model
    weights = unpack_ternary(packed_weights)
    golden = (weights @ activations) * scale

    # RTL simulation
    rtl_sim.write_weights(packed_weights)
    rtl_sim.write_activations(activations)
    rtl_sim.write_scales(scale)
    rtl_sim.run()
    rtl_result = rtl_sim.read_output()

    # Compare
    max_error = np.max(np.abs(golden - rtl_result))
    assert max_error < 1e-3, f"Max error {max_error} exceeds threshold"
```

---

## 12.10 Design Alternatives Considered

### 12.10.1 Bit-Serial vs. Parallel PEs

| Approach | Area | Throughput | Power |
|----------|------|-----------|-------|
| Parallel (chosen) | 720 gates/PE | 1 MAC/cycle | Higher |
| Bit-serial | 200 gates/PE | 1 MAC/8 cycles | Lower |

**Decision:** Parallel PEs chosen because the target clock (1 GHz) provides sufficient throughput, and the area budget (25 mm²) accommodates 16K parallel PEs.

### 12.10.2 Weight-Stationary vs. Output-Stationary

| Dataflow | Weight Traffic | Activation Traffic | Best For |
|----------|---------------|-------------------|----------|
| Weight-stationary (chosen) | Load once | Stream repeatedly | Ternary weights (read-once) |
| Output-stationary | Reuse across tiles | Accumulate locally | Large output channels |

**Decision:** Weight-stationary is optimal for ternary because packed weights are read once from SRAM and reused across all output channels in a tile.

### 12.10.3 On-Chip vs. Off-Chip KV Cache

| Option | Capacity | Bandwidth | Power |
|--------|----------|-----------|-------|
| On-chip only (chosen) | 4 MB | 32 GB/s | Low |
| Off-chip (HBM) | 100+ GB | 1 TB/s | High |

**Decision:** On-chip KV cache for edge target (seq_len ≤ 4096). For server target, add HBM2e interface for larger KV caches.

---

## 12.11 Cost Analysis

| Item | Cost |
|------|------|
| 7nm wafer (12-inch) | ~$10,000/wafer |
| Dies per wafer (~25 mm²) | ~600 |
| Yield (7nm mature) | ~80% |
| Good dies per wafer | ~480 |
| Die cost | ~$21 |
| Package + test | ~$5 |
| **Total per chip** | **~$26** |
| At 100K volume | ~$15-20 per chip |

---

## 12.12 Comparison with Existing Accelerators

| Metric | This Design | Google EdgeTPU | NVIDIA A100 | Apple ANE |
|--------|------------|----------------|-------------|-----------|
| Weight format | Ternary (1.585b) | INT8 | FP16/INT8 | INT8 |
| Model size (1B) | 200 MB | 1 GB | 2 GB | 1 GB |
| Decode (1B) | ~50 µs | ~500 µs | ~30 µs | ~200 µs |
| Power | 4W | 2W | 400W | ~3W |
| Tokens/joule | ~5,000 | ~400 | ~75 | ~150 |
| Process | 7nm | 28nm | 7nm | 5mm |
| Customizable | Yes (RTL) | No | No | No |

**Key advantage:** 10-30× better energy efficiency than any existing accelerator for ternary transformer inference, at the cost of being a fixed-function accelerator rather than a programmable processor.
