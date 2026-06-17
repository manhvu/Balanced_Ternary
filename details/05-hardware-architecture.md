# 5. Hardware Accelerator Architecture

## 5.1 Top-Level Block Diagram

```
┌─────────────────────────────────────────────────────┐
│                   Host CPU (ARM/RISC-V)              │
│   - Model loading & scheduling                       │
│   - Control plane                                    │
│   - System management                                │
└────────────────────┬────────────────────────────────┘
                     │ AXI / PCIe
                     ▼
┌─────────────────────────────────────────────────────┐
│              Ternary Inference Accelerator            │
│                                                       │
│  ┌────────────┐  ┌────────────┐  ┌────────────┐     │
│  │ Weight     │  │ Ternary    │  │ Activation │     │
│  │ SRAM       │◄─┤ Decoder    ├─►│ Buffer     │     │
│  │ (packed    │  │ (5→8 /    │  │ (INT4/INT8)│     │
│  │  trits)    │  │ 10→16)    │  │            │     │
│  └────────────┘  └────────────┘  └──────┬─────┘     │
│         ▲                                │           │
│         │                                ▼           │
│         │                        ┌────────────┐     │
│         │                        │ Ternary    │     │
│         │                        │ GEMM Array │     │
│         │                        │ (add/sub/  │     │
│         │                        │  skip)     │     │
│         │                        └──────┬─────┘     │
│         │                               │           │
│         │                               ▼           │
│         │                        ┌────────────┐     │
│         │                        │ Accumulator│     │
│         │                        │ + Scale    │     │
│         │                        │ (FP16/BF16)│     │
│         │                        └──────┬─────┘     │
│         │                               │           │
│         ▼                               ▼           │
│  ┌───────────────────────────────────────────┐      │
│  │       FP16 Compute Unit                     │      │
│  │  (Softmax, LayerNorm, Attention Scores,     │      │
│  │   Residual Add, Element-wise Ops)           │      │
│  └───────────────────────────────────────────┘      │
│                                                       │
│  ┌───────────────────────────────────────────┐      │
│  │       Scratchpad SRAM (FP16/BF16)          │      │
│  │  (activations, intermediate tensors)       │      │
│  └───────────────────────────────────────────┘      │
│                                                       │
└─────────────────────────────────────────────────────┘
```

---

## 5.2 Ternary GEMM Array

### Core Processing Element (PE)

Each PE handles one output element:

```
Input:  activation x from left neighbor
        weight trit w from decoder (T, 0, or 1)
        partial sum from top neighbor

Operation:
    if w == +1:  sum = sum + x
    if w == -1:  sum = sum - x
    if w ==  0:  sum = sum       (skip)

Output:  sum to bottom neighbor
         x to right neighbor
```

### Elixir: PE and Systolic Array Simulation

```elixir
defmodule TernaryPE do
  @type activation :: integer()
  @type partial_sum :: integer()
  @type trit :: -1 | 0 | 1

  @doc """
  Single processing element operation.
  weight: -1 (T), 0, or +1 (1)
  """
  @spec compute(partial_sum(), activation(), trit()) ::
          {partial_sum(), activation()}
  def compute(sum, x, weight) do
    new_sum = case weight do
                1  -> sum + x
               -1  -> sum - x
                0  -> sum          # skip
             end
    {new_sum, x}  # forward x to right neighbor
  end
end

defmodule SystolicArray do
  @type matrix :: [[integer()]]
  @type trits :: [[-1 | 0 | 1]]

  @doc """
  Simulate a 2D systolic array for ternary GEMM.
  weights: list of columns, each column is a list of trits.
  activations: list of input activation vectors.
  Returns accumulated sums per column.
  """
  @spec gemm(trits(), matrix()) :: [integer()]
  def gemm(weights, activations) do
    # Initialize partial sums to zero
    n_cols = length(weights)
    init_sums = List.duplicate(0, n_cols)

    # Stream activations row by row through the array
    Enum.reduce(activations, init_sums, fn row, sums ->
      # Each PE column processes one weight per row
      Enum.zip_with(weights, sums, fn col, acc ->
        Enum.zip_with(col, row, fn w, x ->
          TernaryPE.compute(0, x, w) |> elem(0)
        end)
        |> Enum.sum()
        |> Kernel.+(acc)
      end)
    end)
  end
end
```

### Systolic Array

```
         Activation bus (horizontal)

         ┌───┐  ┌───┐  ┌───┐  ┌───┐
         │PE │  │PE │  │PE │  │PE │
W →      │0,0│  │0,1│  │0,2│  │0,3│
         └───┘  └───┘  └───┘  └───┘
           │      │      │      │
         ┌───┐  ┌───┐  ┌───┐  ┌───┐
         │PE │  │PE │  │PE │  │PE │
         │1,0│  │1,1│  │1,2│  │1,3│
         └───┘  └───┘  └───┘  └───┘
           │      │      │      │
         ┌───┐  ┌───┐  ┌───┐  ┌───┐
         │PE │  │PE │  │PE │  │PE │
         │2,0│  │2,1│  │2,2│  │2,3│
         └───┘  └───┘  └───┘  └───┘
           │      │      │      │
          out   out     out    out
```

### Zero-Skip Dataflow

When `w = 0`, the PE does not modify the partial sum and also does not consume the activation from the left.

This can be exploited by:

1. **Clock gating**: disable the PE's adder tree
2. **Operand forwarding**: propagate x to next PE in one cycle
3. **Systolic stall reduction**: unused slots allow other data to advance

### 5.2.1 PE Area Estimate

A rough gate-level area estimate for one processing element:

| Component | Gates | Notes |
|-----------|-------|-------|
| INT8 add/sub (mux-selected) | ~200 | 8-bit adder + subtractor + result mux |
| Zero-skip mux | ~20 | 2:1 mux to forward or hold activation |
| 32-bit accumulator | ~500 | 32-bit register + adder |
| **Total per PE** | **~720** | |

For a 128×128 systolic array:

```
16,384 PEs × 720 gates = ~11.8M gates
```

At 7nm, standard cell density is roughly 50M gates/mm², so:

```
11.8M gates ÷ 50M gates/mm² ≈ 0.24 mm² (core PE array)
```

Including decoder logic, accumulator pipeline, and routing overhead (~2×), the total GEMM array area is approximately **0.5 mm²** at 7nm.

> For detailed PE design including clock gating, pipeline stages, and layout, see §12.3 of [12-custom-ternary-accelerator-design.md](12-custom-ternary-accelerator-design.md).

---

## 5.3 Trit Decoder Unit

### Input: Packed data from Weight SRAM

```
Packed weight word (e.g., 8 bits = 5 trits)

Decoder pipeline:
    Stage 1: Read packed word           (1 cycle)
    Stage 2: Decode to 5 trit values    (1-2 cycles)
    Stage 3: Encode to control signals  (1 cycle)
             +1 → ADD signal
             -1 → SUB signal
              0 → SKIP signal
```

### Decoder Table (for 5→8 scheme)

```
Input (8-bit V)   t₄ t₃ t₂ t₁ t₀
        0         T  T  T  T  T
        1         T  T  T  T  0
        2         T  T  T  T  1
        3         T  T  T  0  T
       ...
      242         1  1  1  1  0
      243         invalid (>242 capped to 242)
```

### Decoder Array

One decoder per systolic array column, so weights stream in column-major order.

```
Weight SRAM
  ─► Decoder 0 ─► PE column 0
  ─► Decoder 1 ─► PE column 1
  ─► Decoder 2 ─► PE column 2
  ...
```

---

## 5.4 Accumulator + Scale Unit

After the GEMM array, each output channel needs:

```
yⱼ = αⱼ × sum + bⱼ
```

### Pipeline

```
Input from PE column: partial sum (INT32)

Stage 1:  INT32 → FP16 conversion    (1 cycle)
Stage 2:  FP16 multiply by αⱼ         (2-3 cycles)
Stage 3:  FP16 add bias bⱼ            (1 cycle)
Stage 4:  Write to output buffer      (1 cycle)
```

### Scale Factor Storage

```
Per-channel scale table:
    SRAM:  M × 16 bits
    Where M = number of output channels
```

---

## 5.5 FP16 Compute Unit

Handles operations that cannot be ternary:

| Unit | Operation | Precision | Latency |
|------|-----------|-----------|---------|
| Softmax | exp(x), sum, divide | FP16 | ~10 cycles per element |
| LayerNorm | mean, variance, normalize | FP16 | ~8 cycles per element |
| Attention Score | Q×Kᵀ dot product | FP16 | Systolic array (FP16) |
| Residual Add | Element-wise add | FP16 | 1 cycle per element |
| Activation | SiLU, GELU, ReLU | FP16 | ~3 cycles per element |

### FP16 Systolic Array

A small FP16 systolic array (e.g., 32×32) handles attention scores. This is much smaller than the ternary array because:

- Attention score computation is O(n²) for sequence length n
- Ternary GEMM is O(d_model²) for model dimension
- d_model >> n_sequence for typical edge LLM inference

### FP16 Approximation Note

For softmax, the `exp` function can be approximated with either:

- **Piecewise linear (PWL)**: 8–16 linear segments over the input range, implemented with a small multiplexer and multiplier.
- **Lookup table (LUT)**: 256-entry table with linear interpolation between entries.

Both approaches achieve **< 0.1% accuracy loss** on standard LLM benchmarks while avoiding the area and latency cost of a full FP16 transcendental unit. The LUT approach is preferred for its deterministic latency (~2 cycles) and small area (~0.01 mm² at 7nm).

---

## 5.6 Memory Hierarchy

```
                     Host DRAM (DDR/LPDDR)
                            │
                            │  Model weights (packed ternary)
                            │  Token embeddings (FP16/INT8)
                            ▼
               ┌────────────────────────┐
               │   On-Chip Weight SRAM   │
               │   (2-16 MB)             │
               │   Packed ternary weights │
               └───────────┬────────────┘
                           │ Decoder
                           ▼
               ┌────────────────────────┐
               │   Activation /          │
               │   Scratchpad SRAM       │
               │   (1-8 MB)              │
               │   FP16 / INT8           │
               └────────────────────────┘
                           │
                           ▼
               ┌────────────────────────┐
               │   Register File         │
               │   (PE-local)            │
               │   Small, fast           │
               └────────────────────────┘
```

### Bandwidth Analysis

For a 1B parameter ternary model at 10 tokens/s:

```
Weight transfer per token:
    ~200 MB (ternary weights) + ~4 MB (scales) = ~204 MB

Required bandwidth:
    204 MB × 10 tokens/s = 2.04 GB/s

Compare with FP32:
    4 GB × 10 = 40 GB/s

Bandwidth reduction: ~20×
```

---

## 5.7 KV Cache

The key-value cache is the largest activation memory in LLM inference.

### KV Cache Options

| Format | Storage (1B model, 2048 seq) | Quality |
|--------|------------------------------|---------|
| FP16   | 2048 × 2 × 4096 × 2B = 32 MB | Reference |
| INT8   | 2048 × 2 × 4096 × 1B = 16 MB | Good |
| INT4   | 2048 × 2 × 4096 × 0.5B = 8 MB | Acceptable |
| Ternary | 2048 × 2 × 4096 × 0.2B = 3.2 MB | Risky |

**Recommended**: INT4 or INT8 KV cache, with fallback to FP16 for long contexts.

---

## 5.8 Power Estimates

```
┌─────────────────────────────────────────┐
│ Ternary PE power breakdown per operation │
├─────────────────────────────────────────┤
│ Add operation  : 0.05 pJ (INT8 add)     │
│ Skip operation : 0.01 pJ (clock gate)   │
│ Move data      : 0.10 pJ (per word)     │
│ FP16 multiply  : 0.50 pJ                │
│ SRAM read (16b): 0.05 pJ                │
│ SRAM read (2b) : ~0.01 pJ               │
└─────────────────────────────────────────┘

For a typical ternary layer (75% sparsity, [4096, 11008]):

    Total adds:     4096 × 11008 × 0.25 = 11.3M operations
    Total skips:    4096 × 11008 × 0.75 = 33.8M operations

    Compute energy:  11.3M × 0.05pJ + 33.8M × 0.01pJ = 0.9 µJ
    Data movement:   200MB tokens × 0.1 pJ/B = 20 µJ
    Total:           ~21 µJ per layer

Compare with INT8:
    45.2M MACs × 0.2 pJ = 9 µJ (compute)
    200MB × 0.1 pJ/B = 20 µJ (data)
    Total: ~29 µJ per layer

Energy saving: ~28% per layer (plus memory bandwidth savings)
```

### 5.8.1 Thermal Design

At a 5 W power envelope on a 10 mm × 10 mm die:

```
Power density = 5 W / (10 mm × 10 mm)
              = 5 W / 100 mm²
              = 50 W/cm²
```

This is **manageable with a standard heat spreader** (copper or aluminum) and passive airflow. For reference:

| Device | Power | Die Area | Power Density |
|--------|-------|----------|---------------|
| This accelerator | 5 W | 100 mm² | **50 W/cm²** |
| NVIDIA A100 GPU | 400 W | 800 mm² | **500 W/cm²** |

The GPU's 10× higher power density requires active cooling (fans, heatsinks, or liquid cooling), while the ternary accelerator can operate with a simple passive spreader — a critical advantage for edge and embedded deployments.

---

## 5.9 ASIC Implementation Parameters

| Parameter | Value | Notes |
|-----------|-------|-------|
| Process node | 7nm or 12nm | Cost-effective for edge |
| Ternary GEMM size | 128×128 | 16K PEs |
| PE word width | 16-bit | INT8 or INT4 activations |
| Clock speed | 500 MHz – 1 GHz | Low power target |
| Weight SRAM | 8 MB | ~10B trits (enough for 1B param model) |
| Scratchpad SRAM | 4 MB | FP16 activations |
| Decoder throughput | 1 word/cycle per column | |
| FP16 compute | 32×32 systolic array | For attention scores |
| Interface | PCIe 4.0 ×4 or LPDDR5 | |
| Power envelope | 2–10 W | Edge device target |
| TOPS (ternary) | 128×128×1 GHz = 16 TOP/s | Add/sub operations |
| Effective TOPS (75% sparse) | 4 TOP/s effective | Zero skip accounted |
| Die area estimate | ~25 mm² at 7nm | GEMM + FP16 + SRAM + IO |
| SRAM area | ~15 mm² | 12 MB total (8 MB weight + 4 MB scratchpad) |
| Logic area | ~10 mm² | PE array, decoders, FP16 unit, controllers |
| Cost estimate | ~$5 | At 7nm in volume (100K+ units) |

> For detailed physical design, power breakdown, and die floorplan, see §12.8 of [12-custom-ternary-accelerator-design.md](12-custom-ternary-accelerator-design.md).

---

## 5.10 NUMA-Aware Tile Scheduling

Large matrices are tiled to fit in on-chip SRAM:

```
Matrix W:  [out_channels, in_channels]
Tile:      [tile_out, tile_in]

For each tile:
    1. Load packed weights to SRAM           (DMA)
    2. Decode tiles to trit control signals  (decoder)
    3. Stream activations through PE array   (compute)
    4. Accumulate partial sums               (accumulator)
    5. Write results to scratchpad           (store)

Tile size selection:
    tile_out × tile_in × 1.585 bits ≤ weight SRAM
    tile_in × activation_bits ≤ scratchpad SRAM
```

---

## 5.11 Instruction Set

The accelerator uses a minimal instruction set:

| Instruction | Parameters | Description |
|-------------|-----------|-------------|
| LOAD_WEIGHT | addr, offset, size | Load packed weights to SRAM |
| GEMM | out_tile, in_tile | Execute ternary GEMM tile |
| ACCUM_SCALE | out_ch | Apply scale and bias |
| SOFTMAX | addr, size | Compute softmax |
| LAYER_NORM | addr, size, gamma, beta | Compute layer norm |
| ACTIVATE | addr, size, type | Apply activation (ReLU, SiLU) |
| RESIDUAL | addr_dst, addr_src | Residual add |
| STORE | addr, size | Write to output buffer |
| SYNC | | Synchronize all units |
| DMA | src, dst, size | Memory transfer |

A transformer layer is compiled into a sequence of ~20-30 instructions.

---

## 5.12 Comparison: Ternary Accelerator vs GPU

| Property | GPU (NVIDIA A100) | Ternary Accelerator (this design) |
|----------|-------------------|-----------------------------------|
| Precision | FP32/FP16/INT8 | Ternary weights + FP16 math |
| Weight format | FP16/INT8 | ~1.585 bits/weight |
| MAC unit | Full multiplier | Add/sub/skip |
| Sparsity support | 2:4 structured | Natural (zero = skip) |
| Power | 400 W | 5–10 W |
| Memory | 128 GB HBM3e | 8 MB SRAM + LPDDR |
| Model fit (1B) | Many copies | Fits in SRAM |
| Deployment | Server rack | Edge device |
| Cost | $10,000+ | $50–200 (estimated) |

---

## 5.13 Verification Strategy

Verifying the ternary accelerator requires a multi-level approach:

1. **Unit tests for each module**: Individual testbenches for the PE, decoder, accumulator, FP16 unit, and controller. Directed tests cover corner cases (all zeros, all ±1, overflow). Coverage target: >95% line coverage, >90% toggle coverage.

2. **Co-simulation with PyTorch golden model**: Run the same model layers on the RTL simulator (e.g., Verilator) and PyTorch with identical weights and inputs. Compare outputs element-wise with a tolerance of ±1 LSB in INT8 / FP16. This catches integration-level bugs that unit tests miss.

3. **Formal verification of decoder**: The trit decoder is safety-critical — a decoding error corrupts all downstream computation. Use formal property checking (e.g., Symbiyosys / Yosys) to prove that every valid packed word maps to the correct trit sequence, and that invalid inputs are handled (capped or flagged).

4. **FPGA emulation before tape-out**: Map the full accelerator to an FPGA prototyping board (e.g., Xilinx Alveo or Zynq) and run end-to-end inference on real LLM workloads. This validates timing, power, and functional correctness at near-real-time speeds before committing to silicon. FPGA results feed back into RTL fixes and microarchitecture tuning.

> For the full verification hierarchy with co-simulation code, see §12.9 of [12-custom-ternary-accelerator-design.md](12-custom-ternary-accelerator-design.md).