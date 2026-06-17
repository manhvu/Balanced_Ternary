# 16. Balanced Ternary FPGA Experiment Guide

## 16.1 Overview

This document provides a practical, step-by-step guide for experimenting with balanced ternary neural network inference on FPGA hardware. It covers board selection, RTL design, synthesis, deployment, and benchmarking — from first prototype to production-ready accelerator.

**Target audience**: Hardware engineers, researchers, and ML practitioners who want to validate ternary inference on real hardware before committing to ASIC tape-out.

---

## 16.2 Why FPGA for Ternary Experiments?

| Property | GPU/NPU | FPGA | ASIC |
|----------|---------|------|------|
| Custom ternary datapath | No (emulated) | **Yes (native)** | Yes (native) |
| Packed ternary decode | Software | **Hardware** | Hardware |
| Zero-skip clock gating | No | **Yes** | Yes |
| Development time | Hours | **Weeks** | Months |
| Unit cost | Low | Medium | Low (volume) |
| Reconfigurable | No | **Yes** | No |
| Risk | None | **Low** | High ($1–3M tape-out) |

FPGAs are the **only current platform** that can implement the ternary add/sub/skip datapath natively. This makes them ideal for:
- Validating ternary PE array designs before ASIC
- Measuring real power/latency numbers for ternary inference
- Prototyping the trit decoder and weight loading pipeline
- Testing mixed-precision strategies (ternary weights + INT8 activations)

---

## 16.3 Board Selection Guide

### Recommended Boards

| Board | FPGA | LUTs | DSP | BRAM | HBM | Cost | Best For |
|-------|------|------|-----|------|-----|------|----------|
| Xilinx ZCU104 | Zynq UltraScale+ | 274K | 1,728 | 32 Mb | No | $350 | Entry-level prototyping |
| Xilinx Alveo U250 | Virtex UltraScale+ | 1.7M | 6,840 | 265 Mb | No | $3K | Large PE arrays |
| Xilinx Alveo U55C | Versal AI Core | 1.3M | 4,000 | 130 Mb | HBM2e (8 GB) | $8K | HBM-enabled designs |
| Intel Agilex 7 F-Series | Agilex | 1.4M | 5,760 | 128 Mb | No | $5K | Intel ecosystem |
| Lattice Certus-NX | Certus-NX | 19K | 24 | 3.8 Mb | No | $50 | Ultra-low power edge |
| QuickLogic EOS S3 | EOS S3 | 8K | 8 | 2 Mb | No | $15 | IoT/MCU integration |

### Selection Criteria

```
Decision tree:

1. Budget < $100?
   → Lattice Certus-NX or QuickLogic EOS S3
   → Suitable for: small models (ResNet-18, MobileNet-V2)

2. Budget < $500?
   → Xilinx ZCU104
   → Suitable for: medium models (ResNet-50, YOLOv5-nano)
   → ARM host for software stack

3. Budget < $5K?
   → Xilinx Alveo U250 or Intel Agilex 7
   → Suitable for: large models (ViT-B, YOLOv8)
   → 128×128+ PE arrays

4. Budget < $10K?
   → Xilinx Alveo U55C (with HBM2e)
   → Suitable for: LLM inference (1B+ parameters)
   → On-chip HBM for weight storage

5. Need maximum flexibility?
   → Cloud FPGA (AWS F1, Azure Alveo)
   → Pay-per-hour, no upfront cost
   → Good for initial exploration
```

---

## 16.4 Ternary PE Array Design

### 16.4.1 Core PE (Processing Element)

> For the authoritative ASIC-targeted PE design with detailed timing, clock gating, and pipeline analysis, see §12.3.1 of [12-custom-ternary-accelerator-design.md](12-custom-ternary-accelerator-design.md). Below is the FPGA-adapted version with simplified control logic:

```verilog
module ternary_pe (
    input  wire        clk,
    input  wire        rst_n,
    input  wire [1:0]  trit,
    input  wire [7:0]  activation,
    input  wire        valid,
    output reg  [31:0] accumulator
);

    wire signed [7:0] act = $signed(activation);

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n)
            accumulator <= 32'sd0;
        else if (valid)
            accumulator <= accumulator +
                (trit == 2'b10 ? {{24{act[7]}}, act} :
                 trit == 2'b01 ? -{{24{act[7]}}, act} : 32'sd0);
    end

endmodule
```

**Resource usage**: ~15 LUTs + 1 DSP + 32 FFs per PE

### 16.4.2 PE Array (Systolic)

```verilog
module ternary_pe_array #(
    parameter ROWS = 64,
    parameter COLS = 64
)(
    input  wire clk,
    input  wire rst_n,
    input  wire [1:0]  trits   [ROWS-1:0][COLS-1:0],  // Weight trits
    input  wire [7:0]  acts    [COLS-1:0],             // Input activations
    input  wire        valid,
    output wire [31:0] results [ROWS-1:0]              // Output partial sums
);

    genvar r, c;
    generate
        for (r = 0; r < ROWS; r++) begin : row
            for (c = 0; c < COLS; c++) begin : col
                ternary_pe pe_i (
                    .clk(clk),
                    .rst_n(rst_n),
                    .trit(trits[r][c]),
                    .activation(acts[c]),
                    .valid(valid),
                    .accumulator(results[r]),
                    .done()
                );
            end
        end
    endgenerate

endmodule
```

### 16.4.3 Resource Estimate

| PE Array Size | LUTs | DSPs | BRAM | FPGA Required |
|---------------|------|------|------|---------------|
| 16×16 | 4K | 256 | 0 | ZCU104 ✓ |
| 32×32 | 16K | 1,024 | 0 | ZCU104 ✓ |
| 64×64 | 64K | 4,096 | 0 | Alveo U250 ✓ |
| 128×128 | 256K | 16,384 | 0 | Alveo U250 ✓ |
| 256×256 | 1M | 65,536 | 0 | Multiple FPGAs |

---

## 16.5 Trit Decoder Design

### 16.5.1 5→8 Decoder (Byte-Aligned)

```verilog
module trit_decoder_5to8 (
    input  wire [7:0]  packed_in,
    output wire [9:0]  trits_out,  // 5 trits × 2 bits each
    output wire        valid
);

    // Base-3 digit extraction
    wire [2:0] d0, d1, d2, d3, d4;

    assign d0 = packed_in % 8'd3;
    assign d1 = (packed_in / 8'd3) % 8'd3;
    assign d2 = (packed_in / 8'd9) % 8'd3;
    assign d3 = (packed_in / 8'd27) % 8'd3;
    assign d4 = (packed_in / 8'd81) % 8'd3;

    // Map to 2-bit encoding: 00=0, 01=-1, 10=+1
    assign trits_out[1:0] = (d0 == 0) ? 2'b01 : (d0 == 1) ? 2'b00 : 2'b10;
    assign trits_out[3:2] = (d1 == 0) ? 2'b01 : (d1 == 1) ? 2'b00 : 2'b10;
    assign trits_out[5:4] = (d2 == 0) ? 2'b01 : (d2 == 1) ? 2'b00 : 2'b10;
    assign trits_out[7:6] = (d3 == 0) ? 2'b01 : (d3 == 1) ? 2'b00 : 2'b10;
    assign trits_out[9:8] = (d4 == 0) ? 2'b01 : (d4 == 1) ? 2'b00 : 2'b10;

    assign valid = 1'b1;

endmodule
```

**Resource usage**: ~50 LUTs per 5→8 decoder

### 16.5.2 10→16 Decoder (High Density)

> For the GPU/CPU decoder implementation, see §11.5.4 of [11-current-hardware-gpu-cpu-npu.md](11-current-hardware-gpu-cpu-npu.md).

```verilog
module trit_decoder_10to16 (
    input  wire [15:0] packed_in,
    output wire [19:0] trits_out,  // 10 trits × 2 bits each
    output wire        valid
);

    // Sequential base-3 digit extraction
    reg [15:0] v;
    reg [1:0]  trit_buf [0:9];
    integer    i;

    always @(*) begin
        v = packed_in;
        for (i = 0; i < 10; i = i + 1) begin
            case (v % 16'd3)
                0: trit_buf[i] = 2'b01;  // -1
                1: trit_buf[i] = 2'b00;  //  0
                2: trit_buf[i] = 2'b10;  // +1
            endcase
            v = v / 16'd3;
        end
    end

    assign trits_out = {trit_buf[9], trit_buf[8], trit_buf[7], trit_buf[6],
                        trit_buf[5], trit_buf[4], trit_buf[3], trit_buf[2],
                        trit_buf[1], trit_buf[0]};
    assign valid = 1'b1;

endmodule
```

**Resource usage**: ~500 LUTs per 10→16 decoder

---

## 16.6 Weight Loading Pipeline

### 16.6.1 Memory Hierarchy

```
┌─────────────────────────────────────────────────┐
│ External Memory (DDR4 / HBM)                    │
│ Packed ternary weights (200 MB for 1B model)    │
└──────────────────┬──────────────────────────────┘
                   │ AXI DMA
                   ▼
┌─────────────────────────────────────────────────┐
│ On-chip Weight Buffer (BRAM)                     │
│ 32 KB per PE column (holds 16K trits)            │
└──────────────────┬──────────────────────────────┘
                   │ Weight Bus
                   ▼
┌─────────────────────────────────────────────────┐
│ Trit Decoder Array                               │
│ 10→16 decoders (one per PE column)               │
└──────────────────┬──────────────────────────────┘
                   │ Decoded trits (2 bits each)
                   ▼
┌─────────────────────────────────────────────────┐
│ PE Array (64×64)                                 │
│ Weighted multiply-accumulate                     │
└─────────────────────────────────────────────────┘
```

### 16.6.2 Weight Loading Verilog

```verilog
module weight_loader #(
    parameter NUM_COLUMNS = 64,
    parameter WORDS_PER_COLUMN = 1024  // 10K trits per column
)(
    input  wire clk,
    input  wire rst_n,
    input  wire start,
    input  wire [31:0] ddr_addr,
    output wire        ddr_rd_en,
    input  wire [255:0] ddr_rd_data,
    output wire [1:0]   trits_out [NUM_COLUMNS-1:0][15:0],
    output wire         trits_valid
);

    // State machine
    localparam IDLE = 0, LOAD = 1, DECODE = 2, OUTPUT = 3;
    reg [1:0] state;

    reg [31:0] word_count;
    reg [255:0] buffer;

    // Decode 16 packed words to 10 trits each
    trit_decoder_10to16 decoders [NUM_COLUMNS-1:0] (
        .packed_in(buffer[15:0]),
        .trits_out(trits_out[0]),
        .valid(trits_valid)
    );

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            word_count <= 0;
        end else begin
            case (state)
                IDLE: if (start) state <= LOAD;
                LOAD: begin
                    // Request 256-bit word from DDR
                    ddr_rd_en <= 1;
                    buffer <= ddr_rd_data;
                    state <= DECODE;
                end
                DECODE: begin
                    // Decoder latency (1-2 cycles)
                    state <= OUTPUT;
                end
                OUTPUT: begin
                    // Emit decoded trits to PE array
                    word_count <= word_count + 1;
                    if (word_count < WORDS_PER_COLUMN)
                        state <= LOAD;
                    else
                        state <= IDLE;
                end
            endcase
        end
    end

endmodule
```

---

## 16.7 Host Interface (Zynq PS)

For Zynq-based boards (ZCU104), the ARM PS (Processing System) handles:
- Model loading from SD card or network
- Weight DMA transfer to PL (Programmable Logic)
- Inference orchestration
- Result readout

### 16.7.1 C Driver (PS Side)

```c
#include "xternary_accelerator.h"

// Initialize accelerator
XAxiDma dma;
Xternary_accelerator accel;

void init_accelerator() {
    Xternary_accelerator_Initialize(&accel, 0);
    XAxiDma_Initialize(&dma, 0);

    // Configure PE array dimensions
    Xternary_accelerator_Set_rows(&accel, 64);
    Xternary_accelerator_Set_cols(&accel, 64);
}

// Load model weights from file
void load_model(const char* filename) {
    FILE* f = fopen(filename, "rb");
    fseek(f, 0, SEEK_END);
    long size = ftell(f);
    fseek(f, 0, SEEK_SET);

    uint8_t* buffer = (uint8_t*)malloc(size);
    fread(buffer, 1, size, f);
    fclose(f);

    // DMA transfer to accelerator
    XAxiDma_SimpleTransfer(&dma, (UINTPTR)buffer, size, XAXIDMA_DMA_TO_DEVICE);
    while (XAxiDma_Busy(&dma, XAXIDMA_DMA_TO_DEVICE));

    free(buffer);
}

// Run inference
float* run_inference(int8_t* input, int input_size) {
    // Set input activation
    XAxiDma_SimpleTransfer(&dma, (UINTPTR)input,
                           input_size * sizeof(int8_t),
                           XAXIDMA_DMA_TO_DEVICE);

    // Start computation
    Xternary_accelerator_Start(&accel);

    // Wait for completion
    while (!Xternary_accelerator_IsDone(&accel));

    // Read output
    static float output[1000];
    XAxiDma_SimpleTransfer(&dma, (UINTPTR)output,
                           1000 * sizeof(float),
                           XAXIDMA_DMA_TO_DEVICE);

    return output;
}
```

### 16.7.2 Python Host (Alternative)

```python
import pynq
from pynq import Overlay, allocate

# Load bitstream
ol = Overlay("ternary_accel.bit")
ol.download()

# Allocate DMA buffers
input_buffer = allocate(shape=(224*224*3,), dtype=np.int8)
output_buffer = allocate(shape=(1000,), dtype=np.float32)

# Load model weights
weights = np.fromfile("model.tbin", dtype=np.uint16)
weight_buffer = allocate(shape=weights.shape, dtype=np.uint16)
np.copyto(weight_buffer, weights)

# Run inference
def ternary_inference(image):
    np.copyto(input_buffer, image.flatten())
    ol.dma_send.transfer(input_buffer)
    ol.dma_send.wait()
    ol.ternary_accelerator.write(0x00, 1)  # Start
    while ol.ternary_accelerator.read(0x00) & 0x2 == 0:
        pass  # Wait for done
    ol.dma_recv.transfer(output_buffer)
    ol.dma_recv.wait()
    return output_buffer.copy()
```

---

## 16.8 Benchmarking Methodology

### 16.8.1 Performance Metrics

| Metric | How to Measure | Target (ResNet-50) |
|--------|---------------|-------------------|
| Latency (single image) | Timer from input ready to output valid | < 5 ms |
| Throughput (batched) | Images processed per second | > 200 FPS |
| Power consumption | On-board power monitor (Pynq) or external meter | < 1 W |
| Energy per inference | Power × Latency | < 5 mJ |
| Accuracy | Top-1 accuracy on ImageNet | > 74% (ternary) |
| Resource utilization | Synthesis report (LUTs, DSPs, BRAMs) | < 80% of FPGA |

### 16.8.2 Benchmark Test Cases

| Test | Model | Input | Expected Latency | Purpose |
|------|-------|-------|-----------------|---------|
| 1 | ResNet-18 | 224×224×3 | < 2 ms | Basic validation |
| 2 | ResNet-50 | 224×224×3 | < 5 ms | Full CNN benchmark |
| 3 | MobileNet-V2 | 224×224×3 | < 1 ms | Edge deployment |
| 4 | YOLOv5-nano | 640×640×3 | < 10 ms | Detection benchmark |
| 5 | ViT-B/16 | 224×224×3 | < 15 ms | Transformer benchmark |
| 6 | BERT-base | 128 tokens | < 5 ms | NLP benchmark |

### 16.8.3 Power Measurement Setup

```
┌─────────────────────────────────────────┐
│ FPGA Board (ZCU104 / Alveo)             │
│  ┌─────────┐    ┌──────────────────┐   │
│  │ Power   │    │ Ternary Accel    │   │
│  │ Monitor │───►│ (PL fabric)      │   │
│  │ (Pynq)  │    │                  │   │
│  └─────────┘    └──────────────────┘   │
│       │                                  │
│       ▼                                  │
│  ┌──────────────────────────────────┐   │
│  │ Host PC (USB / JTAG)             │   │
│  │ Python script for measurement    │   │
│  └──────────────────────────────────┘   │
└─────────────────────────────────────────┘
```

**Power measurement code (Pynq):**

```python
import pynq.lib.xrt as xrt

def measure_power(overlay, duration_sec=10):
    """Measure power consumption during inference."""
    power_readings = []

    for _ in range(duration_sec * 10):  # 100ms intervals
        # Read power from on-board sensor
        power = overlay.xadc.read_voltage("vauxp0") * 10  # Simplified
        power_readings.append(power)
        time.sleep(0.1)

    avg_power = np.mean(power_readings)
    peak_power = np.max(power_readings)

    return {
        "avg_power_w": avg_power,
        "peak_power_w": peak_power,
        "energy_per_inference_mj": avg_power * measured_latency_ms / 1000
    }
```

---

## 16.9 Synthesis and Implementation

### 16.9.1 Vivado Flow (Xilinx)

```bash
# Step 1: Create project
vivado -mode batch -source create_project.tcl

# Step 2: Add RTL sources
add_files -norecurse {
    src/ternary_pe.v
    src/ternary_pe_array.v
    src/trit_decoder_5to8.v
    src/trit_decoder_10to16.v
    src/weight_loader.v
    src/ternary_accelerator.v
}

# Step 3: Add constraints
add_files -fileset constrs_1 constraints/ports.xdc

# Step 4: Run synthesis
synth_design -top ternary_accelerator -part xczu7ev-ffvc1156-2-e

# Step 5: Run implementation
place_design
route_design

# Step 6: Generate bitstream
write_bitstream -force ternary_accel.bit

# Step 7: Generate handoff for PS (Zynq only)
write_hw_platform -fixed -include_bit \
    -force -file ternary_accel.xsa
```

### 16.9.2 Resource Utilization Report

```
+----------------------------+-------+-------+-----------+-------+
|        Site Type           | Used  | Fixed | Available | Util% |
+----------------------------+-------+-------+-----------+-------+
| CLB LUTs                   |  4512 |     0 |    174200 |  2.6% |
| CLB Registers              |  2048 |     0 |    348400 |  0.6% |
| Block RAM Tile             |    32 |     0 |       144 | 22.2% |
| DSPs                       |   256 |     0 |      1728 | 14.8% |
+----------------------------+-------+-------+-----------+-------+

Design Summary:
- Ternary PE Array: 16×16 (256 PEs)
- Trit Decoder: 16 × 5→8 decoders
- Weight Buffer: 32 KB BRAM
- Target Frequency: 200 MHz
- Estimated Latency: 8 ms (ResNet-50)
```

### 16.9.3 Timing Closure Tips

| Issue | Solution |
|-------|----------|
| Setup violations on weight bus | Pipeline the weight bus (add registers) |
| Hold violations on trit signals | Add buffer cells on critical paths |
| High fanout on valid signal | Use BUFG or replicate registers |
| Long routing delays | Use Pblock constraints to keep PEs close |
| BRAM timing | Register all BRAM outputs |

---

## 16.10 Example: ResNet-50 on ZCU104

### 16.10.1 Complete Project Structure

```
ternary_fpga_resnet50/
├── rtl/
│   ├── ternary_pe.v
│   ├── ternary_pe_array.v
│   ├── trit_decoder_5to8.v
│   ├── weight_loader.v
│   ├── activation_buffer.v
│   ├── batchnorm_unit.v
│   ├── relu_unit.v
│   ├── pool_unit.v
│   └── ternary_accelerator.v
├── tb/
│   ├── tb_ternary_pe.v
│   ├── tb_decoder.v
│   └── tb_accelerator.v
├── sw/
│   ├── host/main.c          # PS C driver
│   ├── host/ternary_model.h  # Model header
│   └── python/benchmark.py   # Python benchmark
├── constraints/
│   └── ports.xdc
├── vivado/
│   └── create_project.tcl
├── models/
│   ├── resnet50_ternary.tbin  # Packed ternary weights
│   └── resnet50_ternary.scales # Per-channel FP16 scales
├── README.md
└── Makefile
```

### 16.10.2 Layer-by-Layer Mapping

| Layer | Type | Input | Output | PEs Used | Cycles |
|-------|------|-------|--------|----------|--------|
| conv1 | Conv 7×7, stride 2 | 224×224×3 | 112×112×64 | 64×64 | 2.5M |
| bn1 + relu | BN + ReLU | 112×112×64 | 112×112×64 | N/A | 0.5M |
| maxpool | 3×3, stride 2 | 112×112×64 | 56×56×64 | N/A | 0.2M |
| layer1 | 3× bottleneck | 56×56×64 | 56×56×256 | 64×64 | 8M |
| layer2 | 4× bottleneck | 56×56×256 | 28×28×512 | 64×64 | 12M |
| layer3 | 6× bottleneck | 28×28×512 | 14×14×1024 | 64×64 | 16M |
| layer4 | 3× bottleneck | 14×14×1024 | 7×7×2048 | 64×64 | 8M |
| avgpool | Global avg | 7×7×2048 | 1×1×2048 | N/A | 0.1M |
| fc | Linear | 2048 | 1000 | 16×16 | 0.2M |
| **Total** | | | | | **~47.5M** |

**At 200 MHz**: 47.5M cycles / 200 MHz = **237.5 ms** (single PE column)

With 64 PE columns (64×64 array): 237.5 / 64 = **3.7 ms**

### 16.10.3 Accuracy Validation

```python
import numpy as np
from PIL import Image
import onnxruntime as ort

# Load reference model (ONNX)
ref_session = ort.InferenceSession("resnet50.onnx")

# Load ternary model weights
weights = load_tbin("resnet50_ternary.tbin")
scales = load_scales("resnet50_ternary.scales")

# Run inference on ImageNet validation set
correct = 0
total = 0

for image, label in imagenet_val:
    # Ternary inference on FPGA
    fpga_output = ternary_inference(image)

    # Reference output
    ref_output = ref_session.run(None, {"input": image})[0]

    # Compare
    fpga_class = np.argmax(fpga_output)
    ref_class = np.argmax(ref_output)

    if fpga_class == label:
        correct += 1
    total += 1

accuracy = correct / total
print(f"Ternary FPGA Accuracy: {accuracy:.2%}")
# Expected: ~74.5% (vs 76.1% FP32 baseline)
```

---

## 16.11 Debugging and Verification

### 16.11.1 Common Issues and Solutions

| Symptom | Likely Cause | Debug Step | Fix |
|---------|-------------|------------|-----|
| All outputs zero | Decoder not working | Check trit_decoder output with ILA | Fix base-3 extraction logic |
| Random outputs | Weight loading failure | Check DMA transfers with ILA | Fix AXI handshake |
| Correct first layer, wrong rest | Accumulator overflow | Check accumulator width | Use 32-bit accumulator |
| Slow performance | Clock too low | Check timing report | Pipeline critical paths |
| High power | Clock gating not working | Check enable signal | Fix PE valid gating |
| Synthesis fails | Resource exhaustion | Check utilization report | Reduce PE array size |

### 16.11.2 ILA (Integrated Logic Analyzer) Setup

```verilog
// Add to top module for debugging
ila_0 ila_inst (
    .clk(clk),
    .probe0(state),           // 2-bit state
    .probe1(PE_array.trits[0][0]),  // First PE trit
    .probe2(PE_array.acts[0]),      // First activation
    .probe3(PE_array.results[0]),   // First result
    .probe4(ddr_rd_en),       // DDR read enable
    .probe5(word_count)       // Weight loading progress
);
```

### 16.11.3 Bit-Exact Verification

```verilog
module ternary_pe_tb;
    reg clk, rst_n;
    reg [1:0] trit;
    reg [7:0] activation;
    reg valid;
    wire [31:0] accumulator;
    wire done;

    ternary_pe dut (.*);

    initial begin
        clk = 0;
        forever #5 clk = ~clk;
    end

    initial begin
        // Test: +1 × 5 = 5
        rst_n = 0; trit = 2'b10; activation = 8'd5; valid = 0;
        #10 rst_n = 1;
        #10 valid = 1;
        #10 valid = 0;
        #10 assert(accumulator == 32'sd5) else $error("FAIL: +1×5");

        // Test: -1 × 5 = -5
        trit = 2'b01; activation = 8'd5; valid = 1;
        #10 valid = 0;
        #10 assert(accumulator == 32'sd0) else $error("FAIL: -1×5");

        // Test: 0 × 5 = 0 (skip)
        trit = 2'b00; activation = 8'd5; valid = 1;
        #10 valid = 0;
        #10 assert(accumulator == 32'sd0) else $error("FAIL: 0×5");

        $display("All tests passed!");
        $finish;
    end
endmodule
```

---

## 16.12 Cloud FPGA Options

For teams without physical FPGA boards, cloud FPGAs provide a low-risk way to experiment:

### 16.12.1 AWS F1

| Instance | FPGA | LUTs | DSP | BRAM | Cost/Hour |
|----------|------|------|-----|------|-----------|
| f1.2xlarge | Xilinx VU9P | 1.3M | 6,840 | 265 Mb | $0.65 |
| f1.4xlarge | 2× VU9P | 2.6M | 13,680 | 530 Mb | $1.30 |
| f1.16xlarge | 8× VU9P | 10.4M | 54,720 | 2.1 Gb | $5.20 |

**Advantages**: No upfront cost, scale to multiple FPGAs, integrates with AWS ML services.

### 16.12.2 Azure Alveo

| SKU | FPGA | Cost/Hour |
|-----|------|-----------|
| Standard_FPGA | Alveo U250 | $0.80 |
| Standard_FPGA_v2 | Alveo U55C (HBM) | $1.20 |

### 16.12.3 QuickLogic Cloud

| SKU | FPGA | Cost/Hour |
|-----|------|-----------|
| EOS S3 | QuickLogic EOS S3 | $0.10 |

**Best for**: Ultra-low-power edge experiments.

---

## 16.13 Results Template

After running experiments, document results in this format:

```markdown
## Experiment: [Model] on [Board]

### Configuration
- **FPGA Board**: Xilinx ZCU104
- **PE Array**: 64×64 @ 200 MHz
- **Model**: ResNet-50 (ternary)
- **Packing**: 10→16 (10 trits per 16-bit word)
- **Activations**: INT8
- **Scales**: Per-channel FP16

### Results
| Metric | Target | Actual | Notes |
|--------|--------|--------|-------|
| Top-1 Accuracy | >74% | 74.3% | Within expected range |
| Latency (single image) | <5 ms | 4.2 ms | 224×224 input |
| Throughput | >200 FPS | 238 FPS | Batch=1 |
| Power | <1 W | 0.85 W | Measured via XADC |
| Energy/Inference | <5 mJ | 3.6 mJ | Power × Latency |
| LUT Utilization | <80% | 42% | Leaves room for larger array |
| DSP Utilization | <80% | 28% | Underutilized (ternary is DSP-free) |
| BRAM Utilization | <80% | 35% | Weight buffer + line buffers |
| Fmax | >200 MHz | 215 MHz | Timing closed |

### Comparison with GPU
| Metric | FPGA (ZCU104) | GPU (A100) | Ratio |
|--------|---------------|-----------|-------|
| Latency | 4.2 ms | 1.2 ms | 3.5× slower |
| Power | 0.85 W | 300 W | 350× less |
| Energy/Inference | 3.6 mJ | 360 mJ | 100× less |
| Cost | $350 | $10,000 | 28× cheaper |

### Conclusion
The ternary FPGA accelerator demonstrates 100× better energy efficiency than GPU for ternary inference, validating the custom ASIC path for production deployment.
```

---

## 16.14 Next Steps

After successful FPGA prototyping:

1. **Scale up PE array**: 64×64 → 128×128 (requires Alveo U250 or larger)
2. **Add HBM support**: Alveo U55C with HBM2e for 1B+ parameter models
3. **Implement LLM inference**: Transformer layers with KV cache
4. **Optimize pipeline**: Reduce latency via loop unrolling and pipelining
5. **Measure real workloads**: Run actual customer models
6. **Compare with GPU**: Head-to-head energy efficiency comparison
7. **Publish results**: Write paper for FPGA or MICRO conference
8. **Plan ASIC**: Use FPGA numbers to justify tape-out

---

## 16.15 References

| Paper/Resource | Description |
|----------------|-------------|
| BitNet b1.58 (Microsoft, 2024) | Ternary weight training methodology |
| BitNet a4.58 (Microsoft, 2024) | Ternary weights + INT4 activations |
| Xilinx UG1137 | ZCU104 User Guide |
| Xilinx PG302 | Alveo U250 Data Sheet |
| IntelUG-20130 | Agilex 7 User Guide |
| TWN (Li et al., 2016) | Ternary Weight Networks paper |
| TTQ (Zhu et al., 2017) | Trained Ternary Quantization paper |
