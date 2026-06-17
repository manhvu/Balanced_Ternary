# 17. ASIC Implementation Guide for Balanced Ternary Accelerators

## 17.1 Overview

This document provides a comprehensive guide for implementing balanced ternary neural network accelerators in ASIC, covering the complete flow from RTL design to silicon fabrication. It complements the architecture specification in [12-custom-ternary-accelerator-design.md](12-custom-ternary-accelerator-design.md) with practical ASIC implementation details.

---

## 17.2 ASIC Design Flow

### 17.2.1 High-Level Design Flow

```
┌─────────────────────────────────────────────────────────────┐
│                    ASIC Design Flow                          │
│                                                               │
│  Phase 1: Specification                                      │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ • Architecture specification (§12.2)                 │    │
│  │ • Performance/power/area targets                     │    │
│  │ • Interface definitions (AXI4, PCIe)                 │    │
│  │ • Technology selection (foundry, node)                │    │
│  └─────────────────────────────────────────────────────┘    │
│                           │                                   │
│                           ▼                                   │
│  Phase 2: RTL Design                                         │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ • Verilog/SystemVerilog implementation               │    │
│  │ • Module hierarchy (PE, array, memory, control)      │    │
│  │ • Clock domain crossing (CDC) strategy                │    │
│  │ • Reset strategy (synchronous deassert)              │    │
│  └─────────────────────────────────────────────────────┘    │
│                           │                                   │
│                           ▼                                   │
│  Phase 3: Functional Verification                            │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ • UVM testbench development                          │    │
│  │ • Assertion-based verification (ABV)                 │    │
│  │ • Coverage-driven verification                       │    │
│  │ • Co-simulation with PyTorch golden model            │    │
│  └─────────────────────────────────────────────────────┘    │
│                           │                                   │
│                           ▼                                   │
│  Phase 4: Synthesis                                          │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ • RTL → Gate-level netlist                           │    │
│  │ • Technology mapping                                 │    │
│  │ • Timing constraints (SDC)                           │    │
│  │ • Power optimization (clock gating, power islands)   │    │
│  └─────────────────────────────────────────────────────┘    │
│                           │                                   │
│                           ▼                                   │
│  Phase 5: Physical Design                                    │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ • Floorplanning                                      │    │
│  │ • Place and route                                    │    │
│  │ • Clock tree synthesis (CTS)                         │    │
│  │ • Signal integrity (SI) analysis                     │    │
│  │ • Power integrity (IR drop) analysis                 │    │
│  └─────────────────────────────────────────────────────┘    │
│                           │                                   │
│                           ▼                                   │
│  Phase 6: Signoff                                            │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ • Static timing analysis (STA)                       │    │
│  │ • Design rule check (DRC)                            │    │
│  │ • Layout vs. schematic (LVS)                         │    │
│  │ • Electrical rule check (ERC)                        │    │
│  │ • Formal verification                                │    │
│  └─────────────────────────────────────────────────────┘    │
│                           │                                   │
│                           ▼                                   │
│  Phase 7: Tape-out and Fabrication                           │
│  ┌─────────────────────────────────────────────────────┐    │
│  │ • GDSII generation                                   │    │
│  │ • Mask data preparation                              │    │
│  │ • Shuttle run or full wafer lot                      │    │
│  │ • Post-silicon validation                            │    │
│  └─────────────────────────────────────────────────────┘    │
└─────────────────────────────────────────────────────────────┘
```

---

## 17.3 Technology Selection

### 17.3.1 Process Node Comparison

| Process | Gate Density | SRAM Bit Cell | Power Efficiency | Cost (wafer) | Best For |
|---------|--------------|---------------|------------------|--------------|----------|
| 28nm | 10M gates/mm² | 0.10 µm² | 0.5 pJ/MAC | $3,000 | Prototyping |
| 16nm | 30M gates/mm² | 0.05 µm² | 0.2 pJ/MAC | $5,000 | Edge (high volume) |
| 7nm | 100M gates/mm² | 0.02 µm² | 0.05 pJ/MAC | $10,000 | Edge/Server |
| 5nm | 170M gates/mm² | 0.015 µm² | 0.03 pJ/MAC | $15,000 | High-end server |
| 3nm | 300M gates/mm² | 0.01 µm² | 0.02 pJ/MAC | $20,000 | Next-gen (2025+) |

**Recommendation for ternary accelerator:**
- **Edge target**: 7nm (best density/cost balance)
- **Server target**: 5nm (maximum efficiency)
- **Prototyping**: 28nm or 16nm FPGA

### 17.3.2 Foundry Selection

| Foundry | 7nm | 5nm | 3nm | PDK Availability | Shuttle Program |
|---------|-----|-----|-----|------------------|-----------------|
| TSMC | N7 | N5 | N3 | Public | MPW (Multi-Project Wafer) |
| Samsung | 7LPP | 5LPE | 3GAE | Public | MPW |
| Intel | — | Intel 4 | Intel 3 | Limited | IFS |
| GlobalFoundries | 12nm | — | — | Public | Yes |

**Recommended**: TSMC N7 or N5 for first silicon.

---

## 17.4 RTL Design Details

### 17.4.1 Module Hierarchy

```
ternary_accelerator_top
├── control_processor (RISC-V RV64GC)
│   ├── instruction_fetch
│   ├── instruction_decode
│   ├── alu
│   ├── register_file
│   └── csr_unit
├── axi4_interconnect
│   ├── axi_crossbar
│   └── axi_slave_mux
├── ternary_gemm_array
│   ├── weight_decoder_array (128×)
│   │   ├── trit_unpacker_10_16
│   │   └── differential_encoder
│   ├── pe_array (128×128)
│   │   ├── ternary_pe (×16,384)
│   │   │   ├── zero_skip_detector
│   │   │   ├── negation_mux
│   │   │   ├── adder_subtractor
│   │   │   └── accumulator_register
│   │   └── weight_stationary_controller
│   ├── accumulator_array (128×)
│   │   ├── scale_multiplier
│   │   └── bias_adder
│   └── gemm_controller
├── fp16_compute_unit
│   ├── fp16_systolic_array (32×32)
│   ├── softmax_engine
│   ├── layernorm_engine
│   ├── activation_unit (SiLU/GELU/ReLU)
│   └── residual_adder
├── memory_subsystem
│   ├── weight_sram (8 MB)
│   │   ├── sram_bank (×128)
│   │   ├── bank_controller
│   │   └── ecc_encoder_decoder
│   ├── scratchpad_sram (4 MB)
│   │   ├── sram_bank (×4)
│   │   ├── bank_controller
│   │   └── ecc_encoder_decoder
│   └── dma_engine (8-channel)
│       ├── dma_channel (×8)
│       └── arbiter
├── host_interface
│   ├── pcie_controller (PCIe 4.0 ×4)
│   ├── axi4_slave_port
│   └── interrupt_controller
└── test_wrapper
    ├── scan_controller
    ├── bist_engine
    └── jtag_tap
```

### 17.4.2 Clock Domain Strategy

```
┌─────────────────────────────────────────────────────────────┐
│ Clock Domains                                                 │
│                                                               │
│  CLK_CORE (1.0 GHz)                                          │
│  ├── PE array                                                 │
│  ├── FP16 compute unit                                        │
│  ├── Accumulator array                                        │
│  └── Weight decoder array                                     │
│                                                               │
│  CLK_MEM (1.0 GHz)                                            │
│  ├── Weight SRAM                                              │
│  ├── Scratchpad SRAM                                          │
│  └── SRAM controllers                                         │
│                                                               │
│  CLK_DMA (500 MHz)                                            │
│  ├── DMA engine                                               │
│  └── AXI interconnect                                         │
│                                                               │
│  CLK_HOST (250 MHz)                                           │
│  ├── PCIe controller                                          │
│  └── Host interface                                           │
│                                                               │
│  CDC Synchronizers:                                           │
│  ├── Core ↔ Memory: 2-FF synchronizer (same frequency)       │
│  ├── Core ↔ DMA: 2-FF synchronizer (2:1 ratio)              │
│  ├── DMA ↔ Host: 2-FF synchronizer (2:1 ratio)              │
│  └── Async reset synchronizer (all domains)                   │
└─────────────────────────────────────────────────────────────┘
```

### 17.4.3 Reset Strategy

```verilog
// Synchronous deassert, asynchronous assert reset
module ternary_accelerator_top (
    input  wire clk_core,
    input  wire clk_mem,
    input  wire clk_dma,
    input  wire clk_host,
    input  wire rst_n,           // Active-low, async assert
    // ... other ports
);

    // Reset synchronizers for each clock domain
    reg [2:0] rst_sync_core;
    reg [2:0] rst_sync_mem;
    reg [2:0] rst_sync_dma;
    reg [2:0] rst_sync_host;

    always @(posedge clk_core or negedge rst_n) begin
        if (!rst_n) rst_sync_core <= 3'b000;
        else rst_sync_core <= {rst_sync_core[1:0], 1'b1};
    end

    always @(posedge clk_mem or negedge rst_n) begin
        if (!rst_n) rst_sync_mem <= 3'b000;
        else rst_sync_mem <= {rst_sync_mem[1:0], 1'b1};
    end

    // ... similar for other domains

    wire rst_n_core = rst_sync_core[2];
    wire rst_n_mem  = rst_sync_mem[2];
    wire rst_n_dma  = rst_sync_dma[2];
    wire rst_n_host = rst_sync_host[2];

endmodule
```

---

## 17.5 Memory Implementation

### 17.5.1 SRAM Compiler Selection

For the ternary accelerator, we need two types of SRAM:

| SRAM Type | Capacity | Word Width | Ports | Compiler | Technology |
|-----------|----------|------------|-------|----------|------------|
| Weight SRAM | 64 KB banks | 16 bits | 1R | ARM Artisan | TSMC N7 |
| Scratchpad SRAM | 256 KB banks | 128 bits | 2R1W | ARM Artisan | TSMC N7 |

**SRAM compiler options:**
- ARM Artisan Memory compilers (industry standard)
- Synopsys memory compilers
- Samsung/SK Hynix embedded SRAM compilers

### 17.5.2 SRAM Timing

```
Weight SRAM (64 KB, 16-bit word, single-port):
┌─────────────────────────────────────────────────┐
│ Parameter          │ Value    │ Notes            │
├───────────────────┼──────────┼──────────────────┤
│ Read access time   │ 0.4 ns   │ Clock to Q       │
│ Setup time         │ 0.2 ns   │ Address/data     │
│ Hold time          │ 0.05 ns  │                  │
│ Read cycle time    │ 0.6 ns   │ 1.67 GHz max     │
│ Write cycle time   │ 0.5 ns   │ 2.0 GHz max      │
│ Dynamic power/read │ 0.3 pJ   │ Per access       │
│ Leakage (per bank) │ 2 µW     │ @ 0.75V, 25°C    │
└─────────────────────────────────────────────────┘

Scratchpad SRAM (256 KB, 128-bit word, 2R1W):
┌─────────────────────────────────────────────────┐
│ Parameter          │ Value    │ Notes            │
├───────────────────┼──────────┼──────────────────┤
│ Read access time   │ 0.5 ns   │ Clock to Q       │
│ Setup time         │ 0.3 ns   │ Address/data     │
│ Hold time          │ 0.05 ns  │                  │
│ Read cycle time    │ 0.8 ns   │ 1.25 GHz max     │
│ Write cycle time   │ 0.6 ns   │ 1.67 GHz max     │
│ Dynamic power/read │ 1.2 pJ   │ Per access       │
│ Leakage (per bank) │ 8 µW     │ @ 0.75V, 25°C    │
└─────────────────────────────────────────────────┘
```

### 17.5.3 ECC Implementation

All SRAM instances include SEC-DED (Single Error Correct, Double Error Detect) ECC:

```verilog
module ecc_encoder #(
    parameter DATA_WIDTH = 64
)(
    input  wire [DATA_WIDTH-1:0] data_in,
    output wire [DATA_WIDTH+7:0] data_out  // 64 data + 8 ECC bits
);

    // Hsiao SEC-DED ECC encoding
    // Polynomial: x^8 + x^7 + x^6 + x^4 + x^2 + 1
    wire [7:0] parity;

    assign parity[0] = ^(data_in & 64'h00000000000000AA);
    assign parity[1] = ^(data_in & 64'h00000000000000CC);
    assign parity[2] = ^(data_in & 64'h00000000000000F0);
    assign parity[3] = ^(data_in & 64'h000000000000FF00);
    assign parity[4] = ^(data_in & 64'h0000000000FF0000);
    assign parity[5] = ^(data_in & 64'h00000000FF000000);
    assign parity[6] = ^(data_in & 64'h000000FF00000000);
    assign parity[7] = ^(data_in & 64'h00FF000000000000);

    assign data_out = {parity, data_in};

endmodule
```

---

## 17.6 Physical Design

### 17.6.1 Floorplan

```
┌─────────────────────────────────────────────────────────────────┐
│                    Die Floorplan (7nm, 25 mm²)                    │
│                                                                   │
│  ┌───────────────────────────────────────────────────────────┐  │
│  │                    I/O Ring                                 │  │
│  │  ┌─────────────────────────────────────────────────────┐  │  │
│  │  │                  Core Power Grid                      │  │  │
│  │  │                                                       │  │  │
│  │  │  ┌──────────────┐  ┌──────────────────────────────┐ │  │  │
│  │  │  │ Weight SRAM  │  │      Ternary GEMM Array       │ │  │  │
│  │  │  │   8 MB       │  │        128 × 128 PEs          │ │  │  │
│  │  │  │  3.0 × 2.5mm │  │       3.0 × 3.0 mm            │ │  │  │
│  │  │  │  ┌────┐      │  │  ┌────┐ ┌────┐ ┌────┐        │ │  │  │
│  │  │  │  │Bank│ ...   │  │  │PE  │ │PE  │ │PE  │ ...    │ │  │  │
│  │  │  │  │ 0  │       │  │  │0,0 │ │0,1 │ │0,2 │        │ │  │  │
│  │  │  │  └────┘       │  │  └────┘ └────┘ └────┘        │ │  │  │
│  │  │  └──────────────┘  └──────────────────────────────┘ │  │  │
│  │  │                                                       │  │  │
│  │  │  ┌──────────────┐  ┌──────────────────────────────┐ │  │  │
│  │  │  │ Scratchpad   │  │      FP16 Compute Unit        │ │  │  │
│  │  │  │ SRAM 4 MB    │  │  ┌────────┐ ┌────────┐       │ │  │  │
│  │  │  │ 2.0 × 2.0 mm │  │  │Softmax │ │32×32   │       │ │  │  │
│  │  │  └──────────────┘  │  │Engine  │ │FP16    │       │ │  │  │
│  │  │                     │  └────────┘ └────────┘       │ │  │  │
│  │  │  ┌──────────────┐  └──────────────────────────────┘ │  │  │
│  │  │  │ RISC-V +     │                                    │  │  │
│  │  │  │ Interconnect │  ┌──────────────────────────────┐ │  │  │
│  │  │  │ + PCIe I/F   │  │      DMA Engine               │ │  │  │
│  │  │  │  1.5 × 1.5mm │  │       8-channel                │ │  │  │
│  │  │  └──────────────┘  │       1.0 × 0.5 mm             │ │  │  │
│  │  │                     └──────────────────────────────┘ │  │  │
│  │  └─────────────────────────────────────────────────────┘  │  │
│  └───────────────────────────────────────────────────────────┘  │
│                                                                   │
│  Power Grid:                                                     │
│  ├── VDD: 4× horizontal + 4× vertical (M8/M9)                   │
│  ├── VSS: 4× horizontal + 4× vertical (M8/M9)                   │
│  └── Decoupling capacitors: 5% of die area                       │
│                                                                   │
│  Area Breakdown:                                                 │
│  ├── SRAM:     19 mm² (76%)                                      │
│  ├── Logic:     4 mm² (16%)                                      │
│  ├── I/O:       1 mm² (4%)                                       │
│  └── Decap:     1 mm² (4%)                                       │
└─────────────────────────────────────────────────────────────────┘
```

### 17.6.2 Power Grid Design

```
Power Distribution Network:
┌─────────────────────────────────────────────────────────────┐
│ Metal Stack:                                                  │
│ M1-M4:  Local routing (signal)                               │
│ M5-M6:  Intermediate routing                                 │
│ M7:     Power (VDD/VSS stripes, horizontal)                  │
│ M8:     Power (VDD/VSS grid, vertical)                       │
│ M9:     Power (VDD/VSS grid, horizontal)                     │
│                                                               │
│ Power Grid Parameters:                                        │
│ ├── VDD stripe width: 2 µm                                   │
│ ├── VSS stripe width: 2 µm                                   │
│ ├── Stripe pitch: 10 µm                                      │
│ ├── Via density: 1000 vias/mm²                               │
│ └── IR drop budget: 5% of Vnom (0.0375V @ 0.75V)            │
│                                                               │
│ Decoupling Capacitors:                                        │
│ ├── Target: 5% of die area (1.25 mm²)                        │
│ ├── Type: MIM (Metal-Insulator-Metal)                        │
│ └── Density: 2 fF/µm²                                        │
└─────────────────────────────────────────────────────────────┘
```

### 17.6.3 Clock Tree Synthesis (CTS)

```
Clock Tree Architecture:
┌─────────────────────────────────────────────────────────────┐
│                                                               │
│  CLK_IN ──► Clock Divider ──► PLL (optional)                 │
│               │                                              │
│               ├──► CLK_CORE (1.0 GHz)                        │
│               │      ├──► Clock buffer tree                   │
│               │      ├──► H-tree for PE array                 │
│               │      └──► Skew budget: < 50 ps               │
│               │                                              │
│               ├──► CLK_MEM (1.0 GHz)                          │
│               │      └──► SRAM clock distribution             │
│               │                                              │
│               ├──► CLK_DMA (500 MHz)                          │
│               │      └──► DMA engine clock                    │
│               │                                              │
│               └──► CLK_HOST (250 MHz)                         │
│                      └──► PCIe clock                          │
│                                                               │
│  CTS Parameters:                                              │
│  ├── Target skew: 50 ps                                      │
│  ├── Max transition: 150 ps                                  │
│  ├── Max capacitance: 200 fF                                 │
│  └── Buffer library: TSMC N7 CKBD (clock buffer)             │
└─────────────────────────────────────────────────────────────┘
```

---

## 17.7 Timing Constraints (SDC)

### 17.7.1 Clock Definitions

```tcl
# Main clocks
create_clock -name clk_core -period 1.0 -waveform {0.0 0.5} [get_ports clk_core]
create_clock -name clk_mem -period 1.0 -waveform {0.0 0.5} [get_ports clk_mem]
create_clock -name clk_dma -period 2.0 -waveform {0.0 1.0} [get_ports clk_dma]
create_clock -name clk_host -period 4.0 -waveform {0.0 2.0} [get_ports clk_host]

# Clock groups (async domains)
set_clock_groups -asynchronous \
    -group [get_clocks clk_core] \
    -group [get_clocks clk_mem] \
    -group [get_clocks clk_dma] \
    -group [get_clocks clk_host]

# Generated clocks
create_generated_clock -name clk_gemm \
    -source [get_ports clk_core] \
    -master_clock clk_core \
    -div 1 \
    [get_pins ternary_gemm_array/clk_gate/Q]
```

### 17.7.2 Input/Output Constraints

```tcl
# Input delays (relative to clk_core)
set_input_delay -clock clk_core -max 0.3 [get_ports {data_in[*]}]
set_input_delay -clock clk_core -min 0.1 [get_ports {data_in[*]}]

# Output delays (relative to clk_core)
set_output_delay -clock clk_core -max 0.3 [get_ports {data_out[*]}]
set_output_delay -clock clk_core -min 0.1 [get_ports {data_out[*]}]

# False paths (async resets)
set_false_path -from [get_ports rst_n]

# Multicycle paths (FP16 operations)
set_multicycle_path -setup 2 -from [get_pins fp16_compute_unit/*/CK] \
                              -to [get_pins fp16_compute_unit/*/D]
set_multicycle_path -hold 1 -from [get_pins fp16_compute_unit/*/CK] \
                             -to [get_pins fp16_compute_unit/*/D]
```

---

## 17.8 Power Optimization

### 17.8.1 Clock Gating

```verilog
// Integrated Clock Gating Cell (ICG)
module icg (
    input  wire clk_in,
    input  wire enable,
    output wire clk_out
);

    // Latch-based clock gate (TSMC N7 library cell)
    reg enable_latch;

    always @(*) begin
        if (!clk_in)
            enable_latch = enable;
    end

    assign clk_out = clk_in & enable_latch;

endmodule

// Usage in ternary PE
module ternary_pe (
    input  wire clk,
    input  wire weight_is_zero,
    // ... other ports
);

    wire pe_enable = ~weight_is_zero;
    wire clk_gated;

    icg u_icg (
        .clk_in(clk),
        .enable(pe_enable),
        .clk_out(clg_gated)
    );

    // PE logic runs on gated clock
    always @(posedge clk_gated or negedge rst_n) begin
        // ... PE computation
    end

endmodule
```

### 17.8.2 Power Islands

```
Power Island Strategy:
┌─────────────────────────────────────────────────────────────┐
│                                                               │
│  Always-On Domain:                                            │
│  ├── RISC-V control processor                                │
│  ├── PCIe controller                                          │
│  └── Interrupt controller                                     │
│  └── VDD: 0.75V (nominal)                                    │
│                                                               │
│  Switchable Domain 1: Ternary GEMM Array                     │
│  ├── PE array (128×128)                                      │
│  ├── Weight decoder array                                     │
│  └── VDD: 0.75V (can power-gate to 0V)                       │
│                                                               │
│  Switchable Domain 2: FP16 Compute Unit                       │
│  ├── FP16 systolic array                                      │
│  ├── Softmax/LayerNorm engines                                │
│  └── VDD: 0.75V (can power-gate to 0V)                       │
│                                                               │
│  Switchable Domain 3: Memory                                  │
│  ├── Weight SRAM (8 MB)                                      │
│  ├── Scratchpad SRAM (4 MB)                                  │
│  └── VDD: 0.75V (can power-gate to 0V)                       │
│                                                               │
│  Power Gating Implementation:                                 │
│  ├── Header switch: PMOS sleep transistors                    │
│  ├── Retention: None (weights reloaded from host)             │
│  └── Wake-up latency: ~10 µs (SRAM initialization)           │
└─────────────────────────────────────────────────────────────┘
```

### 17.8.3 Voltage Scaling

```
Dynamic Voltage and Frequency Scaling (DVFS):
┌─────────────────────────────────────────────────────────────┐
│                                                               │
│  Mode 1: Full Performance                                     │
│  ├── VDD: 0.75V                                              │
│  ├── Freq: 1.0 GHz                                           │
│  ├── Power: 4.0W                                              │
│  └── Use case: Maximum throughput                             │
│                                                               │
│  Mode 2: Balanced                                             │
│  ├── VDD: 0.65V                                              │
│  ├── Freq: 750 MHz                                            │
│  ├── Power: 2.5W                                              │
│  └── Use case: Balanced performance/power                     │
│                                                               │
│  Mode 3: Low Power                                            │
│  ├── VDD: 0.55V                                              │
│  ├── Freq: 500 MHz                                            │
│  ├── Power: 1.2W                                              │
│  └── Use case: Battery-constrained edge devices               │
│                                                               │
│  Mode 4: Sleep                                                 │
│  ├── VDD: 0V (power-gated)                                   │
│  ├── Freq: 0                                                  │
│  ├── Power: <10 µW (leakage only)                             │
│  └── Use case: Standby                                        │
└─────────────────────────────────────────────────────────────┘
```

---

## 17.9 Signal Integrity

### 17.9.1 Crosstalk Analysis

```
Crosstalk Budget:
┌─────────────────────────────────────────────────────────────┐
│                                                               │
│  Weight Bus (differential pairs):                             │
│  ├── Coupling capacitance: ~50% of ground capacitance        │
│  ├── Noise margin: 150 mV (at 0.75V VDD)                     │
│  └── Mitigation: Differential signaling rejects common-mode  │
│                                                               │
│  Activation Bus (single-ended):                               │
│  ├── Coupling capacitance: ~30% of ground capacitance        │
│  ├── Noise margin: 100 mV                                     │
│  └── Mitigation: Shielding on critical nets                   │
│                                                               │
│  Clock Network:                                               │
│  ├── Shielding: VDD/VSS shields on all clock routes          │
│  ├── Buffer insertion: Every 200 µm                           │
│  └── Skew budget: < 50 ps                                     │
└─────────────────────────────────────────────────────────────┘
```

### 17.9.2 IR Drop Analysis

```
IR Drop Budget:
┌─────────────────────────────────────────────────────────────┐
│                                                               │
│  Target: < 5% of VDD (37.5 mV @ 0.75V)                       │
│                                                               │
│  Analysis Method:                                             │
│  ├── Static IR drop: Calculate worst-case current paths      │
│  ├── Dynamic IR drop: Simulate switching activity             │
│  └── Tool: Cadence Voltus or Synopsys PrimePower             │
│                                                               │
│  Worst-Case Scenario:                                         │
│  ├── All 128×128 PEs switching simultaneously                │
│  ├── Peak current: ~4A (4W @ 1.0V effective)                 │
│  ├── IR drop budget: 37.5 mV                                  │
│  └── Required R_VDD: < 9.4 mΩ                                │
│                                                               │
│  Mitigation:                                                  │
│  ├── Dense power grid (M7-M9)                                 │
│  ├── Decoupling capacitors (5% of die area)                   │
│  ├── Power vias: 1000 vias/mm²                                │
│  └── Voltage-aware placement                                  │
└─────────────────────────────────────────────────────────────┘
```

---

## 17.10 Design for Test (DFT)

### 17.10.1 Scan Chain Architecture

```
Scan Chain Structure:
┌─────────────────────────────────────────────────────────────┐
│                                                               │
│  JTAG TAP Controller                                          │
│       │                                                       │
│       ├──► Scan Chain 0: RISC-V control processor             │
│       │       ├── Instruction fetch logic                     │
│       │       ├── Decode logic                                │
│       │       └── ALU + register file                         │
│       │                                                       │
│       ├──► Scan Chain 1: Ternary GEMM array (PEs)             │
│       │       ├── PE[0,0] → PE[0,1] → ... → PE[127,127]     │
│       │       └── Weight registers (2-bit per PE)             │
│       │                                                       │
│       ├──► Scan Chain 2: Memory controllers                   │
│       │       ├── Weight SRAM controller                      │
│       │       └── Scratchpad SRAM controller                  │
│       │                                                       │
│       ├──► Scan Chain 3: FP16 compute unit                    │
│       │       ├── FP16 systolic array                         │
│       │       └── Softmax/LayerNorm engines                   │
│       │                                                       │
│       └──► Scan Chain 4: I/O and interconnect                 │
│               ├── AXI interconnect                            │
│               ├── PCIe controller                             │
│               └── DMA engine                                   │
│                                                               │
│  Scan Parameters:                                             │
│  ├── Scan chain length: ~500K flip-flops                      │
│  ├── Scan bandwidth: 64 bits                                  │
│  ├── Test time: ~10K cycles                                   │
│  └── Coverage target: > 99% stuck-at                          │
└─────────────────────────────────────────────────────────────┘
```

### 17.10.2 Memory BIST

```verilog
module sram_bist #(
    parameter ADDR_WIDTH = 16,
    parameter DATA_WIDTH = 16
)(
    input  wire clk,
    input  wire rst_n,
    input  wire start,
    output wire done,
    output wire pass,
    // SRAM interface
    output wire [ADDR_WIDTH-1:0] sram_addr,
    output wire [DATA_WIDTH-1:0] sram_wdata,
    input  wire [DATA_WIDTH-1:0] sram_rdata,
    output wire sram_we,
    output wire sram_ce
);

    // March C- algorithm
    // 1. Write 0 to all addresses
    // 2. Read 0 from all addresses
    // 3. Write 1 to all addresses
    // 4. Read 1 from all addresses
    // 5. Write 0 to all addresses (reverse order)
    // 6. Read 0 from all addresses (reverse order)

    localparam IDLE     = 3'd0;
    localparam WRITE_0  = 3'd1;
    localparam READ_0   = 3'd2;
    localparam WRITE_1  = 3'd3;
    localparam READ_1   = 3'd4;
    localparam DONE     = 3'd5;

    reg [2:0] state;
    reg [ADDR_WIDTH-1:0] addr_counter;
    reg pass_flag;

    always @(posedge clk or negedge rst_n) begin
        if (!rst_n) begin
            state <= IDLE;
            addr_counter <= 0;
            pass_flag <= 1'b1;
        end else begin
            case (state)
                IDLE: begin
                    if (start) begin
                        state <= WRITE_0;
                        addr_counter <= 0;
                    end
                end
                WRITE_0: begin
                    sram_addr <= addr_counter;
                    sram_wdata <= {DATA_WIDTH{1'b0}};
                    sram_we <= 1'b1;
                    sram_ce <= 1'b1;
                    if (addr_counter == {ADDR_WIDTH{1'b1}}) begin
                        state <= READ_0;
                        addr_counter <= 0;
                    end else begin
                        addr_counter <= addr_counter + 1;
                    end
                end
                READ_0: begin
                    sram_addr <= addr_counter;
                    sram_we <= 1'b0;
                    sram_ce <= 1'b1;
                    if (sram_rdata != {DATA_WIDTH{1'b0}}) begin
                        pass_flag <= 1'b0;
                    end
                    if (addr_counter == {ADDR_WIDTH{1'b1}}) begin
                        state <= WRITE_1;
                        addr_counter <= 0;
                    end else begin
                        addr_counter <= addr_counter + 1;
                    end
                end
                // ... similar for WRITE_1, READ_1, DONE
            endcase
        end
    end

    assign done = (state == DONE);
    assign pass = pass_flag;

endmodule
```

---

## 17.11 Formal Verification

### 17.11.1 Property Specifications

```systemverilog
// PE properties
property pe_zero_skip;
    @(posedge clk) disable iff (!rst_n)
    (weight_trit == 2'b01) |=> (partial_sum_out == $past(partial_sum_in));
endproperty

property pe_positive_add;
    @(posedge clk) disable iff (!rst_n)
    (weight_trit == 2'b10) |=> (partial_sum_out == $past(partial_sum_in) + $past(activation_x));
endproperty

property pe_negative_sub;
    @(posedge clk) disable iff (!rst_n)
    (weight_trit == 2'b00) |=> (partial_sum_out == $past(partial_sum_in) - $past(activation_x));
endproperty

// Assert properties
assert property (pe_zero_skip) else $error("PE zero-skip failed");
assert property (pe_positive_add) else $error("PE positive-add failed");
assert property (pe_negative_sub) else $error("PE negative-sub failed");

// Memory properties
property sram_read_after_write;
    @(posedge clk) disable iff (!rst_n)
    (sram_we && sram_ce) |=> (sram_rdata == $past(sram_wdata));
endproperty

assert property (sram_read_after_write) else $error("SRAM RAW failed");
```

---

## 17.12 Post-Silicon Validation

### 17.12.1 Test Board Design

```
Test Board Specifications:
┌─────────────────────────────────────────────────────────────┐
│                                                               │
│  PCB Stackup: 10-layer, impedance-controlled                 │
│  ├── Layer 1: Signal (top)                                   │
│  ├── Layer 2: Ground plane                                   │
│  ├── Layer 3: Signal                                         │
│  ├── Layer 4: Power (VDD_0.75V)                              │
│  ├── Layer 5: Ground plane                                   │
│  ├── Layer 6: Signal                                         │
│  ├── Layer 7: Power (VDD_0.75V)                              │
│  ├── Layer 8: Signal                                         │
│  ├── Layer 9: Ground plane                                   │
│  └── Layer 10: Signal (bottom)                               │
│                                                               │
│  Key Components:                                              │
│  ├── Ternary accelerator chip (BGA package)                  │
│  ├── FPGA for test control (Xilinx ZCU104)                   │
│  ├── DDR4 SO-DIMM slot (for host memory)                     │
│  ├── PCIe connector (x4 Gen4)                                │
│  ├── Power regulators (multi-rail, sequenced)                │
│  ├── Clock generators (LVDS, 100 MHz reference)              │
│  ├── JTAG header (for scan testing)                          │
│  ├── UART header (for debug)                                 │
│  └── Thermal solution (heatsink + fan)                        │
│                                                               │
│  Signal Integrity:                                            │
│  ├── Impedance: 50Ω single-ended, 100Ω differential        │
│  ├── Length matching: < 5 mm for differential pairs           │
│  ├── Via stubs: Back-drilled for > 1 GHz signals             │
│  └── Decoupling: 0.1 µF + 1 µF + 10 µF per power pin       │
└─────────────────────────────────────────────────────────────┘
```

### 17.12.2 Validation Test Plan

```
Post-Silicon Validation Test Plan:
┌─────────────────────────────────────────────────────────────┐
│                                                               │
│  Phase 1: Bring-up (Week 1-2)                                │
│  ├── Power sequence verification                              │
│  ├── Clock startup and frequency measurement                  │
│  ├── JTAG connectivity test                                   │
│  ├── Scan chain integrity test                                │
│  └── SRAM BIST (all memories)                                 │
│                                                               │
│  Phase 2: Functional Testing (Week 3-4)                       │
│  ├── GEMM correctness (known inputs/outputs)                  │
│  ├── FP16 unit accuracy (vs. golden model)                    │
│  ├── DMA transfer verification                                │
│  ├── PCIe link training and enumeration                       │
│  └── Interrupt handling                                       │
│                                                               │
│  Phase 3: Performance Validation (Week 5-6)                   │
│  ├── Maximum frequency characterization                       │
│  ├── Throughput measurement (tokens/sec)                      │
│  ├── Latency measurement (ms/token)                           │
│  ├── Memory bandwidth utilization                             │
│  └── Power measurement at various operating points            │
│                                                               │
│  Phase 4: Stress Testing (Week 7-8)                           │
│  ├── Temperature sweep (-40°C to +125°C)                      │
│  ├── Voltage sweep (0.65V to 0.85V)                           │
│  ├── Long-duration reliability test (1000 hours)              │
│  ├── Error injection testing                                  │
│  └── EMI/EMC pre-compliance                                   │
│                                                               │
│  Phase 5: Model Validation (Week 9-10)                        │
│  ├── ResNet-50 inference accuracy                             │
│  ├── BERT-base inference accuracy                             │
│  ├── GPT-2 inference (124M parameters)                        │
│  ├── LLaMA-7B inference (quantized)                           │
│  └── End-to-end latency/throughput vs. targets                │
└─────────────────────────────────────────────────────────────┘
```

---

## 17.13 Manufacturing Considerations

### 17.13.1 Yield Analysis

```
Yield Estimation (Poisson model):
┌─────────────────────────────────────────────────────────────┐
│                                                               │
│  Defect density (D0): 0.1 defects/cm² @ 7nm (mature)         │
│                                                               │
│  Die area: 25 mm² = 0.25 cm²                                  │
│                                                               │
│  Yield = e^(-D0 × A) = e^(-0.1 × 0.25) = e^(-0.025)        │
│        = 0.975 (97.5%)                                        │
│                                                               │
│  Wafer diameter: 300 mm                                        │
│  Wafer area: π × (15)² = 706.86 cm²                           │
│  Gross dies/wafer: 706.86 / 0.25 = 2,827                     │
│  Net dies/wafer: 2,827 × 0.975 = 2,756                       │
│                                                               │
│  Wafer cost: ~$10,000                                         │
│  Die cost: $10,000 / 2,756 = ~$3.63                          │
│  + Packaging: ~$5                                              │
│  + Testing: ~$2                                               │
│  Total per chip: ~$10.63                                       │
│                                                               │
│  At 100K volume: ~$8 per chip (volume discount)               │
└─────────────────────────────────────────────────────────────┘
```

### 17.13.2 Package Selection

```
Package Options:
┌─────────────────────────────────────────────────────────────┐
│                                                               │
│  Option 1: BGA (Ball Grid Array) - Recommended                │
│  ├── Pitch: 0.8 mm                                           │
│  ├── Ball count: 400 (20×20)                                  │
│  ├── Size: 16×16 mm                                           │
│  ├── Thermal: Exposed die pad (solder)                        │
│  └── Cost: ~$3 per package                                    │
│                                                               │
│  Option 2: FC-BGA (Flip-Chip BGA) - High Performance         │
│  ├── Pitch: 0.4 mm                                            │
│  ├── Ball count: 1000 (31×31 + depopulated corners)          │
│  ├── Size: 12×12 mm                                           │
│  ├── Thermal: Flip-chip bumps + lid                           │
│  └── Cost: ~$5 per package                                    │
│                                                               │
│  Option 3: QFN (Quad Flat No-Lead) - Cost Optimized          │
│  ├── Pitch: 0.5 mm                                            │
│  ├── Pin count: 200 (50 per side)                             │
│  ├── Size: 14×14 mm                                           │
│  ├── Thermal: Exposed pad                                     │
│  └── Cost: ~$1 per package                                    │
│                                                               │
│  Recommendation: BGA for first silicon (ease of routing)      │
│                  FC-BGA for production (smaller size)          │
└─────────────────────────────────────────────────────────────┘
```

---

## 17.14 Cost Analysis

### 17.14.1 NRE (Non-Recurring Engineering) Costs

```
NRE Cost Breakdown (7nm):
┌─────────────────────────────────────────────────────────────┐
│                                                               │
│  RTL Design:                                                  │
│  ├── Architecture team (3 engineers × 12 months): $900K      │
│  ├── RTL team (5 engineers × 12 months): $1,500K             │
│  ├── Verification team (3 engineers × 12 months): $900K      │
│  └── Tools (Synopsys/Cadence licenses): $500K                │
│  Total RTL: $3,800K                                           │
│                                                               │
│  Physical Design:                                             │
│  ├── Backend team (2 engineers × 6 months): $400K            │
│  ├── SRAM compiler licenses: $200K                            │
│  └── Physical design tools: $300K                             │
│  Total Physical: $900K                                        │
│                                                               │
│  Mask Set:                                                    │
│  ├── 7nm mask set (approx. 60 layers): $3,000K               │
│  └── E-beam programming (optional): $100K                     │
│  Total Masks: $3,100K                                         │
│                                                               │
│  First Silicon:                                               │
│  ├── MPW (Multi-Project Wafer) shuttle: $50K                 │
│  └── Full wafer lot (25 wafers): $250K                        │
│  Total Silicon: $300K                                         │
│                                                               │
│  Validation:                                                  │
│  ├── Test board design and fabrication: $50K                  │
│  ├── ATE (Automated Test Equipment) time: $100K              │
│  └── Validation engineering (3 months): $200K                 │
│  Total Validation: $350K                                      │
│                                                               │
│  ──────────────────────────────────────────────────────────  │
│  TOTAL NRE: ~$8,450K                                          │
│                                                               │
│  Amortized over 100K units: ~$84.50 per unit                  │
└─────────────────────────────────────────────────────────────┘
```

### 17.14.2 Unit Cost Analysis

```
Unit Cost (at volume):
┌─────────────────────────────────────────────────────────────┐
│                                                               │
│  Die cost (100K volume):                                      │
│  ├── Wafer cost: $10,000                                      │
│  ├── Net dies/wafer: 2,756                                    │
│  ├── Die cost: $3.63                                          │
│  └── Yield-adjusted: $3.75 (accounting for test fallout)     │
│                                                               │
│  Packaging:                                                   │
│  ├── BGA package: $3.00                                       │
│  ├── Assembly: $1.50                                          │
│  └── Total: $4.50                                             │
│                                                               │
│  Testing:                                                     │
│  ├── ATE time: 2 seconds per die                              │
│  ├── ATE cost: $0.50 per die                                  │
│  └── Total: $0.50                                             │
│                                                               │
│  ──────────────────────────────────────────────────────────  │
│  Total unit cost: $8.75                                        │
│  + NRE amortization: $84.50                                    │
│  ──────────────────────────────────────────────────────────  │
│  Total cost per chip: $93.25                                   │
│                                                               │
│  At 1M volume:                                                 │
│  ├── Die cost: $3.25 (higher yield)                           │
│  ├── Packaging: $3.50                                          │
│  ├── Testing: $0.30                                            │
│  └── NRE amortization: $8.45                                   │
│  Total: $15.50 per chip                                        │
└─────────────────────────────────────────────────────────────┘
```

---

## 17.15 Comparison with Existing Solutions

| Metric | Ternary ASIC (This Design) | GPU (NVIDIA A100) | NPU (Google EdgeTPU) | FPGA (Xilinx VU9P) |
|--------|---------------------------|-------------------|----------------------|---------------------|
| Technology | 7nm | 7nm | 28nm | 16nm |
| Die area | 25 mm² | 814 mm² | 100 mm² | 900 mm² |
| Power | 4W | 400W | 2W | 10W |
| Weight format | Ternary (1.585b) | FP16/INT8 | INT8 | Configurable |
| Model size (1B) | 200 MB | 2 GB | 1 GB | 1 GB |
| Decode throughput | ~50K tok/s | ~15K tok/s | ~5K tok/s | ~10K tok/s |
| Tokens/joule | ~12,500 | ~37 | ~2,500 | ~1,000 |
| NRE cost | $8.5M | N/A (commercial) | N/A (commercial) | $1M |
| Unit cost | $15-90 | $10,000 | $50 | $2,000 |
| Flexibility | Fixed-function | Programmable | Fixed-function | Reconfigurable |

---

## 17.16 Risk Assessment

| Risk | Probability | Impact | Mitigation |
|------|-------------|--------|------------|
| Yield below 80% | Medium | High | Multiple mask revisions, MPW first |
| Timing closure failure | Low | High | Aggressive floorplanning, pipeline stages |
| SRAM failure | Low | Critical | ECC on all memories, redundancy |
| Thermal throttling | Medium | Medium | Power gating, DVFS |
| Host interface issues | Low | Medium | Extensive simulation, FPGA prototype |
| Model accuracy degradation | Medium | High | Co-simulation with PyTorch, QAT |
| Mask cost overrun | Low | High | Fixed-price contracts, MPW sharing |
| Schedule delays | Medium | Medium | Parallel verification, early bring-up |

---

## 17.17 Timeline

```
ASIC Implementation Timeline:
┌─────────────────────────────────────────────────────────────┐
│                                                               │
│  Month 1-3: RTL Design                                       │
│  ├── PE and array design                                     │
│  ├── Memory subsystem design                                 │
│  ├── Control processor integration                           │
│  └── Initial verification                                    │
│                                                               │
│  Month 4-6: Verification                                     │
│  ├── UVM testbench development                               │
│  ├── Coverage-driven verification                            │
│  ├── Co-simulation with PyTorch                              │
│  └── Formal verification                                     │
│                                                               │
│  Month 7-9: Physical Design                                  │
│  ├── Floorplanning                                           │
│  ├── Place and route                                         │
│  ├── CTS and signal integrity                                │
│  └── Signoff checks                                          │
│                                                               │
│  Month 10: Tape-out                                          │
│  ├── Final checks (DRC, LVS, ERC)                           │
│  ├── GDSII generation                                        │
│  └── Mask ordering                                           │
│                                                               │
│  Month 11-14: Fabrication and Validation                     │
│  ├── Wafer fabrication (6-8 weeks)                           │
│  ├── Packaging and testing                                   │
│  ├── Board design and fabrication                            │
│  └── Post-silicon validation                                 │
│                                                               │
│  Month 15-16: Production                                     │
│  ├── Production mask set (if MPW successful)                 │
│  ├── Volume production ramp                                  │
│  └── Customer sampling                                        │
└─────────────────────────────────────────────────────────────┘
```

---

## 17.18 References

1. TSMC N7 Design Rule Manual (DRM)
2. ARM Artisan Memory Compiler User Guide
3. Synopsys Design Compiler User Guide
4. Cadence Innovus Implementation User Guide
5. Synopsys PrimeTime Signoff User Guide
6. IEEE 1149.1 (JTAG) Standard
7. IEEE 1500 (Embedded Core Test) Standard
8. JEDEC DDR4 Standard (JESD79-4)
9. PCI Express Base Specification 4.0
10. Ternary Weight Networks (Ma et al., 2016)
11. BitNet b1.58 (Microsoft Research, 2024)
12. Balanced Ternary Neural Network Architecture (this project)

---

## 17.19 Summary

This guide provides a comprehensive roadmap for implementing balanced ternary neural network accelerators in ASIC. Key takeaways:

1. **Technology**: 7nm is the sweet spot for edge/server ternary accelerators
2. **Design flow**: Standard digital ASIC flow (RTL → Synthesis → P&R → Signoff)
3. **Memory**: 8 MB weight SRAM + 4 MB scratchpad SRAM (76% of die area)
4. **Power**: 4W typical, with DVFS modes from 1.2W to 4W
5. **Cost**: ~$15 per chip at 1M volume, ~$90 at 100K volume
6. **Performance**: ~50K tokens/s for 1B model, ~12,500 tokens/joule
7. **Timeline**: 16 months from RTL start to production

The ternary approach offers 10-30× better energy efficiency than existing GPU/NPU solutions for transformer inference, making it ideal for edge AI deployment.