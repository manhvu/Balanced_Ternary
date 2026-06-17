# 9. Differential Trit Encoding for Neural Networks

## 9.1 Encoding Scheme

Differential encoding represents each trit value using two wires (A, B) instead of a single analog voltage level.

```
Trit | Wire A | Wire B | Meaning
-----|--------|--------|--------
T    | 0      | 1      | Negative (subtract)
0    | 0      | 0      | Zero (skip)
1    | 1      | 0      | Positive (add)
```

Invalid state: `Wire A = 1, Wire B = 1` (not used).

---

### 9.1.1 Physical Layer Details

The differential weight bus operates with the following electrical characteristics optimized for on-chip routing:

| Parameter | Value | Notes |
|-----------|-------|-------|
| Differential swing | 0.4 V | Peak-to-peak across the pair (A − B) |
| Common-mode voltage | 0.6 V | Mid-rail bias for noise margin |
| Termination | Parallel 100 Ω to Vcm | At the receiver (PE) end |
| Logic levels | 0.4 V / 0.8 V | Low / high relative to ground |

**Termination scheme:** Each differential pair is terminated at the PE receiver with a 100 Ω resistor across A and B, biased to the 0.6 V common-mode voltage. This provides clean signal edges without reflections on the weight bus, even at multi-gigahertz clock rates. No series termination is needed at the driver because the weight bus is a point-to-point topology (one decoder output driving one PE column).

The 0.4 V differential swing is sufficient for reliable detection by the CMOS comparators inside each PE while keeping dynamic power low. The common-mode voltage of 0.6 V (with Vdd = 1.0 V) provides symmetric noise margin above and below the switching threshold.

---

### Hardware Benefits

| Property | Single Wire (Analog) | Differential (Two Wires) |
|----------|---------------------|--------------------------|
| Reference voltage needed | Yes | No |
| Noise sensitivity | High | Low |
| Wire cost | 1 per trit | 2 per trit |
| Negation | Requires inverter | Swap wires (free) |
| Zero detection | Voltage comparison | Both wires low |
| Timing | Voltage settling | Logic level only |
| Standard CMOS | Requires analog | Fully digital |

### Elixir: Differential Encoder/Decoder

```elixir
defmodule DifferentialEncoding do
  @type trit :: -1 | 0 | 1
  @type wire_pair :: {0 | 1, 0 | 1}  # {wire_a, wire_b}

  @doc """
  Encode a trit into a differential wire pair (A, B).

    T (-1) → {0, 1}  (subtract)
    0      → {0, 0}  (skip)
    1 (+1) → {1, 0}  (add)
  """
  @spec encode(trit()) :: wire_pair()
  def encode(-1), do: {0, 1}
  def encode(0),  do: {0, 0}
  def encode(1),  do: {1, 0}

  @doc """
  Decode a wire pair back to a trit.
  Returns :error on invalid state {1, 1}.
  """
  @spec decode(wire_pair()) :: {:ok, trit()} | {:error, String.t()}
  def decode({0, 0}), do: {:ok, 0}
  def decode({0, 1}), do: {:ok, -1}
  def decode({1, 0}), do: {:ok, 1}
  def decode({1, 1}), do: {:error, "invalid state"}

  @doc """
  Negate a trit by swapping the wire pair.
  This is the key advantage of differential encoding:
  negation is a free wire swap with no inverter needed.
  """
  @spec negate(trit()) :: trit()
  def negate(-1), do: 1
  def negate(0),  do: 0
  def negate(1),  do: -1

  @doc """
  Show negation as wire swap:
    {A, B} → {B, A}
  """
  @spec negate_wires(wire_pair()) :: wire_pair()
  def negate_wires({a, b}), do: {b, a}

  @doc """
  Compute unit: given activation x and wire pair (A,B),
  return the contributed value.

  (0,0) → 0       (skip)
  (0,1) → -x      (subtract)
  (1,0) →  x      (add)
  """
  @spec compute_pe(number(), wire_pair()) :: number()
  def compute_pe(_x, {0, 0}), do: 0
  def compute_pe(x, {0, 1}), do: -x
  def compute_pe(x, {1, 0}), do: x

  @doc """
  Pack a list of trits into a flat list of differential wire pairs
  for routing on the differential weight bus.
  """
  @spec bus_pack([trit()]) :: [{0 | 1, 0 | 1}]
  def bus_pack(trits) do
    Enum.map(trits, &encode/1)
  end
end
```

---

## 9.3 Negation via Wire Swap

Negating a trit is free:

```
Trit   → Negated
A=0,B=1 (T) → A=1,B=0 (1)  → swap wires
A=0,B=0 (0) → A=0,B=0 (0)  → unchanged
A=1,B=0 (1) → A=0,B=1 (T)  → swap wires
```

In a PE, the weight is stored as two bits: one for each differential wire.

Negation is achieved by swapping the two signals going into the adder/subtractor:

```
if weight = +1:  add x
if weight = -1:  subtract x (=> swap wires in transmission)
```

But since weight is stored digitally, wire swapping is just a routing-level swap of the two wires from the weight register.

> **Negation in the accumulator:** Instead of physically swapping wires, the PE can store the sign bit of the weight and conditionally negate the activation value at the accumulator input. A single XOR gate (for two's complement inversion) plus the sign bit controls whether the adder sees +x or −x. This approach is simpler for pipelined designs because it avoids mid-pipeline wire swaps — the sign bit travels alongside the weight data through pipeline stages, and the conditional negate logic sits cleanly at the accumulator boundary without affecting routing.

---

## 9.4 Differential Adder/Subtractor

The PE's core operation:

```
Given:
    - Activation value x (8-bit or 16-bit digital)
    - Weight encoded as (wire_A, wire_B)

Operation:
    case (wire_A, wire_B):
        (0, 0):  skip (do nothing)
        (0, 1):  sum += -x   (subtract)
        (1, 0):  sum +=  x   (add)
        (1, 1):  invalid

    sum accumulates in an internal register (16-bit or 32-bit).
```

---

## 9.5 Differential Weight Bus

Instead of routing decoded weight values within each PE, the differential pair (A, B) can be routed directly on the weight bus.

```
Weight SRAM
    └─► Packed trit word (16 bits = 10 trits)
        └─► Decoder
            └─► 10 differential pairs (A₀,B₀)..(A₉,B₉)
                └─► Distributed to 10 PE columns
                    Each PE sees:
                        A_i: add enable
                        B_i: subtract enable
                        Both low: skip
```

This means each PE receives 2 bits of weight data per cycle instead of decoding internally.

---

## 9.6 Signal Integrity

Differential signaling provides:

```
Noise rejection:  Common-mode noise cancelled
Voltage swing:    Full CMOS logic levels (0/Vdd)
Timing:           No analog settling, logic-level fast
Power:            No static current in CMOS
```

For high-frequency operation (>1 GHz), differential routing on the weight bus can be easier to keep clean than single-ended analog voltage levels.

### 9.6.1 Crosstalk Analysis

Differential pairs have inherently better crosstalk immunity than single-ended signals for two reasons:

1. **Common-mode rejection:** Aggressor noise couples equally onto both wires A and B. Since the receiver detects the *difference* (A − B), the coupled noise cancels out. In a single-ended bus, the same aggressor noise directly corrupts the signal.

2. **Field confinement:** The opposing currents in A and B create opposing magnetic fields that largely cancel at distance, reducing the pair's radiated emissions and susceptibility to far-end crosstalk.

Because of this improved immunity, the differential weight bus can be routed at **tighter pitch** than a single-ended analog bus. Typical spacing rules:

| Bus type | Minimum pitch | Aggressor margin |
|----------|--------------|-----------------|
| Single-ended analog | 3× minimum spacing | High sensitivity |
| Differential digital | 1.5× minimum spacing | Common-mode rejected |

This means the 2× wire count of differential encoding partially pays for itself in routing density — the tighter pitch recovers some of the area lost to the extra wires.

---

## 9.7 Area Comparison

| Component | Analog Weight Bus | Differential Bus |
|-----------|------------------|------------------|
| Wires per weight | 1 | 2 |
| Line drivers | 1 per weight | 2 per weight |
| Termination | voltage reference | no reference needed |
| PE decoder | Voltage comparator + ADC | Simple XOR/AND logic |
| Area overhead | Medium | Low |

**Differential encoding adds ~2× wire count but eliminates analog components.**

> **Wire routing overhead:** While differential encoding doubles the wire count compared to a single-wire scheme, the wires are purely digital and can be routed in standard metal layers (M2–M6) without special shielding, guard rings, or analog-aware design rules. This keeps the physical design flow compatible with standard digital place-and-route tools. The area cost of the extra wires is typically 5–10% of the total PE array area, which is modest compared to the area saved by eliminating ADCs, DACs, and voltage comparators.

---

## 9.8 Integration with Ternary Compute

Within the compute array, the differential encoding integrates naturally:

```
Systolic array column:

    weight bus (differential pair from decoder)
    │
    ▼
  ┌──────┐
  │ PE   │── receives (A, B)
  │      │── if A=1: add input ← input
  │      │── if B=1: add input ← negated input
  │      │── if A=B=0: skip
  └──────┘
```

No ADC, no DAC, no voltage comparator.

---

## 9.9 Comparison with Other Ternary Encoding Methods

| Encoding | Wires/Trit | Negation | Zero Detect | Noise | CMOS Compatible |
|----------|-----------|----------|-------------|-------|-----------------|
| Single analog voltage | 1 | Inverter | Comparator | High | No (analog) |
| 2-bit binary | 2 | Inverter+carry | AND gate | Low | Yes |
| Differential (this) | 2 | Wire swap | NOR gate | Low | Yes |
| Charge-based (memristor) | 1 | Polarity | Zero conductance | Medium | Emerging |
| Frequency-based | 1 | Frequency down | Zero frequency | Medium | Partial |

Differential encoding is the easiest to implement in standard digital CMOS.

---

## 9.10 Why Differential Encoding Works for TNN

Differential encoding is especially well-suited for ternary neural network accelerators because:

1. **Ternary has exact three states** → Two wires can represent them with one redundant state
2. **Negation is the key operation** → Wire swap is the cheapest negation possible
3. **Zero is the skip signal** → Both wires low is naturally the idle state (clock gating friendly)
4. **Standard CMOS** → No process modifications needed
5. **Digital synthesis** → Designs can use standard EDA tools

It is the recommended trit encoding for the ternary accelerators described in this document.

---

## 9.11 Testability and DFT

Testing the differential encoding path requires structured design-for-test (DFT) techniques to ensure manufacturing quality and field reliability. Three complementary strategies cover the full path:

**Note**: These DFT techniques are compatible with modern 3nm/4nm processes and can be integrated with standard EDA toolflows (Synopsys, Cadence, Siemens).

### 1. Built-In Self-Test (BIST) for the Decoder

Each weight decoder includes a BIST engine that exercises all three valid states and the invalid state without external probe access:

```
BIST sequence:
  1. Apply known trit pattern (e.g., [1, 0, -1, 1, -1, ...])
  2. Encode → differential pairs on the bus
  3. Loop back at the PE receiver into a MISR (multi-input signature register)
  4. Compare final signature against golden reference
```

The BIST runs at power-on and can be triggered in-field for periodic health checks. A mismatch flags a decoder fault, enabling the system to remap to a spare PE column.

### 2. Scan Chains for Weight Registers

The two-bit weight registers in each PE are chained into standard scan flip-flops:

```
Scan chain path:
  SO ──► [PE₀ weight_reg] ──► [PE₁ weight_reg] ──► ... ──► [PEₙ weight_reg] ──► SI
```

This allows:
- **Stuck-at fault testing** of the weight storage bits (write 0/1, scan out, verify)
- **At-speed launch-on-capture** testing of the register-to-bus path
- **Debug visibility** — the full weight state can be dumped via JTAG for post-silicon validation

The scan chain adds ~15% area overhead to the weight register file but provides 99%+ fault coverage for the storage elements.

### 3. At-Speed Testing of the Weight Bus

The differential weight bus itself is tested at operating frequency using a pattern generator at the decoder output and a checker at the PE receiver:

```
Test pattern: Alternating (1,0) and (0,1) on each pair → maximum switching
Checker:      Validates correct arrival of both polarities at the PE
```

This catches:
- **Open/short faults** in the differential pair routing
- **Timing violations** (setup/hold at the PE input)
- **Crosstalk-induced bit errors** (by running with adjacent pairs toggling aggressively)

At-speed testing is critical because the weight bus operates at the full accelerator clock rate, and marginal timing defects may only appear at frequency.

| DFT Technique | What It Tests | When Run | Fault Coverage |
|---------------|---------------|----------|----------------|
| Decoder BIST | Encode/decode logic | Power-on, in-field | Decoder logic |
| Scan chains | Weight register stuck-at | Manufacturing test | Storage elements |
| At-speed bus test | Routing, timing, crosstalk | Manufacturing test | Interconnect |