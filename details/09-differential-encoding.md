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

## 9.2 Hardware Benefits

| Property | Single Wire (Analog) | Differential (Two Wires) |
|----------|---------------------|--------------------------|
| Reference voltage needed | Yes | No |
| Noise sensitivity | High | Low |
| Wire cost | 1 per trit | 2 per trit |
| Negation | Requires inverter | Swap wires (free) |
| Zero detection | Voltage comparison | Both wires low |
| Timing | Voltage settling | Logic level only |
| Standard CMOS | Requires analog | Fully digital |

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