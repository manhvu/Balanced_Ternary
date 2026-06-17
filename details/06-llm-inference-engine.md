# 6. LLM Inference Engine Design

## 6.1 Transformer Layer Mapping

A standard transformer decoder layer consists of:

```
1. Attention:
   Q = W_Q × x            (ternary GEMM)
   K = W_K × x            (ternary GEMM)
   V = W_V × x            (ternary GEMM)
   scores = Q @ Kᵀ        (FP16 GEMM)
   attn = softmax(scores)  (FP16)
   out = attn @ V          (FP16 GEMM)
   out = W_O @ out         (ternary GEMM)

2. Residual:
   x = x + out             (FP16 add)

3. LayerNorm:
   x = layernorm(x)        (FP16)

4. MLP:
   gate = W_gate @ x       (ternary GEMM)
   up   = W_up @ x         (ternary GEMM)
   act  = activation(gate) (FP16)
   down = W_down @ (act * up) (ternary GEMM)

5. Residual:
   x = x + down            (FP16 add)

6. LayerNorm:
   x = layernorm(x)        (FP16)
```

### Elixir: Decode-Phase Matrix-Vector Kernel

```elixir
defmodule TernaryDecode do
  @type packed :: non_neg_integer()         # 16-bit packed word
  @type scale :: float()
  @type scale_vec :: [scale()]

  @doc """
  Ternary matrix-vector product for decode phase.
  Weights are stored as packed trit rows (16-bit words),
  each row corresponds to one output channel.
  """
  @spec decode([packed()], scale_vec(), [integer()]) :: [float()]
  def decode(weight_rows, scales, activations) do
    weight_rows
    |> Enum.zip(scales)
    |> Enum.map(fn {packed_row, alpha} ->
      sum = unpack_and_accumulate(packed_row, activations)
      alpha * sum
    end)
  end

  @doc """
  Unpack a 16-bit row of 10 trits and accumulate against
  the activation vector (sparse gather-add).
  Skips zero weights to exploit sparsity.
  """
  defp unpack_and_accumulate(packed, activations) do
    # Unpack 10 trits from the 16-bit word
    trits = unpack10(packed)

    # Only process non-zero trits
    trits
    |> Enum.zip(activations)
    |> Enum.reduce(0, fn
      {1, x}, acc -> acc + x
      {-1, x}, acc -> acc - x
      {0, _}, acc -> acc
    end)
  end

  @doc """
  Unpack a 16-bit value to 10 trits (base-3 extraction).
  """
  defp unpack10(packed) do
    do_unpack10(packed, [])
    |> Enum.reverse()
  end

  defp do_unpack10(0, acc), do: acc ++ List.duplicate(-1, 10 - length(acc))
  defp do_unpack10(v, acc) when length(acc) < 10 do
    digit = rem(v, 3)
    trit = case digit do
             0 -> -1
             1 ->  0
             2 ->  1
           end
    do_unpack10(div(v, 3), [trit | acc])
  end
  defp do_unpack10(_, acc), do: acc

  @doc """
  Batched decode across all layers in the transformer.
  Each layer processes projections sequentially.
  """
  @spec forward([layer()], [integer()]) :: [float()]
  def forward(layers, input_activations) do
    Enum.reduce(layers, input_activations, fn layer, x ->
      x |> layer.q_proj.call()
        |> Kernel.+(layer.residual |> maybe_skip())
    end)
  end

  defp maybe_skip(nil), do: 0
  defp maybe_skip(val), do: val
end
```

---

## 6.2 Prefill vs Decode Phase

LLM inference has two distinct phases.

### Prefill Phase

Processes the entire prompt in parallel.

```
Input:  prompt tokens [token_1, token_2, ..., token_n]
Output: first generated token

Characteristics:
    - High arithmetic intensity
    - Large matrix × large matrix (attended sequence)
    - Good for batch size > 1
    - Ternary GEMM fully utilized
    - Memory bandwidth: less bottleneck
```

### Decode Phase

Generates one token at a time.

```
Input:  single token [token_k]
Output: next token [token_{k+1}]

Characteristics:
    - Low arithmetic intensity
    - Small matrix × vector operations
    - Batch size often 1
    - Ternary GEMM underutilized
    - Memory bandwidth: primary bottleneck
    - KV cache: must be read every step
```

Ternary excels in the **decode phase** because:

- Small weight volume means weights can stay in on-chip SRAM
- No repeated DRAM reads for weights
- KV cache reads dominate, and ternary reduces those too if KV cache is ternarized

### 6.2.1 Continuous Batching

Continuous batching (also called in-flight batching) is a scheduling strategy where new requests are added to a currently running batch without waiting for all in-flight requests to finish. When one request completes (e.g., it produces an <eos> token or hits its max length), a new waiting request immediately takes its slot.

```
Traditional static batching:
  Batch: [Req1, Req2, Req3]
  Req1 finishes at step 5 → idle slots until all done
  Next batch starts only after step 8

Continuous batching:
  Batch: [Req1, Req2, Req3]
  Req1 finishes at step 5 → Req4 immediately inserted
  Batch: [Req4, Req2, Req3]  (no idle slots)
```

Benefits for ternary accelerators:

- **Throughput improvement**: 2–3× for variable-length workloads (e.g., chat, code generation) where request lengths vary significantly.
- **SRAM utilization**: The ternary weight SRAM is already loaded; swapping in a new request only requires updating the activation buffer and KV cache pointers, not reloading weights.
- **Latency**: Tail latency improves because short requests don't wait for long ones to finish.

The main requirement is that the scratchpad SRAM must hold the KV caches for all concurrent requests. With INT4 KV cache and 8 MB SRAM, a typical edge configuration supports 4–8 concurrent requests at sequence length 2048.

---

## 6.3 Ternary GEMM Kernel (Decode Phase)

For single-token decode, the ternary GEMM becomes a matrix-vector product:

```
Input:          W_ternary [M, N]  (packed trits)
                x [N]             (activation vector, INT8)
                α [M]             (per-channel scale, FP16)

Output:         y [M]             (output vector, FP16)

Algorithm:
    For each output channel j:
        sum = 0
        For each input i:
            if W[j,i] == +1:  sum += x[i]
            if W[j,i] == -1:  sum -= x[i]
            if W[j,i] ==  0:  skip
        y[j] = α[j] × sum + b[j]
```

This is a **sparse gather-add** operation when sparsity is high, making it very efficient.

---

## 6.4 KV Cache Management

### KV Cache Growth

```
At step t:
    K_cache[t, :] = K_proj(x_t)    (ternary GEMM)
    V_cache[t, :] = V_proj(x_t)    (ternary GEMM)

At step t+1:
    scores = Q(x_{t+1}) @ K_cache[:t+1]ᵀ    (FP16 GEMM)
    attn = softmax(scores)
    out = attn @ V_cache[:t+1]                (FP16 GEVM)
```

### KV Cache Storage Options

| Format | Memory | Quality | Complexity |
|--------|--------|---------|------------|
| FP16   | 32 MB  | Reference | Baseline |
| INT8   | 16 MB  | Good | Simple |
| INT4   | 8 MB   | Acceptable | Moderate |
| Ternary | 3.2 MB | Risky | Complex |

**Recommended**: INT4 KV cache on edge, INT8 on server.

When the cache exceeds scratchpad SRAM, it spills to LPDDR. The goal is to keep it on-chip.

**Sliding Window Attention**: For long sequences, only the most recent W tokens are kept in the KV cache, reducing memory from O(seq_len) to O(W). A typical window size is W = 4096. This is especially effective for ternary accelerators with limited SRAM: a 4096-token sliding window with INT4 precision requires only ~4 × 512 × 4096 × 0.5 bytes ≈ 4 MB for the KV cache (at n_kv_heads=4, d_head=128), fitting comfortably in on-chip memory even for batch size > 1.

---

## 6.5 Speculative Decoding with Ternary Weights

Speculative decoding generates draft tokens with a small model and verifies with a large model.

Ternary weights can make the **small draft model** extremely fast:

```
Draft model:
    Ternary weights:  ~50-100M parameters
    SRAM fit:         yes (<20 MB)
    Latency per token: ~1 µs (on-chip GEMM)

Target model:
    Ternary weights:  ~1B parameters
    SRAM fit:         yes (~200 MB)
    Latency per token: ~2-3 µs
```

This makes speculative decoding on edge devices practical.

---

## 6.6 Multi-Head Attention Mapping

Standard multi-head attention uses multiple attention heads:

```
d_model = 4096
n_heads = 32
d_head = 128

Q projection: W_Q [4096, 4096] → ternary
  Decomposed into 32 heads: each head [128, 4096]

Score computation (per head):
    score_h = Q_h @ K_hᵀ    (FP16, d_head=128)

Score computation latency:   d_head × seq_len cycles
Total:                       n_heads × d_head × seq_len
```

### Grouped Query Attention (GQA)

Many modern LLMs use GQA where there are fewer K/V heads than Q heads.

```
n_query_heads = 32
n_kv_heads    = 4
group_size    = 8

K projection: W_K [4 × 128, 4096] = [512, 4096]
V projection: W_V [4 × 128, 4096] = [512, 4096]
```

This is important because:

- K/V projection matrices are ~8× smaller
- KV cache is ~8× smaller
- Ternary weight savings on K/V projections are smaller (but still significant)

### 6.6.1 Flash Attention Adaptation

Flash Attention reduces the I/O cost of attention by tiling the softmax computation into blocks that fit in SRAM, avoiding materializing the full [seq_len × seq_len] attention matrix. The key idea is an online softmax update: as each tile of K/V is loaded, the running max and partial sum are updated incrementally.

For ternary weights, the adaptation is straightforward:

- **Q, K, V projections** use the standard ternary GEMM kernel (unchanged).
- **Score computation** (Q @ Kᵀ) is FP16 regardless of weight format — Flash Attention tiles this identically.
- **Online softmax update** is slightly modified: since the non-zero weight pattern in Q/K projections can make some score blocks sparser, the running max update can skip all-zero blocks, saving a small number of FP16 comparisons.
- **I/O complexity improvement remains**: The reduction from O(seq_len²) to O(seq_len² / M) in HBM-to-SRAM traffic is preserved, where M is the SRAM size. For a ternary accelerator with 8 MB SRAM, this means ~8× fewer DRAM reads for the attention matrix on a 2048-token sequence.

The main benefit on a ternary accelerator is that the smaller KV cache (thanks to ternary weight compression) means more of the attention tiles fit in SRAM simultaneously, effectively increasing the tile size and further reducing I/O.

---

## 6.7 MLP Variants

### Classic MLP (GPT-2 style)

```
Gate:   W_gate [4*d_model, d_model]
Up:     W_up   [4*d_model, d_model]
Down:   W_down [d_model, 4*d_model]
```

### SwiGLU MLP (LLaMA style)

```
Gate:   W_gate [8/3*d_model, d_model]
Up:     W_up   [8/3*d_model, d_model]
Down:   W_down [d_model, 8/3*d_model]
```

SwiGLU typically uses `ff_dim = 8/3 × d_model` instead of `4 × d_model`.

### Storage Comparison (d_model=4096)

| MLP Type | Projection | FP32 Size | Ternary Size |
|----------|-----------|-----------|--------------|
| Classic  | W_gate    | 256 MB    | ~13 MB       |
| Classic  | W_up      | 256 MB    | ~13 MB       |
| Classic  | W_down    | 256 MB    | ~13 MB       |
| SwiGLU   | W_gate    | ~170 MB   | ~8.5 MB      |
| SwiGLU   | W_up      | ~170 MB   | ~8.5 MB      |
| SwiGLU   | W_down    | ~170 MB   | ~8.5 MB      |

SwiGLU MLPs are slightly smaller, making them better for ternary on edge devices.

---

## 6.8 Layer Compilation Example

For a single transformer layer, the compiled instruction sequence:

```
// Attention Projections
LOAD_WEIGHT  W_Q                      // Load Q weights to SRAM
GEMM_TILE    |4096→4096|               // Q = ternary_GEMM(W_Q, x)

LOAD_WEIGHT  W_K                      // Load K weights to SRAM (GQA shared heads loaded once across queries if possible)                  
GEMM_TILE    |512→4096|               // K = ternary_GEMM(W_K, x) for each KV pair position during prefill/decode step append                                                                      

LOAD_WEIGHT  W_V |4096→512|           // Load V weights to SRAM                

GEMM_TILE(layer)                      // V = ternary_GEMM(W_V, x)                
                                                                                
// Attention Compute (FP16 systolic)                                           
SOFTMAX      attn_weights              // Compute softmax over attention scores (FP16)
GEMM_ATTN    attn_output              // attn_output = attn_weights @ V (FP16 GEMM)                                                                           
                                                                                
// Output Projection                                                            
LOAD_WEIGHT  W_O                      // Load O projection weights to SRAM     
GEMM_TILE    |4096→4096|               // out = ternary_GEMM(W_O, attn_output)                                                                                 
                                                                                
// Post-Attention                                                              
RESIDUAL_ADD x, out                   // x = x + out (FP16)                    
LAYERNORM    x                        // x = layernorm(x) (FP16)               
                                                                                
// MLP                                                                         
LOAD_WEIGHT  W_gate                   // Load gate projection                  
GEMM_TILE    |11008→4096|             // gate = ternary_GEMM(W_gate, x)        
LOAD_WEIGHT  W_up                     // Load up projection                    
GEMM_TILE    |11008→4096|             // up = ternary_GEMM(W_up, x)            
ACTIVATE     gate, SiLU               // gate = silu(gate) (FP16)              
MUL_ELEMENT  gate, up                 // gate = gate * up (FP16)               
LOAD_WEIGHT  W_down                   // Load down projection                  
GEMM_TILE    |4096→11008|             // down = ternary_GEMM(W_down, gate)                                                                                    
                                                                                
// Post-MLP                                                                   
RESIDUAL_ADD x, down                  // x = x + down (FP16)                   
LAYERNORM    x                        // x = layernorm(x) (FP16)               
                                                                                
// Store                                                                       
STORE        x, output_buffer          // Write output for next layer
```

Total instructions per layer: ~25-30

---

## 6.9 Throughput Estimation

### Assumptions

```
1B parameter model, 12 layers
d_model = 4096, ff_dim = 11008
n_heads = 32, n_kv_heads = 4
Sequence length: 2048 (prefill), 1 (decode)
Ternary array: 128×128
Clock: 1 GHz
Weight SRAM: 8 MB (fit entire model)
Sparsity: 75%
```

### Prefill (2048 tokens)

```
Ternary GEMMs (projections):
    Per layer: 4 × (d_model × d_model) + 3 × (ff_dim × d_model)
    = 4 × 4096² + 3 × 11008 × 4096 ≈ 67M + 135M ≈ 202M operations
    At 75% sparsity: ~50M effective adds/subs
    GEMM array throughput: 128×128 adds per cycle = 16K/cycle
    Cycles per layer: 50M / 16K ≈ 3125 cycles ≈ 3.1 µs
    12 layers: ~37 µs

Attention scores (FP16):
    32 heads × 2048² × 128 ≈ 17B operations
    FP16 systolic array (32×32): 1024/cycle
    Cycles: 17B / 1024 ≈ 16.6M cycles ≈ 16.6 ms

Total prefill: ~17 ms (dominated by attention score compute)
Prefill throughput: 2048 / 0.017 ≈ 120K tokens/s
```

### Decode (single token)

```
Ternary GEMMs:
    Same operations, but now vector × matrix
    202M operations at 75% sparsity = ~50M effective adds
    50M / 16K ≈ 3125 cycles ≈ 3.1 µs per layer
    12 layers: ~37 µs

Attention scores (FP16):
    One query: 32 heads × 2048 (seq) × 128 (d_head) ≈ 8.4M operations
    FP16 systolic: 8.4M / 1024 ≈ 8200 cycles ≈ 8.2 µs

Other (softmax, layernorm, etc.): ~5 µs

Total decode: ~50 µs per token
Decode throughput: ~20,000 tokens/s
```

### Comparison (1B model, edge device)

| Metric | GPU (A100) | Ternary Accelerator |
|--------|-----------|---------------------|
| Prefill throughput (2048 seq) | ~1M tok/s | ~120K tok/s |
| Decode throughput (single) | ~50K tok/s | ~20K tok/s |
| Power | 400W | ~5W |
| Tokens per joule (decode) | 75 | 4,000 |

**Multi-chip scaling**: For 7B+ models whose ternary weights exceed a single chip's SRAM capacity, weights can be split across multiple ternary accelerator chips connected via a high-speed interconnect (e.g., UCIe or a proprietary mesh). Each chip holds a subset of layers or a partition of the weight tensor. The activation vector is broadcast to all chips, and partial results are reduced (summed) across chips. With a 64 GB/s interconnect, the all-reduce for a 4096-element FP16 vector adds only ~1 µs of latency per layer — negligible compared to the GEMM compute time. This approach scales to 13B, 70B, and beyond by adding more chips.

---

## 6.10 Batch Processing

Batch inference improves throughput but requires larger scratchpad memory.

### Batch Size Impact

| Batch Size | Scratchpad Required | Throughput | Notes |
|-----------|-------------------|-----------|-------|
| 1 | 4 MB | 20K tok/s | Fits in on-chip SRAM |
| 4 | 16 MB | 60K tok/s | May spill to LPDDR |
| 8 | 32 MB | 100K tok/s | Significant spilling |
| 16 | 64 MB | 150K tok/s | Mostly DRAM-bound |

For edge devices, batch size 1–4 is most practical.

---

## 6.11 Energy Per Token

```
Ternary GEMM compute (75% sparse):
    50M adds × 0.05 pJ = 2.5 µJ

Ternary weight SRAM reads:
    50M non-zero weights × 1.585 bits × 0.01 pJ/bit = 0.8 µJ

FP16 compute (attention, softmax, layernorm):
    ~20M operations × 0.5 pJ = 10 µJ

Data movement (activations):
    ~1 MB transferred × 0.1 pJ/byte = 100 µJ

Total per layer: ~113 µJ
Total per token (12 layers): ~1.36 mJ

At 20,000 tokens/s:
    Power: 20K × 1.36 mJ ≈ 27 W

With optimization and shared activation reuse: ~5-10 W
```

---

## 6.12 Key Bottlenecks and Mitigations

| Bottleneck | Mitigation |
|------------|-----------|
| KV cache size for long context | Use INT4 KV cache, sliding window attention |
| Attention score compute cost | Use GQA, reduce n_kv_heads |
| Softmax overhead | Use FP16 lookup table for exp approximation |
| Activation memory for batch > 1 | Fuse operations to reduce intermediate storage |
| Scale factor multiply latency | Fuse scale into bias, or use integer multiply-add |
| Zero-sparsity not high enough | Train with stronger sparsity regularization |

---

## 6.13 Multi-Model Support

The accelerator can efficiently switch between different models without reconfiguring the compute array. Since ternary weights are densely packed (10 trits per 16-bit word), model switching is purely a DMA transfer of weight data from external storage (e.g., flash or DRAM) into the on-chip SRAM.

```
Model switch procedure:
  1. Save current context (KV cache, activations) to LPDDR
  2. DMA new model weights from storage → SRAM
  3. Restore new model's context from LPDDR
  4. Resume inference
```

With 8 MB SRAM and a 200 MB ternary model (e.g., a 1B-parameter model at ~1.585 bits/parameter):

```
Load time = Model size / DMA bandwidth
           = 200 MB / 64 GB/s
           ≈ 25 ms
```

This is fast enough for interactive use cases (e.g., switching between a code model and a chat model). The DMA transfer can also be overlapped with computation: while the current model's last few layers are executing, the first layers of the next model can begin loading into a separate SRAM bank (double-buffered weight loading), reducing effective switch time to near zero for pipelined execution.

For multi-tenant edge devices (e.g., a smart hub running both a voice assistant and a translation model), the SRAM can be partitioned: 6 MB for the active model and 2 MB as a staging area for the next model's weights, enabling sub-10 ms hot-swapping.