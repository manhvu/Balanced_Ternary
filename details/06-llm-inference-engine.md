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
| Prefill throughput (2048 seq) | ~500K tok/s | ~120K tok/s |
| Decode throughput (single) | ~30K tok/s | ~20K tok/s |
| Power | 400W | ~5W |
| Tokens per joule (decode) | 75 | 4,000 |

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