# TernaryConverter

A complete Elixir application for converting neural network weights to balanced ternary format `{-1, 0, +1}`.

## Features

- **Ternarization**: Convert FP32 weights to balanced ternary with configurable threshold
- **Per-channel scaling**: MSE-optimal scale factors for accuracy recovery
- **Binary packing**: 10 trits per 16-bit word (base-3 encoding)
- **Sparse encoding**: Index+sign format for high-sparsity layers
- **Sensitivity analysis**: Per-layer quantization sensitivity measurement
- **Mixed precision**: Automatic FP16/ternary layer assignment
- **.tbin format**: Compact binary model format with metadata
- **CLI tool**: Convert, demo, info, and validate commands

## Installation

```bash
cd tools/ternary_converter
mix deps.get
```

## Quick Start

### Run the demo

```bash
mix run -e "TernaryConverter.CLI.main([\"demo\"])"
```

### Convert synthetic model to .tbin

```bash
mix run -e "TernaryConverter.CLI.main([\"convert\", \"--delta\", \"0.5\", \"--output\", \"model.tbin\"])"
```

### Inspect a .tbin file

```bash
mix run -e "TernaryConverter.CLI.main([\"info\", \"--model\", \"model.tbin\"])"
```

### Validate a .tbin file

```bash
mix run -e "TernaryConverter.CLI.main([\"validate\", \"--model\", \"model.tbin\"])"
```

### Run as escript

```bash
mix escript.build
./ternary_converter demo --rows 128 --cols 256 --delta 0.5
./ternary_converter convert --delta 0.5 --output my_model.tbin
./ternary_converter info --model my_model.tbin
```

## Library Usage

```elixir
# Create a ternary layer from weights
w = Nx.random_uniform({256, 512})
layer = TernaryConverter.convert_layer(w, "fc1", delta: 0.5)

# Run inference
input = Nx.random_uniform({1, 512})
output = TernaryConverter.inference([layer], input)

# Convert a full model
weights = %{
  "fc1" => Nx.random_uniform({256, 512}),
  "fc2" => Nx.random_uniform({128, 256}),
  "fc3" => Nx.random_uniform({64, 128})
}
layers = TernaryConverter.convert_all(weights, delta: 0.5)

# Analyze sensitivity
sample_input = Nx.random_uniform({1, 512})
sensitivities = TernaryConverter.analyze_sensitivity(weights, sample_input)

# Auto mixed precision
{assignment, n_ternary, n_fp16} =
  TernaryConverter.auto_mixed_precision(weights, sample_input, 0.5, 0.99)

# Export to .tbin
:ok = TernaryConverter.export(layers, "model.tbin",
  metadata: %{model_name: "my_model", version: "1.0.0"}
)

# Load from .tbin
{:ok, loaded_layers, metadata} = TernaryConverter.load("model.tbin")

# Get statistics
stats = TernaryConverter.stats(layers)
# %{num_layers: 3, total_parameters: ..., compression_ratio: ...}
```

## Running Tests

```bash
mix test
```

## Project Structure

```
ternary_converter/
├── lib/
│   ├── ternary_converter.ex              # Main API
│   └── ternary_converter/
│       ├── quantizer.ex                  # Ternary quantization + scales
│       ├── packer.ex                     # Binary packing (10→16, sparse)
│       ├── layer.ex                      # Layer struct + forward pass
│       ├── exporter.ex                   # .tbin export/import
│       ├── sensitivity.ex                # Per-layer sensitivity analysis
│       └── cli.ex                        # Command-line interface
├── test/
│   ├── quantizer_test.exs
│   ├── packer_test.exs
│   ├── layer_test.exs
│   └── integration_test.exs
├── mix.exs
└── README.md
```

## Module Overview

| Module | Purpose |
|--------|---------|
| `TernaryConverter` | Main API: convert, inference, export, stats, validate |
| `TernaryConverter.Quantizer` | `ternarize/2`, `compute_scales/2`, `auto_threshold/2`, `quality_metrics/2` |
| `TernaryConverter.Packer` | `pack/1`, `unpack/2`, `pack_sparse/1`, `compression_ratio/1` |
| `TernaryConverter.Layer` | `from_dense/3`, `forward/2`, `forward_nx/2`, `to_binary/1` |
| `TernaryConverter.Exporter` | `export/3`, `load/1` for .tbin format |
| `TernaryConverter.Sensitivity` | `analyze/3`, `auto_mixed_precision/4` |
| `TernaryConverter.CLI` | CLI: convert, demo, info, validate |

## .tbin Format

Binary format for storing ternary models:

```
[Header][Layer 1][Layer 2]...[Layer N]

Header:
  magic:       "TBN\0" (4 bytes)
  version:     uint32
  num_layers:  uint32
  meta_len:    uint32
  metadata:    JSON (meta_len bytes)

Per layer:
  name_len:    uint16
  name:        UTF-8 bytes
  out_feat:    uint32
  in_feat:     uint32
  delta:       float32
  sparsity:    float32
  scales:      num_scales × float16
  biases:      num_biases × float16
  weights:     packed trits (10 per 16-bit word)
```

## License

See the project LICENSE file.
