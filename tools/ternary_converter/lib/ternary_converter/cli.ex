defmodule TernaryConverter.CLI do
  @moduledoc """
  Command-line interface for the TernaryConverter tool.

  Usage:
    ternary_converter convert [--delta 0.5] [--output model.tbin]
    ternary_converter demo [--rows 64] [--cols 128] [--delta 0.5]
    ternary_converter info [--model model.tbin]
    ternary_converter validate [--model model.tbin]
  """

  def main(args) do
    case args do
      ["convert" | rest] -> do_convert(rest)
      ["demo" | rest] -> do_demo(rest)
      ["info" | rest] -> do_info(rest)
      ["validate" | rest] -> do_validate(rest)
      _ -> print_usage()
    end
  end

  defp print_usage do
    IO.puts("""
    TernaryConverter — Convert neural network weights to balanced ternary

    Usage:
      ternary_converter convert [--delta 0.5] [--output model.tbin] [--sparsity 0.5]
      ternary_converter demo [--rows 64] [--cols 128] [--delta 0.5] [--batch 16]
      ternary_converter info [--model model.tbin]
      ternary_converter validate [--model model.tbin]

    Commands:
      convert    Convert a model (from safetensors) to .tbin format
      demo       Run a full demo with synthetic weights
      info       Show info about a .tbin model
      validate   Validate a .tbin model (round-trip, inference)

    Options:
      --delta FLOAT     Ternary threshold (default: 0.5)
      --output PATH     Output file path (default: model.tbin)
      --sparsity FLOAT  Target sparsity for auto-threshold (default: 0.5)
      --rows INT        Number of rows for demo weights (default: 64)
      --cols INT        Number of cols for demo weights (default: 128)
      --batch INT       Batch size for demo (default: 16)
      --model PATH      Path to .tbin model file
    """)
  end

  # ── Convert ───────────────────────────────────────────────────

  defp do_convert(args) do
    delta = get_float_arg(args, "--delta", 0.5)
    output = get_string_arg(args, "--output", "model.tbin")
    target_sparsity = get_float_arg(args, "--sparsity", 0.5)

    IO.puts("TernaryConverter — Convert")
    IO.puts("  Delta: #{delta}")
    IO.puts("  Target sparsity: #{target_sparsity}")
    IO.puts("  Output: #{output}")
    IO.puts("")

    # For the CLI demo, create synthetic weights representing a small model
    IO.puts("Creating synthetic model weights...")
    weights = create_synthetic_model()

    IO.puts("Calibrating scales and converting #{map_size(weights)} layers...")

    layers =
      weights
      |> Enum.map(fn {name, w} ->
        {auto_delta, _actual_sparsity} =
          case target_sparsity do
            0.0 -> {delta, 0.0}
            _s -> TernaryConverter.Quantizer.auto_threshold(w, target_sparsity)
          end

        layer = TernaryConverter.convert_layer(w, name, delta: auto_delta)

        IO.puts(
          "  #{name}: shape=#{inspect(layer.shape)}, sparsity=#{Float.round(layer.sparsity * 100, 1)}%, delta=#{Float.round(auto_delta, 3)}"
        )

        layer
      end)

    stats = TernaryConverter.stats(layers)

    IO.puts("\nModel statistics:")
    IO.puts("  Layers: #{stats.num_layers}")
    IO.puts("  Parameters: #{stats.total_parameters}")
    IO.puts("  Sparsity: #{Float.round(stats.overall_sparsity * 100, 1)}%")
    IO.puts("  Original: #{Float.round(stats.original_size_mb, 2)} MB")
    IO.puts("  Packed: #{Float.round(stats.packed_size_mb, 2)} MB")
    IO.puts("  Compression: #{Float.round(stats.compression_ratio, 1)}×")

    :ok = TernaryConverter.export(layers, output, metadata: %{command: "convert", delta: delta})
  end

  # ── Demo ──────────────────────────────────────────────────────

  defp do_demo(args) do
    rows = get_int_arg(args, "--rows", 64)
    cols = get_int_arg(args, "--cols", 128)
    delta = get_float_arg(args, "--delta", 0.5)
    batch = get_int_arg(args, "--batch", 16)

    IO.puts("=== TernaryConverter Demo ===\n")
    Nx.default_backend(Nx.BinaryBackend)

    # Create random weights
    key = Nx.Random.key(42)
    {w_raw, key2} = Nx.Random.uniform(key, 0.0, 1.0, shape: {rows, cols})
    w = Nx.subtract(Nx.multiply(w_raw, 2.0), 1.0)
    {x_raw, _key} = Nx.Random.uniform(key2, 0.0, 1.0, shape: {batch, cols})
    x = Nx.subtract(Nx.multiply(x_raw, 2.0), 1.0)

    IO.puts("Weight tensor shape: #{inspect(Nx.shape(w))}")
    IO.puts("Activation tensor shape: #{inspect(Nx.shape(x))}")

    # Quantize
    w_tern = TernaryConverter.Quantizer.ternarize(w, delta)
    sparsity = Nx.divide(Nx.sum(Nx.equal(w_tern, 0)), Nx.size(w_tern)) |> Nx.to_number()
    IO.puts("Ternary weight sparsity: #{Float.round(sparsity * 100, 2)}% zeros")

    # Scales + GEMM
    scales = TernaryConverter.Quantizer.compute_scales(w, delta)
    scaled_w = Nx.multiply(w_tern, Nx.new_axis(scales, 1))
    result = Nx.dot(x, Nx.transpose(scaled_w))
    IO.puts("GEMM output shape: #{inspect(Nx.shape(result))}")

    # Effective MACs
    nonzero = Nx.sum(Nx.not_equal(w_tern, 0)) |> Nx.to_number()
    total = Nx.size(w_tern) |> Nx.to_number()
    IO.puts("Effective MACs: #{nonzero} / #{total} (#{Float.round(nonzero / total * 100, 2)}%)")

    # Pack
    packed = TernaryConverter.Packer.pack(w_tern)
    IO.puts("Packed size: #{byte_size(packed)} bytes (original: #{rows * cols * 4} bytes f32)")

    # Compression ratio
    {_orig, _packed, ratio} = TernaryConverter.Packer.compression_ratio(w_tern)
    IO.puts("Compression ratio: #{ratio}×")

    # Round-trip verification
    _unpacked = TernaryConverter.Packer.unpack(packed, rows * cols)
    match = TernaryConverter.Packer.verify_roundtrip(w_tern)
    IO.puts("Round-trip match: #{match}")

    # Quality metrics
    metrics = TernaryConverter.Quantizer.quality_metrics(w, delta)
    IO.puts("\nQuality metrics:")
    IO.puts("  Sparsity: #{Float.round(metrics.sparsity * 100, 1)}%")
    IO.puts("  Density: #{Float.round(metrics.density * 100, 1)}%")
    IO.puts("  +1 ratio: #{Float.round(metrics.positive_ratio * 100, 1)}%")
    IO.puts("  -1 ratio: #{Float.round(metrics.negative_ratio * 100, 1)}%")
    IO.puts("  MSE: #{Float.round(metrics.mse, 6)}")
    IO.puts("  SQNR: #{Float.round(metrics.sqnr, 2)} dB")

    IO.puts("\n=== Demo complete ===")
  end

  # ── Info ──────────────────────────────────────────────────────

  defp do_info(args) do
    path = get_string_arg(args, "--model", "model.tbin")

    case TernaryConverter.load(path) do
      {:ok, layers, metadata} ->
        stats = TernaryConverter.stats(layers)

        IO.puts("=== Model Info: #{path} ===\n")
        IO.puts("Metadata:")
        Enum.each(metadata, fn {k, v} -> IO.puts("  #{k}: #{v}") end)

        IO.puts("\nStatistics:")
        IO.puts("  Layers: #{stats.num_layers}")
        IO.puts("  Parameters: #{stats.total_parameters}")
        IO.puts("  Non-zero: #{stats.nonzero_parameters}")
        IO.puts("  Zero: #{stats.zero_parameters}")
        IO.puts("  Sparsity: #{Float.round(stats.overall_sparsity * 100, 1)}%")
        IO.puts("  Original size: #{Float.round(stats.original_size_mb, 2)} MB")
        IO.puts("  Packed size: #{Float.round(stats.packed_size_mb, 2)} MB")
        IO.puts("  Compression: #{Float.round(stats.compression_ratio, 1)}×")

        IO.puts("\nLayers:")

        Enum.each(layers, fn layer ->
          IO.puts(
            "  #{layer.name}: shape=#{inspect(layer.shape)}, sparsity=#{Float.round(layer.sparsity * 100, 1)}%, delta=#{Float.round(layer.delta, 3)}"
          )
        end)

      {:error, reason} ->
        IO.puts("Error loading #{path}: #{inspect(reason)}")
    end
  end

  # ── Validate ──────────────────────────────────────────────────

  defp do_validate(args) do
    path = get_string_arg(args, "--model", "model.tbin")

    case TernaryConverter.load(path) do
      {:ok, layers, _metadata} ->
        IO.puts("=== Validating: #{path} ===\n")

        # Round-trip check
        IO.puts("Round-trip verification:")

        Enum.each(layers, fn layer ->
          ok =
            TernaryConverter.Packer.verify_roundtrip(
              layer.weight_packed
              |> TernaryConverter.Packer.unpack(elem(layer.shape, 0) * elem(layer.shape, 1))
              |> Nx.tensor(type: :s64)
              |> Nx.reshape(layer.shape)
            )

          IO.puts("  #{layer.name}: #{if ok, do: "OK", else: "FAIL"}")
        end)

        # Inference test
        IO.puts("\nInference test:")
        {_out_features, in_features} = List.first(layers).shape
        {input, _} = Nx.Random.uniform(Nx.Random.key(123), 0.0, 1.0, shape: {1, in_features})
        output = TernaryConverter.inference(layers, input)
        IO.puts("  Input shape: #{inspect(Nx.shape(input))}")
        IO.puts("  Output shape: #{inspect(Nx.shape(output))}")

        # Stats
        stats = TernaryConverter.stats(layers)
        IO.puts("\nModel stats:")
        IO.puts("  Layers: #{stats.num_layers}")
        IO.puts("  Parameters: #{stats.total_parameters}")
        IO.puts("  Sparsity: #{Float.round(stats.overall_sparsity * 100, 1)}%")
        IO.puts("  Compression: #{Float.round(stats.compression_ratio, 1)}×")

        IO.puts("\n=== Validation complete ===")

      {:error, reason} ->
        IO.puts("Error loading #{path}: #{inspect(reason)}")
    end
  end

  # ── Helpers ───────────────────────────────────────────────────

  defp create_synthetic_model do
    [
      {"fc1", Nx.Random.key(1), {256, 512}},
      {"fc2", Nx.Random.key(2), {128, 256}},
      {"fc3", Nx.Random.key(3), {64, 128}},
      {"fc4", Nx.Random.key(4), {32, 64}}
    ]
    |> Enum.map(fn {name, key, shape} ->
      {t, _} = Nx.Random.uniform(key, 0.0, 1.0, shape: shape)
      scaled = Nx.subtract(Nx.multiply(t, 2.0), 1.0)
      {name, scaled}
    end)
    |> Map.new()
  end

  defp get_string_arg(args, key, default) do
    case Enum.find_index(args, &(&1 == key)) do
      nil -> default
      idx -> Enum.at(args, idx + 1, default)
    end
  end

  defp get_float_arg(args, key, default) do
    case get_string_arg(args, key, nil) do
      nil -> default
      val -> String.to_float(val)
    end
  end

  defp get_int_arg(args, key, default) do
    case get_string_arg(args, key, nil) do
      nil -> default
      val -> String.to_integer(val)
    end
  end
end
