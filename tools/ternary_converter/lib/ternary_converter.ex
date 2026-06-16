defmodule TernaryConverter do
  @moduledoc """
  Main API for converting pretrained models to balanced ternary format.

  ## Quick Start

      # Create a ternary layer from random weights
      w = Nx.random_uniform({64, 128})
      layer = TernaryConverter.convert_layer(w, "fc1", delta: 0.5)

      # Run inference
      input = Nx.random_uniform({1, 128})
      output = TernaryConverter.inference([layer], input)

      # Export to .tbin
      :ok = TernaryConverter.export([layer], "model.tbin")

      # Load back
      {:ok, layers, metadata} = TernaryConverter.load("model.tbin")

  ## Full Pipeline

      # 1. Create synthetic model weights
      weights = %{
        "fc1" => Nx.random_uniform({256, 512}),
        "fc2" => Nx.random_uniform({128, 256}),
        "fc3" => Nx.random_uniform({64, 128})
      }

      # 2. Analyze sensitivity
      sample_input = Nx.random_uniform({1, 512})
      sens = TernaryConverter.analyze_sensitivity(weights, sample_input)

      # 3. Convert all layers
      layers = TernaryConverter.convert_all(weights, delta: 0.5)

      # 4. Validate
      TernaryConverter.validate(layers, weights, sample_input)

      # 5. Export
      :ok = TernaryConverter.export(layers, "model.tbin")
  """

  alias TernaryConverter.{Packer, Layer, Exporter, Sensitivity}

  # ── Conversion ───────────────────────────────────────────────

  @doc """
  Convert a single weight matrix to a ternary layer.
  """
  @spec convert_layer(Nx.Tensor.t(), String.t(), keyword()) :: Layer.t()
  def convert_layer(weight_tensor, name, opts \\ []) do
    Layer.from_dense(weight_tensor, name, opts)
  end

  @doc """
  Convert all weight tensors to ternary layers.
  """
  @spec convert_all(%{String.t() => Nx.Tensor.t()}, keyword()) :: [Layer.t()]
  def convert_all(weights, opts \\ []) do
    delta = Keyword.get(opts, :delta, 0.5)

    weights
    |> Enum.map(fn {name, w} ->
      Layer.from_dense(w, name, delta: delta)
    end)
    |> Enum.sort_by(& &1.name)
  end

  # ── Inference ────────────────────────────────────────────────

  @doc """
  Run inference through a list of ternary layers.
  """
  @spec inference([Layer.t()], Nx.Tensor.t()) :: Nx.Tensor.t()
  def inference(layers, input_tensor) do
    Enum.reduce(layers, input_tensor, fn layer, acc ->
      Layer.forward_nx(layer, acc)
    end)
  end

  # ── Analysis ─────────────────────────────────────────────────

  @doc """
  Analyze per-layer sensitivity.
  """
  @spec analyze_sensitivity(%{String.t() => Nx.Tensor.t()}, Nx.Tensor.t(), float()) :: [
          {String.t(), float()}
        ]
  def analyze_sensitivity(weights, sample_input, delta \\ 0.5) do
    weights
    |> Enum.to_list()
    |> Sensitivity.analyze(sample_input, delta)
  end

  @doc """
  Automatically determine mixed-precision assignment.
  """
  @spec auto_mixed_precision(%{String.t() => Nx.Tensor.t()}, Nx.Tensor.t(), float(), float()) ::
          {%{String.t() => :ternary | :fp16}, non_neg_integer(), non_neg_integer()}
  def auto_mixed_precision(weights, sample_input, delta, target_similarity) do
    weights
    |> Enum.to_list()
    |> Sensitivity.auto_mixed_precision(sample_input, delta, target_similarity)
  end

  # ── Export / Import ──────────────────────────────────────────

  @doc """
  Export layers to .tbin file.
  """
  @spec export([Layer.t()], String.t(), keyword()) :: :ok | {:error, term()}
  def export(layers, path, opts \\ []) do
    Exporter.export(layers, path, opts)
  end

  @doc """
  Load layers from .tbin file.
  """
  @spec load(String.t()) :: {:ok, [Layer.t()], map()} | {:error, term()}
  def load(path) do
    Exporter.load(path)
  end

  # ── Statistics ───────────────────────────────────────────────

  @doc """
  Get model statistics.
  """
  @spec stats([Layer.t()]) :: map()
  def stats(layers) do
    total_params =
      Enum.reduce(layers, 0, fn l, acc ->
        {o, i} = l.shape
        acc + o * i
      end)

    total_zeros =
      Enum.reduce(layers, 0, fn l, acc ->
        {o, i} = l.shape
        acc + round(o * i * l.sparsity)
      end)

    total_nonzero = total_params - total_zeros

    original_bytes = total_params * 4
    packed_bytes = Enum.reduce(layers, 0, fn l, acc -> acc + byte_size(l.weight_packed) end)
    scale_bytes = Enum.reduce(layers, 0, fn l, acc -> acc + length(l.scales) * 2 end)

    %{
      num_layers: length(layers),
      total_parameters: total_params,
      nonzero_parameters: total_nonzero,
      zero_parameters: total_zeros,
      overall_sparsity: total_zeros / max(total_params, 1),
      original_size_mb: original_bytes / 1024 / 1024,
      packed_size_mb: (packed_bytes + scale_bytes) / 1024 / 1024,
      compression_ratio: original_bytes / max(packed_bytes + scale_bytes, 1)
    }
  end

  # ── Validation ───────────────────────────────────────────────

  @doc """
  Run validation: sparsity, similarity, inference test, round-trip.
  """
  @spec validate([Layer.t()], %{String.t() => Nx.Tensor.t()}, Nx.Tensor.t()) :: map()
  def validate(layers, fp32_weights, sample_input) do
    IO.puts("\n=== Per-Layer Sparsity ===")

    Enum.each(layers, fn layer ->
      IO.puts(
        "  #{layer.name}: #{Float.round(layer.sparsity * 100, 1)}% sparse (shape: #{inspect(layer.shape)})"
      )
    end)

    IO.puts("\n=== Output Similarity ===")

    tern_output = inference(layers, sample_input)

    fp32_output =
      Enum.reduce(fp32_weights, sample_input, fn {_name, w}, acc ->
        Nx.dot(acc, Nx.transpose(w))
      end)

    dot = Nx.sum(Nx.multiply(tern_output, fp32_output)) |> Nx.to_number()
    norm_t = Nx.sqrt(Nx.sum(Nx.pow(tern_output, 2))) |> Nx.to_number()
    norm_f = Nx.sqrt(Nx.sum(Nx.pow(fp32_output, 2))) |> Nx.to_number()
    similarity = dot / (norm_t * norm_f + 1.0e-8)

    IO.puts("  Cosine similarity: #{Float.round(similarity, 4)}")

    IO.puts("\n=== Model Size ===")
    s = stats(layers)
    IO.puts("  Parameters: #{s.total_parameters}")
    IO.puts("  Overall sparsity: #{Float.round(s.overall_sparsity * 100, 1)}%")
    IO.puts("  Original: #{Float.round(s.original_size_mb, 1)} MB")
    IO.puts("  Packed: #{Float.round(s.packed_size_mb, 1)} MB")
    IO.puts("  Compression: #{Float.round(s.compression_ratio, 1)}×")

    IO.puts("\n=== Round-Trip Verification ===")

    roundtrip_ok =
      Enum.all?(layers, fn layer ->
        unpacked = Packer.unpack(layer.weight_packed, elem(layer.shape, 0) * elem(layer.shape, 1))

        original =
          Nx.to_flat_list(
            layer.weight_packed
            |> Packer.unpack(elem(layer.shape, 0) * elem(layer.shape, 1))
          )

        unpacked_list = unpacked
        original == unpacked_list
      end)

    IO.puts("  All layers round-trip OK: #{roundtrip_ok}")

    %{
      cosine_similarity: similarity,
      stats: s,
      roundtrip_ok: roundtrip_ok
    }
  end
end
