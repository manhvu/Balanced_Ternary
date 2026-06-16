defmodule TernaryConverter.Sensitivity do
  @moduledoc """
  Analyzes per-layer sensitivity to ternary quantization.

  Identifies which layers can be safely ternarized and which should
  remain at higher precision (INT8/FP16).
  """

  alias TernaryConverter.Quantizer

  @doc """
  Analyze sensitivity of each quantizable layer.

  For each layer:
    1. Measure output with FP32 weights (reference)
    2. Measure output with ternary weights
    3. Compute MSE between outputs

  Returns a list of `{layer_name, mse}` sorted by sensitivity
  (highest first = most sensitive = keep FP16).
  """
  @spec analyze([{String.t(), Nx.Tensor.t()}], Nx.Tensor.t(), float()) ::
          [{String.t(), float()}]
  def analyze(layer_weights, sample_input, delta \\ 0.5) do
    reference_outputs =
      Enum.map(layer_weights, fn {name, w} ->
        {name, Nx.dot(sample_input, Nx.transpose(w))}
      end)

    ternary_outputs =
      Enum.map(layer_weights, fn {name, w} ->
        scales = Quantizer.compute_scales(w, delta)
        t = Quantizer.ternarize_scaled(w, scales, delta)
        {name, Nx.dot(sample_input, Nx.transpose(t))}
      end)

    Enum.zip(reference_outputs, ternary_outputs)
    |> Enum.map(fn {{name, ref}, {name, tern}} ->
      mse = Nx.mean(Nx.pow(Nx.subtract(ref, tern), 2)) |> Nx.to_number()
      {name, mse}
    end)
    |> Enum.sort_by(fn {_name, mse} -> mse end, :desc)
  end

  @doc """
  Automatically determine mixed-precision assignment.

  Starts with all layers ternary, then greedily promotes the most
  sensitive layer to FP16 until the accuracy target is met.
  """
  @spec auto_mixed_precision(
          [{String.t(), Nx.Tensor.t()}],
          Nx.Tensor.t(),
          float(),
          float()
        ) :: {%{String.t() => :ternary | :fp16}, non_neg_integer(), non_neg_integer()}
  def auto_mixed_precision(layers, sample_input, delta, target_similarity) do
    sensitivities = analyze(layers, sample_input, delta)
    promote_layers(sensitivities, layers, sample_input, delta, target_similarity, %{})
  end

  defp promote_layers([], _layers, _input, _delta, _target, assignment) do
    ternary_count = Enum.count(assignment, fn {_, v} -> v == :ternary end)
    fp16_count = map_size(assignment) - ternary_count
    {assignment, ternary_count, fp16_count}
  end

  defp promote_layers([{name, _mse} | rest], layers, input, delta, target, assignment) do
    assignment = Map.put(assignment, name, :fp16)
    similarity = compute_overall_similarity(layers, input, delta, assignment)

    if similarity >= target do
      remaining = Enum.reduce(rest, assignment, fn {n, _}, acc -> Map.put(acc, n, :ternary) end)
      ternary_count = Enum.count(remaining, fn {_, v} -> v == :ternary end)
      fp16_count = map_size(remaining) - ternary_count
      {remaining, ternary_count, fp16_count}
    else
      promote_layers(rest, layers, input, delta, target, assignment)
    end
  end

  defp compute_overall_similarity(layers, input, delta, assignment) do
    Enum.reduce(layers, 0.0, fn {name, w}, acc ->
      ref = Nx.dot(input, Nx.transpose(w))

      tern =
        case Map.get(assignment, name, :ternary) do
          :fp16 ->
            ref

          :ternary ->
            scales = Quantizer.compute_scales(w, delta)
            t = Quantizer.ternarize_scaled(w, scales, delta)
            Nx.dot(input, Nx.transpose(t))
        end

      cos_sim = cosine_similarity(ref, tern)
      acc + cos_sim
    end) / max(length(layers), 1)
  end

  defp cosine_similarity(a, b) do
    dot = Nx.sum(Nx.multiply(a, b)) |> Nx.to_number()
    norm_a = Nx.sqrt(Nx.sum(Nx.pow(a, 2))) |> Nx.to_number()
    norm_b = Nx.sqrt(Nx.sum(Nx.pow(b, 2))) |> Nx.to_number()
    dot / (norm_a * norm_b + 1.0e-8)
  end
end
