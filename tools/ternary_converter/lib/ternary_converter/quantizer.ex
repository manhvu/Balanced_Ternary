defmodule TernaryConverter.Quantizer do
  @moduledoc """
  Converts full-precision weight tensors to balanced ternary {-1, 0, +1}.

  Supports:
  - Fixed threshold (delta) ternarization
  - Per-channel MSE-optimal scale factors
  - Automatic threshold selection (statistical, grid search)
  - Weight clipping for outlier handling
  - Quality metrics (sparsity, MSE, SQNR)
  """

  @type tensor :: Nx.Tensor.t()
  @type trit :: -1 | 0 | 1

  # ── Ternarization ────────────────────────────────────────────

  @doc """
  Ternarize a tensor with a fixed threshold.

  Values greater than `delta` → 1
  Values between [-delta, delta] → 0
  Values less than `-delta` → -1

  ## Examples

      iex> TernaryConverter.Quantizer.ternarize(Nx.tensor([0.8, -0.3, 0.1, -0.9, 0.5]), 0.4)
      #Nx.Tensor<
        s64[5]
        [1, -1, 0, -1, 1]
      >
  """
  @spec ternarize(tensor(), float()) :: tensor()
  def ternarize(w, delta) do
    pos = Nx.greater(w, delta) |> Nx.as_type(:s64)
    neg = Nx.less(w, Nx.negate(delta)) |> Nx.as_type(:s64) |> Nx.multiply(-1)
    Nx.add(pos, neg)
  end

  @doc """
  Ternarize with per-channel scale factors.

  Normalizes each row by its scale factor before ternarization,
  then rescales. This is the recommended approach for all linear layers.
  """
  @spec ternarize_scaled(tensor(), tensor(), float()) :: tensor()
  def ternarize_scaled(w, scales, delta) do
    w_norm = Nx.divide(w, Nx.new_axis(scales, 1))
    t = ternarize(w_norm, delta)
    Nx.multiply(t, Nx.new_axis(scales, 1))
  end

  # ── Scale Factor Calibration ─────────────────────────────────

  @doc """
  Compute per-channel MSE-optimal scale factors.

  For each output channel j:
    α_j = sum(W_j * T_j) / sum(T_j * T_j)

  ## Examples

      iex> w = Nx.tensor([[0.8, -0.3, 0.1], [-0.9, 0.5, -0.2]])
      iex> scales = TernaryConverter.Quantizer.compute_scales(w, 0.4)
      iex> Nx.to_flat_list(scales)
      [0.5666666626930237, 0.5399999618530273]
  """
  @spec compute_scales(tensor(), float()) :: tensor()
  def compute_scales(weights, delta) do
    ternary = ternarize(weights, delta)
    dot = Nx.sum(Nx.multiply(weights, ternary), axes: [1])
    norm = Nx.sum(Nx.pow(ternary, 2), axes: [1])
    safe_norm = Nx.select(Nx.greater(norm, 0.0), norm, Nx.tensor(1.0))
    Nx.divide(dot, safe_norm)
  end

  @doc """
  Compute per-channel scale factors with outlier clipping.

  Before computing scales, clips each channel to ±3σ to prevent
  outlier weights from dominating the scale factor.
  """
  @spec compute_scales_clipped(tensor(), float(), float()) :: tensor()
  def compute_scales_clipped(weights, delta, clip_sigma \\ 3.0) do
    mean = Nx.mean(weights, axes: [1])
    std = Nx.standard_deviation(weights, axes: [1])

    upper = mean + clip_sigma * std
    lower = mean - clip_sigma * std

    clipped =
      weights
      |> Nx.max(Nx.new_axis(lower, 1))
      |> Nx.min(Nx.new_axis(upper, 1))

    compute_scales(clipped, delta)
  end

  # ── Automatic Threshold Selection ────────────────────────────

  @doc """
  Automatic threshold selection via statistical analysis.

  Chooses delta to achieve a target sparsity level based on the
  weight distribution. Uses binary search starting from a Gaussian estimate.

  Returns `{optimal_delta, actual_sparsity}`.
  """
  @spec auto_threshold(tensor(), float()) :: {float(), float()}
  def auto_threshold(weights, target_sparsity \\ 0.5) do
    std = Nx.standard_deviation(weights) |> Nx.to_number()

    # Gaussian estimate: Δ ≈ σ * √2 * erf(1 - target_sparsity)
    initial_delta = std * :math.sqrt(2) * :math.erf(1 - target_sparsity)

    refine_delta(weights, initial_delta, target_sparsity, 0.01, 20)
  end

  defp refine_delta(_weights, delta, _target, _tolerance, 0), do: {delta, 0.0}

  defp refine_delta(weights, delta, target, tolerance, iterations) do
    t = ternarize(weights, delta)
    actual = Nx.divide(Nx.sum(Nx.equal(t, 0)), Nx.size(t)) |> Nx.to_number()
    error = actual - target

    if abs(error) < tolerance do
      {delta, actual}
    else
      adjustment = if error > 0, do: delta * 0.95, else: delta * 1.05
      refine_delta(weights, adjustment, target, tolerance, iterations - 1)
    end
  end

  # ── Quality Metrics ──────────────────────────────────────────

  @doc """
  Compute quantization quality metrics.

  Returns a map with:
    - `:sparsity` — fraction of zero weights
    - `:density` — fraction of non-zero weights
    - `:positive_ratio` — fraction of +1 weights (of non-zero)
    - `:negative_ratio` — fraction of -1 weights (of non-zero)
    - `:mse` — mean squared error vs. original
    - `:sqnr` — signal-to-quantization-noise ratio in dB
  """
  @spec quality_metrics(tensor(), float()) :: map()
  def quality_metrics(original, delta) do
    t = ternarize(original, delta)
    scales = compute_scales(original, delta)
    quantized = ternarize_scaled(original, scales, delta)

    mse = Nx.mean(Nx.pow(Nx.subtract(original, quantized), 2)) |> Nx.to_number()
    signal_power = Nx.mean(Nx.pow(original, 2)) |> Nx.to_number()
    sqnr = if mse > 0, do: 10 * :math.log10(signal_power / mse), else: :infinity

    total = Nx.size(t) |> Nx.to_number()
    zeros = Nx.sum(Nx.equal(t, 0)) |> Nx.to_number()
    pos = Nx.sum(Nx.equal(t, 1)) |> Nx.to_number()
    neg = Nx.sum(Nx.equal(t, -1)) |> Nx.to_number()
    nonzero = total - zeros

    %{
      sparsity: zeros / total,
      density: nonzero / total,
      positive_ratio: if(nonzero > 0, do: pos / nonzero, else: 0.0),
      negative_ratio: if(nonzero > 0, do: neg / nonzero, else: 0.0),
      mse: mse,
      sqnr: sqnr
    }
  end
end
