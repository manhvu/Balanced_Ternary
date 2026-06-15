defmodule TernaryNx.Quantizer do
  @moduledoc """
  Quantizer — converts full-precision tensors into ternary {-1, 0, +1}.

  Uses plain Nx tensor operations (not Nx.Defn) for broad compatibility.
  """

  @doc """
  Ternarize a tensor element-wise with a threshold `delta`.

  Values greater than `delta` → 1.0
  Values between [-delta, delta] → 0.0
  Values less than `-delta` → -1.0
  """
  def ternarize(tensor, delta) do
    tensor
    |> Nx.greater(delta)
    |> Nx.as_type(:f32)
    |> then(fn pos ->
      neg = Nx.less(tensor, Nx.negate(delta)) |> Nx.as_type(:f32) |> Nx.multiply(-1.0)
      Nx.add(pos, neg)
    end)
  end

  @doc """
  Convenience wrapper — same as `ternarize/2`.
  """
  def ternarize_layer(weight_tensor, delta) do
    ternarize(weight_tensor, delta)
  end

  @doc """
  Compute per-channel MSE-optimal scale factors.

  For each row j: scale_j = sum(W_j * T_j) / sum(T_j * T_j)

  where T is the ternarized weight matrix.
  """
  def compute_scales(weights, delta) do
    ternary = ternarize(weights, delta)

    dot = Nx.sum(Nx.multiply(weights, ternary), axes: [1])
    norm = Nx.sum(Nx.pow(ternary, 2), axes: [1])

    # Avoid division by zero — use 1.0 where norm == 0
    safe_norm = Nx.select(Nx.greater(norm, 0.0), norm, 1.0)
    Nx.divide(dot, safe_norm)
  end
end
