defmodule TernaryNx.Mac do
  @moduledoc """
  MAC (Multiply-Accumulate) operations for ternary neural network layers.

  Implements a ternary GEMM: Y = (scales * ternary_weight) @ activations^T
  plus sparsity analysis utilities.
  """

  @doc """
  Full ternary GEMM forward pass.

  Computes: output = (scales .* ternary_weights) @ activations

  - `ternary_weights`: binary tensor of values in {-1, 0, 1}, shape {out_features, in_features}
  - `activations`: f32 tensor, shape {batch, in_features}
  - `scales`: f32 tensor, shape {out_features} — per-output-channel scale factors

  Returns f32 tensor of shape {batch, out_features}.
  """
  def ternary_gemm(ternary_weights, activations, scales) do
    # Expand scales to {out_features, 1} for broadcasting
    scaled_weights =
      ternary_weights
      |> Nx.multiply(Nx.new_axis(scales, 1))

    # Matrix multiply: (batch, in_features) @ (in_features, out_features)
    # We store weights as {out_features, in_features}, so transpose
    Nx.dot(activations, Nx.transpose(scaled_weights))
  end

  @doc """
  Compute the fraction of zero entries in a tensor.
  """
  def zero_sparsity(tensor) do
    total = Nx.size(tensor) |> then(&max(&1, 1))
    zeros = Nx.sum(Nx.equal(tensor, 0.0)) |> Nx.to_number()
    zeros / total
  end

  @doc """
  Count effective non-zero MACs (non-zero entries) in a ternary weight tensor.
  """
  def effective_macs(tensor) do
    Nx.sum(Nx.not_equal(tensor, 0.0)) |> Nx.to_number()
  end
end
