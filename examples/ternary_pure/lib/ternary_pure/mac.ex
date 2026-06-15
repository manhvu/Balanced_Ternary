defmodule TernaryPure.MAC do
  @moduledoc """
  Multiply-Accumulate operations with ternary weights.
  """

  @doc """
  Ternary multiplication: trit_mul(-1, x) = -x, trit_mul(0, x) = 0, trit_mul(1, x) = x.
  """
  @spec trit_mul(-1 | 0 | 1, number()) :: number()
  def trit_mul(-1, x), do: -x
  def trit_mul(0, _x), do: 0
  def trit_mul(1, x), do: x

  @doc """
  Computes the dot product of ternary weights and activations.
  Skips zero-weight terms for efficiency.
  """
  @spec dot_product([-1 | 0 | 1], [number()]) :: number()
  def dot_product(weights, activations) do
    weights
    |> Enum.zip(activations)
    |> Enum.reduce(0, fn {w, a}, acc ->
      case w do
        0 -> acc
        -1 -> acc - a
        1 -> acc + a
      end
    end)
  end

  @doc """
  General matrix multiply: ternary weights (rows × cols) dotted with 2D activations.
  Each row of `weights_2d` is dotted against each column of `activations_2d`.
  Returns a result matrix with dimensions:
    length(weights_2d) × length(hd(activations_2d)).
  """
  @spec gemm([[-1 | 0 | 1]], [[number()]]) :: [[number()]]
  def gemm(weights_2d, activations_2d) do
    activations_t = Enum.zip_with(activations_2d, & &1)

    Enum.map(weights_2d, fn w_row ->
      Enum.map(activations_t, fn a_col ->
        dot_product(w_row, a_col)
      end)
    end)
  end
end
