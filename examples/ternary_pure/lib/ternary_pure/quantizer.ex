defmodule TernaryPure.Quantizer do
  @moduledoc """
  Quantizes floating-point weights to ternary values {-1, 0, +1}.
  """

  @doc """
  Quantizes a single weight to {-1, 0, +1} using a symmetric threshold.
  Returns -1 if `weight < -delta`, +1 if `weight > delta`, otherwise 0.
  """
  @spec ternarize(float(), float()) :: -1 | 0 | 1
  def ternarize(weight, delta) when weight <= -delta, do: -1
  def ternarize(weight, delta) when weight >= delta, do: 1
  def ternarize(_weight, _delta), do: 0

  @doc """
  Quantizes a list of lists (2D matrix) of weights.
  Each row is quantized with the same delta.
  """
  @spec ternarize_layer([[float()]], float()) :: [[-1 | 0 | 1]]
  def ternarize_layer(weights, delta) do
    Enum.map(weights, fn row ->
      Enum.map(row, &ternarize(&1, delta))
    end)
  end
end
