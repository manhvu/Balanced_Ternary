defmodule TernaryPure.Layer do
  @moduledoc """
  A neural network layer with ternary weights, per-channel scales and bias.
  """

  defstruct [:weights, :scales, :bias, :delta]

  @type t :: %__MODULE__{
          weights: [[-1 | 0 | 1]],
          scales: [float()],
          bias: [float()],
          delta: float()
        }

  @doc """
  Forward pass through the ternary layer.
  Each output neuron i:
    output[i] = (sum_j (W_ternary[i][j] * activation[j])) * scale[i] + bias[i]
  """
  @spec forward(%__MODULE__{}, [number()]) :: [number()]
  def forward(%__MODULE__{weights: w, scales: s, bias: b}, activations) do
    w
    |> Enum.zip(Enum.zip(s, b))
    |> Enum.map(fn {w_row, {scale, bias}} ->
      dot = TernaryPure.MAC.dot_product(w_row, activations)
      dot * scale + bias
    end)
  end

  @doc """
  Creates a TernaryLayer from dense (FP32) weights.
  Computes per-channel scale factor minimizing MSE to the original weights.

  For each channel (row):
    1. Quantize weights to ternary with the given delta.
    2. Compute scale = dot(original, ternary) / dot(ternary, ternary)
  """
  @spec from_dense([[float()]], float()) :: %__MODULE__{}
  def from_dense(dense_weights, delta) do
    quantizer = TernaryPure.Quantizer

    {ternary_rows, scales, biases} =
      dense_weights
      |> Enum.map(fn row ->
        t_row = quantizer.ternarize_layer([row], delta) |> hd()

        # Per-channel scale via MSE-optimal least-squares
        t_dot_t = dot_self(t_row)

        scale =
          if t_dot_t == 0 do
            0.0
          else
            dot(t_row, row) / t_dot_t
          end

        bias = 0.0
        {t_row, scale, bias}
      end)
      |> unzip3()

    %__MODULE__{
      weights: ternary_rows,
      scales: scales,
      bias: biases,
      delta: delta
    }
  end

  defp dot(a, b), do: Enum.zip(a, b) |> Enum.reduce(0, fn {x, y}, acc -> acc + x * y end)
  defp dot_self(a), do: Enum.reduce(a, 0, fn x, acc -> acc + x * x end)

  defp unzip3(list),
    do: Enum.reduce(list, {[], [], []}, fn {a, b, c}, {as, bs, cs} -> {[a | as], [b | bs], [c | cs]} end) |> then(fn {a, b, c} -> {Enum.reverse(a), Enum.reverse(b), Enum.reverse(c)} end)
end
