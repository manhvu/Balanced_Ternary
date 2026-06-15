defmodule TernaryPure.Storage do
  @moduledoc """
  Packing and encoding of ternary weights for efficient storage.
  Uses base-3 encoding: T (-1) → 0, 0 → 1, 1 → 2.
  """

  @doc """
  Packs a list of exactly 10 trits into a 16-bit integer.
  Mapping: T → 0, 0 → 1, 1 → 2 (base-3 digits).
  """
  @spec pack10([-1 | 0 | 1]) :: non_neg_integer()
  def pack10(trits) when length(trits) == 10 do
    trits
    |> Enum.map(&trit_to_digit/1)
    |> Enum.reduce(0, fn d, acc -> acc * 3 + d end)
  end

  @doc """
  Unpacks a 16-bit integer back into 10 trits.
  """
  @spec unpack10(non_neg_integer()) :: [-1 | 0 | 1]
  def unpack10(packed) when packed >= 0 do
    packed
    |> do_unpack10([])
    |> pad_trits(10)
  end

  defp do_unpack10(0, acc), do: acc

  defp do_unpack10(rem, acc) do
    digit = rem - 3 * div(rem, 3)
    do_unpack10(div(rem, 3), [digit_to_trit(digit) | acc])
  end

  defp pad_trits(list, n) when length(list) >= n, do: list
  defp pad_trits(list, n), do: pad_trits([-1 | list], n)

  defp trit_to_digit(-1), do: 0
  defp trit_to_digit(0), do: 1
  defp trit_to_digit(1), do: 2

  defp digit_to_trit(0), do: -1
  defp digit_to_trit(1), do: 0
  defp digit_to_trit(2), do: 1

  @doc """
  Packs a matrix (list of rows) where each row is 10 trits.
  Returns a list of 16-bit integers.
  """
  @spec pack_matrix([[-1 | 0 | 1]]) :: [non_neg_integer()]
  def pack_matrix(rows) do
    Enum.map(rows, &pack10/1)
  end

  @doc """
  Sparse-encodes a list of trits as [{index, sign}].
  Only non-zero entries are kept; sign is -1 or +1.
  """
  @spec sparse_encode([-1 | 0 | 1]) :: [{non_neg_integer(), -1 | 1}]
  def sparse_encode(weights) do
    weights
    |> Enum.with_index()
    |> Enum.reduce([], fn
      {0, _idx}, acc -> acc
      {w, idx}, acc -> [{idx, w} | acc]
    end)
    |> Enum.reverse()
  end
end
