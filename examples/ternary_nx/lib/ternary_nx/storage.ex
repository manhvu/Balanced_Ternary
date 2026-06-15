defmodule TernaryNx.Storage do
  @moduledoc """
  Compact binary packing/unpacking for ternary {-1, 0, 1} tensors.

  Packs 10 trits into a 16-bit word (base-3 encoding).
  See Balanced_Ternary/details/04-storage-format.md for the scheme.
  """

  @trits_per_word 10

  @doc """
  Pack an Nx ternary tensor (values in {-1.0, 0.0, 1.0}) into a compact binary.

  Returns a binary where every 10 elements are packed into 2 bytes (16 bits).
  The tensor is flattened row-major; trailing elements are zero-padded.
  """
  def pack_nx(tensor) do
    tensor
    |> Nx.to_flat_list()
    |> Enum.map(&trunc/1)
    |> pack_trits()
    |> :erlang.list_to_binary()
  end

  @doc """
  Unpack a packed binary back into an Nx tensor of the given shape.

  The binary must have been produced by `pack_nx/1`.
  """
  def unpack_nx(binary, shape) do
    words = for <<word::16-integer-native <- binary>>, do: word

    trits =
      words
      |> Enum.flat_map(&unpack_word/1)

    total = Tuple.product(shape)
    trimmed = Enum.take(trits, total)
    padded = trimmed ++ List.duplicate(0.0, total - length(trimmed))

    Nx.tensor(padded, type: :f32) |> Nx.reshape(shape)
  end

  # ── Internal packing ──────────────────────────────────────────

  defp pack_trits(trits) do
    trits
    |> Enum.chunk_every(@trits_per_word)
    |> Enum.map(fn chunk ->
      padded = chunk ++ List.duplicate(0, @trits_per_word - length(chunk))
      encode_word(padded)
    end)
  end

  defp encode_word(trits) do
    trits
    # shift: -1→0, 0→1, +1→2
    |> Enum.map(&(&1 + 1))
    |> Enum.reduce(0, fn digit, acc -> acc * 3 + digit end)
    |> then(&<<&1::16-integer-native>>)
  end

  defp unpack_word(word) do
    digits = extract_base3_digits(word, @trits_per_word, [])

    Enum.map(digits, fn
      0 -> -1.0
      1 -> 0.0
      2 -> 1.0
    end)
  end

  defp extract_base3_digits(_v, n, acc) when length(acc) == n, do: acc

  defp extract_base3_digits(v, n, acc) do
    extract_base3_digits(div(v, 3), n, [rem(v, 3) | acc])
  end
end
