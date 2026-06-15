defmodule TernaryPure.Differential do
  @moduledoc """
  Differential (dual-rail) signalling for ternary values.
  Maps each trit to a {wire_a, wire_b} pair:
    -1 → {0, 1}
     0 → {0, 0}
    +1 → {1, 0}
  Free negation via wire swap.
  """

  @doc """
  Encodes a trit into a {wire_a, wire_b} differential pair.
  """
  @spec encode(-1 | 0 | 1) :: {0 | 1, 0 | 1}
  def encode(-1), do: {0, 1}
  def encode(0), do: {0, 0}
  def encode(1), do: {1, 0}

  @doc """
  Decodes a {wire_a, wire_b} pair back to a trit.
  Returns {:ok, trit} on success, {:error, :invalid} for invalid pairs.
  """
  @spec decode({0 | 1, 0 | 1}) :: {:ok, -1 | 0 | 1} | {:error, :invalid}
  def decode({0, 0}), do: {:ok, 0}
  def decode({0, 1}), do: {:ok, -1}
  def decode({1, 0}), do: {:ok, 1}
  def decode(_), do: {:error, :invalid}

  @doc """
  Negates a trit for free by swapping the differential wires.
  """
  @spec negate(-1 | 0 | 1) :: -1 | 0 | 1
  def negate(-1), do: 1
  def negate(0), do: 0
  def negate(1), do: -1

  @doc """
  Processing element operation on a differential pair.
  Returns 0, -x, or x depending on the pair.
  """
  @spec compute_pe(number(), {0 | 1, 0 | 1}) :: number()
  def compute_pe(_x, {0, 0}), do: 0
  def compute_pe(x, {0, 1}), do: -x
  def compute_pe(x, {1, 0}), do: x
end
