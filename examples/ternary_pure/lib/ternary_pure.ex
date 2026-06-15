defmodule TernaryPure do
  @moduledoc """
  Main module for the TernaryPure demo.
  """

  @doc """
  Runs a full demo pipeline:
  1. Creates a 3×4 layer from random weights
  2. Quantizes it
  3. Packs the ternary weights
  4. Runs a forward pass
  5. Prints all intermediate values
  """
  def run_demo do
    IO.puts("=== TernaryPure Demo ===\n")

    # 1. Random dense weights (3 output channels × 10 inputs — 10 for pack demo)
    dense = [
      [0.42, -0.13, 0.87, -0.55, 0.11, -0.33, 0.72, -0.08, 0.64, -0.29],
      [-0.91, 0.24, 0.03, 0.68, -0.45, 0.17, -0.82, 0.05, 0.39, -0.01],
      [0.15, -0.72, -0.34, 0.09, 0.93, -0.12, 0.04, -0.67, 0.31, 0.55]
    ]

    delta = 0.3

    IO.puts("Dense weights (3×10):")
    IO.inspect(dense, label: "dense")
    IO.puts("")

    # 2. Quantize
    quantized = TernaryPure.Quantizer.ternarize_layer(dense, delta)
    IO.puts("Quantized weights (delta=#{delta}):")
    IO.inspect(quantized, label: "quantized")
    IO.puts("")

    # 3. Pack — each row has exactly 10 trits now
    packed_rows = TernaryPure.Storage.pack_matrix(quantized)
    IO.puts("Packed matrix (16-bit words):")
    IO.inspect(packed_rows, label: "packed")
    unpacked = Enum.map(packed_rows, &TernaryPure.Storage.unpack10/1)
    IO.puts("Unpacked back:")
    IO.inspect(unpacked, label: "unpacked")
    IO.inspect(unpacked == quantized, label: "round_trip_ok?")
    IO.puts("")

    # Sparse encoding (first row only)
    sparse = TernaryPure.Storage.sparse_encode(Enum.at(quantized, 0))
    IO.puts("Sparse encoding of row 0:")
    IO.inspect(sparse, label: "sparse")
    IO.puts("")

    # 4. Forward pass — use first 4 activations for compact demo
    activations = [1.0, -0.5, 0.3, 0.8, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0]
    layer = TernaryPure.Layer.from_dense(dense, delta)
    IO.puts("TernaryLayer struct:")
    IO.inspect(layer, label: "layer")
    IO.puts("")

    # Manually compute with MAC module for comparison
    dp_results =
      Enum.map(quantized, fn w_row ->
        TernaryPure.MAC.dot_product(w_row, activations)
      end)

    IO.puts("Dot products (pre-scale):")
    IO.inspect(dp_results, label: "dot_products")
    IO.puts("")

    output = TernaryPure.Layer.forward(layer, activations)
    IO.puts("Layer output (post scale+bias):")
    IO.inspect(output, label: "output")
    IO.puts("")

    # Differential encoding demo
    IO.puts("Differential encoding demo:")

    [-1, 0, 1]
    |> Enum.each(fn t ->
      pair = TernaryPure.Differential.encode(t)
      {:ok, decoded} = TernaryPure.Differential.decode(pair)
      neg = TernaryPure.Differential.negate(t)
      pe = TernaryPure.Differential.compute_pe(activations |> hd(), pair)

      IO.inspect(%{trit: t, pair: pair, decoded: decoded, negated: neg, pe_result: pe},
        label: "differential"
      )
    end)

    IO.puts("\n=== Demo complete ===")
    :ok
  end
end
