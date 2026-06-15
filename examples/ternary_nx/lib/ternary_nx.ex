defmodule TernaryNx do
  @moduledoc """
  TernaryNx — Ternary quantization and GEMM using Nx tensors.

  Provides a high-level demonstration: random weight → ternary quantization →
  ternary GEMM → sparsity analysis → compact packing → summary.
  """

  alias TernaryNx.{Quantizer, Mac, Storage}

  @doc """
  Run a full demo pipeline:
  1. Create random weight and activation tensors
  2. Quantize weights to ternary {-1, 0, +1}
  3. Run ternary GEMM
  4. Measure sparsity
  5. Pack to compact binary
  6. Print summary
  """
  def run_demo(opts \\ []) do
    rows = Keyword.get(opts, :rows, 64)
    cols = Keyword.get(opts, :cols, 128)
    batch = Keyword.get(opts, :batch, 16)
    delta = Keyword.get(opts, :delta, 0.5)

    Nx.default_backend(Nx.BinaryBackend)

    IO.puts("=== TernaryNx Demo ===\n")

    key = Nx.Random.key(42)
    {w_raw, key2} = Nx.Random.uniform(key, shape: {rows, cols}, type: :f32)
    w = Nx.subtract(Nx.multiply(w_raw, 2.0), 1.0)
    {x_raw, _key} = Nx.Random.uniform(key2, shape: {batch, cols}, type: :f32)
    x = Nx.subtract(Nx.multiply(x_raw, 2.0), 1.0)

    IO.puts("Weight tensor shape: #{inspect(Nx.shape(w))}")
    IO.puts("Activation tensor shape: #{inspect(Nx.shape(x))}")

    # 2. Quantize to ternary
    w_tern = Quantizer.ternarize_layer(w, delta)
    sparsity = Mac.zero_sparsity(w_tern)
    IO.puts("Ternary weight sparsity: #{Float.round(sparsity * 100, 2)}% zeros")

    # 3. Compute per-channel scales and run GEMM
    scales = Quantizer.compute_scales(w, delta)
    result = Mac.ternary_gemm(w_tern, x, scales)
    IO.puts("GEMM output shape: #{inspect(Nx.shape(result))}")

    # 4. Effective MACs
    nonzero = Mac.effective_macs(w_tern)
    total = Nx.size(w_tern)
    IO.puts("Effective MACs: #{nonzero} / #{total} (#{Float.round(nonzero / total * 100, 2)}%)")

    # 5. Pack to compact binary
    packed = Storage.pack_nx(w_tern)
    IO.puts("Packed size: #{byte_size(packed)} bytes (original: #{rows * cols * 4} bytes f32)")

    # 6. Round-trip verification
    unpacked = Storage.unpack_nx(packed, {rows, cols})
    match = Nx.all_close(w_tern, unpacked)
    match_v = Nx.to_number(match)
    IO.puts("Round-trip match: #{match_v}")

    IO.puts("\n=== Demo complete ===")

    :ok
  end
end
