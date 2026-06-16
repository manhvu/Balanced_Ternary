defmodule TernaryConverter.PackerTest do
  use ExUnit.Case, async: true

  alias TernaryConverter.Packer

  describe "pack/1 and unpack/2" do
    test "round-trip for a single 10-trit word" do
      trits = [1, 0, -1, 1, 1, 0, -1, 0, 1, -1]
      t = Nx.tensor(trits, type: :s64)
      packed = Packer.pack(t)
      unpacked = Packer.unpack(packed, 10)
      assert unpacked == trits
    end

    test "round-trip for multiple words" do
      trits = [1, 0, -1, 1, 1, 0, -1, 0, 1, -1, 0, 0, 1, -1, 1]
      t = Nx.tensor(trits, type: :s64)
      packed = Packer.pack(t)
      unpacked = Packer.unpack(packed, 15)
      assert unpacked == trits
    end

    test "round-trip for 2D tensor" do
      t = Nx.tensor([[1, 0, -1], [0, 1, -1]], type: :s64)
      assert Packer.verify_roundtrip(t)
    end

    test "padded zeros for non-multiple-of-10" do
      trits = [1, -1, 0, 1]
      t = Nx.tensor(trits, type: :s64)
      packed = Packer.pack(t)
      unpacked = Packer.unpack(packed, 4)
      assert unpacked == trits
    end

    test "all zeros" do
      trits = [0, 0, 0, 0, 0]
      t = Nx.tensor(trits, type: :s64)
      packed = Packer.pack(t)
      unpacked = Packer.unpack(packed, 5)
      assert unpacked == trits
    end

    test "all ones" do
      trits = [1, 1, 1, 1, 1]
      t = Nx.tensor(trits, type: :s64)
      packed = Packer.pack(t)
      unpacked = Packer.unpack(packed, 5)
      assert unpacked == trits
    end

    test "all negative ones" do
      trits = [-1, -1, -1, -1, -1]
      t = Nx.tensor(trits, type: :s64)
      packed = Packer.pack(t)
      unpacked = Packer.unpack(packed, 5)
      assert unpacked == trits
    end
  end

  describe "compression_ratio/1" do
    test "returns ratio > 1 for ternary packing" do
      {t, _} = Nx.Random.uniform(Nx.Random.key(42), 0.0, 1.0, shape: {100, 100})
      scaled = Nx.subtract(Nx.multiply(t, 2.0), 1.0)
      tern = TernaryConverter.Quantizer.ternarize(scaled, 0.5)
      {original, packed, ratio} = Packer.compression_ratio(tern)
      assert ratio > 1.0
      assert packed < original
    end
  end

  describe "sparse pack/unpack" do
    test "round-trip for sparse encoding" do
      t = Nx.tensor([1, 0, 0, -1, 0, 0, 0, 1, 0, -1], type: :s64)
      packed = Packer.pack_sparse(t)
      unpacked = Packer.unpack_sparse(packed, 10)
      assert Nx.to_flat_list(Nx.as_type(t, :s64)) == unpacked
    end

    test "round-trip for all zeros" do
      t = Nx.tensor([0, 0, 0, 0, 0], type: :s64)
      packed = Packer.pack_sparse(t)
      unpacked = Packer.unpack_sparse(packed, 5)
      assert unpacked == [0, 0, 0, 0, 0]
    end

    test "round-trip for no zeros" do
      t = Nx.tensor([1, -1, 1, -1], type: :s64)
      packed = Packer.pack_sparse(t)
      unpacked = Packer.unpack_sparse(packed, 4)
      assert Nx.to_flat_list(Nx.as_type(t, :s64)) == unpacked
    end
  end
end
