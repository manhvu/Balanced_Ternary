defmodule TernaryPureTest do
  use ExUnit.Case
  doctest TernaryPure

  describe "Quantizer" do
    test "ternarize single values" do
      assert TernaryPure.Quantizer.ternarize(0.5, 0.3) == 1
      assert TernaryPure.Quantizer.ternarize(-0.5, 0.3) == -1
      assert TernaryPure.Quantizer.ternarize(0.2, 0.3) == 0
      assert TernaryPure.Quantizer.ternarize(-0.2, 0.3) == 0
      # strictly greater than
      assert TernaryPure.Quantizer.ternarize(0.3, 0.3) == 1
      assert TernaryPure.Quantizer.ternarize(-0.3, 0.3) == -1
    end

    test "ternarize_layer" do
      weights = [[0.5, -0.1, 0.8], [-0.6, 0.2, 0.0]]
      result = TernaryPure.Quantizer.ternarize_layer(weights, 0.3)
      assert result == [[1, 0, 1], [-1, 0, 0]]
    end
  end

  describe "MAC" do
    test "trit_mul" do
      assert TernaryPure.MAC.trit_mul(-1, 5) == -5
      assert TernaryPure.MAC.trit_mul(0, 5) == 0
      assert TernaryPure.MAC.trit_mul(1, 5) == 5
      assert TernaryPure.MAC.trit_mul(-1, -2.5) == 2.5
    end

    test "dot_product" do
      w = [1, 0, -1, 1]
      a = [2.0, 3.0, 4.0, 1.0]
      # 1*2 + 0*3 + (-1)*4 + 1*1 = 2 - 4 + 1 = -1
      assert TernaryPure.MAC.dot_product(w, a) == -1.0
    end

    test "dot_product with all zeros" do
      assert TernaryPure.MAC.dot_product([0, 0, 0], [1, 2, 3]) == 0
    end

    test "gemm" do
      w = [[1, 0], [-1, 1]]
      a = [[1.0, 2.0], [3.0, 4.0]]
      # result[0][0] = 1*1 + 0*3 = 1
      # result[0][1] = 1*2 + 0*4 = 2
      # result[1][0] = -1*1 + 1*3 = 2
      # result[1][1] = -1*2 + 1*4 = 2
      assert TernaryPure.MAC.gemm(w, a) == [[1.0, 2.0], [2.0, 2.0]]
    end
  end

  describe "Storage" do
    test "pack10 / unpack10 round-trip" do
      trits = [1, 0, -1, 1, 0, -1, 0, 0, 1, -1]
      packed = TernaryPure.Storage.pack10(trits)
      unpacked = TernaryPure.Storage.unpack10(packed)
      assert unpacked == trits
    end

    test "pack10 all zeros" do
      trits = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
      packed = TernaryPure.Storage.pack10(trits)
      # each 0 -> digit 1 -> all ones in base-3 = (3^10 - 1) / 2 = 29524
      assert packed == 29524
      assert TernaryPure.Storage.unpack10(packed) == trits
    end

    test "pack10 all ones" do
      trits = [1, 1, 1, 1, 1, 1, 1, 1, 1, 1]
      packed = TernaryPure.Storage.pack10(trits)
      # all digits 2 → 2*3^9 + 2*3^8 + ... = 2*(3^10-1)/(3-1) = 3^10 - 1 = 59048
      assert packed == 59048
      assert TernaryPure.Storage.unpack10(packed) == trits
    end

    test "pack10 all negative ones" do
      trits = [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
      packed = TernaryPure.Storage.pack10(trits)
      # all digits 0 → 0
      assert packed == 0
      assert TernaryPure.Storage.unpack10(packed) == trits
    end

    test "pack_matrix" do
      rows = [
        [1, 0, -1, 1, 0, -1, 0, 0, 1, -1],
        [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
      ]

      packed = TernaryPure.Storage.pack_matrix(rows)
      assert length(packed) == 2
      assert Enum.at(packed, 1) == 0
    end

    test "sparse_encode" do
      w = [1, 0, -1, 0, 1]
      assert TernaryPure.Storage.sparse_encode(w) == [{0, 1}, {2, -1}, {4, 1}]
    end

    test "sparse_encode all zeros" do
      assert TernaryPure.Storage.sparse_encode([0, 0, 0]) == []
    end
  end

  describe "Differential" do
    test "encode / decode round-trip" do
      for t <- [-1, 0, 1] do
        pair = TernaryPure.Differential.encode(t)
        assert TernaryPure.Differential.decode(pair) == {:ok, t}
      end
    end

    test "decode invalid pair" do
      assert TernaryPure.Differential.decode({1, 1}) == {:error, :invalid}
    end

    test "negate" do
      assert TernaryPure.Differential.negate(-1) == 1
      assert TernaryPure.Differential.negate(0) == 0
      assert TernaryPure.Differential.negate(1) == -1
    end

    test "compute_pe" do
      assert TernaryPure.Differential.compute_pe(3.0, {0, 0}) == 0
      assert TernaryPure.Differential.compute_pe(3.0, {0, 1}) == -3.0
      assert TernaryPure.Differential.compute_pe(3.0, {1, 0}) == 3.0
    end
  end
end
