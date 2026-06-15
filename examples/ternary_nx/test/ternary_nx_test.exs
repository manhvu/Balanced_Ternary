defmodule TernaryNxTest do
  use ExUnit.Case

  alias TernaryNx.{Quantizer, Mac, Storage, Training}

  setup do
    Nx.default_backend(Nx.BinaryBackend)
    :ok
  end

  describe "Quantizer" do
    test "ternarize returns only -1, 0, 1" do
      key = Nx.Random.key(42)
      {t, _key} = Nx.Random.uniform(key, shape: {100, 100}, type: :f32)
      q = Quantizer.ternarize_layer(t, 0.5)

      vals =
        q
        |> Nx.reshape({:auto})
        |> Nx.to_flat_list()
        |> Enum.map(&trunc/1)
        |> Enum.uniq()
        |> MapSet.new()

      assert MapSet.subset?(vals, MapSet.new([-1, 0, 1]))
    end

    test "ternarize with delta=0 is sign function" do
      t = Nx.tensor([[2.0, 0.0, -3.0], [-0.5, 0.1, 5.0]])
      q = Quantizer.ternarize_layer(t, 0.0)
      expected = Nx.tensor([[1.0, 1.0, -1.0], [-1.0, 1.0, 1.0]])
      assert Nx.all_close(q, expected)
    end

    test "compute_scales returns correct length" do
      key = Nx.Random.key(42)
      {w, _key} = Nx.Random.uniform(key, shape: {16, 32}, type: :f32)
      s = Quantizer.compute_scales(w, 0.5)
      assert Nx.shape(s) == {16}
    end
  end

  describe "Mac" do
    test "ternary_gemm produces correct shape" do
      w = Nx.tensor([[1.0, 0.0, -1.0], [0.0, 1.0, 1.0]])
      x = Nx.tensor([[0.5, 1.0, -0.5], [1.0, -1.0, 2.0]])
      s = Nx.tensor([1.0, 2.0])
      result = Mac.ternary_gemm(w, x, s)
      assert Nx.shape(result) == {2, 2}
    end

    test "zero_sparsity returns 0 to 1 range" do
      t = Nx.tensor([1.0, 0.0, 0.0, -1.0, 0.0])
      assert Mac.zero_sparsity(t) == 3.0 / 5.0
    end

    test "effective_macs counts non-zeros" do
      t = Nx.tensor([1.0, 0.0, -1.0, 0.0, 1.0])
      assert Mac.effective_macs(t) == 3
    end
  end

  describe "Storage" do
    test "round-trip pack/unpack" do
      t =
        Nx.tensor(
          [
            [
              1.0,
              0.0,
              -1.0,
              0.0,
              1.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0,
              -1.0,
              1.0,
              0.0,
              1.0,
              -1.0,
              0.0,
              0.0,
              0.0,
              0.0,
              0.0
            ]
          ],
          type: :f32
        )
        |> Nx.reshape({2, 10})

      packed = Storage.pack_nx(t)
      unpacked = Storage.unpack_nx(packed, {2, 10})
      assert Nx.all_close(t, unpacked)
    end
  end

  describe "Training" do
    test "forward returns result and quantized weights" do
      key = Nx.Random.key(42)
      {w_raw, key2} = Nx.Random.uniform(key, shape: {4, 8}, type: :f32)
      w = Nx.subtract(Nx.multiply(w_raw, 2.0), 1.0)
      {x_raw, _key2} = Nx.Random.uniform(key2, shape: {3, 8}, type: :f32)
      x = Nx.subtract(Nx.multiply(x_raw, 2.0), 1.0)
      s = Nx.tensor([1.0, 1.0, 1.0, 1.0])
      {result, qw} = Training.forward(w, s, x, 0.5)
      assert Nx.shape(result) == {3, 4}
      assert Nx.shape(qw) == {4, 8}
    end

    test "sparsity_loss returns scalar" do
      w = Nx.tensor([[0.1, -0.2, 1.5, -0.05]])
      loss = Training.sparsity_loss(w, 0.5)
      assert loss |> Nx.to_number() |> is_number()
    end

    test "update_weights changes values" do
      w = Nx.tensor([1.0, 2.0, 3.0])
      g = Nx.tensor([0.1, 0.1, 0.1])
      w2 = Training.update_weights(w, g, 0.5)
      expected = Nx.tensor([0.95, 1.95, 2.95])
      assert Nx.all_close(w2, expected)
    end
  end
end
