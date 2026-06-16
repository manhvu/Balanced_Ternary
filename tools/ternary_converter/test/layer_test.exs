defmodule TernaryConverter.LayerTest do
  use ExUnit.Case, async: true

  alias TernaryConverter.Layer

  describe "from_dense/3" do
    test "creates layer with correct shape" do
      w = Nx.tensor([[0.8, -0.3, 0.1, -0.9], [0.5, -0.2, 0.7, -0.4]])
      layer = Layer.from_dense(w, "test_fc", delta: 0.4)
      assert layer.name == "test_fc"
      assert layer.shape == {2, 4}
      assert layer.delta == 0.4
      assert is_binary(layer.weight_packed)
      assert length(layer.scales) == 2
      assert length(layer.bias) == 2
    end

    test "computes sparsity correctly" do
      w = Nx.tensor([[0.8, -0.3, 0.1, -0.9], [0.5, -0.2, 0.7, -0.4]])
      layer = Layer.from_dense(w, "test_fc", delta: 0.4)
      assert layer.sparsity >= 0.0 and layer.sparsity <= 1.0
    end
  end

  describe "forward/2" do
    test "computes dot product with add/sub/skip" do
      w = Nx.tensor([[0.8, -0.3, 0.1], [0.5, -0.2, 0.7]])
      layer = Layer.from_dense(w, "test_fc", delta: 0.4)
      activations = [1.0, 2.0, 3.0]
      output = Layer.forward(layer, activations)
      assert length(output) == 2
      assert is_list(output)
      assert Enum.all?(output, &is_float/1)
    end
  end

  describe "serialization round-trip" do
    test "to_binary and from_binary preserve layer" do
      w = Nx.tensor([[0.8, -0.3, 0.1, -0.9], [0.5, -0.2, 0.7, -0.4]])
      layer = Layer.from_dense(w, "test_fc", delta: 0.4)
      binary = Layer.to_binary(layer)
      restored = Layer.from_binary(binary)
      assert restored.name == layer.name
      assert restored.shape == layer.shape
      assert_in_delta restored.delta, layer.delta, 0.001
      assert abs(restored.sparsity - layer.sparsity) < 0.001
    end
  end
end
