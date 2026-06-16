defmodule TernaryConverter.IntegrationTest do
  use ExUnit.Case, async: true

  alias TernaryConverter.{Packer, Layer}

  describe "full pipeline" do
    test "convert → export → load → inference" do
      # 1. Create synthetic weights
      {w1, _} = Nx.Random.uniform(Nx.Random.key(1), 0.0, 1.0, shape: {64, 128})
      {w2, _} = Nx.Random.uniform(Nx.Random.key(2), 0.0, 1.0, shape: {32, 64})
      w1 = Nx.subtract(Nx.multiply(w1, 2.0), 1.0)
      w2 = Nx.subtract(Nx.multiply(w2, 2.0), 1.0)
      weights = %{"fc1" => w1, "fc2" => w2}

      # 2. Convert to ternary
      layers = TernaryConverter.convert_all(weights, delta: 0.5)
      assert length(layers) == 2

      # 3. Export to .tbin
      path = "test_model_tmp.tbin"
      :ok = TernaryConverter.export(layers, path, metadata: %{test: true})

      # 4. Load back
      {:ok, loaded_layers, metadata} = TernaryConverter.load(path)
      assert length(loaded_layers) == 2
      assert metadata["test"] == true

      # 5. Run inference
      {input, _} = Nx.Random.uniform(Nx.Random.key(3), 0.0, 1.0, shape: {1, 128})
      output = TernaryConverter.inference(loaded_layers, input)
      assert elem(Nx.shape(output), 0) == 1
      assert elem(Nx.shape(output), 1) == 32

      # Cleanup
      File.rm(path)
    end

    test "packer round-trip for all layers" do
      {w, _} = Nx.Random.uniform(Nx.Random.key(4), 0.0, 1.0, shape: {100, 200})
      w = Nx.subtract(Nx.multiply(w, 2.0), 1.0)
      _layer = Layer.from_dense(w, "test", delta: 0.5)
      assert Packer.verify_roundtrip(TernaryConverter.Quantizer.ternarize(w, 0.5))
    end

    test "compression ratio is meaningful" do
      {w, _} = Nx.Random.uniform(Nx.Random.key(5), 0.0, 1.0, shape: {256, 512})
      w = Nx.subtract(Nx.multiply(w, 2.0), 1.0)
      layer = Layer.from_dense(w, "test", delta: 0.5)
      packed_size = byte_size(layer.weight_packed)
      original_size = 256 * 512 * 4
      assert packed_size < original_size
    end
  end
end
