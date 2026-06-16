defmodule TernaryConverter.QuantizerTest do
  use ExUnit.Case, async: true

  alias TernaryConverter.Quantizer

  describe "ternarize/2" do
    test "basic ternarization with delta" do
      w = Nx.tensor([0.8, -0.3, 0.1, -0.9, 0.5])
      result = Quantizer.ternarize(w, 0.4)
      # 0.8 > 0.4 → 1, -0.3 > -0.4 → 0, 0.1 ≤ 0.4 → 0, -0.9 < -0.4 → -1, 0.5 > 0.4 → 1
      assert Nx.to_flat_list(result) == [1, 0, 0, -1, 1]
    end

    test "all zeros when delta is very large" do
      w = Nx.tensor([0.1, -0.2, 0.3])
      result = Quantizer.ternarize(w, 1.0)
      assert Nx.to_flat_list(result) == [0, 0, 0]
    end

    test "no zeros when delta is zero" do
      w = Nx.tensor([0.1, -0.2, 0.3])
      result = Quantizer.ternarize(w, 0.0)
      assert Nx.to_flat_list(result) == [1, -1, 1]
    end

    test "2D tensor ternarization" do
      w = Nx.tensor([[0.8, -0.3, 0.1], [-0.9, 0.5, -0.2]])
      result = Quantizer.ternarize(w, 0.4)
      # Row 0: 0.8>0.4→1, -0.3>-0.4→0, 0.1≤0.4→0
      # Row 1: -0.9<-0.4→-1, 0.5>0.4→1, -0.2>-0.4→0
      assert Nx.to_flat_list(result) == [1, 0, 0, -1, 1, 0]
    end

    test "exact boundary values" do
      w = Nx.tensor([0.5, -0.5, 0.0])
      result = Quantizer.ternarize(w, 0.5)
      # 0.5 is NOT > 0.5, so → 0; -0.5 is NOT < -0.5, so → 0; 0.0 → 0
      assert Nx.to_flat_list(result) == [0, 0, 0]
    end

    test "just above/below boundary" do
      w = Nx.tensor([0.51, -0.51])
      result = Quantizer.ternarize(w, 0.5)
      assert Nx.to_flat_list(result) == [1, -1]
    end
  end

  describe "compute_scales/2" do
    test "computes per-channel scales" do
      w = Nx.tensor([[0.8, -0.3, 0.1], [-0.9, 0.5, -0.2]])
      scales = Quantizer.compute_scales(w, 0.4)
      list = Nx.to_flat_list(scales)
      assert length(list) == 2
      assert Enum.all?(list, &(&1 > 0))
    end

    test "handles all-zero channel" do
      w = Nx.tensor([[0.01, 0.02, 0.03], [0.8, -0.3, 0.1]])
      scales = Quantizer.compute_scales(w, 0.4)
      list = Nx.to_flat_list(scales)
      # First channel: all values within [-0.4, 0.4], so all ternarized to 0
      # norm = 0, so scale should be 0.0 (dot product is 0)
      assert Enum.at(list, 0) == 0.0
    end
  end

  describe "quality_metrics/2" do
    test "returns valid metrics" do
      w = Nx.tensor([[0.8, -0.3, 0.1, -0.9], [0.5, -0.2, 0.7, -0.4]])
      metrics = Quantizer.quality_metrics(w, 0.4)

      assert is_map(metrics)
      assert Map.has_key?(metrics, :sparsity)
      assert Map.has_key?(metrics, :density)
      assert Map.has_key?(metrics, :mse)
      assert Map.has_key?(metrics, :sqnr)

      assert metrics.sparsity >= 0.0 and metrics.sparsity <= 1.0
      assert metrics.density >= 0.0 and metrics.density <= 1.0
      assert metrics.mse >= 0.0
    end
  end
end
