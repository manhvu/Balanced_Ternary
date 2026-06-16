defmodule TernaryConverter.Layer do
  @moduledoc """
  A neural network layer with ternary weights.

  Supports:
  - Forward pass with add/sub/skip (no multiplication)
  - Per-channel scale factors and bias
  - Conversion from dense (FP32) weights
  - Packed weight storage (10 trits per 16-bit word)
  - Serialization to/from binary format
  """

  defstruct [
    :name,
    :weight_packed,
    :scales,
    :bias,
    :shape,
    :delta,
    :sparsity
  ]

  @type t :: %__MODULE__{
          name: String.t(),
          weight_packed: binary(),
          scales: [float()],
          bias: [float()],
          shape: {non_neg_integer(), non_neg_integer()},
          delta: float(),
          sparsity: float()
        }

  alias TernaryConverter.{Quantizer, Packer}

  @doc """
  Create a TernaryLayer from a dense (FP32) weight matrix.

  Automatically computes per-channel scale factors and packs weights.

  ## Examples

      iex> w = Nx.tensor([[0.8, -0.3, 0.1, -0.9], [0.5, -0.2, 0.7, -0.4]])
      iex> layer = TernaryConverter.Layer.from_dense(w, "fc1", delta: 0.4)
      iex> layer.shape
      {2, 4}
      iex> layer.name
      "fc1"
  """
  @spec from_dense(Nx.Tensor.t(), String.t(), keyword()) :: t()
  def from_dense(weight_tensor, name, opts \\ []) do
    delta = Keyword.get(opts, :delta, 0.5)
    {out_features, in_features} = Nx.shape(weight_tensor)

    scales = Quantizer.compute_scales(weight_tensor, delta)
    scales_list = Nx.to_flat_list(scales)
    ternary = Quantizer.ternarize(weight_tensor, delta)

    zeros = Nx.sum(Nx.equal(ternary, 0)) |> Nx.to_number()
    total = Nx.size(ternary) |> Nx.to_number()
    sparsity = zeros / total

    packed = Packer.pack(ternary)
    bias = List.duplicate(0.0, out_features)

    %__MODULE__{
      name: name,
      weight_packed: packed,
      scales: scales_list,
      bias: bias,
      shape: {out_features, in_features},
      delta: delta,
      sparsity: sparsity
    }
  end

  @doc """
  Forward pass through the ternary layer.

  Computes: output[i] = scale[i] * sum_j(W_ternary[i][j] * x[j]) + bias[i]

  Uses add/sub/skip — no multiplication for the weight-activation product.

  ## Examples

      iex> w = Nx.tensor([[0.8, -0.3, 0.1], [0.5, -0.2, 0.7]])
      iex> layer = TernaryConverter.Layer.from_dense(w, "test", delta: 0.4)
      iex> TernaryConverter.Layer.forward(layer, [1.0, 2.0, 3.0])
      [0.5666666626930237, 1.619999885559082]
  """
  @spec forward(t(), [number()]) :: [number()]
  def forward(%__MODULE__{} = layer, activations) do
    {out_features, in_features} = layer.shape

    trits =
      Packer.unpack(layer.weight_packed, out_features * in_features)

    weight_rows = Enum.chunk_every(trits, in_features)

    weight_rows
    |> Enum.zip(Enum.zip(layer.scales, layer.bias))
    |> Enum.map(fn {w_row, {scale, bias}} ->
      dot =
        Enum.zip(w_row, activations)
        |> Enum.reduce(0.0, fn
          {1, x}, acc -> acc + x
          {-1, x}, acc -> acc - x
          {0, _x}, acc -> acc
        end)

      dot * scale + bias
    end)
  end

  @doc """
  Forward pass with Nx tensors (for batched inference).
  """
  @spec forward_nx(t(), Nx.Tensor.t()) :: Nx.Tensor.t()
  def forward_nx(%__MODULE__{} = layer, input_tensor) do
    {out_features, in_features} = layer.shape

    trits =
      Packer.unpack(layer.weight_packed, out_features * in_features)

    weight_matrix =
      trits
      |> Enum.chunk_every(in_features)
      |> Nx.tensor(type: :f32)

    scales = Nx.tensor(layer.scales) |> Nx.new_axis(1)
    bias = Nx.tensor(layer.bias)

    scaled_weights = Nx.multiply(weight_matrix, scales)

    Nx.dot(input_tensor, Nx.transpose(scaled_weights))
    |> Nx.add(Nx.new_axis(bias, 0))
  end

  @doc """
  Serialize layer to binary (.tbin format).
  """
  @spec to_binary(t()) :: binary()
  def to_binary(%__MODULE__{} = layer) do
    {out, in_feat} = layer.shape

    scales_binary = for s <- layer.scales, into: <<>>, do: <<s::float-16>>
    bias_binary = for b <- layer.bias, into: <<>>, do: <<b::float-16>>

    <<
      byte_size(layer.name)::16,
      layer.name::binary,
      out::32,
      in_feat::32,
      layer.delta::float-32,
      layer.sparsity::float-32,
      length(layer.scales)::32,
      scales_binary::binary,
      length(layer.bias)::32,
      bias_binary::binary,
      byte_size(layer.weight_packed)::32,
      layer.weight_packed::binary
    >>
  end

  @doc """
  Deserialize layer from binary.
  """
  @spec from_binary(binary()) :: t()
  def from_binary(<<
        name_len::16,
        name::binary-size(name_len),
        out::32,
        in_feat::32,
        delta::float-32,
        sparsity::float-32,
        num_scales::32,
        scales_binary::binary-size(num_scales)-unit(16),
        num_bias::32,
        bias_binary::binary-size(num_bias)-unit(16),
        packed_size::32,
        packed::binary-size(packed_size)
      >>) do
    scales = for <<s::float-16 <- scales_binary>>, do: s
    bias = for <<b::float-16 <- bias_binary>>, do: b

    %__MODULE__{
      name: name,
      weight_packed: packed,
      scales: scales,
      bias: bias,
      shape: {out, in_feat},
      delta: delta,
      sparsity: sparsity
    }
  end
end
