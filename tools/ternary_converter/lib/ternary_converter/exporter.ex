defmodule TernaryConverter.Exporter do
  @moduledoc """
  Exports and imports ternary models to/from .tbin binary format.

  The .tbin format:
    [Header][Layer 1][Layer 2]...[Layer N]

    Header:
      magic:       4 bytes  ("TBN\\0")
      version:     4 bytes  (uint32)
      num_layers:  4 bytes  (uint32)
      meta_len:    4 bytes  (uint32)
      metadata:    meta_len bytes (UTF-8 JSON)

    Per layer: binary serialized TernaryConverter.Layer
  """

  @tbin_magic "TBN\0"
  @tbin_version 1

  @doc """
  Export a list of ternary layers to a .tbin file.

  ## Options
    - `:metadata` — Map of metadata (model name, version, etc.)
    - `:compress` — Whether to compress with gzip (default: false)
  """
  @spec export([TernaryConverter.Layer.t()], String.t(), keyword()) :: :ok | {:error, term()}
  def export(layers, output_path, opts \\ []) do
    metadata = Keyword.get(opts, :metadata, %{})
    compress = Keyword.get(opts, :compress, false)

    binary = build_tbin(layers, metadata, compress)

    case File.write(output_path, binary) do
      :ok ->
        size_mb = byte_size(binary) / 1024 / 1024

        IO.puts(
          "Exported #{length(layers)} layers to #{output_path} (#{Float.round(size_mb, 1)} MB)"
        )

        :ok

      {:error, reason} ->
        {:error, reason}
    end
  end

  defp build_tbin(layers, metadata, compress) do
    metadata_binary = Jason.encode!(metadata)

    layer_binaries =
      Enum.map(layers, fn layer ->
        bin = TernaryConverter.Layer.to_binary(layer)
        <<byte_size(bin)::32, bin::binary>>
      end)

    layers_binary = :erlang.list_to_binary(layer_binaries)

    binary = <<
      @tbin_magic::binary,
      @tbin_version::32,
      length(layers)::32,
      byte_size(metadata_binary)::32,
      metadata_binary::binary,
      layers_binary::binary
    >>

    if compress, do: :zlib.gzip(binary), else: binary
  end

  @doc """
  Load a .tbin file back into a list of layers.
  """
  @spec load(String.t()) ::
          {:ok, [TernaryConverter.Layer.t()], map()} | {:error, term()}
  def load(input_path) do
    with {:ok, binary} <- File.read(input_path),
         binary <- maybe_decompress(binary),
         {:ok, layers, metadata} <- parse_tbin(binary) do
      {:ok, layers, metadata}
    else
      {:error, reason} -> {:error, reason}
    end
  end

  defp maybe_decompress(binary) do
    case binary do
      <<31, 139, _rest::binary>> -> :zlib.gunzip(binary)
      _ -> binary
    end
  end

  defp parse_tbin(<<
         @tbin_magic::binary,
         @tbin_version::32,
         num_layers::32,
         meta_len::32,
         metadata_binary::binary-size(meta_len),
         layers_binary::binary
       >>) do
    metadata = Jason.decode!(metadata_binary)
    layers = parse_layers(layers_binary, num_layers, [])
    {:ok, Enum.reverse(layers), metadata}
  end

  defp parse_tbin(_), do: {:error, "invalid .tbin format"}

  defp parse_layers(<<>>, 0, acc), do: acc

  defp parse_layers(binary, remaining, acc) do
    {layer, rest} = parse_single_layer(binary)
    parse_layers(rest, remaining - 1, [layer | acc])
  end

  defp parse_single_layer(binary) do
    <<layer_size::32, layer_binary::binary-size(layer_size), rest::binary>> = binary
    layer = TernaryConverter.Layer.from_binary(layer_binary)
    {layer, rest}
  end
end
