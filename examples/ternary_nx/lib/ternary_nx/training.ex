defmodule TernaryNx.Training do
  @moduledoc """
  QAT (Quantization-Aware Training) simulation helpers.

  Provides a straight-through estimator (STE) forward pass, sparsity loss,
  and a simple SGD update step — all using plain Nx tensor operations.
  """

  alias TernaryNx.Quantizer

  @doc """
  STE forward pass: quantize weights in the forward pass but let gradients
  flow through un-quantized in the backward pass (simulated by using the
  quantized weights for the result but returning the shadow weights unchanged).

  Returns `{result, quantized_weights}`.

  - `w_shadow`: full-precision shadow weights, shape {out, in}
  - `scale`: per-channel scale vector, shape {out}
  - `x`: activations, shape {batch, in}
  - `delta`: quantization threshold
  """
  def forward(w_shadow, scale, x, delta) do
    w_tern = Quantizer.ternarize_layer(w_shadow, delta)

    scaled_w =
      w_tern
      |> Nx.multiply(Nx.new_axis(scale, 1))

    result = Nx.dot(x, Nx.transpose(scaled_w))
    {result, w_tern}
  end

  @doc """
  Sparsity regularization loss: encourages more weights to be exactly zero.

  Uses a differentiable approximation:
    loss = mean(tanh(|w| / delta))

  When |w| is small (below delta), tanh is ~|w|/delta → pushes toward zero.
  When |w| is large, tanh saturates at 1 → no gradient.
  """
  def sparsity_loss(w, delta) do
    w
    |> Nx.abs()
    |> Nx.divide(delta)
    |> Nx.tanh()
    |> Nx.mean()
  end

  @doc """
  Simple SGD weight update: w = w - lr * grad
  """
  def update_weights(w, grad, lr) do
    Nx.subtract(w, Nx.multiply(grad, lr))
  end
end
