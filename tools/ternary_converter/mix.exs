defmodule TernaryConverter.MixProject do
  use Mix.Project

  def project do
    [
      app: :ternary_converter,
      version: "0.1.0",
      elixir: "~> 1.15",
      start_permanent: Mix.env() == :prod,
      deps: deps(),
      escript: escript()
    ]
  end

  def application do
    [extra_applications: [:logger]]
  end

  defp deps do
    [
      {:nx, "~> 0.9"},
      {:jason, "~> 1.4"}
    ]
  end

  defp escript do
    [main_module: TernaryConverter.CLI]
  end
end
