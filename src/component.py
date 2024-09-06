"""
"""
from typing import Tuple, Optional

import torch


class MultilayerPerceptronEncoder(torch.nn.Module):
    """
    Define the PyTorch module that parametrizes q(z|x_i) with an MLP
    """

    def __init__(self, input_dim: int, hidden_dim: int, z_dim: int) -> None:
        super(MultilayerPerceptronEncoder, self).__init__()

        self._input_dim: int = input_dim
        self._hidden_dim: int = hidden_dim
        self.z_dim: int = z_dim

        self.net: torch.nn.Module = torch.nn.Sequential(
            torch.nn.Linear(
                in_features=input_dim,
                out_features=hidden_dim
            ),
            torch.nn.SiLU()
        )
        self.z_loc_layer: torch.nn.Module = torch.nn.Sequential(
            torch.nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(in_features=hidden_dim, out_features=z_dim))
        self.z_scale_layer: torch.nn.Module = torch.nn.Sequential(
            torch.nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(in_features=hidden_dim, out_features=z_dim))

    def forward(self, inputs: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        hidden: torch.Tensor = self.net(inputs)
        z_loc: torch.Tensor = self.z_loc_layer(hidden)
        z_scale: torch.Tensor = torch.exp(self.z_scale_layer(hidden))

        return z_loc, z_scale


class MultilayerPerceptronDecoder(torch.nn.Module):
    """
    Define the PyTorch module that parametrizes p(x|z) with an MLP
    """

    def __init__(self, input_dim: int, hidden_dim: int, z_dim: int) -> None:
        super(MultilayerPerceptronDecoder, self).__init__()

        self._input_dim: int = input_dim
        self._hidden_dim: int = hidden_dim
        self.z_dim: int = z_dim

        self.net: torch.nn.Module = torch.nn.Sequential(
            torch.nn.Linear(in_features=z_dim, out_features=hidden_dim),
            torch.nn.SiLU()
        )
        self.input_layer: torch.nn.Module = torch.nn.Sequential(
            torch.nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(in_features=hidden_dim, out_features=input_dim)
        )

    def forward(self, latent_space: torch.Tensor) -> torch.Tensor:
        hidden: torch.Tensor = self.net(latent_space)
        recon_inputs: torch.Tensor = self.input_layer(hidden)

        return recon_inputs


class MultilayerPerceptron(torch.nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, output_dim: int, device: str) -> None:
        super(MultilayerPerceptron, self).__init__()

        self._input_dim: int = input_dim
        self._hidden_dim: int = hidden_dim
        self._output_dim: int = output_dim

        self._device = device

        self.net: torch.nn.Module = torch.nn.Sequential(
            torch.nn.Linear(in_features=input_dim, out_features=hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(in_features=hidden_dim, out_features=hidden_dim),
            torch.nn.SiLU(),
            torch.nn.Linear(in_features=hidden_dim, out_features=output_dim)
        )

        self.to(device)

    def forward(self, inputs: Optional[torch.Tensor]) -> torch.Tensor:
        features: torch.Tensor = self.net(inputs)

        return features
