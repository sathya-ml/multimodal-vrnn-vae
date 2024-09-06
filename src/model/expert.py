from abc import abstractmethod, ABC
from typing import Tuple

import torch


class Expert(torch.nn.Module, ABC):
    def __init__(self):
        """
        Base class for an expert in a mixture of experts model.
        """
        super(Expert, self).__init__()

    @abstractmethod
    def forward(self, loc: torch.Tensor, scale: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass through the expert.

        Args:
            loc (torch.Tensor): Tensor of shape (M, D) representing the means of M experts.
            scale (torch.Tensor): Tensor of shape (M, D) representing the scales (standard deviations) of M experts.

        Returns:
            Tuple[torch.Tensor, ...]: A tuple of tensors representing the expert's output.
            The exact content and shape of the tuple depend on the specific implementation of the expert.
        """
        pass


class ProductOfExperts(Expert):
    """
    Implements the Product of Experts model for combining multiple independent experts.
    See https://www.cs.toronto.edu/~hinton/absps/icann-99.pdf for a detailed breakdown of the method.

    Args:
        num_const (float): A small constant added for numerical stability (e.g., in division). Default is 1e-6.
    """

    def __init__(self, num_const: float = 1e-6) -> None:
        """
        Initializes the ProductOfExperts class with a numerical stability constant.

        Args:
            num_const (float): Constant for numerical stability, default is 1e-6.
        """
        super(ProductOfExperts, self).__init__()

        # Constant for numerical stability (e.g. in division)
        self._eps: float = num_const

    def forward(self, loc: torch.Tensor, scale: torch.Tensor) -> Tuple[torch.Tensor, ...]:
        """
        Forward pass through the product of experts, combining the means and scales of independent experts.

        Args:
            loc (torch.Tensor): Tensor of shape (M, D) representing the means of M experts.
            scale (torch.Tensor): Tensor of shape (M, D) representing the scales (standard deviations) of M experts.

        Returns:
            Tuple[torch.Tensor, ...]: A tuple containing:
                - product_loc (torch.Tensor): Tensor of shape (D,) representing the combined mean of the experts.
                - product_scale (torch.Tensor): Tensor of shape (D,) representing the combined scale of the experts.

        Notes:
            - `scale` is adjusted by adding a small constant `self._eps` for numerical stability.
            - `precision` is computed as the inverse of `scale`, and is used to calculate the combined mean and scale.
        """
        scale += self._eps
        # Precision of i-th Gaussian expert (T = 1/sigma^2)
        precision = 1.0 / scale

        product_loc: torch.Tensor = torch.sum(loc * precision, dim=0) / torch.sum(precision, dim=0)
        product_scale: torch.Tensor = 1.0 / torch.sum(precision, dim=0)

        return product_loc, product_scale
