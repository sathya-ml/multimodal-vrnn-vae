from typing import Tuple, Dict

import torch
import torch.nn.functional as torch_functional
import torch.nn as nn

from model.expert import Expert


class MultimodalVariationalAutoencoder(torch.nn.Module):
    """
    A flexible multimodal variational autoencoder that can handle multiple modalities.

    This class implements a variational autoencoder (VAE) capable of processing and reconstructing
    data from multiple modalities. It uses separate encoders and decoders for each modality and
    combines the results using a latent space learned from the data. The model can be trained on
    either CPU or GPU.

    Attributes:
        encoders (Dict[str, torch.nn.Module]): A dictionary mapping modality names to their respective encoder modules.
        decoders (Dict[str, torch.nn.Module]): A dictionary mapping modality names to their respective decoder modules.
        _loss_weights (Dict[str, float]): A dictionary of weights for each modality's reconstruction loss.
        _expert (Expert): An instance of an Expert class used for decision-making in the latent space.
        _latent_space_dim (int): Dimensionality of the latent space.
        use_cuda (bool): Whether to use GPU for computations.
    """

    def __init__(
            self,
            encoders: Dict[str, torch.nn.Module],
            decoders: Dict[str, torch.nn.Module],
            loss_weights: Dict[str, float],
            expert: Expert,
            latent_space_dim: int,
            use_cuda: bool = False
    ) -> None:
        """
        Initializes the MultimodalVariationalAutoencoder with given parameters.

        Args:
            encoders (Dict[str, torch.nn.Module]): A dictionary mapping modality names to encoder modules.
            decoders (Dict[str, torch.nn.Module]): A dictionary mapping modality names to decoder modules.
            loss_weights (Dict[str, float]): A dictionary of weights for each modality's reconstruction loss.
            expert (Expert): An instance of an Expert class for making decisions in the latent space.
            latent_space_dim (int): Dimensionality of the latent space.
            use_cuda (bool, optional): Whether to use GPU for computations. Defaults to False.
        """
        super(MultimodalVariationalAutoencoder, self).__init__()

        self.encoders = nn.ModuleDict(encoders)
        self.decoders = nn.ModuleDict(decoders)
        self._loss_weights = loss_weights
        self._expert = expert
        self._latent_space_dim = latent_space_dim

        # Train on GPU
        self.use_cuda = use_cuda
        if self.use_cuda:
            self.cuda()

    def loss_function(
            self,
            inputs: Dict[str, torch.Tensor],
            reconstructions: Dict[str, torch.Tensor],
            z_loc: torch.Tensor,
            z_scale: torch.Tensor,
            beta: float
    ) -> Dict[str, torch.Tensor]:
        """
        Computes the loss function for the variational autoencoder, including reconstruction loss and
        KL divergence:
            KL(N(\mu, \sigma), N(0, 1)) = \log \frac{1}{\sigma} + \frac{\sigma^2 + \mu^2}{2} - \frac{1}{2}

        Args:
            inputs (Dict[str, torch.Tensor]): A dictionary of input tensors for each modality.
            reconstructions (Dict[str, torch.Tensor]): A dictionary of reconstructed tensors for each modality.
            z_loc (torch.Tensor): Mean of the latent variable distribution.
            z_scale (torch.Tensor): Standard deviation of the latent variable distribution.
            beta (float): Weighting factor for the KL divergence term.

        Returns:
            Dict[str, torch.Tensor]: A dictionary containing:
                - 'total_loss': The total loss, including reconstruction loss and KL divergence.
                - 'reconstruction_loss': The weighted reconstruction loss.
                - 'kld_loss': The KL divergence loss.
        """

        reconstruction_loss = torch.Tensor([0.0]).cuda() if self.use_cuda else torch.Tensor([0.0])

        for modality in inputs.keys():
            if inputs[modality] is not None:
                modality_loss = torch_functional.mse_loss(reconstructions[modality], inputs[modality])
                if self.use_cuda:
                    modality_loss = modality_loss.cuda()
                reconstruction_loss += self._loss_weights[modality] * modality_loss

        # Calculate the KLD loss
        log_var = torch.log(torch.square(z_scale))
        kld_loss = torch.mean(-0.5 * torch.sum(1 + log_var - z_loc ** 2 - log_var.exp(), dim=1), dim=0)

        total_loss = reconstruction_loss + beta * kld_loss

        return {
            "total_loss": total_loss,
            "reconstruction_loss": reconstruction_loss,
            "kld_loss": kld_loss,
        }

    def _extract_batch_size_from_data(self, inputs: Dict[str, torch.Tensor]) -> int:
        """
        Extracts the batch size from the input data.

        Args:
            inputs (Dict[str, torch.Tensor]): A dictionary where keys are modality names and values are tensors of
                input data.

        Returns:
            int: The batch size extracted from the input tensors. Returns 0 if no valid tensors are found.
        """
        for modality in inputs.values():
            if modality is not None:
                return modality.shape[0]
        return 0

    def infer_latent(
            self,
            inputs: Dict[str, torch.Tensor]
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Infers the latent distribution parameters for each modality.

        Args:
            inputs (Dict[str, torch.Tensor]): A dictionary where keys are modality names and values are tensors of
                input data.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]: A tuple containing:
                - The latent means inferred by the expert.
                - The latent scales inferred by the expert.
                - The latent means for each modality.
                - The latent scales for each modality.
        """
        batch_size = self._extract_batch_size_from_data(inputs)

        z_loc, z_scale = (
            torch.zeros([1, batch_size, self._latent_space_dim]),
            torch.ones([1, batch_size, self._latent_space_dim])
        )
        if self.use_cuda:
            z_loc = z_loc.cuda()
            z_scale = z_scale.cuda()

        for modality, encoder in self.encoders.items():
            if inputs[modality] is not None:
                modality_z_loc, modality_z_scale = encoder(inputs[modality])
                z_loc = torch.cat((z_loc, modality_z_loc.unsqueeze(0)), dim=0)
                z_scale = torch.cat((z_scale, modality_z_scale.unsqueeze(0)), dim=0)

        # Give the inferred parameters to the expert to arrive at a unique decision
        z_loc_expert, z_scale_expert = self._expert(z_loc, z_scale)

        return z_loc_expert, z_scale_expert, z_loc, z_scale

    def sample_latent(self, z_loc: torch.Tensor, z_scale: torch.Tensor) -> torch.Tensor:
        """
        Samples from the latent space using the reparameterization trick.

        Args:
            z_loc (torch.Tensor): Mean of the latent distribution.
            z_scale (torch.Tensor): Standard deviation of the latent distribution.

        Returns:
            torch.Tensor: A sample from the latent distribution N(mu, var).
        """
        epsilon: torch.Tensor = torch.randn_like(z_loc)
        latent_sample: torch.Tensor = z_loc + epsilon * z_scale

        return latent_sample

    def generate(self, latent_sample: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Generates reconstructions for each modality from a latent sample.

        Args:
            latent_sample (torch.Tensor): A sample from the latent space.

        Returns:
            Dict[str, torch.Tensor]: A dictionary where keys are modality names and values are tensors of generated
                reconstructions.
        """
        reconstructions = dict()
        for modality, decoder in self.decoders.items():
            reconstructions[modality] = decoder(latent_sample)
        return reconstructions

    def forward(self, inputs) -> Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]:
        """
        Performs a forward pass through the model to infer latent variables, sample from the latent space, and
        generate reconstructions.

        Args:
            **inputs: A variable-length keyword argument list where keys are modality names and values are tensors
                of input data.

        Returns:
            Tuple[Dict[str, torch.Tensor], torch.Tensor, torch.Tensor]: A tuple containing:
                - A dictionary with reconstructed tensors for each modality.
                - The latent means inferred by the expert.
                - The latent scales inferred by the expert.
        """
        # Infer the latent distribution parameters
        z_loc_expert, z_scale_expert, _, _ = self.infer_latent(inputs)

        # Sample from the latent space
        latent_sample: torch.Tensor = self.sample_latent(
            z_loc=z_loc_expert,
            z_scale=z_scale_expert
        )

        # Reconstruct inputs based on that Gaussian sample
        reconstructions = self.generate(latent_sample)

        return reconstructions, z_loc_expert, z_scale_expert
