"""
Implementation of the Variational Recurrent
Neural Network (VRNN) from https://arxiv.org/abs/1506.02216
using unimodal isotropic gaussian distributions for
inference, prior, and generation.
"""
from typing import Any, List, Dict, Tuple, Optional

import torch
import torch.nn.functional as torch_functional
import torch.nn as nn


class MultimodalVariationalRecurrentNeuralNetwork(torch.nn.Module):
    """Multimodal Variational Recurrent Neural Network (VRNN) for processing multiple modalities.

    This class implements a VRNN that can handle multiple input modalities by encoding them
    into a shared latent space, followed by decoding for each modality. The model can be
    trained for tasks requiring joint representation learning from different types of data,
    such as text, images, and audio.

    Args:
        modality_input_modules (Dict[str, torch.nn.Module]):
            A dictionary where the keys are modality names (e.g., 'text', 'image', 'audio'),
            and the values are the corresponding input feature extraction modules for each modality.
        latent_feature_extraction_module (torch.nn.Module):
            The module responsible for extracting features from the latent representation.
        encoding_network (torch.nn.Module):
            The encoding network that encodes the fused features from all modalities into a latent space.
        decoding_networks (Dict[str, torch.nn.Module]):
            A dictionary where the keys are modality names and the values are the corresponding
            decoding networks for each modality.
        prior_network (torch.nn.Module):
            The network that calculates the prior distribution in the latent space.
        fusion_network (torch.nn.Module):
            The network responsible for fusing features from different modalities before encoding.
        rnn_module (torch.nn.Module):
            The recurrent neural network (RNN) module used for processing the sequence of fused features.
        rnn_num_layers (int):
            The number of layers in the RNN.
        rnn_hidden_dim (int):
            The hidden dimension size of the RNN.
        device (Any, optional):
            The device on which to run the model (e.g., 'cuda', 'cpu'). Default is 'cuda'.

    Attributes:
        modality_input_modules (Dict[str, torch.nn.Module]):
            Stores the input feature extraction modules for each modality.
        latent_feature_extraction_module (torch.nn.Module):
            Module to extract features from the latent representation.
        encoding_network (torch.nn.Module):
            Encodes fused features into the latent space.
        decoding_networks (Dict[str, torch.nn.Module]):
            Stores the decoding networks for each modality.
        prior_network (torch.nn.Module):
            Calculates the prior distribution.
        fusion_network (torch.nn.Module):
            Fuses features from different modalities.
        rnn (torch.nn.Module):
            The RNN module.
        rnn_num_layers (int):
            Number of layers in the RNN.
        rnn_hidden_dim (int):
            Hidden dimension size of the RNN.
        device (str):
            The device on which the model is running.
    """
    def __init__(
            self,
            modality_input_modules: Dict[str, torch.nn.Module],
            modality_input_module_output_dimensions: Dict[str, int],
            latent_feature_extraction_module: torch.nn.Module,
            encoding_network: torch.nn.Module,
            decoding_networks: Dict[str, torch.nn.Module],
            prior_network: torch.nn.Module,
            fusion_network: torch.nn.Module,
            rnn_module: torch.nn.Module,
            rnn_num_layers: int,
            rnn_hidden_dim: int,
            device: Any = "cuda"
    ):
        super(MultimodalVariationalRecurrentNeuralNetwork, self).__init__()

        self.modality_input_modules = nn.ModuleDict(modality_input_modules)
        self.modality_input_module_output_dimensions = modality_input_module_output_dimensions
        self.latent_feature_extraction_module = latent_feature_extraction_module
        self.encoding_network = encoding_network
        self.decoding_networks = nn.ModuleDict(decoding_networks)
        self.prior_network = prior_network
        self.fusion_network = fusion_network
        self.rnn = rnn_module
        self.rnn_num_layers = rnn_num_layers
        self.rnn_hidden_dim = rnn_hidden_dim
        self.device = device

        # Transfer to device
        self.to(device)

    @staticmethod
    def sample_latent(mean: torch.Tensor, std: torch.Tensor) -> torch.Tensor:
        """Samples a latent vector using the reparameterization trick.

        This method samples from a Gaussian distribution defined by the input `mean` and `std`
        (standard deviation) tensors. The reparameterization trick allows backpropagation through
        stochastic nodes by sampling from a standard normal distribution and shifting it by the
        `mean` and scaling by `std`.

        Args:
            mean (torch.Tensor):
                The mean tensor of the Gaussian distribution from which to sample.
                Should have the same shape as `std`.
            std (torch.Tensor):
                The standard deviation tensor of the Gaussian distribution from which to sample.
                Should have the same shape as `mean`.

        Returns:
            torch.Tensor:
                A sampled tensor from the Gaussian distribution defined by `mean` and `std`,
                having the same shape as the input tensors.
        """
        epsilon = torch.randn_like(mean)

        return mean + epsilon * std

    @staticmethod
    def kl_divergence_loss(mean_1, std_1, mean_2, std_2):
        """Computes the Kullback-Leibler Divergence (KLD) between two Gaussian distributions.

        This method calculates the KLD between two isotropic Gaussian distributions,
        defined by their respective means and standard deviations. The KLD measures how one
        probability distribution diverges from another.

        Args:
            mean_1 (torch.Tensor):
                The mean tensor of the first Gaussian distribution.
            std_1 (torch.Tensor):
                The standard deviation tensor of the first Gaussian distribution.
            mean_2 (torch.Tensor):
                The mean tensor of the second Gaussian distribution.
            std_2 (torch.Tensor):
                The standard deviation tensor of the second Gaussian distribution.

        Returns:
            torch.Tensor:
                The computed KLD value as a scalar tensor.
        """
        log_var_1 = 2 * torch.log(std_1)
        log_var_2 = 2 * torch.log(std_2)
        kld_element = (
                log_var_2 - log_var_1 + (torch.exp(log_var_1) + (mean_1 - mean_2).pow(2)) / torch.exp(log_var_2) - 1
        )
        
        return 0.5 * torch.sum(kld_element)

    @staticmethod
    def reconstruction_loss(predicted: torch.Tensor, observation: torch.Tensor) -> torch.Tensor:
        """Computes the reconstruction loss between the predicted and observed tensors.

        This method calculates the Mean Squared Error (MSE) loss between the predicted output
        and the ground truth observation. The reconstruction loss is commonly used to measure
        how well the model's output matches the actual data.

        Args:
            predicted (torch.Tensor):
                The predicted tensor generated by the model, with the same shape as `observation`.
            observation (torch.Tensor):
                The ground truth tensor that the model is attempting to reconstruct,
                with the same shape as `predicted`.

        Returns:
            torch.Tensor:
                The computed MSE loss as a scalar tensor.
        """
        return torch_functional.mse_loss(predicted, observation)

    @staticmethod
    def loss_function(
            observations: Dict[str, torch.Tensor],
            reconstruction_lists: Dict[str, List[torch.Tensor]],
            prior_mean_list: List[torch.Tensor],
            prior_std_list: List[torch.Tensor],
            encoder_mean_list: List[torch.Tensor],
            encoder_std_list: List[torch.Tensor],
    ):
        """Computes the total reconstruction and Kullback-Leibler divergence (KLD) losses.

        This method calculates the overall reconstruction loss across all modalities and time
        steps, as well as the KLD loss between the encoder and prior distributions.

        Args:
            observations (Dict[str, torch.Tensor]):
                A dictionary mapping modality names to their corresponding observed tensors.
                Each tensor is expected to have the shape (batch_size, timestep, input_dim).
            reconstruction_lists (Dict[str, List[torch.Tensor]]):
                A dictionary mapping modality names to lists of reconstructed tensors,
                where each list contains the reconstructed outputs for each time step.
            prior_mean_list (List[torch.Tensor]):
                A list of prior means for each time step, where each tensor has the shape
                (batch_size, latent_dim).
            prior_std_list (List[torch.Tensor]):
                A list of prior standard deviations for each time step, where each tensor
                has the shape (batch_size, latent_dim).
            encoder_mean_list (List[torch.Tensor]):
                A list of encoder means for each time step, where each tensor has the shape
                (batch_size, latent_dim).
            encoder_std_list (List[torch.Tensor]):
                A list of encoder standard deviations for each time step, where each tensor
                has the shape (batch_size, latent_dim).

        Returns:
            Tuple[torch.Tensor, torch.Tensor]:
                A tuple containing two scalar tensors: the total reconstruction loss and the
                total KLD loss.
        """
        reconstruction_loss = 0
        kld_loss = 0

        num_timesteps = len(prior_mean_list)

        for idx in range(num_timesteps):
            # Compute reconstruction loss for each modality for the current timestep
            for modality, observation in observations.items():
                if observation is not None:
                    observation = observation.permute(1, 0, 2)  # (timestep, batch_size, input_dim)
                    modality_reconstruction_loss = MultimodalVariationalRecurrentNeuralNetwork.reconstruction_loss(
                        predicted=reconstruction_lists[modality][idx], observation=observation[idx]
                    )
                    reconstruction_loss += modality_reconstruction_loss

            # Compute KLD loss for the current timestep
            kld_loss += MultimodalVariationalRecurrentNeuralNetwork.kl_divergence_loss(
                mean_1=encoder_mean_list[idx],
                std_1=encoder_std_list[idx],
                mean_2=prior_mean_list[idx],
                std_2=prior_std_list[idx]
            )

        return reconstruction_loss, kld_loss

    def encode(
            self,
            observations: Dict[str, Optional[torch.Tensor]]
    ) -> torch.Tensor:
        """
        Encodes a sequence of observations into a latent encoding tensor using a Recurrent Neural Network (RNN).

        This method processes input observations from multiple modalities, each of which may be represented
        as a tensor with dimensions corresponding to batch size and number of timesteps. The method first
        initializes the hidden state of the RNN and iterates over each timestep to compute features from the
        observations. These features are fused and passed through a series of networks to compute the latent
        encoding for each timestep. The final latent encoding tensor is returned as the output.

        Args:
            observations (Dict[str, Optional[torch.Tensor]]): A dictionary where keys are modality names (strings)
                and values are tensors of shape (batch_size, num_timesteps, ...) or None. Each tensor represents
                the observations for a given modality over time. If a modality's observations are None, the method
                handles this case by providing a default missing signal to the fusion network.

        Returns:
            torch.Tensor: A tensor of shape (batch_size, rnn_hidden_dim + latent_feature_dim) containing the latent
                encoding. This tensor combines the features extracted from the observations and the RNN hidden state.

        Raises:
            ValueError: If all values in the `observations` dictionary are None, indicating that no valid observations
                were provided.

        """
        # Get the batch size and num_timesteps from the first non-None observation
        for observation in observations.values():
            if observation is not None:
                batch_size = observation.shape[0]
                num_timesteps = observation.shape[1]
                break
        else:
            raise ValueError("All modalities cannot be None.")

        # Initialize RNN hidden state
        rnn_hidden_state = torch.zeros(
            self.rnn_num_layers, batch_size, self.rnn_hidden_dim, device=self.device
        )

        latent_encoding_tensor = torch.Tensor().to(self.device)

        for timestep in range(num_timesteps):
            features_list = list()

            # Extract features for each modality
            for modality, observation in observations.items():
                if observation is not None:
                    features = self.modality_input_modules[modality](observation[:, timestep])
                else:
                    features = self.modality_input_modules[modality].get_missing_signal(batch_size=batch_size)
                features_list.append(features)

            # Fuse features from all modalities
            fused_features = self.fusion_network(torch.cat(features_list, dim=1))

            # Compute encoding distribution parameters
            encoding_mean, encoding_std = self.encoding_network(
                torch.cat([fused_features, rnn_hidden_state[-1]], dim=1)
            )

            # Compute prior distribution parameters
            prior_mean, prior_std = self.prior_network(rnn_hidden_state[-1])

            # Sample from the encoding distribution and extract latent features
            latent_sample = self.sample_latent(encoding_mean, encoding_std)
            latent_features = self.latent_feature_extraction_module(latent_sample)

            # Combine latent features with RNN hidden state
            latent_encoding_tensor = torch.cat([latent_features, rnn_hidden_state[-1]], dim=1)

            # Update RNN hidden state
            _, rnn_hidden_state = self.rnn(
                torch.cat([fused_features, latent_features], 1).unsqueeze(0),
                rnn_hidden_state
            )

        return latent_encoding_tensor

    def forward(
            self,
            observations: Dict[str, Optional[torch.Tensor]]
    ) -> Tuple:
        """
        Performs a forward pass through the multimodal variational RNN, processing input observations to compute
        latent encodings and reconstruct the observations.

        This method takes in observations from various modalities, processes them through a series of networks
        and an RNN, and outputs lists of means and standard deviations for the prior and encoder distributions,
        as well as reconstructed observations for each modality.

        Args:
            observations (Dict[str, Optional[torch.Tensor]]): A dictionary where keys are modality names (strings)
                and values are tensors of shape (batch_size, num_timesteps, ...) or None. Each tensor represents
                the observations for a given modality over time. If a modality's observations are None, the method
                provides a default missing signal.

        Returns:
            Tuple: A tuple containing the following elements:
                - List of torch.Tensor: Prior means for each timestep, one for each modality.
                - List of torch.Tensor: Prior standard deviations for each timestep, one for each modality.
                - List of torch.Tensor: Encoder means for each timestep, one for each modality.
                - List of torch.Tensor: Encoder standard deviations for each timestep, one for each modality.
                - Dict[str, List[torch.Tensor]]: A dictionary where keys are modality names and values are lists
                  of tensors representing the reconstructed observations for each timestep.

        Raises:
            ValueError: If all values in the `observations` dictionary are None, indicating that no valid observations
                were provided.

        """
        # Get the batch size and num_timesteps from the first non-None observation
        for observation in observations.values():
            if observation is not None:
                batch_size = observation.shape[0]
                num_timesteps = observation.shape[1]
                break
        else:
            raise ValueError("All modalities cannot be None.")
        # Initialize RNN hidden state
        rnn_hidden_state = torch.zeros(
            self.rnn_num_layers, batch_size, self.rnn_hidden_dim, device=self.device
        )

        # Initialize lists to store prior and encoder distributions
        prior_mean_list: List[torch.Tensor] = list()
        prior_std_list: List[torch.Tensor] = list()
        encoder_mean_list: List[torch.Tensor] = list()
        encoder_std_list: List[torch.Tensor] = list()

        # Initialize dictionary to store reconstructions for each modality
        reconstruction_lists = {modality: list() for modality in observations.keys()}

        # Process each timestep
        for timestep in range(num_timesteps):
            features_list = list()

            # Extract features for each modality
            for modality, observation in observations.items():
                if observation is not None:
                    features = self.modality_input_modules[modality](observation[:, timestep])
                else:
                    # Determine the input size from the modality input module
                    input_size = self.modality_input_module_output_dimensions[modality]
                    # Generate a tensor of zeros with the appropriate size
                    features = torch.zeros(batch_size, input_size, device=self.device)

                features_list.append(features)

            # Fuse features from all modalities
            fused_features = self.fusion_network(torch.cat(features_list, dim=1))

            # Compute encoder distribution
            encoding_mean, encoding_std = self.encoding_network(
                torch.cat([fused_features, rnn_hidden_state[-1]], dim=1)
            )

            # Compute prior distribution
            prior_mean, prior_std = self.prior_network(rnn_hidden_state[-1])

            # Sample from the encoder distribution and extract latent features
            latent_sample = self.sample_latent(encoding_mean, encoding_std)
            latent_features = self.latent_feature_extraction_module(latent_sample)

            # Decode latent features for each modality
            for modality in observations.keys():
                decoded = self.decoding_networks[modality](
                    torch.cat([latent_features, rnn_hidden_state[-1]], dim=1)
                )
                reconstruction_lists[modality].append(decoded)

            # Update RNN hidden state
            _, rnn_hidden_state = self.rnn(
                torch.cat([fused_features, latent_features], 1).unsqueeze(0),
                rnn_hidden_state
            )

            # Store distributions for this timestep
            prior_mean_list.append(prior_mean)
            prior_std_list.append(prior_std)
            encoder_mean_list.append(encoding_mean)
            encoder_std_list.append(encoding_std)

        return (
            prior_mean_list,
            prior_std_list,
            encoder_mean_list,
            encoder_std_list,
            reconstruction_lists
        )
