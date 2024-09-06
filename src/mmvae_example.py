import itertools
from typing import Dict, Tuple, List, Generator

import numpy
import torch
import tqdm
from torch.utils.data import Dataset, DataLoader
from loguru import logger

from model import multimodal_vae, expert
import component

import torch
import matplotlib.pyplot as plt

from dataset import (
    QuadraticSinusoidalDataset,
    generate_quadratic_sinusoidal_dataset,
    plot_quadratic_sinusoidal_dataset,
)


def build_model(
    input_dims: dict,
    latent_space_dim: int,
    hidden_dim: int,
    loss_weights: dict,
    use_cuda: bool,
) -> torch.nn.Module:
    # Build the first modality components
    first_modality_encoder: torch.nn.Module = component.MultilayerPerceptronEncoder(
        input_dim=input_dims["first_modality"],
        hidden_dim=hidden_dim,
        z_dim=latent_space_dim,
    )
    first_modality_decoder: torch.nn.Module = component.MultilayerPerceptronDecoder(
        input_dim=input_dims["first_modality"],
        hidden_dim=hidden_dim,
        z_dim=latent_space_dim,
    )

    # Build the second modality components
    second_modality_encoder: torch.nn.Module = component.MultilayerPerceptronEncoder(
        input_dim=input_dims["second_modality"],
        hidden_dim=hidden_dim,
        z_dim=latent_space_dim,
    )
    second_modality_decoder: torch.nn.Module = component.MultilayerPerceptronDecoder(
        input_dim=input_dims["second_modality"],
        hidden_dim=hidden_dim,
        z_dim=latent_space_dim,
    )

    # Create the expert
    expert_instance = expert.ProductOfExperts()

    # Build the model
    exteroceptive_mmvae: torch.nn.Module = (
        multimodal_vae.MultimodalVariationalAutoencoder(
            encoders={
                "first_modality": first_modality_encoder,
                "second_modality": second_modality_encoder,
            },
            decoders={
                "first_modality": first_modality_decoder,
                "second_modality": second_modality_decoder,
            },
            loss_weights=loss_weights,
            expert=expert_instance,
            latent_space_dim=latent_space_dim,
            use_cuda=use_cuda,
        )
    )

    return exteroceptive_mmvae


def eval_model_training(
    model: torch.nn.Module,
    optimizer: torch.optim.Optimizer,
    beta: float,
    data: Dict[str, torch.Tensor],
) -> dict:
    # Zero the parameter gradients
    optimizer.zero_grad()

    (reconstructions, z_loc_expert, z_scale_expert) = model(data)

    loss = model.loss_function(
        inputs=data,
        reconstructions=reconstructions,
        z_loc=z_loc_expert,
        z_scale=z_scale_expert,
        beta=beta,
    )

    loss["total_loss"].backward()
    optimizer.step()

    return loss


def train(
    mmvae_model: torch.nn.Module,
    dataset: Dataset,
    learning_rate: float,
    optim_betas: Tuple[float, float],
    kl_weight: float,
    modality_dropout_prob: float,
    num_epochs: int,
    batch_size: int,
    seed: int,
    use_cuda: bool,
) -> None:
    logger.info("Currently using Torch version: " + torch.__version__)
    torch.manual_seed(seed=seed)

    # Debug: Print the number of parameters
    total_params = sum(p.numel() for p in mmvae_model.parameters() if p.requires_grad)
    logger.info(f"Total trainable parameters: {total_params}")

    # Setup the optimizer
    adam_args = {"lr": learning_rate, "betas": optim_betas}
    optimizer = torch.optim.Adam(params=mmvae_model.parameters(), **adam_args)

    # Create data loader
    data_loader = DataLoader(dataset=dataset, batch_size=batch_size, shuffle=True)

    training_losses: List[List[float]] = list()

    # Training loop
    for epoch_num in range(num_epochs):
        # Initialize loss accumulator and the progress bar
        epoch_losses: List[List[float]] = list()
        progress_bar = tqdm.tqdm(data_loader)

        logger.info(f"Starting epoch {epoch_num + 1}.")

        # Do a training epoch over each mini-batch returned
        # by the data loader
        for data in progress_bar:
            if len(epoch_losses) > 0:
                batch_loss_mean = numpy.nanmean(epoch_losses[-1])
                progress_bar.set_description(
                    f"Epoch {epoch_num + 1:3}; Batch loss: {batch_loss_mean:3.5}"
                )
            else:
                progress_bar.set_description(
                    f"Epoch {epoch_num + 1:3}; Batch loss: Nan"
                )

            # If on GPU put the mini-batch into CUDA memory
            if use_cuda:
                for key, value in data.items():
                    data[key] = value.cuda()

            # Implement modality dropout
            modalities = list(data.keys())
            
            if numpy.random.random() < modality_dropout_prob and len(modalities) > 1:
                # Sample the number of modalities to drop
                num_modalities_to_drop = numpy.random.randint(1, len(modalities))
                # Randomly choose modalities to drop
                dropped_modalities = numpy.random.choice(modalities, size=num_modalities_to_drop, replace=False)
                data_with_dropout = {k: v if k not in dropped_modalities else None for k, v in data.items()}
            else:
                data_with_dropout = data

            losses = eval_model_training(
                model=mmvae_model,
                optimizer=optimizer,
                beta=kl_weight,
                data=data_with_dropout,
            )
            epoch_losses.append(
                float(losses["total_loss"].cpu().detach().numpy().mean())
            )

        training_losses.extend(epoch_losses)

    return training_losses


def main() -> None:
    use_cuda = torch.cuda.is_available()

    # Generate a toy dataset and plot it
    mod1_data, mod2_data = generate_quadratic_sinusoidal_dataset(
        num_samples=5000, noise_level=0.1
    )
    plot_quadratic_sinusoidal_dataset(mod1_data, mod2_data)

    training_dataset = QuadraticSinusoidalDataset(mod1_data, mod2_data)

    # Build the model
    input_dims: dict = {"first_modality": 2, "second_modality": 2}
    loss_weights: dict = {"first_modality": 1.0, "second_modality": 1.0}
    model: torch.nn.Module = build_model(
        input_dims=input_dims,
        latent_space_dim=5,
        hidden_dim=10,
        loss_weights=loss_weights,
        use_cuda=use_cuda,
    )

    # Train the model
    training_losses = train(
        mmvae_model=model,
        dataset=training_dataset,
        learning_rate=0.001,
        optim_betas=(0.9, 0.999),
        kl_weight=0.1,
        modality_dropout_prob=0.5,
        num_epochs=400,
        batch_size=100,
        seed=42,
        use_cuda=use_cuda,
    )

    # Plot the training losses
    plt.figure(figsize=(10, 6))
    plt.plot(range(1, len(training_losses) + 1), training_losses, marker="o")
    plt.title("Training Losses Over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)
    plt.show()

    # Test conditional generation on a new sample
    new_mod1_data, new_mod2_data = generate_quadratic_sinusoidal_dataset(num_samples=1)

    # Prepare the input for the model
    new_sample = {"first_modality": new_mod1_data, "second_modality": new_mod2_data}

    # If the model is on GPU, move the new sample there
    if use_cuda:
        new_sample = {k: v.cuda() for k, v in new_sample.items()}

    # Generate reconstructions
    with torch.no_grad():
        reconstructions, _, _ = model(new_sample)

    # Extract the generated data
    generated_mod1 = reconstructions["first_modality"].cpu().numpy()
    # generated_mod2 = reconstructions["second_modality"].cpu().numpy()

    # Generate second modality conditioned on the first
    with torch.no_grad():
        data_without_second_modality = {
            "first_modality": new_sample["first_modality"],
            "second_modality": None,
        }
        z_loc_expert, z_scale_expert, _, _ = model.infer_latent(
            data_without_second_modality
        )
        latent_sample = model.sample_latent(z_loc_expert, z_scale_expert)
        conditional_mod2 = (
            model.generate(latent_sample)["second_modality"].cpu().numpy()
        )

    # Plot the results
    fig, axs = plt.subplots(2, 2, figsize=(12, 12))
    fig.suptitle("Conditional Generation Results")

    # Plot the entire dataset with small dots
    axs[0, 0].scatter(mod1_data[:, 0], mod1_data[:, 1], c="lightgray", s=1, alpha=0.5)
    axs[0, 0].scatter(
        new_mod1_data[:, 0], new_mod1_data[:, 1], c="blue", s=50, label="Original"
    )
    axs[0, 0].set_title("Original First Modality")
    axs[0, 0].legend()

    axs[0, 1].scatter(mod2_data[:, 0], mod2_data[:, 1], c="lightgray", s=1, alpha=0.5)
    axs[0, 1].scatter(
        new_mod2_data[:, 0], new_mod2_data[:, 1], c="blue", s=50, label="Original"
    )
    axs[0, 1].set_title("Original Second Modality")
    axs[0, 1].legend()

    axs[1, 0].scatter(mod1_data[:, 0], mod1_data[:, 1], c="lightgray", s=1, alpha=0.5)
    axs[1, 0].scatter(
        generated_mod1[:, 0], generated_mod1[:, 1], c="red", s=50, label="Generated"
    )
    axs[1, 0].set_title("Reconstructed First Modality")
    axs[1, 0].legend()

    axs[1, 1].scatter(mod2_data[:, 0], mod2_data[:, 1], c="lightgray", s=1, alpha=0.5)
    axs[1, 1].scatter(
        conditional_mod2[:, 0],
        conditional_mod2[:, 1],
        c="green",
        s=50,
        label="Conditional",
    )
    axs[1, 1].set_title("Second Modality (Conditioned on First)")
    axs[1, 1].legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
