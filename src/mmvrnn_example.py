from itertools import combinations
from typing import Tuple, List

import numpy
import torch
import tqdm
from torch.utils.data import Dataset, DataLoader
from loguru import logger

from model import multimodal_vrnn
import component

import matplotlib.pyplot as plt

from dataset import SpiralsDataset, generate_spirals_dataset, plot_spirals_dataset


def build_model(
    input_dims: dict,
    latent_space_dim: int,
    va_mlp_hidden_dim: int,
    feature_space_dim: int,
    rnn_hidden_dim: int,
    rnn_num_layers: int,
    rnn_bias: bool,
    use_cuda: bool,
) -> torch.nn.Module:
    if use_cuda:
        device: str = "cuda"
    else:
        device: str = "cpu"

    first_modality_feature_extractor: torch.nn.Module = (
        component.MultilayerPerceptron(
            input_dim=input_dims["first_modality"],
            hidden_dim=va_mlp_hidden_dim,
            output_dim=feature_space_dim,
            device=device,
        )
    )
    second_modality_feature_extractor: torch.nn.Module = (
        component.MultilayerPerceptron(
            input_dim=input_dims["second_modality"],
            hidden_dim=va_mlp_hidden_dim,
            output_dim=feature_space_dim,
            device=device,
        )
    )

    first_modality_decoder: torch.nn.Module = component.MultilayerPerceptronDecoder(
        input_dim=input_dims["first_modality"],
        hidden_dim=va_mlp_hidden_dim,
        z_dim=rnn_hidden_dim * 2,
    )
    second_modality_decoder: torch.nn.Module = component.MultilayerPerceptronDecoder(
        input_dim=input_dims["second_modality"],
        hidden_dim=va_mlp_hidden_dim,
        z_dim=rnn_hidden_dim * 2,
    )

    fusion_network: torch.nn.Module = component.MultilayerPerceptron(
        input_dim=feature_space_dim * 2,  # We have two modalities
        hidden_dim=va_mlp_hidden_dim,
        output_dim=rnn_hidden_dim,
        device=device,
    )
    encoding_network: torch.nn.Module = component.MultilayerPerceptronEncoder(
        input_dim=feature_space_dim + rnn_hidden_dim,
        hidden_dim=va_mlp_hidden_dim,
        z_dim=latent_space_dim,
    )
    prior_network: torch.nn.Module = component.MultilayerPerceptronEncoder(
        input_dim=rnn_hidden_dim, hidden_dim=va_mlp_hidden_dim, z_dim=latent_space_dim
    )
    latent_feature_extraction_network: torch.nn.Module = (
        component.MultilayerPerceptron(
            input_dim=latent_space_dim,
            hidden_dim=va_mlp_hidden_dim,
            output_dim=rnn_hidden_dim,
            device=device,
        )
    )

    # Recurrence
    rnn_network: torch.nn.Module = torch.nn.GRU(
        # We multiply by 2 as we take both the input features and latent features
        # for the recurrence relation
        input_size=feature_space_dim * 2,
        hidden_size=rnn_hidden_dim,
        num_layers=rnn_num_layers,
        bias=rnn_bias,
    )

    multimodal_variational_rnn: torch.nn.Module = (
        multimodal_vrnn.MultimodalVariationalRecurrentNeuralNetwork(
            modality_input_modules={
                "first_modality": first_modality_feature_extractor,
                "second_modality": second_modality_feature_extractor,
            },
            modality_input_module_output_dimensions={
                "first_modality": feature_space_dim,
                "second_modality": feature_space_dim,
            },
            latent_feature_extraction_module=latent_feature_extraction_network,
            encoding_network=encoding_network,
            decoding_networks={
                "first_modality": first_modality_decoder,
                "second_modality": second_modality_decoder,
            },
            prior_network=prior_network,
            fusion_network=fusion_network,
            rnn_module=rnn_network,
            rnn_num_layers=rnn_num_layers,
            rnn_hidden_dim=rnn_hidden_dim,
            device=device,
        )
    )

    return multimodal_variational_rnn


def eval_model_training(model, data, optimizer, beta) -> torch.Tensor:
    optimizer.zero_grad()

    (
        prior_mean_list,
        prior_std_list,
        encoder_mean_list,
        encoder_std_list,
        reconstruction_lists,
    ) = model(data)

    reconstruction_loss, kld_loss = model.loss_function(
        observations=data,
        reconstruction_lists=reconstruction_lists,
        prior_mean_list=prior_mean_list,
        prior_std_list=prior_std_list,
        encoder_mean_list=encoder_mean_list,
        encoder_std_list=encoder_std_list,
    )
    total_loss = reconstruction_loss + beta * kld_loss

    total_loss.backward()
    optimizer.step()

    return total_loss


def train(
    mmvae_model: multimodal_vrnn.MultimodalVariationalRecurrentNeuralNetwork,
    dataset: Dataset,
    learning_rate: float,
    optim_betas: Tuple[float, float],
    num_epochs: int,
    batch_size: int,
    kl_weight: float,
    modality_dropout_prob: float,
    seed: int,
    use_cuda: bool,
) -> None:
    logger.info("Currently using Torch version: " + torch.__version__)
    torch.manual_seed(seed=seed)

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

        # Do a training epoch over each mini-batch returned
        #   by the data loader
        for data_batch in progress_bar:
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
                for modality in data_batch.keys():
                    data_batch[modality] = data_batch[modality].cuda()

            # Implement modality dropout
            modalities = list(data_batch.keys())
            
            if numpy.random.random() < modality_dropout_prob and len(modalities) > 1:
                # Sample the number of modalities to drop
                num_modalities_to_drop = numpy.random.randint(1, len(modalities))
                # Randomly choose modalities to drop
                dropped_modalities = numpy.random.choice(modalities, size=num_modalities_to_drop, replace=False)
                data_with_dropout = {k: v if k not in dropped_modalities else None for k, v in data_batch.items()}
            else:
                data_with_dropout = data_batch

            losses = eval_model_training(
                model=mmvae_model,
                optimizer=optimizer,
                beta=kl_weight,
                data=data_with_dropout,
            )
            epoch_losses.append(
                float(losses.cpu().detach().numpy().mean())
            )

        training_losses.append(epoch_losses)



        epoch_losses: numpy.ndarray = numpy.array(epoch_losses)
        training_losses.extend(epoch_losses)

    return training_losses


def main() -> None:
    # Create the toy dataset
    modality1, modality2 = generate_spirals_dataset(
        num_samples=1000, sequence_length=50, noise_level=0.25
    )

    plot_spirals_dataset(modality1, modality2, sample_idx=10)

    training_dataset = SpiralsDataset(modality1, modality2)

    use_cuda = torch.cuda.is_available()

    # Define the model architecture
    input_dims: dict = {"first_modality": 2, "second_modality": 2}
    latent_space_dim: int = 5
    va_mlp_hidden_dim: int = 10
    feature_space_dim: int = 5
    rnn_hidden_dim: int = 5
    rnn_num_layers: int = 2
    rnn_bias: bool = True

    # Define the training parameters
    learning_rate: float = 0.001
    optim_betas: Tuple[float, float] = (0.9, 0.999)
    kl_weight: float = 0.01
    modality_dropout_prob: float = 0.5
    num_epochs: int = 60
    batch_size: int = 64
    seed: int = 42

    model: torch.nn.Module = build_model(
        input_dims=input_dims,
        latent_space_dim=latent_space_dim,
        va_mlp_hidden_dim=va_mlp_hidden_dim,
        feature_space_dim=feature_space_dim,
        rnn_hidden_dim=rnn_hidden_dim,
        rnn_num_layers=rnn_num_layers,
        rnn_bias=rnn_bias,
        use_cuda=use_cuda,
    )

    training_losses = train(
        mmvae_model=model,
        dataset=training_dataset,
        learning_rate=learning_rate,
        optim_betas=optim_betas,
        num_epochs=num_epochs,
        batch_size=batch_size,
        kl_weight=kl_weight,
        modality_dropout_prob=modality_dropout_prob,
        seed=seed,
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

    # Test conditional generation
    # Generate a new sample
    new_mod1_data, new_mod2_data = generate_spirals_dataset(
        num_samples=1, sequence_length=50, noise_level=0.25
    )
    new_sample = {"first_modality": new_mod1_data, "second_modality": new_mod2_data}

    # If the model is on GPU, move the new sample there
    if use_cuda:
        new_sample = {k: torch.Tensor(v).cuda() for k, v in new_sample.items()}

    # Generate reconstructions
    with torch.no_grad():
        _, _, _, _, reconstructions = model(new_sample)

    # As we get back a list over time steps, we stack the list of tensors into a single tensor 
    # over the time dimension, and then move the result to the CPU and convert to a numpy array.
    reconstructed_mod1 = (
        torch.stack(reconstructions["first_modality"], dim=1).cpu().numpy()
    )

    # Generate the second modality by conditioning on the first
    model.eval()  # Set the model to evaluation mode

    # Prepare input for generation
    input_data = {
        "first_modality": new_sample["first_modality"],
        "second_modality": None,  # We'll generate this
    }

    # Generate the second modality
    with torch.no_grad():
        _, _, _, _, generated_output = model(input_data)

    # Extract the generated second modality data
    generated_mod2 = (
        torch.stack(generated_output["second_modality"], dim=1).cpu().numpy()
    )

    # Plot the results
    plt.figure(figsize=(15, 5))

    # Plot original data
    plt.subplot(1, 3, 1)
    plt.scatter(
        new_mod1_data[0, :, 0],
        new_mod1_data[0, :, 1],
        s=1,
        alpha=0.7,
        label="First Modality",
    )
    plt.scatter(
        new_mod2_data[0, :, 0],
        new_mod2_data[0, :, 1],
        s=1,
        alpha=0.7,
        label="Second Modality",
    )
    plt.title("Original Data")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()

    # Plot reconstructed first modality
    plt.subplot(1, 3, 2)
    plt.scatter(
        new_mod1_data[0, :, 0], new_mod1_data[0, :, 1], s=1, alpha=0.7, label="Original"
    )
    plt.scatter(
        reconstructed_mod1[0, :, 0],
        reconstructed_mod1[0, :, 1],
        s=3,
        alpha=0.7,
        label="Reconstructed",
    )
    plt.title("Reconstructed First Modality")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()

    # Plot generated second modality
    plt.subplot(1, 3, 3)
    plt.scatter(
        new_mod2_data[0, :, 0], new_mod2_data[0, :, 1], s=1, alpha=0.7, label="Original"
    )
    plt.scatter(
        generated_mod2[0, :, 0],
        generated_mod2[0, :, 1],
        s=3,
        alpha=0.7,
        label="Generated",
    )
    plt.title("Generated Second Modality")
    plt.xlabel("X")
    plt.ylabel("Y")
    plt.legend()

    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    main()
