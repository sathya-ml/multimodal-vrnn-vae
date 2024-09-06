import matplotlib.pyplot as plt
import numpy as np
import torch

from typing import Tuple

from torch.utils.data import Dataset


def generate_spirals_dataset(
    num_samples: int, sequence_length: int, noise_level: float = 0.3
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Create a toy dataset with two two-dimensional modalities, where one depends on the other.
    Both modalities have temporal dependence and added noise.

    Args:
        num_samples (int): Number of samples in the dataset.
        sequence_length (int): Length of each sequence.
        noise_level (float): Level of noise to add to both modalities.

    Returns:
        Tuple[np.ndarray, np.ndarray]: Two arrays representing the two modalities.
    """
    # Generate the first modality: a spiral
    t = np.linspace(0, 4 * np.pi, sequence_length)
    modality1 = np.zeros((num_samples, sequence_length, 2))

    for i in range(num_samples):
        # Random initial conditions
        A = np.random.uniform(0.5, 1.5)

        # Spiral equation
        x = A * t * np.cos(t)
        y = A * t * np.sin(t)

        modality1[i, :, 0] = x
        modality1[i, :, 1] = y

    # Add noise to the first modality
    modality1 += np.random.normal(0, noise_level, modality1.shape)

    # Generate the second modality based on the first
    modality2 = np.zeros_like(modality1)
    modality2[:, :, 0] = modality1[:, :, 0] * 0.5 + modality1[:, :, 1] * 0.5
    modality2[:, :, 1] = -modality1[:, :, 0] * 0.5 + modality1[:, :, 1] * 0.5

    # Add noise to the second modality
    modality2 += np.random.normal(0, noise_level, modality2.shape)

    return modality1, modality2


def plot_spirals_dataset(modality1: np.ndarray, modality2: np.ndarray, sample_idx: int = 0):
    """
    Plot the toy dataset for a given sample.

    Args:
        modality1 (np.ndarray): First modality data.
        modality2 (np.ndarray): Second modality data.
        sample_idx (int): Index of the sample to plot.
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    # Plot first modality
    ax1.scatter(
        modality1[sample_idx, :, 0],
        modality1[sample_idx, :, 1],
        c=np.arange(modality1.shape[1]),
        cmap="viridis",
        s=10,
    )
    ax1.set_title("Modality 1: Noisy Spiral")
    ax1.set_xlabel("X")
    ax1.set_ylabel("Y")

    # Plot second modality
    ax2.scatter(
        modality2[sample_idx, :, 0],
        modality2[sample_idx, :, 1],
        c=np.arange(modality2.shape[1]),
        cmap="viridis",
        s=10,
    )
    ax2.set_title("Modality 2: Derived from Modality 1")
    ax2.set_xlabel("X")
    ax2.set_ylabel("Y")

    plt.tight_layout()
    plt.show()


class SpiralsDataset(torch.utils.data.Dataset):
    def __init__(self, modality1: np.ndarray, modality2: np.ndarray):
        self.modality1 = torch.FloatTensor(modality1)
        self.modality2 = torch.FloatTensor(modality2)
        self.num_samples = modality1.shape[0]

    def __len__(self):
        return self.num_samples

    def __getitem__(self, idx):
        return {
            "first_modality": self.modality1[idx],
            "second_modality": self.modality2[idx],
        }


def generate_quadratic_sinusoidal_dataset(num_samples=1000, noise_level=0.1):
    # Generate data for the first modality: 2D points
    x1 = np.linspace(-1, 1, num_samples)
    y1 = x1**2 + np.random.normal(0, noise_level, num_samples)
    modality1 = torch.tensor(np.column_stack((x1, y1)), dtype=torch.float32)

    # Generate data for the second modality: 2D points
    # Make it interdependent with the first modality
    x2 = y1 + np.random.normal(0, noise_level, num_samples)
    y2 = np.sin(2 * np.pi * x1) + 0.5 * y1 + np.random.normal(0, noise_level, num_samples)
    modality2 = torch.tensor(np.column_stack((x2, y2)), dtype=torch.float32)

    return modality1, modality2


def plot_quadratic_sinusoidal_dataset(mod1_data, mod2_data):
    plt.figure(figsize=(15, 5))

    plt.subplot(1, 3, 1)
    plt.scatter(mod1_data[:, 0], mod1_data[:, 1], alpha=0.5)
    plt.title("Modality 1: 2D Points")
    plt.xlabel("X1")
    plt.ylabel("Y1")

    plt.subplot(1, 3, 2)
    plt.scatter(mod2_data[:, 0], mod2_data[:, 1], alpha=0.5)
    plt.title("Modality 2: 2D Points")
    plt.xlabel("X2")
    plt.ylabel("Y2")

    plt.subplot(1, 3, 3)
    plt.scatter(mod1_data[:, 1], mod2_data[:, 0], alpha=0.5)
    plt.title("Interdependence: Y1 vs X2")
    plt.xlabel("Y1 (Modality 1)")
    plt.ylabel("X2 (Modality 2)")

    plt.tight_layout()
    plt.show()


class QuadraticSinusoidalDataset(Dataset):
    def __init__(self, mod1_data, mod2_data):
        self.mod1_data = mod1_data
        self.mod2_data = mod2_data

    def __len__(self):
        return len(self.mod1_data)

    def __getitem__(self, idx):
        return {"first_modality": self.mod1_data[idx], "second_modality": self.mod2_data[idx]}
