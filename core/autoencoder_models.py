"""Autoencoder models for 2D latent space manifold exploration and image denoising."""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict

class MNIST2DAutoencoder(nn.Module):
    """Deep Autoencoder with a 2-dimensional bottleneck latent space (z1, z2).

    Enables direct 2D Cartesian plane visualization and continuous generative decoding.
    """

    def __init__(self):
        super().__init__()
        # Encoder: 784 -> 256 -> 64 -> 2
        self.encoder = nn.Sequential(
            nn.Linear(784, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 2)  # 2D Latent Bottleneck
        )

        # Decoder: 2 -> 64 -> 256 -> 784
        self.decoder = nn.Sequential(
            nn.Linear(2, 64),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
            nn.Linear(64, 256),
            nn.BatchNorm1d(256),
            nn.LeakyReLU(0.2),
            nn.Linear(256, 784),
            nn.Sigmoid()  # Pixel intensities in [0, 1]
        )

    def encode(self, x: torch.Tensor) -> torch.Tensor:
        """Projects image tensor [B, 1, 28, 28], [B, 28, 28] or [B, 784] to 2D latent coordinates."""
        if x.dim() > 2:
            x = x.reshape(x.size(0), -1)
        return self.encoder(x.float())

    def decode(self, z: torch.Tensor) -> torch.Tensor:
        """Generates synthetic 28x28 images from 2D latent coordinates."""
        out = self.decoder(z)
        return out.reshape(-1, 1, 28, 28)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Full reconstruction pass."""
        z = self.encode(x)
        return self.decode(z)


def add_noise_to_image(img_np: np.ndarray, noise_type: str = "gaussian", intensity: float = 0.3) -> np.ndarray:
    """Injects synthetic corruption into a 28x28 image for denoising experiments."""
    noisy = img_np.copy()

    if noise_type == "gaussian":
        noise = np.random.normal(0.0, intensity, size=noisy.shape)
        noisy = np.clip(noisy + noise, 0.0, 1.0)

    elif noise_type == "salt_pepper":
        prob = intensity * 0.5
        # Salt (1.0)
        salt_mask = np.random.rand(*noisy.shape) < prob
        noisy[salt_mask] = 1.0
        # Pepper (0.0)
        pepper_mask = np.random.rand(*noisy.shape) < prob
        noisy[pepper_mask] = 0.0

    elif noise_type == "occlusion":
        # Block out a rectangular patch
        box_size = int(round(14 * intensity)) + 4
        h, w = noisy.shape
        top = np.random.randint(0, max(1, h - box_size))
        left = np.random.randint(0, max(1, w - box_size))
        noisy[top : top + box_size, left : left + box_size] = 0.0

    return noisy


def compute_reconstruction_metrics(original: np.ndarray, reconstructed: np.ndarray) -> Dict[str, float]:
    """Computes Mean Squared Error (MSE) and Peak Signal-to-Noise Ratio (PSNR)."""
    mse = float(np.mean((original - reconstructed) ** 2))
    if mse < 1e-10:
        psnr = 99.0
    else:
        psnr = float(10.0 * np.log10(1.0 / mse))
    return {"mse": mse, "psnr": psnr}
