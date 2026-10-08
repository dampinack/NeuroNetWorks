"""Multi-Layer Perceptron and 2D Dataset generators for Decision Boundary Playground."""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict

class DecisionBoundaryMLP(nn.Module):
    """Configurable Multi-Layer Perceptron for non-linear 2D classification."""

    def __init__(
        self,
        hidden_layers: List[int] = [16, 16],
        activation: str = "relu"
    ):
        super().__init__()
        self.hidden_layers = hidden_layers
        self.activation_name = activation

        act_cls = {
            "relu": nn.ReLU,
            "tanh": nn.Tanh,
            "sigmoid": nn.Sigmoid,
            "gelu": nn.GELU,
            "leaky_relu": nn.LeakyReLU
        }.get(activation.lower(), nn.ReLU)

        layers = []
        prev_dim = 2
        for h_dim in hidden_layers:
            layers.append(nn.Linear(prev_dim, h_dim))
            layers.append(act_cls())
            prev_dim = h_dim

        # Final classification layer (2 classes)
        layers.append(nn.Linear(prev_dim, 2))
        self.net = nn.Sequential(*layers)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def predict_proba(self, x_np: np.ndarray) -> np.ndarray:
        """Evaluates class 1 probabilities for an array of 2D coordinates."""
        self.eval()
        with torch.no_grad():
            x_tensor = torch.from_numpy(x_np.astype(np.float32))
            logits = self(x_tensor)
            probs = torch.softmax(logits, dim=-1)[:, 1].numpy()
        return probs


def generate_2d_dataset(
    dataset_type: str = "spirals",
    n_samples: int = 400,
    noise: float = 0.1,
    random_state: int = 42
) -> Tuple[np.ndarray, np.ndarray]:
    """Generates 2D synthetic datasets for classification playground.

    Returns:
        X: [N, 2] float coordinates in approx [-2, 2]
        y: [N] int binary labels (0 or 1)
    """
    rng = np.random.RandomState(random_state)
    n_per_class = n_samples // 2

    if dataset_type == "spirals":
        # Dual Archimedean spirals
        theta = np.sqrt(rng.rand(n_per_class)) * 2.5 * np.pi
        r0 = 2.0 * theta / (2.5 * np.pi)
        x0 = r0 * np.cos(theta) + rng.randn(n_per_class) * noise * 0.4
        y0 = r0 * np.sin(theta) + rng.randn(n_per_class) * noise * 0.4

        x1 = -r0 * np.cos(theta) + rng.randn(n_per_class) * noise * 0.4
        y1 = -r0 * np.sin(theta) + rng.randn(n_per_class) * noise * 0.4

        X = np.vstack([np.column_stack([x0, y0]), np.column_stack([x1, y1])])
        y = np.array([0] * n_per_class + [1] * n_per_class)

    elif dataset_type == "moons":
        # Two interleaving half circles
        theta0 = np.linspace(0, np.pi, n_per_class)
        x0 = np.cos(theta0) + rng.randn(n_per_class) * noise * 0.3
        y0 = np.sin(theta0) + rng.randn(n_per_class) * noise * 0.3

        theta1 = np.linspace(0, np.pi, n_per_class)
        x1 = 1.0 - np.cos(theta1) + rng.randn(n_per_class) * noise * 0.3
        y1 = 0.5 - np.sin(theta1) - 0.2 + rng.randn(n_per_class) * noise * 0.3

        X = np.vstack([np.column_stack([x0, y0]), np.column_stack([x1, y1])])
        # Scale to [-1.5, 1.5]
        X = (X - np.mean(X, axis=0)) / (np.std(X, axis=0) + 1e-6) * 1.2
        y = np.array([0] * n_per_class + [1] * n_per_class)

    elif dataset_type == "circles":
        # Concentric circles
        r_inner = 0.6 + rng.randn(n_per_class) * noise * 0.3
        theta_inner = rng.rand(n_per_class) * 2 * np.pi
        x0 = r_inner * np.cos(theta_inner)
        y0 = r_inner * np.sin(theta_inner)

        r_outer = 1.4 + rng.randn(n_per_class) * noise * 0.3
        theta_outer = rng.rand(n_per_class) * 2 * np.pi
        x1 = r_outer * np.cos(theta_outer)
        y1 = r_outer * np.sin(theta_outer)

        X = np.vstack([np.column_stack([x0, y0]), np.column_stack([x1, y1])])
        y = np.array([0] * n_per_class + [1] * n_per_class)

    elif dataset_type == "xor":
        # 4-quadrant XOR
        pts = rng.uniform(-1.5, 1.5, size=(n_samples, 2))
        labels = ((pts[:, 0] > 0) ^ (pts[:, 1] > 0)).astype(int)
        # Add noise jitter
        pts += rng.randn(n_samples, 2) * noise * 0.2
        X, y = pts, labels

    else:  # Gaussian clusters
        c0 = rng.randn(n_per_class, 2) * (0.4 + noise * 0.5) + np.array([-0.8, -0.8])
        c1 = rng.randn(n_per_class, 2) * (0.4 + noise * 0.5) + np.array([0.8, 0.8])
        X = np.vstack([c0, c1])
        y = np.array([0] * n_per_class + [1] * n_per_class)

    # Shuffle
    idx = rng.permutation(len(X))
    return X[idx].astype(np.float32), y[idx].astype(np.int64)
