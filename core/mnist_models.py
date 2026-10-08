"""MNIST Neural Network Models with intermediate activation inspection and saliency map explainability."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional

class MNISTCNN(nn.Module):
    """Convolutional Neural Network for MNIST digit recognition with full activation telemetry."""

    def __init__(self):
        super().__init__()
        # Layer 1: 1 -> 16 channels, 3x3 kernel with padding
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Layer 2: 16 -> 32 channels, 3x3 kernel with padding
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)

        # Fully connected layers
        self.fc1 = nn.Linear(32 * 7 * 7, 128)
        self.relu3 = nn.ReLU()
        self.dropout = nn.Dropout(0.2)
        self.fc2 = nn.Linear(128, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.pool1(self.relu1(self.conv1(x)))
        x = self.pool2(self.relu2(self.conv2(x)))
        x = x.reshape(x.size(0), -1)
        x = self.dropout(self.relu3(self.fc1(x)))
        x = self.fc2(x)
        return x

    def forward_with_activations(self, x: torch.Tensor) -> Dict[str, np.ndarray]:
        """Performs forward pass and captures layer-by-layer internal activations for live inspection."""
        self.eval()
        with torch.no_grad():
            c1 = self.conv1(x)
            r1 = self.relu1(c1)
            p1 = self.pool1(r1)

            c2 = self.conv2(p1)
            r2 = self.relu2(c2)
            p2 = self.pool2(r2)

            flat = p2.reshape(p2.size(0), -1)
            f1 = self.fc1(flat)
            r3 = self.relu3(f1)
            f2 = self.fc2(r3)
            probs = F.softmax(f2, dim=-1)

        return {
            "input": x[0, 0].detach().cpu().numpy(),
            "conv1_raw": c1[0].detach().cpu().numpy(),
            "conv1_act": r1[0].detach().cpu().numpy(),
            "pool1": p1[0].detach().cpu().numpy(),
            "conv2_raw": c2[0].detach().cpu().numpy(),
            "conv2_act": r2[0].detach().cpu().numpy(),
            "pool2": p2[0].detach().cpu().numpy(),
            "fc1": r3[0].detach().cpu().numpy(),
            "logits": f2[0].detach().cpu().numpy(),
            "probabilities": probs[0].detach().cpu().numpy(),
        }

    def compute_saliency(self, x: torch.Tensor, target_class: Optional[int] = None) -> np.ndarray:
        """Computes gradient-based saliency (Vanilla Backprop pixel attribution)

        Reveals which input pixels most strongly influenced the network's prediction.
        """
        self.eval()
        x_clone = x.clone().detach().requires_grad_(True)
        logits = self(x_clone)

        if target_class is None:
            target_class = int(logits.argmax(dim=-1)[0].item())

        score = logits[0, target_class]
        score.backward()

        # Magnitude of gradient across pixels
        saliency = x_clone.grad.data[0, 0].abs().cpu().numpy()

        # Min-max normalization for crisp visualization
        s_min = saliency.min()
        s_max = saliency.max()
        if s_max - s_min > 1e-8:
            saliency = (saliency - s_min) / (s_max - s_min)
        else:
            saliency = np.zeros_like(saliency)

        return saliency


class MNISTMLP(nn.Module):
    """Dense Multi-Layer Perceptron for MNIST comparison."""

    def __init__(self, hidden_dim: int = 128):
        super().__init__()
        self.fc1 = nn.Linear(784, hidden_dim)
        self.relu1 = nn.ReLU()
        self.fc2 = nn.Linear(hidden_dim, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, 10)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = x.reshape(x.size(0), -1)
        x = self.relu1(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x

    def forward_with_activations(self, x: torch.Tensor) -> Dict[str, np.ndarray]:
        self.eval()
        with torch.no_grad():
            x_flat = x.reshape(x.size(0), -1)
            h1 = self.relu1(self.fc1(x_flat))
            h2 = self.relu2(self.fc2(h1))
            logits = self.fc3(h2)
            probs = F.softmax(logits, dim=-1)

        return {
            "input": x[0, 0].detach().cpu().numpy(),
            "fc1": h1[0].detach().cpu().numpy(),
            "fc2": h2[0].detach().cpu().numpy(),
            "logits": logits[0].detach().cpu().numpy(),
            "probabilities": probs[0].detach().cpu().numpy(),
        }
