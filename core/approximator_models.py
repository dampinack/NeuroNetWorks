"""Universal Function Approximator (1D Regression) MLP with individual neuron basis decomposition."""

import torch
import torch.nn as nn
import numpy as np
from typing import List, Tuple, Dict, Callable

class SineActivation(nn.Module):
    """SIREN Sine activation function: sin(w0 * x)."""
    def __init__(self, w0: float = 3.0):
        super().__init__()
        self.w0 = w0

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return torch.sin(self.w0 * x)


class FunctionApproximatorMLP(nn.Module):
    """Arbitrary-depth MLP for 1D continuous function approximation."""

    def __init__(
        self,
        hidden_dim: int = 32,
        num_layers: int = 2,
        activation: str = "tanh"
    ):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_layers = num_layers
        self.activation_name = activation.lower()

        def make_act():
            if self.activation_name == "relu":
                return nn.ReLU()
            elif self.activation_name == "sigmoid":
                return nn.Sigmoid()
            elif self.activation_name == "siren":
                return SineActivation(3.0)
            elif self.activation_name == "gelu":
                return nn.GELU()
            else:
                return nn.Tanh()

        layers = []
        layers.append(nn.Linear(1, hidden_dim))
        layers.append(make_act())

        for _ in range(num_layers - 1):
            layers.append(nn.Linear(hidden_dim, hidden_dim))
            layers.append(make_act())

        self.hidden_net = nn.Sequential(*layers)
        self.out_linear = nn.Linear(hidden_dim, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.hidden_net(x)
        return self.out_linear(h)

    def get_basis_neuron_outputs(self, x_np: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Extracts individual basis curves from the final hidden layer feeding into output linear layer.

        f(x) = sum_i [ W_i * h_i(x) ] + b
        Returns:
            basis_curves: [H, len(x)] individual scaled neuron outputs W_i * h_i(x)
            total_prediction: [len(x)] sum of all neurons + bias
        """
        self.eval()
        with torch.no_grad():
            x_t = torch.from_numpy(x_np.astype(np.float32)).unsqueeze(-1)
            h = self.hidden_net(x_t)  # [N, H]
            W = self.out_linear.weight[0]  # [H]
            b = self.out_linear.bias.item()

            # Individual scaled contributions: [H, N]
            basis = (h * W.unsqueeze(0)).t().cpu().numpy()
            total = (self.out_linear(h)).squeeze(-1).cpu().numpy()

        return basis, total


def get_target_function(name: str) -> Callable[[np.ndarray], np.ndarray]:
    """Generates standard 1D continuous and discontinuous target benchmark curves."""
    name = name.lower()
    if "damped" in name:
        return lambda x: np.exp(-1.2 * (x + 1.0)) * np.sin(4.0 * np.pi * x)
    elif "runge" in name:
        return lambda x: 1.0 / (1.0 + 25.0 * (x ** 2))
    elif "harmonics" in name or "mixture" in name:
        return lambda x: 0.7 * np.sin(2.0 * np.pi * x) + 0.3 * np.cos(6.0 * np.pi * x)
    elif "step" in name or "square" in name:
        return lambda x: np.sign(np.sin(2.0 * np.pi * x)) * 0.75
    elif "sawtooth" in name:
        return lambda x: 2.0 * (x - np.floor(x + 0.5))
    else:  # Standard Sine
        return lambda x: np.sin(2.0 * np.pi * x)
