"""2D Toy Diffusion Model & Flow Matching engine with interactive particle trajectories."""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, Tuple, List, Optional

class SinusoidalPositionalEmbedding(nn.Module):
    """Sinusoidal time embedding module."""
    def __init__(self, dim: int = 16):
        super().__init__()
        self.dim = dim

    def forward(self, t: torch.Tensor) -> torch.Tensor:
        # t is [B, 1] in range [0, 1]
        half_dim = self.dim // 2
        emb = np.log(10000) / (half_dim - 1)
        emb = torch.exp(torch.arange(half_dim, dtype=torch.float32, device=t.device) * -emb)
        emb = t * emb[None, :]
        return torch.cat((emb.sin(), emb.cos()), dim=-1)


class FlowFieldMLP(nn.Module):
    """Small MLP predicting 2D velocity vector field v(x, t) for Flow Matching / Diffusion."""

    def __init__(self, hidden_dim: int = 64, time_dim: int = 16):
        super().__init__()
        self.time_mlp = SinusoidalPositionalEmbedding(dim=time_dim)
        self.net = nn.Sequential(
            nn.Linear(2 + time_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.SiLU(),
            nn.Linear(hidden_dim, 2)
        )

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        # x: [B, 2], t: [B, 1]
        t_emb = self.time_mlp(t)
        feat = torch.cat([x, t_emb], dim=-1)
        return self.net(feat)


def generate_2d_shape_points(shape_name: str, n_points: int = 400) -> np.ndarray:
    """Generates 2D benchmark point manifolds scaled roughly in [-1.8, 1.8]."""
    shape_name = shape_name.lower()
    rng = np.random.RandomState(42)

    if "heart" in shape_name:
        theta = rng.uniform(0, 2 * np.pi, n_points)
        x = 16 * (np.sin(theta) ** 3)
        y = 13 * np.cos(theta) - 5 * np.cos(2 * theta) - 2 * np.cos(3 * theta) - np.cos(4 * theta)
        pts = np.column_stack([x / 11.0, y / 11.0])
        pts += rng.randn(n_points, 2) * 0.04

    elif "swiss" in shape_name or "roll" in shape_name:
        theta = 1.5 * np.pi * (1 + 2 * rng.rand(n_points))
        x = 0.8 * theta * np.cos(theta) / 10.0
        y = 0.8 * theta * np.sin(theta) / 10.0
        pts = np.column_stack([x, y]) * 1.5
        pts += rng.randn(n_points, 2) * 0.03

    elif "rings" in shape_name or "circles" in shape_name:
        n_half = n_points // 2
        th1 = rng.uniform(0, 2 * np.pi, n_half)
        r1 = 0.6 + rng.randn(n_half) * 0.05
        th2 = rng.uniform(0, 2 * np.pi, n_points - n_half)
        r2 = 1.4 + rng.randn(n_points - n_half) * 0.05
        x = np.concatenate([r1 * np.cos(th1), r2 * np.cos(th2)])
        y = np.concatenate([r1 * np.sin(th1), r2 * np.sin(th2)])
        pts = np.column_stack([x, y])

    else:  # Two Moons default
        n_half = n_points // 2
        th1 = np.linspace(0, np.pi, n_half) + rng.randn(n_half) * 0.05
        x1 = np.cos(th1)
        y1 = np.sin(th1)
        th2 = np.linspace(0, np.pi, n_points - n_half) + rng.randn(n_points - n_half) * 0.05
        x2 = 1.0 - np.cos(th2)
        y2 = 0.5 - np.sin(th2) - 0.2
        pts = np.vstack([np.column_stack([x1, y1]), np.column_stack([x2, y2])])
        pts = (pts - np.mean(pts, axis=0)) * 1.4

    return pts.astype(np.float32)


class DiffusionFlowSimulator:
    """Manages the Flow Matching / Diffusion training loop and ODE particle integrator."""

    def __init__(self, shape_name: str = "heart", device: torch.device = torch.device("cpu")):
        self.device = device
        self.shape_name = shape_name
        self.target_pts = generate_2d_shape_points(shape_name, n_points=500)
        self.model = FlowFieldMLP().to(device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.005)
        self._pretrain_quick(epochs=120)

    def _pretrain_quick(self, epochs: int = 120):
        """Pretrains flow matching velocity field so particle self-assembly is immediately visible."""
        self.model.train()
        x1_all = torch.from_numpy(self.target_pts).to(self.device)
        n = len(x1_all)

        for _ in range(epochs):
            # Target data x1
            idx = torch.randint(0, n, (128,))
            x1 = x1_all[idx]
            # Standard Gaussian noise source x0
            x0 = torch.randn_like(x1)
            # Continuous time t ~ Uniform[0, 1]
            t = torch.rand((128, 1), device=self.device)

            # Linear optimal transport flow: x_t = (1 - t) x0 + t x1
            xt = (1.0 - t) * x0 + t * x1
            # Ground truth velocity target: v = x1 - x0
            v_target = x1 - x0

            pred_v = self.model(xt, t)
            loss = torch.mean((pred_v - v_target) ** 2)

            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

    def set_shape(self, shape_name: str):
        self.shape_name = shape_name
        self.target_pts = generate_2d_shape_points(shape_name, n_points=500)
        self.model = FlowFieldMLP().to(self.device)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.005)
        self._pretrain_quick(epochs=120)

    def compute_sampling_trajectory(self, n_particles: int = 400, steps: int = 40) -> np.ndarray:
        """Integrates ODE dx/dt = v(x, t) from t=0 (noise) to t=1 (generated shape).

        Returns:
            trajectory: [steps + 1, n_particles, 2] float coordinates
        """
        self.model.eval()
        with torch.no_grad():
            x = torch.randn((n_particles, 2), device=self.device)
            traj = [x.cpu().numpy()]
            dt = 1.0 / steps

            for s in range(steps):
                t_val = float(s / steps)
                t_tensor = torch.full((n_particles, 1), t_val, device=self.device)
                v = self.model(x, t_tensor)
                x = x + v * dt
                traj.append(x.cpu().numpy())

        return np.array(traj, dtype=np.float32)

    def get_vector_field_grid(self, t_val: float, grid_res: int = 14) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Calculates 2D velocity vectors across a grid at time t for vector flow visualization."""
        self.model.eval()
        with torch.no_grad():
            xs = np.linspace(-2.2, 2.2, grid_res)
            ys = np.linspace(-2.2, 2.2, grid_res)
            gx, gy = np.meshgrid(xs, ys)
            coords = np.column_stack([gx.ravel(), gy.ravel()]).astype(np.float32)

            c_t = torch.from_numpy(coords).to(self.device)
            t_t = torch.full((len(coords), 1), t_val, device=self.device)
            v_t = self.model(c_t, t_t).cpu().numpy()

            vx = v_t[:, 0].reshape((grid_res, grid_res))
            vy = v_t[:, 1].reshape((grid_res, grid_res))

        return gx, gy, np.stack([vx, vy], axis=-1)
