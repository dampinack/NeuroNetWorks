"""Physics simulation environment and Deep Q-Network (DQN) for Cart-Pole balancing."""

import torch
import torch.nn as nn
import numpy as np
import random
from collections import deque
from typing import Tuple, List, Dict

class CartPoleEnv:
    """Rigorous Newtonian physics simulation of an inverted pendulum mounted on a movable cart.

    Supports custom gravity, pole length, and interactive user physical disturbances.
    """

    def __init__(
        self,
        gravity: float = 9.8,
        mass_cart: float = 1.0,
        mass_pole: float = 0.1,
        length: float = 0.5,  # half-length of pole
        force_mag: float = 12.0,
        tau: float = 0.02  # seconds per simulation frame
    ):
        self.gravity = gravity
        self.mass_cart = mass_cart
        self.mass_pole = mass_pole
        self.total_mass = mass_cart + mass_pole
        self.length = length
        self.polemass_length = mass_pole * length
        self.force_mag = force_mag
        self.tau = tau

        # State limits
        self.x_threshold = 2.4
        self.theta_threshold_radians = 12 * 2 * np.pi / 360  # ~0.209 rad (12 degrees)

        self.state = np.zeros(4, dtype=np.float32)  # [x, x_dot, theta, theta_dot]
        self.steps = 0
        self.reset()

    def reset(self, random_init: bool = True) -> np.ndarray:
        if random_init:
            self.state = np.random.uniform(low=-0.05, high=0.05, size=(4,)).astype(np.float32)
        else:
            self.state = np.zeros(4, dtype=np.float32)
        self.steps = 0
        return self.state.copy()

    def step(self, action: int, external_disturbance: float = 0.0) -> Tuple[np.ndarray, float, bool, Dict]:
        """Runs one physics integration step.

        action: 0 (push Left) or 1 (push Right)
        external_disturbance: impulsive force injected by human user (Newtons)
        """
        x, x_dot, theta, theta_dot = self.state
        force = (self.force_mag if action == 1 else -self.force_mag) + external_disturbance

        costheta = np.cos(theta)
        sintheta = np.sin(theta)

        # Standard inverted pendulum equations of motion
        temp = (force + self.polemass_length * theta_dot ** 2 * sintheta) / self.total_mass
        thetaacc = (self.gravity * sintheta - costheta * temp) / (
            self.length * (4.0 / 3.0 - self.mass_pole * costheta ** 2 / self.total_mass)
        )
        xacc = temp - self.polemass_length * thetaacc * costheta / self.total_mass

        # Euler-Cromer symplectic integration
        x_dot = x_dot + self.tau * xacc
        x = x + self.tau * x_dot
        theta_dot = theta_dot + self.tau * thetaacc
        theta = theta + self.tau * theta_dot

        self.state = np.array([x, x_dot, theta, theta_dot], dtype=np.float32)
        self.steps += 1

        # Check termination condition
        terminated = bool(
            x < -self.x_threshold
            or x > self.x_threshold
            or theta < -self.theta_threshold_radians
            or theta > self.theta_threshold_radians
        )

        reward = 1.0 if not terminated else 0.0

        return self.state.copy(), reward, terminated, {}


class QNetwork(nn.Module):
    """Deep Q-Network approximating Q(s, a)."""

    def __init__(self, state_dim: int = 4, action_dim: int = 2, hidden_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state: torch.Tensor) -> torch.Tensor:
        return self.net(state)


class DQNAgent:
    """DQN Agent with replay buffer, target network, and epsilon-greedy policy."""

    def __init__(
        self,
        state_dim: int = 4,
        action_dim: int = 2,
        lr: float = 0.001,
        gamma: float = 0.99,
        epsilon: float = 1.0,
        epsilon_min: float = 0.02,
        epsilon_decay: float = 0.995,
        buffer_capacity: int = 10000,
        batch_size: int = 64
    ):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.gamma = gamma
        self.epsilon = epsilon
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size

        self.q_net = QNetwork(state_dim, action_dim)
        self.target_net = QNetwork(state_dim, action_dim)
        self.target_net.load_state_dict(self.q_net.state_dict())

        self.optimizer = torch.optim.Adam(self.q_net.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        self.replay_buffer = deque(maxlen=buffer_capacity)

    def select_action(self, state: np.ndarray, evaluate: bool = False) -> Tuple[int, np.ndarray]:
        """Returns chosen discrete action (0 or 1) and raw Q-values for visual display."""
        with torch.no_grad():
            s_t = torch.from_numpy(state.astype(np.float32)).unsqueeze(0)
            q_values = self.q_net(s_t).squeeze(0).cpu().numpy()

        if not evaluate and random.random() < self.epsilon:
            action = random.randint(0, self.action_dim - 1)
        else:
            action = int(np.argmax(q_values))

        return action, q_values

    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def train_step(self) -> float:
        """Samples a mini-batch from experience replay and performs one gradient descent step."""
        if len(self.replay_buffer) < self.batch_size:
            return 0.0

        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)

        states_t = torch.tensor(np.array(states), dtype=torch.float32)
        actions_t = torch.tensor(actions, dtype=torch.int64).unsqueeze(1)
        rewards_t = torch.tensor(rewards, dtype=torch.float32).unsqueeze(1)
        next_states_t = torch.tensor(np.array(next_states), dtype=torch.float32)
        dones_t = torch.tensor(dones, dtype=torch.float32).unsqueeze(1)

        # Q(s, a)
        curr_q = self.q_net(states_t).gather(1, actions_t)

        with torch.no_grad():
            max_next_q = self.target_net(next_states_t).max(1, keepdim=True)[0]
            target_q = rewards_t + (1 - dones_t) * self.gamma * max_next_q

        loss = self.criterion(curr_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Decay exploration
        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

        return float(loss.item())

    def update_target_network(self):
        self.target_net.load_state_dict(self.q_net.state_dict())
