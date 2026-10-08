"""Offline Trainer and Asset Generator for NeuroNetWorks.

Trains and caches compact pretrained PyTorch weights and offline MNIST sample datasets.
"""

import os
import time
import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torchvision import datasets, transforms

from core.mnist_models import MNISTCNN
from core.autoencoder_models import MNIST2DAutoencoder
from core.signal_models import SignalDenoising1DCNN, generate_synthetic_signal, inject_noise
from core.cartpole_models import CartPoleEnv, DQNAgent

WEIGHTS_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "models_weights")
DATA_DIR = os.path.join(os.path.dirname(os.path.dirname(__file__)), "data")

def ensure_dirs():
    os.makedirs(WEIGHTS_DIR, exist_ok=True)
    os.makedirs(DATA_DIR, exist_ok=True)

def train_and_export_mnist_cnn(epochs: int = 2):
    """Trains MNIST CNN to >98% accuracy and exports weights."""
    print(">>> Training MNIST CNN...")
    device = torch.device("cpu")
    model = MNISTCNN().to(device)

    train_data = datasets.MNIST(DATA_DIR, train=True, download=True, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=0.002)
    criterion = nn.CrossEntropyLoss()

    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        for imgs, lbls in train_loader:
            imgs, lbls = imgs.to(device), lbls.to(device)
            optimizer.zero_grad()
            out = model(imgs)
            loss = criterion(out, lbls)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"MNIST CNN Epoch {ep+1}/{epochs} - Avg Loss: {total_loss / len(train_loader):.4f}")

    # Evaluate accuracy
    test_data = datasets.MNIST(DATA_DIR, train=False, download=False, transform=transforms.ToTensor())
    test_loader = torch.utils.data.DataLoader(test_data, batch_size=256, shuffle=False)
    model.eval()
    correct, total = 0, 0
    with torch.no_grad():
        for imgs, lbls in test_loader:
            preds = model(imgs).argmax(dim=-1)
            correct += (preds == lbls).sum().item()
            total += len(lbls)
    acc = correct / total
    print(f"MNIST CNN Test Accuracy: {acc * 100:.2f}%")

    out_path = os.path.join(WEIGHTS_DIR, "mnist_cnn.pth")
    torch.save(model.state_dict(), out_path)
    print(f"Saved: {out_path}")
    return model

def export_mnist_samples(num_per_class: int = 100):
    """Saves curated offline test samples for instant testing in the UI."""
    print(">>> Exporting curated MNIST test samples...")
    test_data = datasets.MNIST(DATA_DIR, train=False, download=False, transform=transforms.ToTensor())
    by_class = {i: [] for i in range(10)}

    for img, lbl in test_data:
        lbl = int(lbl)
        if len(by_class[lbl]) < num_per_class:
            by_class[lbl].append(img.squeeze(0).numpy())
        if all(len(v) == num_per_class for v in by_class.values()):
            break

    images = []
    labels = []
    for lbl in range(10):
        for img in by_class[lbl]:
            images.append(img)
            labels.append(lbl)

    images = np.array(images, dtype=np.float32)
    labels = np.array(labels, dtype=np.int64)

    out_path = os.path.join(DATA_DIR, "mnist_samples.npz")
    np.savez_compressed(out_path, images=images, labels=labels)
    print(f"Saved {len(images)} samples to: {out_path}")

def train_and_export_mnist_autoencoder(epochs: int = 4):
    """Trains MNIST 2D Latent Bottleneck Autoencoder."""
    print(">>> Training MNIST 2D Autoencoder...")
    device = torch.device("cpu")
    model = MNIST2DAutoencoder().to(device)

    train_data = datasets.MNIST(DATA_DIR, train=True, download=False, transform=transforms.ToTensor())
    train_loader = torch.utils.data.DataLoader(train_data, batch_size=128, shuffle=True)

    optimizer = optim.Adam(model.parameters(), lr=0.003)
    criterion = nn.MSELoss()

    for ep in range(epochs):
        model.train()
        total_loss = 0.0
        for imgs, _ in train_loader:
            imgs = imgs.to(device)
            optimizer.zero_grad()
            recon = model(imgs)
            loss = criterion(recon, imgs)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"Autoencoder Epoch {ep+1}/{epochs} - Recon Loss: {total_loss / len(train_loader):.4f}")

    out_path = os.path.join(WEIGHTS_DIR, "mnist_autoencoder.pth")
    torch.save(model.state_dict(), out_path)
    print(f"Saved: {out_path}")
    return model

def train_and_export_signal_denoiser(iterations: int = 600):
    """Trains 1D-CNN signal denoising filter on synthetic physical signals."""
    print(">>> Training 1D-CNN Signal Denoiser...")
    device = torch.device("cpu")
    model = SignalDenoising1DCNN().to(device)
    optimizer = optim.Adam(model.parameters(), lr=0.003)
    criterion = nn.MSELoss()

    model.train()
    time_pts = np.linspace(0, 1, 512, endpoint=False)
    types = ["harmonics", "ecg", "chirp"]

    for step in range(iterations):
        batch_clean = []
        batch_noisy = []
        for _ in range(32):
            stype = np.random.choice(types)
            freq = np.random.uniform(2.0, 10.0)
            clean = generate_synthetic_signal(stype, time_pts, freq=freq)
            noisy, _ = inject_noise(clean, noise_level=np.random.uniform(0.15, 0.45), noise_mode="gaussian")
            batch_clean.append(clean)
            batch_noisy.append(noisy)

        clean_t = torch.tensor(np.array(batch_clean), dtype=torch.float32).unsqueeze(1).to(device)
        noisy_t = torch.tensor(np.array(batch_noisy), dtype=torch.float32).unsqueeze(1).to(device)

        optimizer.zero_grad()
        pred = model(noisy_t)
        loss = criterion(pred, clean_t)
        loss.backward()
        optimizer.step()

        if (step + 1) % 200 == 0:
            print(f"Signal Denoiser Step {step+1}/{iterations} - Loss: {loss.item():.5f}")

    out_path = os.path.join(WEIGHTS_DIR, "signal_denoiser.pth")
    torch.save(model.state_dict(), out_path)
    print(f"Saved: {out_path}")
    return model

def train_and_export_cartpole_dqn(target_episodes: int = 160):
    """Trains Cart-Pole DQN agent until robust balance is achieved."""
    print(">>> Training Cart-Pole DQN Agent...")
    env = CartPoleEnv()
    agent = DQNAgent(state_dim=4, action_dim=2, lr=0.002, epsilon_decay=0.985)

    scores = []
    for ep in range(target_episodes):
        state = env.reset()
        score = 0
        done = False
        while not done and score < 400:
            action, _ = agent.select_action(state)
            next_state, reward, done, _ = env.step(action)
            # Custom shaped reward for keeping near center and upright
            shaped_reward = reward - 0.5 * abs(state[0]) - 1.0 * abs(state[2])
            agent.remember(state, action, shaped_reward, next_state, done)
            agent.train_step()
            state = next_state
            score += 1

        scores.append(score)
        if ep % 10 == 0:
            agent.update_target_network()
            recent_avg = np.mean(scores[-10:])
            print(f"CartPole Ep {ep}/{target_episodes} - Score: {score}, Recent Avg: {recent_avg:.1f}, Eps: {agent.epsilon:.3f}")

        if len(scores) >= 15 and np.mean(scores[-15:]) >= 300:
            print(f"CartPole solved at episode {ep}!")
            break

    out_path = os.path.join(WEIGHTS_DIR, "cartpole_dqn.pth")
    torch.save(agent.q_net.state_dict(), out_path)
    print(f"Saved: {out_path}")

def run_all_pretraining():
    ensure_dirs()
    t0 = time.time()
    train_and_export_mnist_cnn(epochs=2)
    export_mnist_samples(num_per_class=100)
    train_and_export_mnist_autoencoder(epochs=3)
    train_and_export_signal_denoiser(iterations=500)
    train_and_export_cartpole_dqn(target_episodes=150)
    print(f"\nAll models and datasets successfully generated in {time.time() - t0:.1f}s!")

if __name__ == "__main__":
    run_all_pretraining()
