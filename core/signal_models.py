"""1D-CNN Waveform Denoising and real-time audio/signal synthesis."""

import torch
import torch.nn as nn
import numpy as np
from typing import Tuple, Dict

class SignalDenoising1DCNN(nn.Module):
    """1D Fully-Convolutional Denoising Network with residual skip connection."""

    def __init__(self):
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=9, padding=4),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 64, kernel_size=7, padding=3),
            nn.BatchNorm1d(64),
            nn.LeakyReLU(0.2),
        )
        self.decoder = nn.Sequential(
            nn.Conv1d(64, 32, kernel_size=7, padding=3),
            nn.BatchNorm1d(32),
            nn.LeakyReLU(0.2),
            nn.Conv1d(32, 16, kernel_size=5, padding=2),
            nn.LeakyReLU(0.2),
            nn.Conv1d(16, 1, kernel_size=5, padding=2),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Predicts the clean signal directly
        features = self.encoder(x)
        out = self.decoder(features)
        return out


def generate_synthetic_signal(
    signal_type: str,
    time_points: np.ndarray,
    freq: float = 5.0
) -> np.ndarray:
    """Generates continuous physical or biological 1D waveforms."""
    signal_type = signal_type.lower()
    t = time_points

    if "ecg" in signal_type:
        # Synthetic ECG heartbeat (P, Q, R, S, T waves)
        period = 1.0  # 60 bpm
        phase = (t % period) / period
        # Baseline
        ecg = np.zeros_like(t)
        # P wave
        ecg += 0.15 * np.exp(-((phase - 0.2) ** 2) / (2 * 0.02 ** 2))
        # Q wave
        ecg -= 0.15 * np.exp(-((phase - 0.35) ** 2) / (2 * 0.005 ** 2))
        # R peak (tall spike)
        ecg += 1.00 * np.exp(-((phase - 0.38) ** 2) / (2 * 0.007 ** 2))
        # S wave
        ecg -= 0.25 * np.exp(-((phase - 0.41) ** 2) / (2 * 0.007 ** 2))
        # T wave
        ecg += 0.30 * np.exp(-((phase - 0.65) ** 2) / (2 * 0.04 ** 2))
        return ecg

    elif "chirp" in signal_type:
        # Linear frequency sweep (Chirp)
        f0 = freq * 0.5
        f1 = freq * 4.0
        T = t[-1] - t[0] if len(t) > 1 and (t[-1] - t[0]) > 0 else 1.0
        k = (f1 - f0) / T
        return np.sin(2 * np.pi * (f0 * t + 0.5 * k * (t ** 2)))

    elif "square" in signal_type:
        # Band-limited smooth square wave
        return np.sign(np.sin(2 * np.pi * freq * t)) * 0.8

    else:  # Acoustic Harmonics / Chord
        s = np.sin(2 * np.pi * freq * t)
        s += 0.5 * np.sin(2 * np.pi * (freq * 2.0) * t + 0.3)
        s += 0.25 * np.sin(2 * np.pi * (freq * 3.0) * t + 0.7)
        return s / 1.75


def inject_noise(
    clean: np.ndarray,
    noise_level: float = 0.3,
    noise_mode: str = "gaussian",
    hum_freq: float = 60.0,
    time_points: np.ndarray = None
) -> Tuple[np.ndarray, np.ndarray]:
    """Injects noise components (White Noise, Impulsive Spikes, AC Hum).

    Returns:
        noisy_signal: clean + noise
        noise_only: pure noise component
    """
    n_samples = len(clean)
    pure_noise = np.random.normal(0, noise_level, size=n_samples)

    if noise_mode == "spikes":
        # Random impulsive pops
        num_spikes = int(n_samples * 0.03)
        spike_indices = np.random.choice(n_samples, size=num_spikes, replace=False)
        pure_noise[spike_indices] += np.random.choice([-1.0, 1.0], size=num_spikes) * noise_level * 3.0

    elif noise_mode == "hum" and time_points is not None:
        # 60Hz powerline hum
        hum = np.sin(2 * np.pi * hum_freq * time_points) * noise_level * 1.2
        pure_noise += hum

    noisy = clean + pure_noise
    return noisy, pure_noise


def compute_snr_gain(clean: np.ndarray, noisy: np.ndarray, denoised: np.ndarray) -> Dict[str, float]:
    """Computes Signal-to-Noise Ratio (SNR) in dB and net SNR improvement."""
    p_signal = np.mean(clean ** 2) + 1e-12
    p_noise_in = np.mean((noisy - clean) ** 2) + 1e-12
    p_noise_out = np.mean((denoised - clean) ** 2) + 1e-12

    snr_in = float(10.0 * np.log10(p_signal / p_noise_in))
    snr_out = float(10.0 * np.log10(p_signal / p_noise_out))
    gain = snr_out - snr_in

    return {
        "snr_in": snr_in,
        "snr_out": snr_out,
        "gain": gain
    }
