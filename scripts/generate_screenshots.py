#!/usr/bin/env python3
"""Automated Screen Capture Generator for NeuroNetWorks GitHub Showcase."""

import sys
import os
import time

# Ensure workspace root is in path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt, QTimer
from ui.main_window import MainWindow

SCREENSHOTS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "docs", "screenshots")

def generate():
    os.makedirs(SCREENSHOTS_DIR, exist_ok=True)
    print("📸 Initializing headless Qt application to capture screenshots...")

    app = QApplication(sys.argv)
    app.setApplicationName("NeuroNetWorks Screenshot Generator")

    win = MainWindow()
    win.resize(1360, 880)
    win.show()

    # Process events to complete initial layout
    app.processEvents()

    # 1. Capture Module 0: MNIST Digit Recognizer with loaded digit
    print("📸 Capturing Module 0: MNIST Digit Recognizer...")
    win._switch_module(0)
    win.view_mnist._load_sample_digit(7)
    win.view_mnist.run_inference()
    app.processEvents()
    time.sleep(0.1)
    win.grab().save(os.path.join(SCREENSHOTS_DIR, "01_mnist_recognizer.png"))

    # 2. Capture Auxiliary Neural Activity Inspector
    print("📸 Capturing Auxiliary Neural Activity Inspector...")
    win.aux_window.resize(1080, 760)
    win.aux_window.show()
    win.aux_window.update_telemetry(win.model_cnn_telemetry if hasattr(win, "model_cnn_telemetry") else {})
    app.processEvents()
    time.sleep(0.1)
    win.aux_window.grab().save(os.path.join(SCREENSHOTS_DIR, "02_neural_activity_inspector.png"))
    win.aux_window.hide()

    # 3. Capture Module 1: 2D Decision Boundaries (train for a few steps to show non-linear separation)
    print("📸 Capturing Module 1: 2D Decision Boundary Playground...")
    win._switch_module(1)
    for _ in range(35):
        win.view_boundary._training_step()
    app.processEvents()
    time.sleep(0.1)
    win.grab().save(os.path.join(SCREENSHOTS_DIR, "03_decision_boundaries.png"))

    # 4. Capture Module 2: Autoencoder Latent Space Explorer
    print("📸 Capturing Module 2: Autoencoder Latent Space...")
    win._switch_module(2)
    win.view_autoencoder.sub_tabs.setCurrentIndex(0)  # Latent Manifold tab
    win.view_autoencoder._on_latent_point_moved(0.8, -1.2)
    app.processEvents()
    time.sleep(0.1)
    win.grab().save(os.path.join(SCREENSHOTS_DIR, "04_autoencoder_latent.png"))

    # 5. Capture Module 2 Tab 2: Denoising Laboratory
    print("📸 Capturing Module 2: Denoising Laboratory...")
    win.view_autoencoder.sub_tabs.setCurrentIndex(1)  # Denoising tab
    win.view_autoencoder._update_denoising()
    app.processEvents()
    time.sleep(0.1)
    win.grab().save(os.path.join(SCREENSHOTS_DIR, "05_autoencoder_denoising.png"))

    # 6. Capture Module 3: Universal Function Approximator (train to show curve fit and basis neurons)
    print("📸 Capturing Module 3: Universal Function Approximator...")
    win._switch_module(3)
    for _ in range(40):
        win.view_approximator._training_step()
    app.processEvents()
    time.sleep(0.1)
    win.grab().save(os.path.join(SCREENSHOTS_DIR, "06_function_approximator.png"))

    # 7. Capture Module 4: Real-Time Signal & Audio Waveform Denoiser
    print("📸 Capturing Module 4: Signal & Waveform Denoiser...")
    win._switch_module(4)
    for _ in range(10):
        win.view_signal._stream_step()
    app.processEvents()
    time.sleep(0.1)
    win.grab().save(os.path.join(SCREENSHOTS_DIR, "07_signal_oscilloscope.png"))

    # 8. Capture Module 5: Reinforcement Learning Cart-Pole Balancer
    print("📸 Capturing Module 5: RL Cart-Pole Balancer...")
    win._switch_module(5)
    for _ in range(15):
        win.view_cartpole._sim_step()
    app.processEvents()
    time.sleep(0.1)
    win.grab().save(os.path.join(SCREENSHOTS_DIR, "08_cartpole_dqn.png"))

    print(f"\n✅ All 8 screen captures successfully generated in: {SCREENSHOTS_DIR}")
    for f in sorted(os.listdir(SCREENSHOTS_DIR)):
        if f.endswith(".png"):
            size_kb = os.path.getsize(os.path.join(SCREENSHOTS_DIR, f)) / 1024
            print(f"   • {f} ({size_kb:.1f} KB)")

if __name__ == "__main__":
    generate()
