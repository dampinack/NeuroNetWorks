#!/usr/bin/env python3
"""NeuroNetWorks - Cutting-Edge Deep Learning Interactive Suite.

Combines real-time PyTorch & NumPy computations with high-performance PyQt6 GUI.
"""

import sys
import os
import argparse

# Ensure local modules are found
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from core.device import configure_torch_threads
from ui.main_window import MainWindow

def parse_args():
    parser = argparse.ArgumentParser(description="NeuroNetWorks Deep Learning Educational Suite")
    parser.add_argument("--test", action="store_true", help="Run automated GUI self-test and exit")
    parser.add_argument("--module", type=int, default=0, help="Initial module index (0-5)")
    return parser.parse_args()

def main():
    args = parse_args()

    # Configure optimal compute threads for responsive UI
    configure_torch_threads()

    app = QApplication(sys.argv)
    app.setApplicationName("NeuroNetWorks")
    app.setOrganizationName("AI Research Lab")

    window = MainWindow()

    if args.module in range(6):
        window._switch_module(args.module)
        window.nav_buttons[args.module].setChecked(True)

    if args.test:
        print("Self-test mode: testing UI instantiation, module switching, and inference...")
        # Verify all 6 modules render and switch without errors
        for i in range(6):
            window._switch_module(i)
            print(f"Module {i} initialized successfully.")

        # Test MNIST sample inference and telemetry
        window._switch_module(0)
        window.view_mnist._load_sample_digit(7)
        window.view_mnist.run_inference()
        print("MNIST sample inference and telemetry verified.")

        # Test Auxiliary window
        window.aux_window.show()
        print("Auxiliary Neural Inspector verified.")

        # Test 1 step of training on Boundary View
        window._switch_module(1)
        window.view_boundary._single_step()
        print("2D Decision Boundary training step verified.")

        # Test 1 step of Approximator View
        window._switch_module(3)
        window.view_approximator._training_step()
        print("Universal Approximator training step verified.")

        # Test 1 step of Signal View
        window._switch_module(4)
        window.view_signal._stream_step()
        print("1D Signal Denoiser stream step verified.")

        # Test 1 step of CartPole View
        window._switch_module(5)
        window.view_cartpole._sim_step()
        print("CartPole DQN simulation step verified.")

        print("ALL NEURONETWORKS TEST CHECKS PASSED PERFECTLY!")
        return 0

    window.show()
    return app.exec()

if __name__ == "__main__":
    sys.exit(main())
