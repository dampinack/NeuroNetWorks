# Antigravity Workspace Guidelines: NeuroNetWorks

## Project Overview
NeuroNetWorks is an interactive desktop deep learning educational suite built with PyTorch, NumPy, and PyQt6.

## Running & Debugging
- **Virtual Environment:** Always use `${workspaceFolder}/.venv/bin/python`
- **Main Entrypoint:** `app.py` or `./launch.sh`
- **Automated Verification:** `QT_QPA_PLATFORM=offscreen .venv/bin/python app.py --test`
- **Retrain Models:** `.venv/bin/python -m core.trainer`

## Architecture
- `core/`: PyTorch models, datasets, and physics simulators.
- `ui/`: PyQt6 widgets, themes, main shell (`main_window.py`), and auxiliary inspector (`auxiliary_window.py`).
- `ui/modules/`: The 6 dedicated module views (MNIST, Boundary, Autoencoder, Approximator, Signal, CartPole).
- `models_weights/`: Cached PyTorch checkpoint files (`.pth`).
- `data/`: Offline sample datasets (`mnist_samples.npz`).
