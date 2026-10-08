"""Hardware and device utilities for PyTorch computation in NeuroNetWorks."""

import torch
import os

def get_device() -> torch.device:
    """Returns the optimal PyTorch device (CUDA, MPS, or CPU)."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")

def get_device_name() -> str:
    """Returns a user-friendly name of the active compute device."""
    dev = get_device()
    if dev.type == "cuda":
        return f"CUDA: {torch.cuda.get_device_name(0)}"
    elif dev.type == "mps":
        return "Apple Silicon MPS"
    else:
        threads = torch.get_num_threads()
        return f"Intel / CPU ({threads} threads)"

def configure_torch_threads():
    """Configures optimal thread count for responsive desktop GUI execution."""
    cpu_count = os.cpu_count() or 4
    # Leave at least 1 core for GUI rendering responsiveness
    optimal_threads = max(1, min(cpu_count - 1, 8))
    torch.set_num_threads(optimal_threads)
