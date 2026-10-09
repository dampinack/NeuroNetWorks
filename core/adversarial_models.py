"""Adversarial Attack algorithms (FGSM, PGD, Targeted Attacks) and robustness evaluation."""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, Tuple, Optional

def generate_fgsm_attack(
    model: nn.Module,
    image_tensor: torch.Tensor,
    epsilon: float = 0.15,
    target_class: Optional[int] = None,
    true_class: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Generates an adversarial example using the Fast Gradient Sign Method (Goodfellow et al., 2014).

    Args:
        model: PyTorch classifier
        image_tensor: [1, 1, 28, 28] float input tensor in [0, 1]
        epsilon: Perturbation magnitude (L_inf budget)
        target_class: If specified, performs targeted attack (minimizing loss for target)
        true_class: Ground truth class for untargeted attack (maximizing loss)

    Returns:
        adv_image_np: [28, 28] adversarial image
        noise_np: [28, 28] pure perturbation noise
        metrics: Dictionary of distortion metrics
    """
    model.eval()
    x = image_tensor.clone().detach().requires_grad_(True)
    logits = model(x)

    if true_class is None:
        true_class = int(logits.argmax(dim=-1)[0].item())

    if target_class is not None:
        # Targeted Attack: minimize loss with respect to desired wrong target
        loss = F.cross_entropy(logits, torch.tensor([target_class], device=x.device))
        loss.backward()
        # Step in OPPOSITE direction of gradient to minimize target loss
        perturbation = -epsilon * x.grad.data.sign()
    else:
        # Untargeted Attack: maximize loss with respect to true class
        loss = F.cross_entropy(logits, torch.tensor([true_class], device=x.device))
        loss.backward()
        perturbation = epsilon * x.grad.data.sign()

    adv_x = torch.clamp(x + perturbation, 0.0, 1.0)

    orig_np = x[0, 0].detach().cpu().numpy()
    adv_np = adv_x[0, 0].detach().cpu().numpy()
    noise_np = adv_np - orig_np

    l_inf = float(np.max(np.abs(noise_np)))
    l2 = float(np.linalg.norm(noise_np))
    mse = float(np.mean(noise_np ** 2))

    return adv_np, noise_np, {"l_inf": l_inf, "l2": l2, "mse": mse}


def generate_pgd_attack(
    model: nn.Module,
    image_tensor: torch.Tensor,
    epsilon: float = 0.15,
    alpha: float = 0.03,
    steps: int = 10,
    target_class: Optional[int] = None,
    true_class: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray, Dict[str, float]]:
    """Projected Gradient Descent (Madry et al., 2018) iterative multi-step attack."""
    model.eval()
    x_orig = image_tensor.clone().detach()
    adv_x = x_orig.clone().detach()

    # Small initial random perturbation within epsilon ball
    adv_x = adv_x + torch.FloatTensor(*adv_x.shape).uniform_(-epsilon, epsilon).to(x_orig.device)
    adv_x = torch.clamp(adv_x, 0.0, 1.0)

    if true_class is None:
        with torch.no_grad():
            true_class = int(model(x_orig).argmax(dim=-1)[0].item())

    for _ in range(steps):
        adv_x.requires_grad = True
        logits = model(adv_x)

        if target_class is not None:
            loss = F.cross_entropy(logits, torch.tensor([target_class], device=adv_x.device))
            loss.backward()
            step = -alpha * adv_x.grad.data.sign()
        else:
            loss = F.cross_entropy(logits, torch.tensor([true_class], device=adv_x.device))
            loss.backward()
            step = alpha * adv_x.grad.data.sign()

        # Update and project back to epsilon ball around original image
        adv_x = adv_x.detach() + step
        eta = torch.clamp(adv_x - x_orig, min=-epsilon, max=epsilon)
        adv_x = torch.clamp(x_orig + eta, min=0.0, max=1.0)

    orig_np = x_orig[0, 0].cpu().numpy()
    adv_np = adv_x[0, 0].cpu().numpy()
    noise_np = adv_np - orig_np

    l_inf = float(np.max(np.abs(noise_np)))
    l2 = float(np.linalg.norm(noise_np))
    mse = float(np.mean(noise_np ** 2))

    return adv_np, noise_np, {"l_inf": l_inf, "l2": l2, "mse": mse}
