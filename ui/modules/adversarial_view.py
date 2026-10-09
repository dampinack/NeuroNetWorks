"""Module 8: Adversarial Attack Studio (FGSM & PGD Robustness Evaluation)."""

import os
import numpy as np
import torch
import torch.nn.functional as F

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QFrame, QGridLayout, QProgressBar, QSplitter
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QImage, QPixmap

from core.mnist_models import MNISTCNN
from core.adversarial_models import generate_fgsm_attack, generate_pgd_attack
from core.device import get_device
from ui.theme import (
    COLOR_BG_DARK, COLOR_BG_CARD, COLOR_BG_SURFACE, COLOR_BORDER,
    COLOR_ACCENT_CYAN, COLOR_ACCENT_PURPLE, COLOR_ACCENT_GREEN,
    COLOR_ACCENT_AMBER, COLOR_ACCENT_RED, COLOR_TEXT_PRIMARY,
    COLOR_TEXT_SECONDARY, COLOR_TEXT_MUTED
)
from ui.custom_widgets import Card, MetricBadge, LabeledSlider, EducationalPanel

WEIGHTS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models_weights", "mnist_cnn.pth")
SAMPLES_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "mnist_samples.npz")

class AdversarialView(QWidget):
    """Main Adversarial Attack Studio & Robustness Evaluation View."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.device = get_device()
        self.model = MNISTCNN().to(self.device)

        if os.path.exists(WEIGHTS_PATH):
            try:
                self.model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=self.device))
                self.model.eval()
            except Exception as e:
                print("Error loading CNN weights:", e)

        # Samples
        self.sample_images = None
        self.sample_labels = None
        if os.path.exists(SAMPLES_PATH):
            try:
                data = np.load(SAMPLES_PATH)
                self.sample_images = data["images"]
                self.sample_labels = data["labels"]
            except Exception as e:
                print("Error loading samples:", e)

        self.current_img = self.sample_images[0] if self.sample_images is not None else np.zeros((28, 28), dtype=np.float32)
        self.current_label = int(self.sample_labels[0]) if self.sample_labels is not None else 0
        self.attack_method = "FGSM"
        self.epsilon = 0.15
        self.target_class = None

        self._init_ui()
        self._run_attack()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(16)

        # Header Title
        hdr_layout = QHBoxLayout()
        title_box = QVBoxLayout()
        title = QLabel("ADVERSARIAL ATTACK STUDIO & MODEL ROBUSTNESS")
        title.setStyleSheet("font-size: 20px; font-weight: 800; color: #FFFFFF;")
        subtitle = QLabel("Exploiting High-Dimensional Gradient Geometry with FGSM & Iterative PGD Attacks")
        subtitle.setStyleSheet(f"font-size: 12px; color: {COLOR_TEXT_SECONDARY};")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        hdr_layout.addLayout(title_box)
        hdr_layout.addStretch()
        main_layout.addLayout(hdr_layout)

        # Controls Bar Card
        ctrl_card = Card("ATTACK CONFIGURATION & PERTURBATION BUDGET (ε)")
        ctrl_row = QHBoxLayout()

        lbl_method = QLabel("Attack Algorithm:")
        lbl_method.setStyleSheet(f"font-weight: 600; color: {COLOR_TEXT_SECONDARY};")
        ctrl_row.addWidget(lbl_method)

        self.combo_method = QComboBox()
        self.combo_method.addItems(["FGSM (Fast Gradient Sign Method - 1 Step)", "PGD (Projected Gradient Descent - 10 Steps)"])
        self.combo_method.currentIndexChanged.connect(self._on_method_changed)
        ctrl_row.addWidget(self.combo_method)

        lbl_target = QLabel("Attack Objective:")
        lbl_target.setStyleSheet(f"font-weight: 600; color: {COLOR_TEXT_SECONDARY};")
        ctrl_row.addWidget(lbl_target)

        self.combo_target = QComboBox()
        self.combo_target.addItems(["Untargeted (Cause Any Misclassification)"] + [f"Targeted: Force Classification as '{i}'" for i in range(10)])
        self.combo_target.currentIndexChanged.connect(self._on_target_changed)
        ctrl_row.addWidget(self.combo_target, stretch=1)

        ctrl_card.addLayout(ctrl_row)

        self.slider_epsilon = LabeledSlider("Perturbation Magnitude (L_inf Budget ε)", 0.0, 0.40, 0.15, step=0.01, is_float=True)
        self.slider_epsilon.valueChanged.connect(self._on_epsilon_changed)
        ctrl_card.addWidget(self.slider_epsilon)
        main_layout.addWidget(ctrl_card)

        # Visual Comparison Grid: Original -> Noise -> Adversarial -> Probability Spectrum
        vis_layout = QHBoxLayout()
        vis_layout.setSpacing(16)

        # Card 1: Original Clean
        c1 = Card("1. CLEAN INPUT IMAGE")
        self.lbl_clean = QLabel()
        self.lbl_clean.setFixedSize(160, 160)
        self.lbl_clean.setStyleSheet("background: #000; border: 1px solid #334155; border-radius: 8px;")
        self.lbl_clean.setAlignment(Qt.AlignmentFlag.AlignCenter)
        c1.addWidget(self.lbl_clean)
        self.badge_clean = MetricBadge("ORIGINAL PREDICTION", "--", "", COLOR_ACCENT_GREEN)
        c1.addWidget(self.badge_clean)
        vis_layout.addWidget(c1)

        # Card 2: Perturbation Noise
        c2 = Card("2. PERTURBATION NOISE (x5)")
        self.lbl_noise = QLabel()
        self.lbl_noise.setFixedSize(160, 160)
        self.lbl_noise.setStyleSheet("background: #000; border: 1px solid #334155; border-radius: 8px;")
        self.lbl_noise.setAlignment(Qt.AlignmentFlag.AlignCenter)
        c2.addWidget(self.lbl_noise)
        self.badge_linf = MetricBadge("L_INF DISTORTION", "0.00", "", COLOR_ACCENT_AMBER)
        c2.addWidget(self.badge_linf)
        vis_layout.addWidget(c2)

        # Card 3: Adversarial Image
        c3 = Card("3. ADVERSARIAL RESULT")
        self.lbl_adv = QLabel()
        self.lbl_adv.setFixedSize(160, 160)
        self.lbl_adv.setStyleSheet("background: #000; border: 1px solid #334155; border-radius: 8px;")
        self.lbl_adv.setAlignment(Qt.AlignmentFlag.AlignCenter)
        c3.addWidget(self.lbl_adv)
        self.badge_adv = MetricBadge("ATTACKED PREDICTION", "--", "", COLOR_ACCENT_RED)
        c3.addWidget(self.badge_adv)
        vis_layout.addWidget(c3)

        # Card 4: Collapsed Confidence Spectrum
        c4 = Card("4. ATTACKED PROBABILITY SPECTRUM")
        self.prob_bars = []
        self.prob_labels = []
        bars_layout = QVBoxLayout()
        bars_layout.setSpacing(4)
        for d in range(10):
            row = QHBoxLayout()
            lbl = QLabel(f"{d}:")
            lbl.setFixedWidth(18)
            lbl.setStyleSheet("font-weight: 700; font-size: 11px;")
            pbar = QProgressBar()
            pbar.setRange(0, 1000)
            pbar.setTextVisible(False)
            pbar.setFixedHeight(10)
            val_lbl = QLabel("0%")
            val_lbl.setFixedWidth(40)
            val_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
            val_lbl.setStyleSheet(f"font-size: 10px; color: {COLOR_TEXT_MUTED};")
            row.addWidget(lbl)
            row.addWidget(pbar, stretch=1)
            row.addWidget(val_lbl)
            bars_layout.addLayout(row)
            self.prob_bars.append(pbar)
            self.prob_labels.append(val_lbl)
        c4.addLayout(bars_layout)
        vis_layout.addWidget(c4, stretch=1)

        main_layout.addLayout(vis_layout)

        # Sample Carousel
        sample_card = Card("PRELOADED TEST BENCHMARK DIGITS")
        sample_row = QHBoxLayout()
        sample_row.setSpacing(6)
        for d in range(10):
            btn = QPushButton(f"Digit {d}")
            btn.clicked.connect(lambda checked, digit=d: self._load_sample(digit))
            sample_row.addWidget(btn)
        sample_card.addLayout(sample_row)
        main_layout.addWidget(sample_card)

        # Educational Panel
        edu = EducationalPanel(
            "The Linearity Hypothesis & Adversarial Vulnerability (Goodfellow et al., 2014)",
            "**Why Deep Neural Networks Fail on Imperceptible Noise:**\n\n"
            "- **The Linearity Hypothesis:** Contrary to intuition, neural networks fail not from extreme non-linearity, but from their *linear behavior in high-dimensional space*. For an activation $\\mathbf{w}^T \\tilde{\\mathbf{x}} = \\mathbf{w}^T \\mathbf{x} + \\mathbf{w}^T \\mathbf{\\eta}$, if each pixel changes by only $\\epsilon$, the total activation shifts by $\\epsilon \\sum |w_i| \\approx \\epsilon n m$. In high dimensions ($n=784$), subtle imperceptible noise adds up to massive activation shifts!\n"
            "- **Fast Gradient Sign Method (FGSM):**\n"
            "$$\\mathbf{x}_{\\text{adv}} = \\text{clip}\\left(\\mathbf{x} + \\epsilon \\cdot \\text{sign}\\left(\\nabla_{\\mathbf{x}} \\mathcal{L}(\\theta, \\mathbf{x}, y)\\right), 0, 1\\right)$$\n"
            "- **Projected Gradient Descent (PGD):** Iteratively takes small gradient steps inside the $\\epsilon$-ball boundary, finding the strongest localized adversarial perturbation."
        )
        main_layout.addWidget(edu)

    def _on_method_changed(self, idx: int):
        self.attack_method = "FGSM" if idx == 0 else "PGD"
        self._run_attack()

    def _on_target_changed(self, idx: int):
        self.target_class = None if idx == 0 else (idx - 1)
        self._run_attack()

    def _on_epsilon_changed(self, val: float):
        self.epsilon = val
        self._run_attack()

    def _load_sample(self, digit: int):
        if self.sample_images is None:
            return
        matches = np.where(self.sample_labels == digit)[0]
        if len(matches) > 0:
            idx = np.random.choice(matches)
            self.current_img = self.sample_images[idx]
            self.current_label = digit
            self._run_attack()

    def _run_attack(self):
        img_t = torch.from_numpy(self.current_img).unsqueeze(0).unsqueeze(0).to(self.device)

        # Baseline original prediction
        self.model.eval()
        with torch.no_grad():
            orig_logits = self.model(img_t)
            orig_probs = F.softmax(orig_logits, dim=-1)[0].cpu().numpy()
            orig_pred = int(np.argmax(orig_probs))
            orig_conf = float(orig_probs[orig_pred]) * 100.0

        self.badge_clean.set_value(f"{orig_pred} ({orig_conf:.1f}%)")

        # Generate adversarial example
        if self.attack_method == "FGSM":
            adv_np, noise_np, metrics = generate_fgsm_attack(
                self.model, img_t, epsilon=self.epsilon,
                target_class=self.target_class, true_class=self.current_label
            )
        else:
            adv_np, noise_np, metrics = generate_pgd_attack(
                self.model, img_t, epsilon=self.epsilon, steps=10,
                target_class=self.target_class, true_class=self.current_label
            )

        # Attacked prediction
        adv_t = torch.from_numpy(adv_np).unsqueeze(0).unsqueeze(0).to(self.device)
        with torch.no_grad():
            adv_logits = self.model(adv_t)
            adv_probs = F.softmax(adv_logits, dim=-1)[0].cpu().numpy()
            adv_pred = int(np.argmax(adv_probs))
            adv_conf = float(adv_probs[adv_pred]) * 100.0

        # Update badges
        self.badge_adv.set_value(f"{adv_pred} ({adv_conf:.1f}%)")
        if adv_pred != orig_pred:
            self.badge_adv.val_widget.setStyleSheet(f"color: {COLOR_ACCENT_RED}; font-size: 18px; font-weight: 800;")
        else:
            self.badge_adv.val_widget.setStyleSheet(f"color: {COLOR_ACCENT_GREEN}; font-size: 18px; font-weight: 800;")

        self.badge_linf.set_value(f"{metrics['l_inf']:.3f}")

        # Update images
        self.lbl_clean.setPixmap(self._numpy_to_pixmap(self.current_img, colormap="mono"))
        self.lbl_noise.setPixmap(self._numpy_to_pixmap(np.abs(noise_np) * 4.0, colormap="hot"))
        self.lbl_adv.setPixmap(self._numpy_to_pixmap(adv_np, colormap="mono"))

        # Update confidence bars
        for d in range(10):
            p = float(adv_probs[d])
            self.prob_bars[d].setValue(int(p * 1000))
            self.prob_labels[d].setText(f"{p * 100:.1f}%")
            if d == adv_pred:
                color = COLOR_ACCENT_RED if adv_pred != orig_pred else COLOR_ACCENT_GREEN
                self.prob_bars[d].setStyleSheet(f"QProgressBar::chunk {{ background: {color}; }}")
            else:
                self.prob_bars[d].setStyleSheet(f"QProgressBar::chunk {{ background: {COLOR_BG_SURFACE}; }}")

    def _numpy_to_pixmap(self, arr: np.ndarray, colormap: str = "mono") -> QPixmap:
        arr = np.clip(np.nan_to_num(arr), 0.0, 1.0)
        h, w = arr.shape
        img = QImage(w, h, QImage.Format.Format_RGB32)
        for y in range(h):
            for x in range(w):
                v = float(arr[y, x])
                if colormap == "hot":
                    r = int(min(1.0, v * 2.0) * 255)
                    g = int(max(0.0, v - 0.2) * 255)
                    b = int(max(0.0, v - 0.6) * 255)
                else:
                    val = int(v * 255)
                    r, g, b = val, val, val
                img.setPixelColor(x, y, QColor(r, g, b))
        pix = QPixmap.fromImage(img)
        return pix.scaled(160, 160, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation)
