"""Module 4: Interactive Universal Function Approximator (1D Regression)."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QFrame, QSplitter
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QPointF
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QImage

from core.approximator_models import FunctionApproximatorMLP, get_target_function
from core.device import get_device
from ui.theme import (
    COLOR_BG_DARK, COLOR_BG_CARD, COLOR_BG_SURFACE, COLOR_BORDER,
    COLOR_ACCENT_CYAN, COLOR_ACCENT_PURPLE, COLOR_ACCENT_GREEN,
    COLOR_ACCENT_AMBER, COLOR_ACCENT_RED, COLOR_TEXT_PRIMARY,
    COLOR_TEXT_SECONDARY, COLOR_TEXT_MUTED
)
from ui.custom_widgets import Card, MetricBadge, LabeledSlider, EducationalPanel

class CurveCanvas(QWidget):
    """Real-time 2D canvas displaying the Target Function vs Neural Network Approximation."""

    def __init__(self, size_w: int = 560, size_h: int = 340, parent=None):
        super().__init__(parent)
        self.setMinimumSize(size_w, size_h)
        self.x_vals = np.linspace(-1.0, 1.0, 200, dtype=np.float32)
        self.target_y = np.zeros_like(self.x_vals)
        self.approx_y = None
        self.basis_curves = None

    def set_data(self, target_y: np.ndarray, approx_y: np.ndarray = None, basis_curves: np.ndarray = None):
        self.target_y = target_y
        self.approx_y = approx_y
        self.basis_curves = basis_curves
        self.update()

    def _val_to_pixel(self, x: float, y: float) -> QPointF:
        w, h = self.width(), self.height()
        # x is in [-1.0, 1.0], y is mapped in [-1.5, 1.5]
        px = (x + 1.0) / 2.0 * (w - 40) + 20
        py = (1.5 - y) / 3.0 * (h - 40) + 20
        return QPointF(px, py)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()

        # Canvas Background
        painter.fillRect(0, 0, w, h, QColor(COLOR_BG_DARK))

        # Coordinate Axes
        painter.setPen(QPen(QColor("#1E293B"), 1, Qt.PenStyle.DashLine))
        origin = self._val_to_pixel(0.0, 0.0)
        painter.drawLine(20, int(origin.y()), w - 20, int(origin.y()))
        painter.drawLine(int(origin.x()), 20, int(origin.x()), h - 20)

        # 1. Draw Individual Basis Neurons (subtle semi-transparent curves)
        if self.basis_curves is not None:
            n_basis = min(12, len(self.basis_curves))
            for i in range(n_basis):
                pen_basis = QPen(QColor(168, 85, 247, 60), 1, Qt.PenStyle.SolidLine)
                painter.setPen(pen_basis)
                curve = self.basis_curves[i]
                for j in range(len(self.x_vals) - 1):
                    p1 = self._val_to_pixel(self.x_vals[j], curve[j])
                    p2 = self._val_to_pixel(self.x_vals[j + 1], curve[j + 1])
                    painter.drawLine(p1, p2)

        # 2. Draw Target Function Ground Truth (dotted white line)
        if self.target_y is not None:
            pen_target = QPen(QColor(COLOR_TEXT_MUTED), 2, Qt.PenStyle.DashLine)
            painter.setPen(pen_target)
            for j in range(len(self.x_vals) - 1):
                p1 = self._val_to_pixel(self.x_vals[j], self.target_y[j])
                p2 = self._val_to_pixel(self.x_vals[j + 1], self.target_y[j + 1])
                painter.drawLine(p1, p2)

        # 3. Draw Neural Network Fitted Prediction (vivid glowing cyan line)
        if self.approx_y is not None:
            pen_approx = QPen(QColor(COLOR_ACCENT_CYAN), 3, Qt.PenStyle.SolidLine)
            painter.setPen(pen_approx)
            for j in range(len(self.x_vals) - 1):
                p1 = self._val_to_pixel(self.x_vals[j], self.approx_y[j])
                p2 = self._val_to_pixel(self.x_vals[j + 1], self.approx_y[j + 1])
                painter.drawLine(p1, p2)

        # Legend Overlay
        painter.setPen(QPen(QColor(COLOR_TEXT_MUTED), 1))
        painter.setFont(painter.font())
        painter.drawText(30, 32, "--- Target Curve (Ground Truth)")
        painter.setPen(QPen(QColor(COLOR_ACCENT_CYAN), 1))
        painter.drawText(240, 32, "━ Neural Approximation f(x)")
        painter.setPen(QPen(QColor(COLOR_ACCENT_PURPLE), 1))
        painter.drawText(430, 32, "┈ Hidden Neuron Basis Σ [wᵢ·σ(vᵢx)]")

        # Border
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor(COLOR_BORDER), 2))
        painter.drawRoundedRect(0, 0, w, h, 8, 8)


class ApproximatorView(QWidget):
    """Main Interactive Universal Function Approximator View."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.device = get_device()
        self.is_training = False
        self.step_count = 0

        # Domain samples
        self.x_train = np.linspace(-1.0, 1.0, 200, dtype=np.float32)
        self.fn_target = get_target_function("sine")
        self.y_train = self.fn_target(self.x_train)

        # Hyperparameters
        self.hidden_dim = 32
        self.num_layers = 2
        self.activation = "tanh"
        self.lr = 0.02

        self.model = None
        self.optimizer = None
        self.criterion = nn.MSELoss()

        self._init_ui()
        self._rebuild_model()

        # Training timer (40 updates per second)
        self.train_timer = QTimer(self)
        self.train_timer.timeout.connect(self._training_step)

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(16)

        # Header Title
        hdr_layout = QHBoxLayout()
        title_box = QVBoxLayout()
        title = QLabel("UNIVERSAL FUNCTION APPROXIMATOR (1D REGRESSION)")
        title.setStyleSheet("font-size: 20px; font-weight: 800; color: #FFFFFF;")
        subtitle = QLabel("Visual Proof of the Universal Approximation Theorem with Hidden Neuron Basis Decomposition")
        subtitle.setStyleSheet(f"font-size: 12px; color: {COLOR_TEXT_SECONDARY};")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        hdr_layout.addLayout(title_box)
        hdr_layout.addStretch()

        # Controls (Play/Pause/Step/Reset)
        self.btn_play = QPushButton("▶ Start Training")
        self.btn_play.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLOR_ACCENT_GREEN};
                color: #000;
                font-weight: 800;
                padding: 10px 18px;
                border-radius: 8px;
            }}
            QPushButton:hover {{
                background-color: #34D399;
            }}
        """)
        self.btn_play.clicked.connect(self._toggle_training)
        hdr_layout.addWidget(self.btn_play)

        self.btn_step = QPushButton("⏭ Step")
        self.btn_step.clicked.connect(self._training_step)
        hdr_layout.addWidget(self.btn_step)

        self.btn_reset = QPushButton("↺ Reset")
        self.btn_reset.clicked.connect(self._rebuild_model)
        hdr_layout.addWidget(self.btn_reset)

        main_layout.addLayout(hdr_layout)

        # Content Splitter (Left: Curve Canvas, Right: Controls & Metrics)
        content_layout = QHBoxLayout()
        content_layout.setSpacing(20)

        # Left: Curve Canvas Card
        canvas_card = Card("1D REGRESSION APPROXIMATION MANIFOLD")
        self.canvas = CurveCanvas(560, 360)
        canvas_card.addWidget(self.canvas)
        content_layout.addWidget(canvas_card, stretch=2)

        # Right: Parameters & Metrics Panel
        right_panel = QVBoxLayout()
        right_panel.setSpacing(12)

        # Metrics Card
        metrics_card = Card("CONVERGENCE TELEMETRY")
        m_row = QHBoxLayout()
        self.badge_steps = MetricBadge("ITERATIONS", "0", "", COLOR_TEXT_PRIMARY)
        self.badge_loss = MetricBadge("MSE LOSS", "0.0000", "", COLOR_ACCENT_CYAN)
        self.badge_r2 = MetricBadge("R² FIT", "0.00%", "", COLOR_ACCENT_GREEN)
        m_row.addWidget(self.badge_steps)
        m_row.addWidget(self.badge_loss)
        m_row.addWidget(self.badge_r2)
        metrics_card.addLayout(m_row)
        right_panel.addWidget(metrics_card)

        # Target Function Card
        fn_card = Card("TARGET CONTINUOUS FUNCTION")
        row_fn = QHBoxLayout()
        lbl_fn = QLabel("Target Curve:")
        lbl_fn.setStyleSheet(f"font-weight: 600; color: {COLOR_TEXT_SECONDARY};")
        row_fn.addWidget(lbl_fn)

        self.combo_target = QComboBox()
        self.combo_target.addItems([
            "Sine Wave: sin(2πx)",
            "Harmonic Superposition: sin(2πx) + 0.5cos(6πx)",
            "Damped Oscillator: exp(-1.2x)sin(4πx)",
            "Runge's Function: 1 / (1 + 25x²)",
            "Discontinuous Square / Step Wave",
            "Sawtooth Wave"
        ])
        self.combo_target.currentIndexChanged.connect(self._on_target_changed)
        row_fn.addWidget(self.combo_target, stretch=1)
        fn_card.addLayout(row_fn)
        right_panel.addWidget(fn_card)

        # Architecture Card
        arch_card = Card("NETWORK CAPACITY & ACTIVATION BASIS")

        row_depth = QHBoxLayout()
        lbl_depth = QLabel("Hidden Depth:")
        lbl_depth.setStyleSheet(f"font-weight: 600; color: {COLOR_TEXT_SECONDARY};")
        row_depth.addWidget(lbl_depth)

        self.combo_depth = QComboBox()
        self.combo_depth.addItems(["1 Hidden Layer (Theorem Pure Form)", "2 Hidden Layers", "3 Hidden Layers", "4 Hidden Layers"])
        self.combo_depth.setCurrentIndex(1)
        self.combo_depth.currentIndexChanged.connect(self._on_arch_changed)
        row_depth.addWidget(self.combo_depth, stretch=1)
        arch_card.addLayout(row_depth)

        self.slider_neurons = LabeledSlider("Neurons per Layer", 4, 128, 32, step=4)
        self.slider_neurons.valueChanged.connect(self._on_arch_changed)
        arch_card.addWidget(self.slider_neurons)

        row_act = QHBoxLayout()
        lbl_act = QLabel("Basis Function:")
        lbl_act.setStyleSheet(f"font-weight: 600; color: {COLOR_TEXT_SECONDARY};")
        row_act.addWidget(lbl_act)

        self.combo_act = QComboBox()
        self.combo_act.addItems(["Tanh (Smooth Sigmoidal)", "ReLU (Piecewise Linear)", "GELU (Smooth Gaussian)", "SIREN (Periodic Sine)", "Sigmoid (Logistic)"])
        self.combo_act.currentIndexChanged.connect(self._on_arch_changed)
        row_act.addWidget(self.combo_act, stretch=1)
        arch_card.addLayout(row_act)

        self.slider_lr = LabeledSlider("Learning Rate", 0.002, 0.08, 0.02, step=0.002, is_float=True)
        self.slider_lr.valueChanged.connect(self._update_lr)
        arch_card.addWidget(self.slider_lr)

        right_panel.addWidget(arch_card)
        content_layout.addLayout(right_panel, stretch=1)
        main_layout.addLayout(content_layout)

        # Educational Panel
        edu = EducationalPanel(
            "The Universal Approximation Theorem (Hornik, 1991)",
            "**Theoretical Foundation:**\n\n"
            "- **The Theorem:** A single hidden layer feedforward neural network with a non-polynomial activation function $\\sigma$ can approximate any continuous function $f \\in C(\\mathbb{R})$ on compact subsets to arbitrary precision $\\epsilon > 0$, given sufficient hidden neurons $H$.\n"
            "$$f(x) \\approx \\sum_{i=1}^H w_i \\sigma(v_i x + b_i) + b_0$$\n"
            "- **Geometric Basis Intuition:** Notice the purple dashed curves in the background! Each curve is an individual hidden neuron $w_i \\sigma(v_i x + b_i)$. With **ReLU**, each neuron adds a slope change hinge; with **Tanh**, each neuron adds a smooth sigmoidal step; with **SIREN**, each neuron adds a Fourier harmonic wave."
        )
        main_layout.addWidget(edu)

    def _on_target_changed(self, idx: int):
        names = ["sine", "harmonics", "damped", "runge", "square", "sawtooth"]
        self.fn_target = get_target_function(names[idx])
        self.y_train = self.fn_target(self.x_train)
        self._rebuild_model()

    def _on_arch_changed(self):
        self.num_layers = self.combo_depth.currentIndex() + 1
        self.hidden_dim = int(self.slider_neurons.value())

        act_text = self.combo_act.currentText()
        if "Tanh" in act_text:
            self.activation = "tanh"
        elif "ReLU" in act_text:
            self.activation = "relu"
        elif "GELU" in act_text:
            self.activation = "gelu"
        elif "SIREN" in act_text:
            self.activation = "siren"
        else:
            self.activation = "sigmoid"

        self._rebuild_model()

    def _update_lr(self, val: float):
        self.lr = val
        if self.optimizer:
            for g in self.optimizer.param_groups:
                g['lr'] = self.lr

    def _rebuild_model(self):
        self.model = FunctionApproximatorMLP(
            hidden_dim=self.hidden_dim,
            num_layers=self.num_layers,
            activation=self.activation
        ).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.step_count = 0
        self.badge_steps.set_value("0")
        self.badge_loss.set_value("0.0000")
        self.badge_r2.set_value("0.00%")
        self._update_display()

    def _toggle_training(self):
        if self.is_training:
            self.is_training = False
            self.train_timer.stop()
            self.btn_play.setText("▶ Start Training")
            self.btn_play.setStyleSheet(f"background-color: {COLOR_ACCENT_GREEN}; color: #000; font-weight: 800; padding: 10px 18px; border-radius: 8px;")
        else:
            self.is_training = True
            self.train_timer.start(25)
            self.btn_play.setText("⏸ Pause")
            self.btn_play.setStyleSheet(f"background-color: {COLOR_ACCENT_AMBER}; color: #000; font-weight: 800; padding: 10px 18px; border-radius: 8px;")

    def _training_step(self):
        if self.model is None:
            return

        self.model.train()
        x_t = torch.from_numpy(self.x_train).unsqueeze(-1).to(self.device)
        y_t = torch.from_numpy(self.y_train).unsqueeze(-1).to(self.device)

        # 5 micro-steps per UI tick for smooth convergence
        for _ in range(5):
            self.optimizer.zero_grad()
            preds = self.model(x_t)
            loss = self.criterion(preds, y_t)
            loss.backward()
            self.optimizer.step()
            self.step_count += 1

        self.badge_steps.set_value(str(self.step_count))
        self.badge_loss.set_value(f"{loss.item():.5f}")

        # Compute R2 score
        y_pred = preds.squeeze(-1).detach().cpu().numpy()
        ss_res = np.sum((self.y_train - y_pred) ** 2)
        ss_tot = np.sum((self.y_train - np.mean(self.y_train)) ** 2) + 1e-10
        r2 = max(0.0, 1.0 - (ss_res / ss_tot)) * 100.0
        self.badge_r2.set_value(f"{r2:.1f}%")

        self._update_display()

    def _update_display(self):
        if self.model is None:
            return
        basis, total = self.model.get_basis_neuron_outputs(self.x_train)
        self.canvas.set_data(self.y_train, total, basis)
