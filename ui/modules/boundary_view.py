"""Module 2: 2D Decision Boundary Playground (Multi-Layer Perceptron)."""

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from typing import Tuple, List

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QFrame, QSplitter
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QPointF
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QImage, QPixmap

from core.boundary_models import DecisionBoundaryMLP, generate_2d_dataset
from core.device import get_device
from ui.theme import (
    COLOR_BG_DARK, COLOR_BG_CARD, COLOR_BG_SURFACE, COLOR_BORDER,
    COLOR_ACCENT_CYAN, COLOR_ACCENT_PURPLE, COLOR_ACCENT_GREEN,
    COLOR_ACCENT_AMBER, COLOR_ACCENT_RED, COLOR_TEXT_PRIMARY,
    COLOR_TEXT_SECONDARY, COLOR_TEXT_MUTED
)
from ui.custom_widgets import Card, MetricBadge, LabeledSlider, EducationalPanel

class BoundaryCanvas(QWidget):
    """Interactive 2D coordinate plane with live decision probability contour and data points."""
    point_added = pyqtSignal(float, float, int)

    def __init__(self, size: int = 420, parent=None):
        super().__init__(parent)
        self.canvas_size = size
        self.setFixedSize(size, size)

        self.grid_res = 50  # 50x50 evaluation grid for real-time 60fps rendering
        self.grid_probs = None
        self.X_data = np.empty((0, 2), dtype=np.float32)
        self.y_data = np.empty((0,), dtype=np.int64)

    def set_data(self, X: np.ndarray, y: np.ndarray):
        self.X_data = X
        self.y_data = y
        self.update()

    def set_heatmap(self, probs_grid: np.ndarray):
        self.grid_probs = probs_grid
        self.update()

    def _coord_to_pixel(self, x: float, y: float) -> QPointF:
        """Maps [-2.2, 2.2] coordinates to canvas pixels."""
        w, h = self.width(), self.height()
        px = (x + 2.2) / 4.4 * w
        py = (2.2 - y) / 4.4 * h
        return QPointF(px, py)

    def _pixel_to_coord(self, px: float, py: float) -> Tuple:
        w, h = self.width(), self.height()
        x = (px / w) * 4.4 - 2.2
        y = 2.2 - (py / h) * 4.4
        return float(x), float(y)

    def mousePressEvent(self, event):
        px = event.position().x()
        py = event.position().y()
        x, y = self._pixel_to_coord(px, py)
        # Left click = class 0 (Cyan), Right click = class 1 (Purple)
        cls = 0 if event.button() == Qt.MouseButton.LeftButton else 1
        self.point_added.emit(x, y, cls)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()

        # 1. Draw Decision Boundary Surface Heatmap
        if self.grid_probs is not None:
            gw, gh = self.grid_probs.shape
            img = QImage(gw, gh, QImage.Format.Format_RGB32)
            for gy in range(gh):
                for gx in range(gw):
                    p = float(self.grid_probs[gy, gx])  # prob of class 1
                    # Gradient: Cyan (Class 0, p=0) -> Dark Navy (p=0.5) -> Magenta/Purple (Class 1, p=1)
                    if p < 0.5:
                        factor = (0.5 - p) / 0.5
                        r = int(10 + factor * 0)
                        g = int(20 + factor * 160)
                        b = int(40 + factor * 180)
                    else:
                        factor = (p - 0.5) / 0.5
                        r = int(10 + factor * 180)
                        g = int(20 + factor * 30)
                        b = int(40 + factor * 170)
                    img.setPixelColor(gx, gy, QColor(r, g, b))

            scaled_img = img.scaled(w, h, Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation)
            painter.drawImage(0, 0, scaled_img)
        else:
            painter.fillRect(0, 0, w, h, QColor(COLOR_BG_DARK))

        # 2. Draw subtle coordinate grid lines
        painter.setPen(QPen(QColor("#334155"), 1, Qt.PenStyle.DashLine))
        center_x = w / 2
        center_y = h / 2
        painter.drawLine(int(center_x), 0, int(center_x), h)
        painter.drawLine(0, int(center_y), w, int(center_y))

        # 3. Draw Data Points
        if len(self.X_data) > 0:
            for pt, label in zip(self.X_data, self.y_data):
                pix_pt = self._coord_to_pixel(pt[0], pt[1])
                if label == 0:
                    painter.setBrush(QBrush(QColor(COLOR_ACCENT_CYAN)))
                    painter.setPen(QPen(QColor("#FFFFFF"), 1.5))
                else:
                    painter.setBrush(QBrush(QColor(COLOR_ACCENT_PURPLE)))
                    painter.setPen(QPen(QColor("#FFFFFF"), 1.5))

                painter.drawEllipse(pix_pt, 4.5, 4.5)

        # Border
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor(COLOR_BORDER), 2))
        painter.drawRoundedRect(0, 0, w, h, 8, 8)


class BoundaryView(QWidget):
    """Main interactive 2D Decision Boundary Playground view."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.device = get_device()
        self.is_training = False
        self.epoch = 0

        # Model & Training state
        self.X = None
        self.y = None
        self.model = None
        self.optimizer = None
        self.criterion = nn.CrossEntropyLoss()

        self.lr = 0.03
        self.weight_decay = 0.0001
        self.hidden_dim = 16
        self.num_layers = 2
        self.activation = "relu"
        self.optimizer_name = "adam"

        # Evaluation grid coordinates for [-2.2, 2.2]
        res = 50
        xs = np.linspace(-2.2, 2.2, res)
        ys = np.linspace(2.2, -2.2, res)  # inverted for y-down screen coords
        grid_x, grid_y = np.meshgrid(xs, ys)
        self.grid_coords = np.column_stack([grid_x.ravel(), grid_y.ravel()]).astype(np.float32)
        self.grid_shape = (res, res)

        self._init_ui()
        self._load_dataset("spirals")
        self._rebuild_model()

        # Training timer (30 updates per second)
        self.train_timer = QTimer(self)
        self.train_timer.timeout.connect(self._training_step)

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(16)

        # Header Title
        hdr_layout = QHBoxLayout()
        title_box = QVBoxLayout()
        title = QLabel("2D DECISION BOUNDARY PLAYGROUND")
        title.setStyleSheet("font-size: 20px; font-weight: 800; color: #FFFFFF;")
        subtitle = QLabel("Non-Linear Classification & Representation Learning with PyTorch Multi-Layer Perceptrons")
        subtitle.setStyleSheet(f"font-size: 12px; color: {COLOR_TEXT_SECONDARY};")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        hdr_layout.addLayout(title_box)
        hdr_layout.addStretch()

        # Control Buttons (Play, Pause, Step, Reset)
        self.btn_play = QPushButton("▶ Start Training")
        self.btn_play.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLOR_ACCENT_GREEN};
                color: #000000;
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
        self.btn_step.clicked.connect(self._single_step)
        hdr_layout.addWidget(self.btn_step)

        self.btn_reset = QPushButton("↺ Reset")
        self.btn_reset.clicked.connect(self._reset_network)
        hdr_layout.addWidget(self.btn_reset)

        main_layout.addLayout(hdr_layout)

        # Splitter Layout
        content_layout = QHBoxLayout()
        content_layout.setSpacing(20)

        # Left: Interactive Canvas Card
        canvas_card = Card("INTERACTIVE 2D DECISION MANIFOLD")
        self.canvas = BoundaryCanvas(420)
        self.canvas.point_added.connect(self._add_user_point)
        canvas_card.addWidget(self.canvas)

        tip_lbl = QLabel("💡 Tip: Click Left mouse to add Class 0 (Cyan), Right mouse for Class 1 (Purple).")
        tip_lbl.setStyleSheet(f"font-size: 11px; color: {COLOR_TEXT_MUTED};")
        canvas_card.addWidget(tip_lbl)
        content_layout.addWidget(canvas_card)

        # Right: Parameters & Metrics Panel
        right_panel = QVBoxLayout()
        right_panel.setSpacing(12)

        # Telemetry Metrics Card
        metrics_card = Card("TRAINING TELEMETRY")
        metric_row = QHBoxLayout()
        self.badge_epoch = MetricBadge("EPOCH", "0", "", COLOR_TEXT_PRIMARY)
        self.badge_loss = MetricBadge("LOSS", "0.000", "", COLOR_ACCENT_CYAN)
        self.badge_accuracy = MetricBadge("ACCURACY", "0.0", "%", COLOR_ACCENT_GREEN)
        metric_row.addWidget(self.badge_epoch)
        metric_row.addWidget(self.badge_loss)
        metric_row.addWidget(self.badge_accuracy)
        metrics_card.addLayout(metric_row)
        right_panel.addWidget(metrics_card)

        # Dataset Configuration Card
        data_card = Card("DATASET TOPOLOGY")
        data_row = QHBoxLayout()
        data_lbl = QLabel("Benchmark Pattern:")
        data_lbl.setStyleSheet(f"font-weight: 600; color: {COLOR_TEXT_SECONDARY};")
        data_row.addWidget(data_lbl)

        self.combo_dataset = QComboBox()
        self.combo_dataset.addItems(["Interlocking Spirals", "Two Moons", "Concentric Circles", "4-Quadrant XOR", "Gaussian Clusters"])
        self.combo_dataset.currentIndexChanged.connect(self._on_dataset_changed)
        data_row.addWidget(self.combo_dataset, stretch=1)
        data_card.addLayout(data_row)

        self.slider_noise = LabeledSlider("Noise Jitter", 0.0, 0.4, 0.1, step=0.02, is_float=True)
        self.slider_noise.valueChanged.connect(lambda _: self._on_dataset_changed(self.combo_dataset.currentIndex()))
        data_card.addWidget(self.slider_noise)
        right_panel.addWidget(data_card)

        # Hyperparameters Card
        hyp_card = Card("NETWORK ARCHITECTURE & OPTIMIZATION")

        row_arch = QHBoxLayout()
        lbl_arch = QLabel("Hidden Layers:")
        lbl_arch.setStyleSheet(f"font-weight: 600; color: {COLOR_TEXT_SECONDARY};")
        row_arch.addWidget(lbl_arch)

        self.combo_layers = QComboBox()
        self.combo_layers.addItems(["1 Hidden Layer", "2 Hidden Layers", "3 Hidden Layers"])
        self.combo_layers.setCurrentIndex(1)
        self.combo_layers.currentIndexChanged.connect(self._on_architecture_changed)
        row_arch.addWidget(self.combo_layers, stretch=1)
        hyp_card.addLayout(row_arch)

        self.slider_neurons = LabeledSlider("Neurons per Layer", 4, 64, 16, step=4)
        self.slider_neurons.valueChanged.connect(self._on_architecture_changed)
        hyp_card.addWidget(self.slider_neurons)

        row_act = QHBoxLayout()
        lbl_act = QLabel("Activation Function:")
        lbl_act.setStyleSheet(f"font-weight: 600; color: {COLOR_TEXT_SECONDARY};")
        row_act.addWidget(lbl_act)

        self.combo_act = QComboBox()
        self.combo_act.addItems(["ReLU", "Tanh", "Sigmoid", "GELU", "LeakyReLU"])
        self.combo_act.currentIndexChanged.connect(self._on_architecture_changed)
        row_act.addWidget(self.combo_act, stretch=1)
        hyp_card.addLayout(row_act)

        self.slider_lr = LabeledSlider("Learning Rate", 0.005, 0.1, 0.03, step=0.005, is_float=True)
        self.slider_lr.valueChanged.connect(self._update_lr)
        hyp_card.addWidget(self.slider_lr)

        right_panel.addWidget(hyp_card)
        content_layout.addLayout(right_panel, stretch=1)
        main_layout.addLayout(content_layout)

        # Educational Panel
        edu = EducationalPanel(
            "Universal Approximation & Non-Linear Boundary Formation",
            "**Why Single Perceptrons Fail on Spirals:**\n\n"
            "- **The XOR / Spiral Paradox:** A single linear layer computes $y = \\mathbf{w}^T \\mathbf{x} + b$, which defines a straight planar hyperplane. Non-convex, intertwined topologies (like spirals and concentric rings) are fundamentally linearly inseparable.\n"
            "- **Layered Representation:** Hidden layers with non-linear activations (ReLU, Tanh, GELU) warp and project the 2D input space into higher-dimensional representations where the classes become linearly separable.\n"
            "- **Activation Geometry:** Notice how **ReLU** produces sharp, polygonal geometric boundaries, whereas **Tanh/GELU** produces smooth, flowing equipotential curves."
        )
        main_layout.addWidget(edu)

    def _load_dataset(self, name: str):
        type_map = {
            "Interlocking Spirals": "spirals",
            "Two Moons": "moons",
            "Concentric Circles": "circles",
            "4-Quadrant XOR": "xor",
            "Gaussian Clusters": "gaussian"
        }
        dataset_key = type_map.get(name, "spirals")
        noise = self.slider_noise.value()
        self.X, self.y = generate_2d_dataset(dataset_key, n_samples=360, noise=noise)
        self.canvas.set_data(self.X, self.y)
        self._evaluate_and_render_heatmap()

    def _on_dataset_changed(self, idx: int):
        self._load_dataset(self.combo_dataset.currentText())
        self._reset_network()

    def _on_architecture_changed(self):
        self.num_layers = self.combo_layers.currentIndex() + 1
        self.hidden_dim = int(self.slider_neurons.value())
        self.activation = self.combo_act.currentText().lower()
        self._rebuild_model()

    def _update_lr(self, val: float):
        self.lr = val
        if self.optimizer:
            for g in self.optimizer.param_groups:
                g['lr'] = self.lr

    def _rebuild_model(self):
        hidden_dims = [self.hidden_dim] * self.num_layers
        self.model = DecisionBoundaryMLP(hidden_dims, activation=self.activation).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        self.epoch = 0
        self.badge_epoch.set_value("0")
        self.badge_loss.set_value("0.000")
        self.badge_accuracy.set_value("0.0")
        self._evaluate_and_render_heatmap()

    def _reset_network(self):
        self._rebuild_model()

    def _toggle_training(self):
        if self.is_training:
            self.is_training = False
            self.train_timer.stop()
            self.btn_play.setText("▶ Start Training")
            self.btn_play.setStyleSheet(f"background-color: {COLOR_ACCENT_GREEN}; color: #000; font-weight: 800; padding: 10px 18px; border-radius: 8px;")
        else:
            self.is_training = True
            self.train_timer.start(25)  # 40 fps
            self.btn_play.setText("⏸ Pause")
            self.btn_play.setStyleSheet(f"background-color: {COLOR_ACCENT_AMBER}; color: #000; font-weight: 800; padding: 10px 18px; border-radius: 8px;")

    def _single_step(self):
        self._training_step()

    def _add_user_point(self, x: float, y: float, cls: int):
        new_pt = np.array([[x, y]], dtype=np.float32)
        new_lbl = np.array([cls], dtype=np.int64)
        self.X = np.vstack([self.X, new_pt])
        self.y = np.concatenate([self.y, new_lbl])
        self.canvas.set_data(self.X, self.y)
        self._evaluate_and_render_heatmap()

    def _training_step(self):
        if self.model is None or len(self.X) == 0:
            return

        self.model.train()
        X_t = torch.from_numpy(self.X).to(self.device)
        y_t = torch.from_numpy(self.y).to(self.device)

        self.optimizer.zero_grad()
        logits = self.model(X_t)
        loss = self.criterion(logits, y_t)
        loss.backward()
        self.optimizer.step()

        self.epoch += 1

        # Accuracy
        preds = logits.argmax(dim=-1)
        acc = float((preds == y_t).float().mean().item()) * 100.0

        self.badge_epoch.set_value(str(self.epoch))
        self.badge_loss.set_value(f"{loss.item():.4f}")
        self.badge_accuracy.set_value(f"{acc:.1f}")

        # Refresh decision surface every 2 epochs
        if self.epoch % 2 == 0:
            self._evaluate_and_render_heatmap()

    def _evaluate_and_render_heatmap(self):
        """Computes probabilities on 2D grid and updates canvas."""
        if self.model is None:
            return
        probs = self.model.predict_proba(self.grid_coords)
        grid_probs = probs.reshape(self.grid_shape)
        self.canvas.set_heatmap(grid_probs)
