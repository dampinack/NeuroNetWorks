"""Module 1: Interactive MNIST Handwriting Recognition Canvas with Live Network Inference."""

import os
import numpy as np
import torch
import torch.nn.functional as F
from scipy.ndimage import center_of_mass

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout, QLabel,
    QPushButton, QProgressBar, QCheckBox, QComboBox, QFrame, QSplitter
)
from PyQt6.QtCore import Qt, pyqtSignal, QPoint, QSize
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QImage, QPixmap

from core.mnist_models import MNISTCNN, MNISTMLP
from core.device import get_device
from ui.theme import (
    COLOR_BG_CARD, COLOR_BG_SURFACE, COLOR_BG_HOVER, COLOR_BORDER, COLOR_ACCENT_CYAN,
    COLOR_ACCENT_GREEN, COLOR_ACCENT_PURPLE, COLOR_ACCENT_AMBER,
    COLOR_TEXT_PRIMARY, COLOR_TEXT_SECONDARY, COLOR_TEXT_MUTED
)
from ui.custom_widgets import Card, MetricBadge, LabeledSlider, EducationalPanel

WEIGHTS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models_weights", "mnist_cnn.pth")
SAMPLES_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "mnist_samples.npz")

class DrawingCanvas(QWidget):
    """High-resolution anti-aliased drawing canvas for digit input."""
    drawing_updated = pyqtSignal()

    def __init__(self, size: int = 280, parent=None):
        super().__init__(parent)
        self.canvas_size = size
        self.setFixedSize(size, size)
        self.brush_size = 20
        self.last_point = None

        # Pixel image representation (280x280)
        self.image = QImage(size, size, QImage.Format.Format_Grayscale8)
        self.clear_canvas()

    def clear_canvas(self):
        self.image.fill(QColor(0, 0, 0))  # Black background
        self.update()
        self.drawing_updated.emit()

    def set_brush_size(self, size: int):
        self.brush_size = size

    def mousePressEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.last_point = event.position()
            self._draw_stroke(self.last_point, self.last_point)

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.MouseButton.LeftButton and self.last_point:
            curr_point = event.position()
            self._draw_stroke(self.last_point, curr_point)
            self.last_point = curr_point

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.LeftButton:
            self.last_point = None
            self.drawing_updated.emit()

    def _draw_stroke(self, p1, p2):
        painter = QPainter(self.image)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        pen = QPen(QColor(255, 255, 255), self.brush_size, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap, Qt.PenJoinStyle.RoundJoin)
        painter.setPen(pen)
        painter.drawLine(p1.toPoint(), p2.toPoint())
        painter.end()
        self.update()
        self.drawing_updated.emit()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        # Background border
        painter.drawImage(0, 0, self.image)
        painter.setPen(QPen(QColor(COLOR_BORDER), 2))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(0, 0, self.width(), self.height(), 8, 8)

    def load_array(self, arr_28x28: np.ndarray):
        """Loads a 28x28 float array [0, 1] onto the drawing canvas."""
        scaled = (np.clip(arr_28x28, 0.0, 1.0) * 255).astype(np.uint8)
        img = QImage(scaled.data, 28, 28, 28, QImage.Format.Format_Grayscale8)
        self.image = img.scaled(self.canvas_size, self.canvas_size, Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation)
        self.update()
        self.drawing_updated.emit()

    def get_mnist_tensor(self, center_mass: bool = True) -> np.ndarray:
        """Converts canvas drawing to standard MNIST-normalized 28x28 float matrix."""
        # Scale 280x280 down to 28x28
        scaled_img = self.image.scaled(28, 28, Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.SmoothTransformation)
        ptr = scaled_img.constBits()
        ptr.setsize(28 * 28)
        raw_arr = np.frombuffer(ptr, np.uint8).reshape((28, 28)).astype(np.float32) / 255.0

        if not center_mass or np.max(raw_arr) < 0.05:
            return raw_arr

        # Center-of-mass alignment: standard MNIST normalization protocol
        cy, cx = center_of_mass(raw_arr)
        if np.isnan(cx) or np.isnan(cy):
            return raw_arr

        shift_x = int(round(13.5 - cx))
        shift_y = int(round(13.5 - cy))

        shifted = np.roll(raw_arr, shift_y, axis=0)
        shifted = np.roll(shifted, shift_x, axis=1)

        # Clear wrap-around border artifacts
        if shift_y > 0:
            shifted[:shift_y, :] = 0
        elif shift_y < 0:
            shifted[shift_y:, :] = 0
        if shift_x > 0:
            shifted[:, :shift_x] = 0
        elif shift_x < 0:
            shifted[:, shift_x:] = 0

        return shifted


class MNISTView(QWidget):
    """Main interactive MNIST handwriting recognition screen."""
    telemetry_ready = pyqtSignal(dict)
    open_auxiliary_requested = pyqtSignal()

    def __init__(self, parent=None):
        super().__init__(parent)
        self.device = get_device()
        self.model_cnn = MNISTCNN().to(self.device)
        self.model_mlp = MNISTMLP().to(self.device)
        self.active_model_name = "CNN"

        # Load weights if available
        if os.path.exists(WEIGHTS_PATH):
            try:
                self.model_cnn.load_state_dict(torch.load(WEIGHTS_PATH, map_location=self.device))
                self.model_cnn.eval()
            except Exception as e:
                print("Error loading CNN weights:", e)

        # Load offline sample dataset
        self.samples_data = None
        if os.path.exists(SAMPLES_PATH):
            try:
                data = np.load(SAMPLES_PATH)
                self.samples_data = (data["images"], data["labels"])
            except Exception as e:
                print("Error loading samples:", e)

        self._init_ui()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(16)

        # Header Title Banner
        hdr_layout = QHBoxLayout()
        title_box = QVBoxLayout()
        title = QLabel("MNIST HANDWRITING DIGIT RECOGNIZER")
        title.setObjectName("HeaderTitle")
        title.setStyleSheet("font-size: 20px; font-weight: 800; color: #FFFFFF;")
        subtitle = QLabel("Real-time PyTorch Deep Convolutional Inference with Centroid Normalization & Live Telemetry")
        subtitle.setStyleSheet(f"font-size: 12px; color: {COLOR_TEXT_SECONDARY};")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        hdr_layout.addLayout(title_box)
        hdr_layout.addStretch()

        self.btn_aux = QPushButton("🔍 Open Neural Activity Inspector")
        self.btn_aux.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLOR_BG_SURFACE};
                border: 1px solid {COLOR_ACCENT_CYAN};
                color: {COLOR_ACCENT_CYAN};
                font-weight: 700;
                padding: 10px 18px;
                border-radius: 8px;
            }}
            QPushButton:hover {{
                background-color: {COLOR_ACCENT_CYAN};
                color: #000000;
            }}
        """)
        self.btn_aux.clicked.connect(self.open_auxiliary_requested.emit)
        hdr_layout.addWidget(self.btn_aux)
        main_layout.addLayout(hdr_layout)

        # Splitter Layout (Left: Canvas & Controls, Right: Prediction Probabilities & Telemetry)
        content_layout = QHBoxLayout()
        content_layout.setSpacing(20)

        # Left Column: Canvas Card
        canvas_card = Card("INTERACTIVE DRAWING PAD (280×280)")
        
        self.canvas = DrawingCanvas(280)
        self.canvas.drawing_updated.connect(self._on_canvas_modified)
        canvas_card.addWidget(self.canvas)

        # Canvas Controls
        controls_layout = QHBoxLayout()
        self.btn_clear = QPushButton("🗑️ Clear")
        self.btn_clear.clicked.connect(self.canvas.clear_canvas)
        controls_layout.addWidget(self.btn_clear)

        self.check_center = QCheckBox("Centroid Alignment")
        self.check_center.setChecked(True)
        self.check_center.setStyleSheet(f"color: {COLOR_TEXT_SECONDARY}; font-weight: 600;")
        self.check_center.toggled.connect(self.run_inference)
        controls_layout.addWidget(self.check_center)
        canvas_card.addLayout(controls_layout)

        # Brush size slider
        self.brush_slider = LabeledSlider("Brush Stroke Width", 10, 34, 20, step=2)
        self.brush_slider.valueChanged.connect(lambda v: self.canvas.set_brush_size(int(v)))
        canvas_card.addWidget(self.brush_slider)

        # Sample Digits Carousel (0 - 9)
        sample_card_layout = QVBoxLayout()
        sample_lbl = QLabel("PRE-LOADED REAL MNIST TEST SAMPLES:")
        sample_lbl.setStyleSheet(f"font-size: 10px; font-weight: 700; color: {COLOR_TEXT_MUTED};")
        sample_card_layout.addWidget(sample_lbl)

        sample_btn_layout = QHBoxLayout()
        sample_btn_layout.setSpacing(4)
        for digit in range(10):
            btn = QPushButton(str(digit))
            btn.setFixedSize(26, 26)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {COLOR_BG_SURFACE};
                    border: 1px solid {COLOR_BORDER};
                    border-radius: 4px;
                    font-size: 11px;
                    font-weight: 700;
                    padding: 0;
                }}
                QPushButton:hover {{
                    border-color: {COLOR_ACCENT_CYAN};
                    color: {COLOR_ACCENT_CYAN};
                }}
            """)
            btn.clicked.connect(lambda checked, d=digit: self._load_sample_digit(d))
            sample_btn_layout.addWidget(btn)

        btn_rand = QPushButton("🎲")
        btn_rand.setToolTip("Random Sample")
        btn_rand.setFixedSize(26, 26)
        btn_rand.clicked.connect(lambda: self._load_sample_digit(np.random.randint(0, 10)))
        sample_btn_layout.addWidget(btn_rand)
        sample_card_layout.addLayout(sample_btn_layout)
        canvas_card.addLayout(sample_card_layout)

        content_layout.addWidget(canvas_card)

        # Right Column: Prediction Metrics & Probability Spectrum
        metrics_card = Card("PYTORCH PREDICTION & CONFIDENCE DISTRIBUTION")

        # Top Badge Indicators
        top_badges = QHBoxLayout()
        self.badge_prediction = MetricBadge("WINNING CLASS", "--", "", COLOR_ACCENT_GREEN)
        self.badge_confidence = MetricBadge("CONFIDENCE", "--", "%", COLOR_ACCENT_CYAN)
        self.badge_entropy = MetricBadge("UNCERTAINTY", "--", "bits", COLOR_ACCENT_AMBER)
        top_badges.addWidget(self.badge_prediction)
        top_badges.addWidget(self.badge_confidence)
        top_badges.addWidget(self.badge_entropy)
        metrics_card.addLayout(top_badges)

        # Model Selector
        model_row = QHBoxLayout()
        model_lbl = QLabel("Active Architecture:")
        model_lbl.setStyleSheet(f"font-weight: 600; color: {COLOR_TEXT_SECONDARY};")
        model_row.addWidget(model_lbl)

        self.model_combo = QComboBox()
        self.model_combo.addItems(["Deep CNN (Conv2D + MaxPool + Dense)", "MLP Baseline (784 -> 128 -> 64 -> 10)"])
        self.model_combo.currentIndexChanged.connect(self._on_model_changed)
        model_row.addWidget(self.model_combo, stretch=1)
        metrics_card.addLayout(model_row)

        # 10 Class Probability Bars
        bars_layout = QVBoxLayout()
        bars_layout.setSpacing(6)
        self.prob_bars = []
        self.prob_labels = []

        for digit in range(10):
            row = QHBoxLayout()
            row.setSpacing(8)

            digit_lbl = QLabel(f"Digit {digit}:")
            digit_lbl.setFixedWidth(54)
            digit_lbl.setStyleSheet("font-weight: 700; font-size: 12px;")
            row.addWidget(digit_lbl)

            pbar = QProgressBar()
            pbar.setRange(0, 1000)
            pbar.setValue(0)
            pbar.setTextVisible(False)
            pbar.setFixedHeight(12)
            row.addWidget(pbar, stretch=1)

            pct_lbl = QLabel("0.0%")
            pct_lbl.setFixedWidth(48)
            pct_lbl.setAlignment(Qt.AlignmentFlag.AlignRight)
            pct_lbl.setStyleSheet(f"color: {COLOR_TEXT_MUTED}; font-size: 11px;")
            row.addWidget(pct_lbl)

            bars_layout.addLayout(row)
            self.prob_bars.append(pbar)
            self.prob_labels.append(pct_lbl)

        metrics_card.addLayout(bars_layout)
        content_layout.addWidget(metrics_card, stretch=1)
        main_layout.addLayout(content_layout)

        # Educational Insights Panel
        edu = EducationalPanel(
            "Convolutional Neural Networks & Feature Hierarchies",
            "**Key Mathematical Principles:**\n\n"
            "- **Weight Sharing & Translation Invariance:** Unlike fully-connected networks where each pixel has separate weights, 2D convolutional kernels slide across the image, recognizing local edges regardless of position.\n"
            "- **Centroid Normalization:** Real MNIST images are centered by center-of-mass $(\\bar{x}, \\bar{y}) = \\frac{1}{M}\\sum x_i I(x_i)$. Toggling Centroid Alignment ensures your custom hand-drawn digit aligns with MNIST training distribution.\n"
            "- **Information Entropy (Uncertainty):** Measured as $H(p) = -\\sum_{i=0}^9 p_i \\log_2 p_i$. A confident prediction has entropy near 0 bits, while high entropy indicates confusion between ambiguous digits (e.g. 4 vs 9)."
        )
        main_layout.addWidget(edu)

    def _on_canvas_modified(self):
        self.run_inference()

    def _on_model_changed(self, idx: int):
        self.active_model_name = "CNN" if idx == 0 else "MLP"
        self.run_inference()

    def _load_sample_digit(self, digit: int):
        if self.samples_data is None:
            return
        images, labels = self.samples_data
        matching = np.where(labels == digit)[0]
        if len(matching) > 0:
            idx = np.random.choice(matching)
            img = images[idx]
            self.canvas.load_array(img)

    def run_inference(self):
        """Runs PyTorch forward pass, updates UI probability bars, and pushes telemetry to auxiliary inspector."""
        tensor_28x28 = self.canvas.get_mnist_tensor(center_mass=self.check_center.isChecked())

        # If canvas is completely blank
        if np.max(tensor_28x28) < 0.05:
            self.badge_prediction.set_value("--")
            self.badge_confidence.set_value("--")
            self.badge_entropy.set_value("--")
            for bar, lbl in zip(self.prob_bars, self.prob_labels):
                bar.setValue(0)
                lbl.setText("0.0%")
            return

        x_tensor = torch.from_numpy(tensor_28x28).unsqueeze(0).unsqueeze(0).to(self.device)

        if self.active_model_name == "CNN":
            telemetry = self.model_cnn.forward_with_activations(x_tensor)
            saliency = self.model_cnn.compute_saliency(x_tensor)
            telemetry["saliency"] = saliency
        else:
            telemetry = self.model_mlp.forward_with_activations(x_tensor)

        probs = telemetry["probabilities"]
        top_digit = int(np.argmax(probs))
        top_conf = float(probs[top_digit]) * 100.0

        # Shannon Entropy in bits
        valid_probs = probs[probs > 1e-12]
        entropy = float(-np.sum(valid_probs * np.log2(valid_probs)))

        # Update metrics badges
        self.badge_prediction.set_value(str(top_digit))
        self.badge_confidence.set_value(f"{top_conf:.1f}")
        self.badge_entropy.set_value(f"{entropy:.2f}")

        # Update probability bars
        for d in range(10):
            p = float(probs[d])
            self.prob_bars[d].setValue(int(p * 1000))
            self.prob_labels[d].setText(f"{p * 100:.1f}%")

            if d == top_digit:
                self.prob_bars[d].setStyleSheet(f"""
                    QProgressBar::chunk {{
                        background: {COLOR_ACCENT_CYAN};
                    }}
                """)
                self.prob_labels[d].setStyleSheet(f"color: {COLOR_ACCENT_CYAN}; font-weight: 700; font-size: 11px;")
            else:
                self.prob_bars[d].setStyleSheet(f"""
                    QProgressBar::chunk {{
                        background: {COLOR_BG_HOVER};
                    }}
                """)
                self.prob_labels[d].setStyleSheet(f"color: {COLOR_TEXT_MUTED}; font-size: 11px;")

        # Emit telemetry to Auxiliary window
        self.telemetry_ready.emit(telemetry)
