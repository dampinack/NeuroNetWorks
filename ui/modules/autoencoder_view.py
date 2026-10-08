"""Module 3: Autoencoder Latent Space Explorer & Denoising (Unsupervised Learning)."""

import os
import numpy as np
import torch
from typing import Tuple, List

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QFrame, QTabWidget, QGridLayout, QSplitter
)
from PyQt6.QtCore import Qt, pyqtSignal, QPointF
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QImage, QPixmap

from core.autoencoder_models import (
    MNIST2DAutoencoder, add_noise_to_image, compute_reconstruction_metrics
)
from core.device import get_device
from ui.theme import (
    COLOR_BG_DARK, COLOR_BG_CARD, COLOR_BG_SURFACE, COLOR_BORDER,
    COLOR_ACCENT_CYAN, COLOR_ACCENT_PURPLE, COLOR_ACCENT_GREEN,
    COLOR_ACCENT_AMBER, COLOR_ACCENT_RED, COLOR_TEXT_PRIMARY,
    COLOR_TEXT_SECONDARY, COLOR_TEXT_MUTED
)
from ui.custom_widgets import Card, MetricBadge, LabeledSlider, EducationalPanel

WEIGHTS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models_weights", "mnist_autoencoder.pth")
SAMPLES_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "mnist_samples.npz")

DIGIT_COLORS = [
    QColor("#EF4444"),  # 0 Red
    QColor("#F97316"),  # 1 Orange
    QColor("#FBBF24"),  # 2 Amber
    QColor("#10B981"),  # 3 Green
    QColor("#06B6D4"),  # 4 Cyan
    QColor("#3B82F6"),  # 5 Blue
    QColor("#8B5CF6"),  # 6 Indigo
    QColor("#EC4899"),  # 7 Pink
    QColor("#A855F7"),  # 8 Purple
    QColor("#14B8A6"),  # 9 Teal
]

class LatentSpaceCanvas(QWidget):
    """Interactive 2D Cartesian plane of the Autoencoder latent bottleneck (z1, z2)."""
    point_selected = pyqtSignal(float, float)

    def __init__(self, size: int = 380, parent=None):
        super().__init__(parent)
        self.canvas_size = size
        self.setFixedSize(size, size)

        self.z_points = None
        self.labels = None
        self.cursor_z = [0.0, 0.0]
        self.z_range = 6.0  # [-6, 6] coordinate bounds

    def set_latent_data(self, z_points: np.ndarray, labels: np.ndarray):
        self.z_points = z_points
        self.labels = labels
        self.update()

    def set_cursor(self, z1: float, z2: float):
        self.cursor_z = [z1, z2]
        self.update()

    def _z_to_pixel(self, z1: float, z2: float) -> QPointF:
        w, h = self.width(), self.height()
        px = (z1 + self.z_range) / (2 * self.z_range) * w
        py = (self.z_range - z2) / (2 * self.z_range) * h
        return QPointF(px, py)

    def _pixel_to_z(self, px: float, py: float) -> Tuple:
        w, h = self.width(), self.height()
        z1 = (px / w) * (2 * self.z_range) - self.z_range
        z2 = self.z_range - (py / h) * (2 * self.z_range)
        return float(z1), float(z2)

    def mousePressEvent(self, event):
        self._handle_mouse(event)

    def mouseMoveEvent(self, event):
        if event.buttons() & Qt.MouseButton.LeftButton:
            self._handle_mouse(event)

    def _handle_mouse(self, event):
        px = event.position().x()
        py = event.position().y()
        z1, z2 = self._pixel_to_z(px, py)
        self.cursor_z = [z1, z2]
        self.update()
        self.point_selected.emit(z1, z2)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()

        # Background
        painter.fillRect(0, 0, w, h, QColor(COLOR_BG_DARK))

        # Axis Lines
        painter.setPen(QPen(QColor("#334155"), 1, Qt.PenStyle.DashLine))
        center_x = w / 2
        center_y = h / 2
        painter.drawLine(int(center_x), 0, int(center_x), h)
        painter.drawLine(0, int(center_y), w, int(center_y))

        # Scatter plot of latent projections
        if self.z_points is not None:
            for pt, lbl in zip(self.z_points, self.labels):
                pix_pt = self._z_to_pixel(pt[0], pt[1])
                color = DIGIT_COLORS[int(lbl) % 10]
                painter.setBrush(QBrush(color))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawEllipse(pix_pt, 2.5, 2.5)

        # Crosshair cursor at active (z1, z2)
        cur_pt = self._z_to_pixel(self.cursor_z[0], self.cursor_z[1])
        # Outer glow ring
        glow = QColor(COLOR_ACCENT_CYAN)
        glow.setAlpha(80)
        painter.setBrush(QBrush(glow))
        painter.setPen(QPen(QColor(COLOR_ACCENT_CYAN), 2))
        painter.drawEllipse(cur_pt, 9, 9)
        # Center dot
        painter.setBrush(QBrush(QColor("#FFFFFF")))
        painter.drawEllipse(cur_pt, 3, 3)

        # Border
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor(COLOR_BORDER), 2))
        painter.drawRoundedRect(0, 0, w, h, 8, 8)


class AutoencoderView(QWidget):
    """Main Autoencoder Latent Space Explorer & Image Denoising View."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.device = get_device()
        self.model = MNIST2DAutoencoder().to(self.device)

        if os.path.exists(WEIGHTS_PATH):
            try:
                self.model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=self.device))
                self.model.eval()
            except Exception as e:
                print("Error loading autoencoder weights:", e)

        # Offline test digits
        self.sample_images = None
        self.sample_labels = None
        self.sample_latents = None

        if os.path.exists(SAMPLES_PATH):
            try:
                data = np.load(SAMPLES_PATH)
                self.sample_images = data["images"]  # [1000, 28, 28]
                self.sample_labels = data["labels"]  # [1000]
                self._compute_offline_latents()
            except Exception as e:
                print("Error loading sample dataset:", e)

        self.current_denoise_img = None
        if self.sample_images is not None:
            self.current_denoise_img = self.sample_images[0]

        self._init_ui()

    def _compute_offline_latents(self):
        """Computes 2D latent coordinates for all test samples."""
        self.model.eval()
        with torch.no_grad():
            imgs_t = torch.from_numpy(self.sample_images).to(self.device)
            z_t = self.model.encode(imgs_t)
            self.sample_latents = z_t.cpu().numpy()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(16)

        # Header Title
        hdr_layout = QHBoxLayout()
        title_box = QVBoxLayout()
        title = QLabel("AUTOENCODER LATENT SPACE EXPLORER & DENOISING")
        title.setStyleSheet("font-size: 20px; font-weight: 800; color: #FFFFFF;")
        subtitle = QLabel("Unsupervised Representation Learning, Generative Manifold Traversal & Image Restoration")
        subtitle.setStyleSheet(f"font-size: 12px; color: {COLOR_TEXT_SECONDARY};")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        hdr_layout.addLayout(title_box)
        hdr_layout.addStretch()
        main_layout.addLayout(hdr_layout)

        # Main Sub-Tabs: 1. Latent Space Explorer, 2. Denoising Laboratory
        self.sub_tabs = QTabWidget()
        main_layout.addWidget(self.sub_tabs, stretch=1)

        # Tab 1: Latent Space Explorer
        self.tab_latent = self._create_latent_tab()
        self.sub_tabs.addTab(self.tab_latent, "🌌 2D Latent Bottleneck Manifold")

        # Tab 2: Denoising Laboratory
        self.tab_denoise = self._create_denoising_tab()
        self.sub_tabs.addTab(self.tab_denoise, "✨ Denoising & Reconstruction Lab")

        # Educational Panel
        edu = EducationalPanel(
            "Autoencoder Bottlenecks & Denoising Manifolds",
            "**Key Concepts in Unsupervised Autoencoders:**\n\n"
            "- **Information Bottleneck:** Compressing 784 pixels ($28\\times 28$) down to a 2D coordinate $(z_1, z_2)$ forces the network to discover the most dominant generative manifold traits (loops, vertical strokes, curvature).\n"
            "- **Continuous Generative Decoding:** Any point $(z_1, z_2)$ in the Cartesian plane can be passed to the decoder $G(z)$ to synthesize a novel smooth digit.\n"
            "- **Denoising Projection:** Adding noise knocks the image off the data manifold in high-dimensional space. The Denoising Autoencoder projects corrupted vectors orthogonal back onto the true low-dimensional handwritten data manifold."
        )
        main_layout.addWidget(edu)

    def _create_latent_tab(self) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(20)

        # Left: Latent Canvas Card
        canvas_card = Card("2D LATENT SPACE MANIFOLD (z₁, z₂)")
        self.latent_canvas = LatentSpaceCanvas(380)
        if self.sample_latents is not None:
            self.latent_canvas.set_latent_data(self.sample_latents, self.sample_labels)
        self.latent_canvas.point_selected.connect(self._on_latent_point_moved)
        canvas_card.addWidget(self.latent_canvas)

        hint = QLabel("💡 Drag crosshair cursor to generate continuous synthetic digits across the manifold.")
        hint.setStyleSheet(f"font-size: 11px; color: {COLOR_TEXT_MUTED};")
        canvas_card.addWidget(hint)
        layout.addWidget(canvas_card)

        # Right: Generator Preview & Latent Interpolation Tool
        right_panel = QVBoxLayout()
        right_panel.setSpacing(16)

        # Synthetic Reconstructed Output Card
        gen_card = Card("GENERATIVE DECODER OUTPUT (28×28)")
        gen_box = QHBoxLayout()

        self.lbl_gen_image = QLabel()
        self.lbl_gen_image.setFixedSize(180, 180)
        self.lbl_gen_image.setStyleSheet("background: #000; border: 1px solid #334155; border-radius: 8px;")
        self.lbl_gen_image.setAlignment(Qt.AlignmentFlag.AlignCenter)
        gen_box.addWidget(self.lbl_gen_image)

        coord_box = QVBoxLayout()
        self.badge_z1 = MetricBadge("LATENT Z₁", "0.00", "", COLOR_ACCENT_CYAN)
        self.badge_z2 = MetricBadge("LATENT Z₂", "0.00", "", COLOR_ACCENT_PURPLE)
        coord_box.addWidget(self.badge_z1)
        coord_box.addWidget(self.badge_z2)
        gen_box.addLayout(coord_box)

        gen_card.addLayout(gen_box)
        right_panel.addWidget(gen_card)

        # Latent Morphing / Interpolation Tool
        morph_card = Card("CONTINUOUS MANIFOLD INTERPOLATION (MORPHING)")
        morph_row = QHBoxLayout()

        lbl_from = QLabel("From Digit:")
        lbl_from.setStyleSheet(f"color: {COLOR_TEXT_SECONDARY}; font-weight: 600;")
        morph_row.addWidget(lbl_from)

        self.combo_from = QComboBox()
        self.combo_from.addItems([f"Digit {i}" for i in range(10)])
        self.combo_from.setCurrentIndex(3)
        morph_row.addWidget(self.combo_from)

        lbl_to = QLabel("To Digit:")
        lbl_to.setStyleSheet(f"color: {COLOR_TEXT_SECONDARY}; font-weight: 600;")
        morph_row.addWidget(lbl_to)

        self.combo_to = QComboBox()
        self.combo_to.addItems([f"Digit {i}" for i in range(10)])
        self.combo_to.setCurrentIndex(8)
        morph_row.addWidget(self.combo_to)
        morph_card.addLayout(morph_row)

        self.slider_morph = LabeledSlider("Interpolation Alpha (t)", 0.0, 1.0, 0.0, step=0.02, is_float=True)
        self.slider_morph.valueChanged.connect(self._on_morph_slider_changed)
        morph_card.addWidget(self.slider_morph)

        right_panel.addWidget(morph_card)
        layout.addLayout(right_panel, stretch=1)

        # Initial decode
        self._on_latent_point_moved(0.0, 0.0)
        return widget

    def _create_denoising_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        # Top Controls: Noise Type, Noise Level, Test Digit Selector
        ctrl_card = Card("NOISE INJECTION CONFIGURATION")
        ctrl_row = QHBoxLayout()

        lbl_sample = QLabel("Select Test Digit:")
        lbl_sample.setStyleSheet(f"font-weight: 600; color: {COLOR_TEXT_SECONDARY};")
        ctrl_row.addWidget(lbl_sample)

        self.combo_denoise_digit = QComboBox()
        self.combo_denoise_digit.addItems([f"MNIST Sample #{i} (Class: {self.sample_labels[i] if self.sample_labels is not None else i})" for i in range(10)])
        self.combo_denoise_digit.currentIndexChanged.connect(self._on_denoise_sample_changed)
        ctrl_row.addWidget(self.combo_denoise_digit, stretch=1)

        lbl_noise = QLabel("Noise Mode:")
        lbl_noise.setStyleSheet(f"font-weight: 600; color: {COLOR_TEXT_SECONDARY};")
        ctrl_row.addWidget(lbl_noise)

        self.combo_noise_type = QComboBox()
        self.combo_noise_type.addItems(["Gaussian White Noise", "Salt & Pepper Noise", "Occlusion Patch Mask"])
        self.combo_noise_type.currentIndexChanged.connect(lambda _: self._update_denoising())
        ctrl_row.addWidget(self.combo_noise_type, stretch=1)

        btn_new_noise = QPushButton("🎲 Randomize Noise")
        btn_new_noise.clicked.connect(self._update_denoising)
        ctrl_row.addWidget(btn_new_noise)

        ctrl_card.addLayout(ctrl_row)

        self.slider_noise_level = LabeledSlider("Corruption Intensity", 0.05, 0.8, 0.35, step=0.05, is_float=True)
        self.slider_noise_level.valueChanged.connect(lambda _: self._update_denoising())
        ctrl_card.addWidget(self.slider_noise_level)
        layout.addWidget(ctrl_card)

        # Visual Comparison Grid: Clean -> Noisy -> Denoised -> Residual Error
        grid_layout = QHBoxLayout()
        grid_layout.setSpacing(16)

        # Card 1: Clean
        c1 = Card("1. CLEAN ORIGINAL")
        self.lbl_clean = QLabel()
        self.lbl_clean.setFixedSize(160, 160)
        self.lbl_clean.setStyleSheet("background: #000; border: 1px solid #334155; border-radius: 8px;")
        self.lbl_clean.setAlignment(Qt.AlignmentFlag.AlignCenter)
        c1.addWidget(self.lbl_clean)
        grid_layout.addWidget(c1)

        # Card 2: Noisy
        c2 = Card("2. CORRUPTED INPUT")
        self.lbl_noisy = QLabel()
        self.lbl_noisy.setFixedSize(160, 160)
        self.lbl_noisy.setStyleSheet("background: #000; border: 1px solid #334155; border-radius: 8px;")
        self.lbl_noisy.setAlignment(Qt.AlignmentFlag.AlignCenter)
        c2.addWidget(self.lbl_noisy)
        grid_layout.addWidget(c2)

        # Card 3: Denoised
        c3 = Card("3. NEURAL RECONSTRUCTION")
        self.lbl_reconstructed = QLabel()
        self.lbl_reconstructed.setFixedSize(160, 160)
        self.lbl_reconstructed.setStyleSheet("background: #000; border: 1px solid #334155; border-radius: 8px;")
        self.lbl_reconstructed.setAlignment(Qt.AlignmentFlag.AlignCenter)
        c3.addWidget(self.lbl_reconstructed)
        grid_layout.addWidget(c3)

        # Card 4: Residual Error
        c4 = Card("4. RESIDUAL ERROR MAP")
        self.lbl_error = QLabel()
        self.lbl_error.setFixedSize(160, 160)
        self.lbl_error.setStyleSheet("background: #000; border: 1px solid #334155; border-radius: 8px;")
        self.lbl_error.setAlignment(Qt.AlignmentFlag.AlignCenter)
        c4.addWidget(self.lbl_error)
        grid_layout.addWidget(c4)

        layout.addLayout(grid_layout)

        # Denoising Metrics
        metrics_row = QHBoxLayout()
        self.badge_in_psnr = MetricBadge("INPUT PSNR", "--", "dB", COLOR_ACCENT_AMBER)
        self.badge_out_psnr = MetricBadge("RECONSTRUCTED PSNR", "--", "dB", COLOR_ACCENT_CYAN)
        self.badge_psnr_gain = MetricBadge("PSNR RECOVERY GAIN", "--", "dB", COLOR_ACCENT_GREEN)
        metrics_row.addWidget(self.badge_in_psnr)
        metrics_row.addWidget(self.badge_out_psnr)
        metrics_row.addWidget(self.badge_psnr_gain)
        layout.addLayout(metrics_row)

        self._update_denoising()
        return widget

    def _on_latent_point_moved(self, z1: float, z2: float):
        """Passes 2D coordinate through decoder and renders reconstructed digit."""
        self.badge_z1.set_value(f"{z1:.2f}")
        self.badge_z2.set_value(f"{z2:.2f}")

        z_t = torch.tensor([[z1, z2]], dtype=torch.float32).to(self.device)
        self.model.eval()
        with torch.no_grad():
            img_t = self.model.decode(z_t)
            img_np = img_t.squeeze().cpu().numpy()

        pixmap = self._numpy_to_pixmap(img_np, size=180, colormap="cyan")
        self.lbl_gen_image.setPixmap(pixmap)

    def _on_morph_slider_changed(self, alpha: float):
        """Linearly interpolates latent vectors between two digit class centroids."""
        if self.sample_latents is None:
            return
        d_from = self.combo_from.currentIndex()
        d_to = self.combo_to.currentIndex()

        pts_from = self.sample_latents[self.sample_labels == d_from]
        pts_to = self.sample_latents[self.sample_labels == d_to]

        if len(pts_from) == 0 or len(pts_to) == 0:
            return

        mean_from = np.mean(pts_from, axis=0)
        mean_to = np.mean(pts_to, axis=0)

        z_interp = (1.0 - alpha) * mean_from + alpha * mean_to
        self.latent_canvas.set_cursor(float(z_interp[0]), float(z_interp[1]))
        self._on_latent_point_moved(float(z_interp[0]), float(z_interp[1]))

    def _on_denoise_sample_changed(self, idx: int):
        if self.sample_images is not None and idx < len(self.sample_images):
            self.current_denoise_img = self.sample_images[idx]
            self._update_denoising()

    def _update_denoising(self):
        if self.current_denoise_img is None:
            return

        noise_mode_map = {
            0: "gaussian",
            1: "salt_pepper",
            2: "occlusion"
        }
        mode = noise_mode_map.get(self.combo_noise_type.currentIndex(), "gaussian")
        intensity = self.slider_noise_level.value()

        # Clean image
        clean = self.current_denoise_img
        # Noisy image
        noisy = add_noise_to_image(clean, noise_type=mode, intensity=intensity)

        # Run Autoencoder Denoising forward pass
        self.model.eval()
        with torch.no_grad():
            noisy_t = torch.from_numpy(noisy.astype(np.float32)).unsqueeze(0).to(self.device)
            recon_t = self.model(noisy_t)
            recon = recon_t.squeeze().cpu().numpy()

        # Residual error
        residual = np.abs(recon - clean)

        # Metrics
        m_in = compute_reconstruction_metrics(clean, noisy)
        m_out = compute_reconstruction_metrics(clean, recon)
        gain = m_out["psnr"] - m_in["psnr"]

        self.badge_in_psnr.set_value(f"{m_in['psnr']:.1f}")
        self.badge_out_psnr.set_value(f"{m_out['psnr']:.1f}")
        self.badge_psnr_gain.set_value(f"+{gain:.1f}" if gain >= 0 else f"{gain:.1f}")

        # Update displays
        self.lbl_clean.setPixmap(self._numpy_to_pixmap(clean, size=160, colormap="mono"))
        self.lbl_noisy.setPixmap(self._numpy_to_pixmap(noisy, size=160, colormap="mono"))
        self.lbl_reconstructed.setPixmap(self._numpy_to_pixmap(recon, size=160, colormap="cyan"))
        self.lbl_error.setPixmap(self._numpy_to_pixmap(residual, size=160, colormap="hot"))

    def _numpy_to_pixmap(self, arr: np.ndarray, size: int = 160, colormap: str = "mono") -> QPixmap:
        arr = np.clip(np.nan_to_num(arr), 0.0, 1.0)
        h, w = arr.shape
        img = QImage(w, h, QImage.Format.Format_RGB32)

        for y in range(h):
            for x in range(w):
                v = float(arr[y, x])
                if colormap == "cyan":
                    r, g, b = int(v * 20), int(v * 229), int(v * 255)
                elif colormap == "hot":
                    r = int(min(1.0, v * 2.0) * 255)
                    g = int(max(0.0, v - 0.3) * 255)
                    b = int(max(0.0, v - 0.7) * 255)
                else:  # monochrome
                    val = int(v * 255)
                    r, g, b = val, val, val
                img.setPixelColor(x, y, QColor(r, g, b))

        pixmap = QPixmap.fromImage(img)
        return pixmap.scaled(size, size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation)
