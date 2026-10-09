"""Module 7: Vision Transformer (ViT) & Multi-Head Self-Attention Explorer."""

import os
import numpy as np
import torch

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QFrame, QGridLayout, QSplitter
)
from PyQt6.QtCore import Qt, pyqtSignal, QPointF
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QImage, QPixmap

from core.transformer_models import MiniVisionTransformer
from core.device import get_device
from ui.theme import (
    COLOR_BG_DARK, COLOR_BG_CARD, COLOR_BG_SURFACE, COLOR_BORDER,
    COLOR_ACCENT_CYAN, COLOR_ACCENT_PURPLE, COLOR_ACCENT_GREEN,
    COLOR_ACCENT_AMBER, COLOR_ACCENT_RED, COLOR_TEXT_PRIMARY,
    COLOR_TEXT_SECONDARY, COLOR_TEXT_MUTED
)
from ui.custom_widgets import Card, MetricBadge, EducationalPanel

WEIGHTS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models_weights", "mnist_vit.pth")
SAMPLES_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "data", "mnist_samples.npz")

class PatchAttentionCanvas(QWidget):
    """Renders the 28x28 digit with interactive 4x4 patch grid and attention heat overlay."""
    patch_selected = pyqtSignal(int)  # 0 to 15 (patches) or -1 ([CLS] token)

    def __init__(self, size: int = 280, parent=None):
        super().__init__(parent)
        self.canvas_size = size
        self.setFixedSize(size, size)

        self.input_img = np.zeros((28, 28), dtype=np.float32)
        self.attention_weights = np.zeros(16, dtype=np.float32)
        self.selected_patch = 0  # 0 to 15
        self.is_cls_query = True

    def set_data(self, img_28x28: np.ndarray, patch_attn: np.ndarray, selected_patch: int = 0, is_cls: bool = True):
        self.input_img = img_28x28
        self.attention_weights = patch_attn
        self.selected_patch = selected_patch
        self.is_cls_query = is_cls
        self.update()

    def mousePressEvent(self, event):
        px = event.position().x()
        py = event.position().y()
        cell_size = self.canvas_size / 4.0
        col = int(px // cell_size)
        row = int(py // cell_size)
        col = max(0, min(3, col))
        row = max(0, min(3, row))
        idx = row * 4 + col
        self.selected_patch = idx
        self.is_cls_query = False
        self.patch_selected.emit(idx)
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()

        # 1. Draw base 28x28 image scaled to canvas
        scaled_img = (np.clip(self.input_img, 0.0, 1.0) * 255).astype(np.uint8)
        qimg = QImage(scaled_img.data, 28, 28, 28, QImage.Format.Format_Grayscale8)
        pix = QPixmap.fromImage(qimg).scaled(w, h, Qt.AspectRatioMode.IgnoreAspectRatio, Qt.TransformationMode.FastTransformation)
        painter.drawPixmap(0, 0, pix)

        # 2. Draw 4x4 Patch Grid Lines & Attention Glow
        cell_w = w / 4.0
        cell_h = h / 4.0

        max_attn = max(float(np.max(self.attention_weights)), 1e-6)

        for row in range(4):
            for col in range(4):
                idx = row * 4 + col
                x = col * cell_w
                y = row * cell_h

                attn = float(self.attention_weights[idx]) if idx < len(self.attention_weights) else 0.0
                norm_attn = attn / max_attn

                # Heat overlay: glowing cyan semi-transparent box
                if not (self.selected_patch == idx and not self.is_cls_query):
                    alpha = int(norm_attn * 180)
                    painter.fillRect(int(x), int(y), int(cell_w), int(cell_h), QColor(0, 229, 255, alpha))

                # Grid borders
                painter.setPen(QPen(QColor(COLOR_BORDER), 1))
                painter.setBrush(Qt.BrushStyle.NoBrush)
                painter.drawRect(int(x), int(y), int(cell_w), int(cell_h))

                # Patch Index label
                painter.setPen(QColor(255, 255, 255, 160))
                painter.drawText(int(x + 4), int(y + 14), f"P{idx}")

        # 3. Highlight Selected Query Patch (Gold / Amber box)
        if not self.is_cls_query and self.selected_patch in range(16):
            q_row = self.selected_patch // 4
            q_col = self.selected_patch % 4
            qx = q_col * cell_w
            qy = q_row * cell_h
            painter.setPen(QPen(QColor(COLOR_ACCENT_AMBER), 3))
            painter.setBrush(QBrush(QColor(245, 158, 11, 60)))
            painter.drawRect(int(qx), int(qy), int(cell_w), int(cell_h))

        # Canvas Border
        painter.setPen(QPen(QColor(COLOR_BORDER), 2))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRoundedRect(0, 0, w, h, 8, 8)


class AttentionMatrixCanvas(QWidget):
    """Renders the full 17x17 token-to-token Attention Heatmap matrix."""

    def __init__(self, size: int = 260, parent=None):
        super().__init__(parent)
        self.setFixedSize(size, size)
        self.matrix = np.zeros((17, 17), dtype=np.float32)

    def set_matrix(self, mat: np.ndarray):
        self.matrix = mat
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        w, h = self.width(), self.height()
        n = 17
        dx = w / n
        dy = h / n

        for i in range(n):
            for j in range(n):
                val = float(self.matrix[i, j])
                val = min(1.0, val * 3.0)  # enhance contrast
                # Colormap: Deep navy to vivid cyan
                r = int(val * 20)
                g = int(val * 229)
                b = int(val * 255)
                painter.fillRect(int(j * dx), int(i * dy), int(dx + 1), int(dy + 1), QColor(r, g, b))

        painter.setPen(QPen(QColor(COLOR_BORDER), 1))
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.drawRect(0, 0, w - 1, h - 1)


class TransformerView(QWidget):
    """Main Vision Transformer (ViT) & Multi-Head Attention Explorer View."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.device = get_device()
        self.model = MiniVisionTransformer().to(self.device)

        if os.path.exists(WEIGHTS_PATH):
            try:
                self.model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=self.device))
                self.model.eval()
            except Exception as e:
                print("Error loading ViT weights:", e)

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
        self.current_head = 0  # 0 to 3, or 4 for Average
        self.selected_patch = 0
        self.is_cls_query = True
        self.telemetry = None

        self._init_ui()
        self._run_inference()

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(16)

        # Header Title
        hdr_layout = QHBoxLayout()
        title_box = QVBoxLayout()
        title = QLabel("VISION TRANSFORMER (ViT) & SELF-ATTENTION EXPLORER")
        title.setStyleSheet("font-size: 20px; font-weight: 800; color: #FFFFFF;")
        subtitle = QLabel("Multi-Head Scaled Dot-Product Attention Softmax(QKᵀ / √dₖ)V with Interactive Patch Queries")
        subtitle.setStyleSheet(f"font-size: 12px; color: {COLOR_TEXT_SECONDARY};")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        hdr_layout.addLayout(title_box)
        hdr_layout.addStretch()
        main_layout.addLayout(hdr_layout)

        # Main Layout
        content_layout = QHBoxLayout()
        content_layout.setSpacing(20)

        # Left Column: Interactive Patch Attention Canvas Card
        canvas_card = Card("IMAGE PATCH GRID & ATTENTION RECEPTIVE FIELD (16 PATCHES)")
        self.patch_canvas = PatchAttentionCanvas(280)
        self.patch_canvas.patch_selected.connect(self._on_patch_clicked)
        canvas_card.addWidget(self.patch_canvas)

        hint = QLabel("💡 Click ANY patch on the digit to make it the Attention Query! Gold = Query patch, Cyan = Attended regions.")
        hint.setWordWrap(True)
        hint.setStyleSheet(f"font-size: 11px; color: {COLOR_TEXT_MUTED};")
        canvas_card.addWidget(hint)

        # Query Switch Buttons
        query_row = QHBoxLayout()
        self.btn_cls_query = QPushButton("🎯 Query = [CLS] Token")
        self.btn_cls_query.setStyleSheet(f"background-color: {COLOR_ACCENT_PURPLE}; color: #FFF; font-weight: 700; border-radius: 6px; padding: 6px 12px;")
        self.btn_cls_query.clicked.connect(self._select_cls_query)
        query_row.addWidget(self.btn_cls_query)
        canvas_card.addLayout(query_row)

        # Sample Carousel
        sample_lbl = QLabel("PRE-LOADED MNIST TEST DIGITS:")
        sample_lbl.setStyleSheet(f"font-size: 10px; font-weight: 700; color: {COLOR_TEXT_MUTED};")
        canvas_card.addWidget(sample_lbl)

        sample_row = QHBoxLayout()
        sample_row.setSpacing(4)
        for d in range(10):
            btn = QPushButton(str(d))
            btn.setFixedSize(26, 26)
            btn.clicked.connect(lambda checked, digit=d: self._load_sample(digit))
            sample_row.addWidget(btn)
        canvas_card.addLayout(sample_row)

        content_layout.addWidget(canvas_card)

        # Right Column: Attention Telemetry & Heatmap Matrix
        right_panel = QVBoxLayout()
        right_panel.setSpacing(12)

        # Telemetry Badges
        metrics_card = Card("PREDICTION & ATTENTION DYNAMICS")
        m_row = QHBoxLayout()
        self.badge_pred = MetricBadge("PREDICTED CLASS", "--", "", COLOR_ACCENT_GREEN)
        self.badge_conf = MetricBadge("CONFIDENCE", "--", "%", COLOR_ACCENT_CYAN)
        self.badge_query = MetricBadge("ACTIVE QUERY", "[CLS] Token", "", COLOR_ACCENT_AMBER)
        m_row.addWidget(self.badge_pred)
        m_row.addWidget(self.badge_conf)
        m_row.addWidget(self.badge_query)
        metrics_card.addLayout(m_row)
        right_panel.addWidget(metrics_card)

        # Attention Head Selector & Matrix Card
        attn_card = Card("MULTI-HEAD ATTENTION MATRIX (17×17 TOKENS)")

        head_row = QHBoxLayout()
        lbl_h = QLabel("Attention Head:")
        lbl_h.setStyleSheet(f"font-weight: 600; color: {COLOR_TEXT_SECONDARY};")
        head_row.addWidget(lbl_h)

        self.combo_head = QComboBox()
        self.combo_head.addItems(["Head 1 (Local Edge Specialist)", "Head 2 (Global Contour Specialist)", "Head 3 (Loop / Intersection)", "Head 4 (Background Invariant)", "Average of All Heads"])
        self.combo_head.currentIndexChanged.connect(self._on_head_changed)
        head_row.addWidget(self.combo_head, stretch=1)
        attn_card.addLayout(head_row)

        # Matrix visualizer + explanation
        matrix_box = QHBoxLayout()
        self.matrix_canvas = AttentionMatrixCanvas(240)
        matrix_box.addWidget(self.matrix_canvas)

        matrix_desc = QLabel(
            "<b>Token Attention Matrix A<sub>i,j</sub></b><br><br>"
            "• Row 0: <b>[CLS] token</b> attending to all 16 visual patches.<br>"
            "• Rows 1–16: <b>Patch tokens</b> attending to other patches.<br>"
            "• Notice how different heads specialize in distinct structural aspects (vertical strokes, curves, and loops) without hand-crafted convolutional kernels."
        )
        matrix_desc.setWordWrap(True)
        matrix_desc.setStyleSheet(f"color: {COLOR_TEXT_SECONDARY}; font-size: 11px; line-height: 1.4;")
        matrix_box.addWidget(matrix_desc, stretch=1)
        attn_card.addLayout(matrix_box)

        right_panel.addWidget(attn_card)
        content_layout.addLayout(right_panel, stretch=1)
        main_layout.addLayout(content_layout)

        # Educational Panel
        edu = EducationalPanel(
            "Vision Transformers (ViT) & Inductive Bias vs CNNs",
            "**Key Mathematical Principles of Vision Transformers:**\n\n"
            "- **Patchification:** The image $X \\in \\mathbb{R}^{28\\times 28}$ is sliced into $N=16$ non-overlapping patches ($7\\times 7$). Each patch is flattened and linearly projected into an embedding dimension $D=64$.\n"
            "- **The Scaled Dot-Product Attention:**\n"
            "$$\\text{Attention}(\\mathbf{Q}, \\mathbf{K}, \\mathbf{V}) = \\text{Softmax}\\left(\\frac{\\mathbf{Q}\\mathbf{K}^T}{\\sqrt{d_k}}\\right)\\mathbf{V}$$\n"
            "- **Inductive Bias vs. Flexibility:** CNNs enforce hard translational equivariance and local connectivity. In contrast, Vision Transformers have *no local bias*—every patch can attend to any other patch across the entire canvas from layer 1, learning global relationships directly from data."
        )
        main_layout.addWidget(edu)

    def _load_sample(self, digit: int):
        if self.sample_images is None:
            return
        matches = np.where(self.sample_labels == digit)[0]
        if len(matches) > 0:
            idx = np.random.choice(matches)
            self.current_img = self.sample_images[idx]
            self._run_inference()

    def _select_cls_query(self):
        self.is_cls_query = True
        self.badge_query.set_value("[CLS] Token")
        self._update_display()

    def _on_patch_clicked(self, idx: int):
        self.selected_patch = idx
        self.is_cls_query = False
        self.badge_query.set_value(f"Patch #{idx}")
        self._update_display()

    def _on_head_changed(self, idx: int):
        self.current_head = idx
        self._update_display()

    def _run_inference(self):
        img_t = torch.from_numpy(self.current_img).unsqueeze(0).unsqueeze(0).to(self.device)
        self.telemetry = self.model.forward_with_attention(img_t)

        probs = self.telemetry["probabilities"]
        top_d = int(np.argmax(probs))
        conf = float(probs[top_d]) * 100.0

        self.badge_pred.set_value(str(top_d))
        self.badge_conf.set_value(f"{conf:.1f}")

        self._update_display()

    def _update_display(self):
        if self.telemetry is None:
            return

        all_heads_matrix = self.telemetry["attention_weights"]  # [4, 17, 17]

        if self.current_head == 4:  # Average of all heads
            head_mat = np.mean(all_heads_matrix, axis=0)  # [17, 17]
        else:
            head_mat = all_heads_matrix[self.current_head]

        self.matrix_canvas.set_matrix(head_mat)

        # Extract patch attention based on active query
        if self.is_cls_query:
            patch_attn = head_mat[0, 1:]  # [CLS] token attending to patches 1..16
        else:
            query_token_idx = self.selected_patch + 1
            patch_attn = head_mat[query_token_idx, 1:]  # query patch attending to other patches

        self.patch_canvas.set_data(
            self.current_img,
            patch_attn,
            selected_patch=self.selected_patch,
            is_cls=self.is_cls_query
        )
