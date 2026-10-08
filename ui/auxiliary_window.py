"""Auxiliary Neural Network Inspector Window.

Provides deep visual inspection of internal activations, convolutional feature maps,
saliency attribution gradients, and architecture signal flow.
"""

import numpy as np
from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QPushButton, QTabWidget, QScrollArea, QFrame, QSplitter
)
from PyQt6.QtCore import Qt, QTimer, QPointF
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QImage, QPixmap, QFont
from ui.theme import (
    COLOR_BG_DARK, COLOR_BG_CARD, COLOR_BG_SURFACE, COLOR_BORDER,
    COLOR_ACCENT_CYAN, COLOR_ACCENT_PURPLE, COLOR_ACCENT_GREEN,
    COLOR_ACCENT_AMBER, COLOR_TEXT_PRIMARY, COLOR_TEXT_SECONDARY, COLOR_TEXT_MUTED
)
from ui.custom_widgets import Card, MetricBadge

class ArchitectureFlowCanvas(QWidget):
    """Animated visual diagram of the CNN layers with propagating activation pulses."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(240)
        self.layers = [
            {"name": "Input", "shape": "1×28×28", "type": "input", "nodes": 1},
            {"name": "Conv1", "shape": "16×28×28", "type": "conv", "nodes": 4},
            {"name": "Pool1", "shape": "16×14×14", "type": "pool", "nodes": 4},
            {"name": "Conv2", "shape": "32×14×14", "type": "conv", "nodes": 6},
            {"name": "Pool2", "shape": "32×7×7", "type": "pool", "nodes": 6},
            {"name": "FC1 (Dense)", "shape": "128", "type": "fc", "nodes": 8},
            {"name": "Output", "shape": "10 Classes", "type": "output", "nodes": 10},
        ]
        self.pulse_phase = 0.0
        self.is_active = False

        # Pulse animation timer
        self.timer = QTimer(self)
        self.timer.timeout.connect(self._animate_pulse)
        self.timer.start(30)

    def trigger_pulse(self):
        self.pulse_phase = 0.0
        self.is_active = True
        self.update()

    def _animate_pulse(self):
        if self.is_active:
            self.pulse_phase += 0.04
            if self.pulse_phase > 1.2:
                self.is_active = False
            self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        painter.fillRect(0, 0, w, h, QColor(COLOR_BG_DARK))

        n_layers = len(self.layers)
        spacing_x = (w - 120) / max(1, n_layers - 1)
        layer_x_positions = [60 + i * spacing_x for i in range(n_layers)]
        center_y = h / 2

        # Draw connecting signal lines
        for i in range(n_layers - 1):
            x1 = layer_x_positions[i]
            x2 = layer_x_positions[i + 1]
            pen = QPen(QColor("#1E293B"), 2, Qt.PenStyle.DashLine)
            painter.setPen(pen)
            painter.drawLine(int(x1), int(center_y), int(x2), int(center_y))

            # Draw glowing signal pulse
            if self.is_active:
                pulse_x = 60 + self.pulse_phase * (w - 120)
                if abs(pulse_x - ((x1 + x2) / 2)) < spacing_x * 0.7:
                    glow_pen = QPen(QColor(COLOR_ACCENT_CYAN), 3)
                    painter.setPen(glow_pen)
                    painter.drawLine(int(x1), int(center_y), int(x2), int(center_y))

        # Draw layer blocks / nodes
        for i, l in enumerate(self.layers):
            x = layer_x_positions[i]
            node_h = 36 + l["nodes"] * 8
            y = center_y - node_h / 2

            # Pulse highlight
            highlight = False
            if self.is_active:
                curr_layer_phase = i / (n_layers - 1)
                if abs(self.pulse_phase - curr_layer_phase) < 0.15:
                    highlight = True

            # Box background
            if highlight:
                painter.setBrush(QBrush(QColor(COLOR_ACCENT_CYAN)))
                painter.setPen(QPen(QColor("#FFFFFF"), 2))
            else:
                painter.setBrush(QBrush(QColor(COLOR_BG_CARD)))
                painter.setPen(QPen(QColor(COLOR_BORDER), 1))

            painter.drawRoundedRect(int(x - 28), int(y), 56, int(node_h), 8, 8)

            # Node circles inside
            n_circles = min(l["nodes"], 6)
            for ci in range(n_circles):
                cy = y + (ci + 1) * (node_h / (n_circles + 1))
                c_color = QColor(COLOR_ACCENT_PURPLE if not highlight else "#0B0F19")
                painter.setBrush(QBrush(c_color))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawEllipse(int(x - 4), int(cy - 4), 8, 8)

            # Labels
            painter.setPen(QColor(COLOR_TEXT_PRIMARY if not highlight else "#000000"))
            painter.setFont(QFont("sans-serif", 9, QFont.Weight.Bold))
            painter.drawText(int(x - 50), int(y - 12), 100, 16, Qt.AlignmentFlag.AlignCenter, l["name"])

            painter.setPen(QColor(COLOR_TEXT_MUTED if not highlight else "#1E293B"))
            painter.setFont(QFont("sans-serif", 8))
            painter.drawText(int(x - 50), int(y + node_h + 4), 100, 16, Qt.AlignmentFlag.AlignCenter, l["shape"])


class AuxiliaryNeuralWindow(QMainWindow):
    """Stand-alone / Detachable Deep Neural Network Live Telemetry Screen."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("NeuroNetWorks - Neural Activity Live Inspector")
        self.resize(980, 720)
        self.setStyleSheet(f"background-color: {COLOR_BG_DARK}; color: {COLOR_TEXT_PRIMARY};")

        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(16, 16, 16, 16)
        main_layout.setSpacing(12)

        # Header Bar
        hdr_layout = QHBoxLayout()
        title_box = QVBoxLayout()
        lbl_title = QLabel("LIVE NEURAL NETWORK ACTIVATION TELEMETRY")
        lbl_title.setStyleSheet("font-size: 16px; font-weight: 800; color: #FFFFFF; letter-spacing: 1px;")
        lbl_subtitle = QLabel("Real-time forward-pass layer introspection, convolutional filters & autograd saliency")
        lbl_subtitle.setStyleSheet(f"font-size: 11px; color: {COLOR_TEXT_MUTED};")
        title_box.addWidget(lbl_title)
        title_box.addWidget(lbl_subtitle)
        hdr_layout.addLayout(title_box)
        hdr_layout.addStretch()

        self.badge_status = MetricBadge("MODEL INFERENCE", "STANDBY", "", COLOR_ACCENT_AMBER)
        hdr_layout.addWidget(self.badge_status)
        main_layout.addLayout(hdr_layout)

        # Signal Flow Architecture Diagram (always visible on top)
        self.flow_canvas = ArchitectureFlowCanvas()
        flow_card = Card("PIPELINE ARCHITECTURE & LIVE PROPAGATION")
        flow_card.addWidget(self.flow_canvas)
        main_layout.addWidget(flow_card)

        # Multi-tab inspection view
        self.tabs = QTabWidget()
        main_layout.addWidget(self.tabs, stretch=1)

        # Tab 1: Feature Maps (Conv1 & Conv2)
        self.tab_feature_maps = self._create_feature_maps_tab()
        self.tabs.addTab(self.tab_feature_maps, "🔬 Convolutional Feature Maps")

        # Tab 2: Saliency Map & Pixel Attribution
        self.tab_saliency = self._create_saliency_tab()
        self.tabs.addTab(self.tab_saliency, "🎯 Autograd Saliency Map")

        # Tab 3: Dense FC1 Neuron Activation Heatmap
        self.tab_dense = self._create_dense_tab()
        self.tabs.addTab(self.tab_dense, "⚡ Dense Layer Heatmap (128 Neurons)")

    def _create_feature_maps_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(12, 12, 12, 12)

        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet("background-color: transparent; border: none;")

        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)

        # Conv1 Section (16 feature maps)
        lbl_c1 = QLabel("LAYER 1: CONV2D FEATURE MAPS (16 FILTERS @ 14×14)")
        lbl_c1.setStyleSheet(f"font-weight: 700; color: {COLOR_ACCENT_CYAN}; font-size: 12px;")
        scroll_layout.addWidget(lbl_c1)

        self.conv1_grid = QGridLayout()
        self.conv1_grid.setSpacing(8)
        self.conv1_labels = []
        for i in range(16):
            box = QVBoxLayout()
            img_lbl = QLabel()
            img_lbl.setFixedSize(64, 64)
            img_lbl.setStyleSheet("background-color: #000000; border: 1px solid #334155; border-radius: 4px;")
            img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            tag = QLabel(f"Filter #{i}")
            tag.setStyleSheet(f"font-size: 9px; color: {COLOR_TEXT_MUTED};")
            tag.setAlignment(Qt.AlignmentFlag.AlignCenter)
            box.addWidget(img_lbl)
            box.addWidget(tag)

            container = QWidget()
            container.setLayout(box)
            self.conv1_grid.addWidget(container, i // 8, i % 8)
            self.conv1_labels.append(img_lbl)

        scroll_layout.addLayout(self.conv1_grid)

        # Conv2 Section (32 feature maps)
        lbl_c2 = QLabel("LAYER 2: CONV2D FEATURE MAPS (32 FILTERS @ 7×7)")
        lbl_c2.setStyleSheet(f"font-weight: 700; color: {COLOR_ACCENT_PURPLE}; font-size: 12px; margin-top: 16px;")
        scroll_layout.addWidget(lbl_c2)

        self.conv2_grid = QGridLayout()
        self.conv2_grid.setSpacing(8)
        self.conv2_labels = []
        for i in range(32):
            box = QVBoxLayout()
            img_lbl = QLabel()
            img_lbl.setFixedSize(54, 54)
            img_lbl.setStyleSheet("background-color: #000000; border: 1px solid #334155; border-radius: 4px;")
            img_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
            tag = QLabel(f"#{i}")
            tag.setStyleSheet(f"font-size: 8px; color: {COLOR_TEXT_MUTED};")
            tag.setAlignment(Qt.AlignmentFlag.AlignCenter)
            box.addWidget(img_lbl)
            box.addWidget(tag)

            container = QWidget()
            container.setLayout(box)
            self.conv2_grid.addWidget(container, i // 8, i % 8)
            self.conv2_labels.append(img_lbl)

        scroll_layout.addLayout(self.conv2_grid)
        scroll.setWidget(scroll_content)
        layout.addWidget(scroll)
        return widget

    def _create_saliency_tab(self) -> QWidget:
        widget = QWidget()
        layout = QHBoxLayout(widget)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(24)

        # Original Input
        c1 = Card("1. INPUT IMAGE (28×28)")
        self.lbl_saliency_input = QLabel()
        self.lbl_saliency_input.setFixedSize(200, 200)
        self.lbl_saliency_input.setStyleSheet("background: #000; border: 1px solid #334155; border-radius: 8px;")
        self.lbl_saliency_input.setAlignment(Qt.AlignmentFlag.AlignCenter)
        c1.addWidget(self.lbl_saliency_input)
        layout.addWidget(c1)

        # Saliency Heatmap
        c2 = Card("2. ATTRIBUTION GRADIENTS (∂y / ∂x)")
        self.lbl_saliency_heat = QLabel()
        self.lbl_saliency_heat.setFixedSize(200, 200)
        self.lbl_saliency_heat.setStyleSheet("background: #000; border: 1px solid #334155; border-radius: 8px;")
        self.lbl_saliency_heat.setAlignment(Qt.AlignmentFlag.AlignCenter)
        c2.addWidget(self.lbl_saliency_heat)
        layout.addWidget(c2)

        # Attribution Explanation Card
        c3 = Card("EXPLAINABLE AI INSIGHTS")
        exp_text = QLabel(
            "<b>Vanilla Backpropagation Attribution</b><br><br>"
            "The Saliency Map shows the derivative of the winning class logit with respect to each input pixel.<br><br>"
            "• <b>Bright Cyan/White:</b> Pixels that critically increased network confidence.<br>"
            "• <b>Dark Regions:</b> Pixels ignored by the convolutional receptive fields.<br><br>"
            "This confirms whether the model recognizes the digit by genuine structural strokes or background artifacts."
        )
        exp_text.setWordWrap(True)
        exp_text.setStyleSheet(f"color: {COLOR_TEXT_SECONDARY}; line-height: 1.5;")
        c3.addWidget(exp_text)
        layout.addWidget(c3, stretch=1)

        return widget

    def _create_dense_tab(self) -> QWidget:
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(16, 16, 16, 16)

        c = Card("FULLY CONNECTED LAYER 1 (128 ACTIVATION UNITS)")
        desc = QLabel("Neuron activation levels post-ReLU non-linearity. Demonstrates activation sparsity and distributed representation.")
        desc.setStyleSheet(f"color: {COLOR_TEXT_MUTED}; font-size: 11px;")
        c.addWidget(desc)

        self.dense_grid_label = QLabel()
        self.dense_grid_label.setMinimumHeight(180)
        self.dense_grid_label.setStyleSheet("background: #000; border: 1px solid #334155; border-radius: 8px;")
        self.dense_grid_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        c.addWidget(self.dense_grid_label)
        layout.addWidget(c)
        return widget

    def update_telemetry(self, data: dict):
        """Receives live forward pass outputs and refreshes all visual inspectors."""
        self.flow_canvas.trigger_pulse()
        self.badge_status.set_value("ACTIVE")
        self.badge_status.val_widget.setStyleSheet(f"color: {COLOR_ACCENT_GREEN}; font-size: 18px; font-weight: 800;")

        # 1. Update Conv1 feature maps
        if "conv1_act" in data:
            c1_maps = data["conv1_act"]  # [16, 14, 14]
            for i in range(min(16, len(c1_maps))):
                pixmap = self._numpy_to_pixmap(c1_maps[i], size=64, colormap="cyan")
                self.conv1_labels[i].setPixmap(pixmap)

        # 2. Update Conv2 feature maps
        if "conv2_act" in data:
            c2_maps = data["conv2_act"]  # [32, 7, 7]
            for i in range(min(32, len(c2_maps))):
                pixmap = self._numpy_to_pixmap(c2_maps[i], size=54, colormap="purple")
                self.conv2_labels[i].setPixmap(pixmap)

        # 3. Update Saliency
        if "input" in data:
            inp = data["input"]
            pixmap_in = self._numpy_to_pixmap(inp, size=200, colormap="mono")
            self.lbl_saliency_input.setPixmap(pixmap_in)

        if "saliency" in data:
            sal = data["saliency"]
            pixmap_sal = self._numpy_to_pixmap(sal, size=200, colormap="hot")
            self.lbl_saliency_heat.setPixmap(pixmap_sal)

        # 4. Update FC1 Heatmap (128 units as 16x8 matrix)
        if "fc1" in data:
            fc1 = data["fc1"]  # [128]
            # Reshape to 8x16
            fc1_grid = fc1.reshape(8, 16)
            pixmap_fc = self._numpy_to_pixmap(fc1_grid, size=360, colormap="cyan")
            self.dense_grid_label.setPixmap(pixmap_fc)

    def _numpy_to_pixmap(self, arr: np.ndarray, size: int = 64, colormap: str = "mono") -> QPixmap:
        """Converts a 2D float numpy array to a colored high-contrast QPixmap."""
        arr = np.nan_to_num(arr)
        a_min, a_max = arr.min(), arr.max()
        if a_max - a_min > 1e-6:
            norm = (arr - a_min) / (a_max - a_min)
        else:
            norm = np.zeros_like(arr)

        h, w = norm.shape
        img = QImage(w, h, QImage.Format.Format_RGB32)

        for y in range(h):
            for x in range(w):
                v = float(norm[y, x])
                if colormap == "cyan":
                    r = int(v * 20)
                    g = int(v * 229)
                    b = int(v * 255)
                elif colormap == "purple":
                    r = int(v * 168)
                    g = int(v * 85)
                    b = int(v * 247)
                elif colormap == "hot":
                    r = int(min(1.0, v * 1.5) * 255)
                    g = int(max(0.0, v - 0.4) * 255)
                    b = int(max(0.0, v - 0.7) * 255)
                else:  # monochrome
                    val = int(v * 255)
                    r, g, b = val, val, val
                img.setPixelColor(x, y, QColor(r, g, b))

        pixmap = QPixmap.fromImage(img)
        return pixmap.scaled(size, size, Qt.AspectRatioMode.KeepAspectRatio, Qt.TransformationMode.FastTransformation)
