"""Module 9: 2D Toy Diffusion Model & Flow Matching Generative Playground."""

import numpy as np
import torch

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QFrame, QCheckBox, QSplitter
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QPointF
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush

from core.diffusion_models import DiffusionFlowSimulator
from core.device import get_device
from ui.theme import (
    COLOR_BG_DARK, COLOR_BG_CARD, COLOR_BG_SURFACE, COLOR_BORDER,
    COLOR_ACCENT_CYAN, COLOR_ACCENT_PURPLE, COLOR_ACCENT_GREEN,
    COLOR_ACCENT_AMBER, COLOR_ACCENT_RED, COLOR_TEXT_PRIMARY,
    COLOR_TEXT_SECONDARY, COLOR_TEXT_MUTED
)
from ui.custom_widgets import Card, MetricBadge, LabeledSlider, EducationalPanel

class DiffusionCanvas(QWidget):
    """2D coordinate canvas displaying particle self-assembly and velocity vector field lines."""

    def __init__(self, size: int = 440, parent=None):
        super().__init__(parent)
        self.canvas_size = size
        self.setFixedSize(size, size)

        self.particles = np.empty((0, 2), dtype=np.float32)
        self.vector_field = None  # (gx, gy, vxy)
        self.show_vectors = True
        self.t_val = 0.0

    def set_particles(self, pts: np.ndarray, t_val: float):
        self.particles = pts
        self.t_val = t_val
        self.update()

    def set_vector_field(self, field):
        self.vector_field = field
        self.update()

    def _coord_to_pixel(self, x: float, y: float) -> QPointF:
        w, h = self.width(), self.height()
        px = (x + 2.2) / 4.4 * w
        py = (2.2 - y) / 4.4 * h
        return QPointF(px, py)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()

        painter.fillRect(0, 0, w, h, QColor(COLOR_BG_DARK))

        # 1. Background Vector Field Flow Lines
        if self.show_vectors and self.vector_field is not None:
            gx, gy, vxy = self.vector_field
            rows, cols, _ = vxy.shape
            pen_vec = QPen(QColor(100, 116, 139, 70), 1)
            painter.setPen(pen_vec)

            for r in range(rows):
                for c in range(cols):
                    x0, y0 = float(gx[r, c]), float(gy[r, c])
                    vx, vy = float(vxy[r, c, 0]), float(vxy[r, c, 1])
                    v_mag = np.sqrt(vx ** 2 + vy ** 2) + 1e-6
                    # Unit scaled arrow
                    x1 = x0 + (vx / v_mag) * 0.18
                    y1 = y0 + (vy / v_mag) * 0.18

                    p0 = self._coord_to_pixel(x0, y0)
                    p1 = self._coord_to_pixel(x1, y1)
                    painter.drawLine(p0, p1)

        # 2. Coordinate Axis Lines
        painter.setPen(QPen(QColor("#1E293B"), 1, Qt.PenStyle.DashLine))
        center_x = w / 2
        center_y = h / 2
        painter.drawLine(int(center_x), 0, int(center_x), h)
        painter.drawLine(0, int(center_y), w, int(center_y))

        # 3. Draw Diffusion Particles
        if len(self.particles) > 0:
            for pt in self.particles:
                pix = self._coord_to_pixel(pt[0], pt[1])

                # Color transition: Noise (Purple/Pink at t=0) -> Target Shape (Cyan at t=1)
                r = int((1.0 - self.t_val) * 236 + self.t_val * 0)
                g = int((1.0 - self.t_val) * 72 + self.t_val * 229)
                b = int((1.0 - self.t_val) * 153 + self.t_val * 255)

                glow_col = QColor(r, g, b, 90)
                core_col = QColor(r, g, b)

                painter.setBrush(QBrush(glow_col))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawEllipse(pix, 5.0, 5.0)

                painter.setBrush(QBrush(core_col))
                painter.drawEllipse(pix, 2.5, 2.5)

        # Canvas Border
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor(COLOR_BORDER), 2))
        painter.drawRoundedRect(0, 0, w, h, 8, 8)


class DiffusionView(QWidget):
    """Main 2D Toy Diffusion Model & Flow Matching Playground View."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.device = get_device()
        self.sim = DiffusionFlowSimulator(shape_name="heart", device=self.device)

        self.trajectory = None
        self.total_steps = 40
        self.current_step = 0
        self.is_animating = False

        self._init_ui()
        self._compute_trajectory()

        # Animation timer (30 fps)
        self.anim_timer = QTimer(self)
        self.anim_timer.timeout.connect(self._animation_tick)

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(16)

        # Header Title
        hdr_layout = QHBoxLayout()
        title_box = QVBoxLayout()
        title = QLabel("2D TOY DIFFUSION MODEL & FLOW MATCHING PLAYGROUND")
        title.setStyleSheet("font-size: 20px; font-weight: 800; color: #FFFFFF;")
        subtitle = QLabel("Continuous Optimal Transport Flow dx/dt = v(x, t) Generating 2D Manifolds from Gaussian Noise")
        subtitle.setStyleSheet(f"font-size: 12px; color: {COLOR_TEXT_SECONDARY};")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        hdr_layout.addLayout(title_box)
        hdr_layout.addStretch()

        self.btn_animate = QPushButton("▶ Animate Generative Flow")
        self.btn_animate.setStyleSheet(f"""
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
        self.btn_animate.clicked.connect(self._toggle_animation)
        hdr_layout.addWidget(self.btn_animate)
        main_layout.addLayout(hdr_layout)

        # Content Splitter Layout
        content_layout = QHBoxLayout()
        content_layout.setSpacing(20)

        # Left: Diffusion Canvas Card
        canvas_card = Card("2D GENERATIVE FLOW TRAJECTORY")
        self.canvas = DiffusionCanvas(440)
        canvas_card.addWidget(self.canvas)
        content_layout.addWidget(canvas_card)

        # Right: Parameters, Controls & Metrics
        right_panel = QVBoxLayout()
        right_panel.setSpacing(12)

        # Telemetry Metrics Card
        metrics_card = Card("DIFFUSION SAMPLING TELEMETRY")
        m_row = QHBoxLayout()
        self.badge_time = MetricBadge("FLOW TIME (t)", "0.00", "", COLOR_ACCENT_CYAN)
        self.badge_phase = MetricBadge("PHASE", "Gaussian Noise", "", COLOR_ACCENT_PURPLE)
        self.badge_particles = MetricBadge("PARTICLES", "400", "", COLOR_TEXT_PRIMARY)
        m_row.addWidget(self.badge_time)
        m_row.addWidget(self.badge_phase)
        m_row.addWidget(self.badge_particles)
        metrics_card.addLayout(m_row)
        right_panel.addWidget(metrics_card)

        # Target Manifold Card
        shape_card = Card("TARGET GENERATIVE MANIFOLD")
        row_shape = QHBoxLayout()
        lbl_s = QLabel("Target Geometry:")
        lbl_s.setStyleSheet(f"font-weight: 600; color: {COLOR_TEXT_SECONDARY};")
        row_shape.addWidget(lbl_s)

        self.combo_shape = QComboBox()
        self.combo_shape.addItems(["Parametric Heart Curve", "Swiss Roll Spiral", "Two Moons Distribution", "Concentric Rings"])
        self.combo_shape.currentIndexChanged.connect(self._on_shape_changed)
        row_shape.addWidget(self.combo_shape, stretch=1)
        shape_card.addLayout(row_shape)

        self.check_vectors = QCheckBox("Render Velocity Streamlines v(x, t)")
        self.check_vectors.setChecked(True)
        self.check_vectors.setStyleSheet(f"color: {COLOR_TEXT_SECONDARY}; font-weight: 600;")
        self.check_vectors.toggled.connect(self._toggle_vectors)
        shape_card.addWidget(self.check_vectors)
        right_panel.addWidget(shape_card)

        # Interactive Time Slider Card
        slider_card = Card("INTERACTIVE REVERSE TIME CONTROLLER")
        self.slider_time = LabeledSlider("Time Step (0 = Noise → 1 = Target Shape)", 0.0, 1.0, 0.0, step=0.025, is_float=True)
        self.slider_time.valueChanged.connect(self._on_slider_scrubbed)
        slider_card.addWidget(self.slider_time)

        hint = QLabel("💡 Scrub the slider backward and forward to view continuous particle transformation through time.")
        hint.setWordWrap(True)
        hint.setStyleSheet(f"font-size: 11px; color: {COLOR_TEXT_MUTED};")
        slider_card.addWidget(hint)
        right_panel.addWidget(slider_card)

        content_layout.addLayout(right_panel, stretch=1)
        main_layout.addLayout(content_layout)

        # Educational Panel
        edu = EducationalPanel(
            "Continuous Flow Matching & Score-Based Diffusion Models",
            "**Key Principles of Modern Generative Diffusion:**\n\n"
            "- **The Generative Flow Formulation:** Rather than generating images in a single unstable step (like GANs), diffusion models define a continuous trajectory $\\mathbf{x}_t$ connecting pure Gaussian noise $\\mathbf{x}_0 \\sim \\mathcal{N}(0, \\mathbf{I})$ to data distribution $\\mathbf{x}_1$.\n"
            "- **Optimal Transport Flow Matching:** With straight interpolation paths $\\mathbf{x}_t = (1-t)\\mathbf{x}_0 + t\\mathbf{x}_1$, the ground truth velocity is simply $\\mathbf{v}^* = \\mathbf{x}_1 - \\mathbf{x}_0$. A neural network regresses this vector field using simple MSE loss:\n"
            "$$\\mathcal{L}_{\\text{FM}} = \\mathbb{E}_{t, \\mathbf{x}_0, \\mathbf{x}_1} \\| \\mathbf{v}_\\theta(\\mathbf{x}_t, t) - (\\mathbf{x}_1 - \\mathbf{x}_0) \\|^2$$\n"
            "- **Sampling via ODE Integration:** At inference, sampling starts with random noise and integrates along the learned vector field: $\\mathbf{x}_{t+\\Delta t} = \\mathbf{x}_t + \\mathbf{v}_\\theta(\\mathbf{x}_t, t)\\Delta t$."
        )
        main_layout.addWidget(edu)

    def _compute_trajectory(self):
        self.trajectory = self.sim.compute_sampling_trajectory(n_particles=400, steps=self.total_steps)
        self._update_time_slice(0.0)

    def _on_shape_changed(self, idx: int):
        shapes = ["heart", "swiss", "moons", "rings"]
        self.sim.set_shape(shapes[idx])
        self._compute_trajectory()

    def _toggle_vectors(self, checked: bool):
        self.canvas.show_vectors = checked
        self.canvas.update()

    def _on_slider_scrubbed(self, t_val: float):
        if not self.is_animating:
            self._update_time_slice(t_val)

    def _update_time_slice(self, t_val: float):
        t_clamped = max(0.0, min(1.0, t_val))
        step_idx = int(round(t_clamped * self.total_steps))
        step_idx = min(step_idx, len(self.trajectory) - 1)

        particles = self.trajectory[step_idx]
        self.badge_time.set_value(f"{t_clamped:.2f}")

        if t_clamped < 0.2:
            self.badge_phase.set_value("Gaussian Noise")
            self.badge_phase.val_widget.setStyleSheet(f"color: {COLOR_ACCENT_PURPLE}; font-size: 18px; font-weight: 800;")
        elif t_clamped > 0.8:
            self.badge_phase.set_value("Target Manifold")
            self.badge_phase.val_widget.setStyleSheet(f"color: {COLOR_ACCENT_GREEN}; font-size: 18px; font-weight: 800;")
        else:
            self.badge_phase.set_value("Flow Transport")
            self.badge_phase.val_widget.setStyleSheet(f"color: {COLOR_ACCENT_CYAN}; font-size: 18px; font-weight: 800;")

        # Update vector field streamlines for time t
        grid_field = self.sim.get_vector_field_grid(t_clamped, grid_res=14)
        self.canvas.set_vector_field(grid_field)
        self.canvas.set_particles(particles, t_clamped)

    def _toggle_animation(self):
        if self.is_animating:
            self.is_animating = False
            self.anim_timer.stop()
            self.btn_animate.setText("▶ Animate Generative Flow")
            self.btn_animate.setStyleSheet(f"background-color: {COLOR_ACCENT_GREEN}; color: #000; font-weight: 800; padding: 10px 18px; border-radius: 8px;")
        else:
            self.is_animating = True
            self.current_step = 0
            self.anim_timer.start(30)
            self.btn_animate.setText("⏸ Pause Animation")
            self.btn_animate.setStyleSheet(f"background-color: {COLOR_ACCENT_AMBER}; color: #000; font-weight: 800; padding: 10px 18px; border-radius: 8px;")

    def _animation_tick(self):
        if self.current_step > self.total_steps:
            self._toggle_animation()
            return

        t_val = self.current_step / self.total_steps
        self.slider_time.setValue(t_val)
        self._update_time_slice(t_val)
        self.current_step += 1
