"""Module 6: Reinforcement Learning Cart-Pole Balancer (DQN)."""

import os
import numpy as np
import torch

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QFrame, QSplitter
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QPointF
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush, QFont

from core.cartpole_models import CartPoleEnv, DQNAgent
from core.device import get_device
from ui.theme import (
    COLOR_BG_DARK, COLOR_BG_CARD, COLOR_BG_SURFACE, COLOR_BORDER,
    COLOR_ACCENT_CYAN, COLOR_ACCENT_PURPLE, COLOR_ACCENT_GREEN,
    COLOR_ACCENT_AMBER, COLOR_ACCENT_RED, COLOR_TEXT_PRIMARY,
    COLOR_TEXT_SECONDARY, COLOR_TEXT_MUTED
)
from ui.custom_widgets import Card, MetricBadge, LabeledSlider, EducationalPanel

WEIGHTS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models_weights", "cartpole_dqn.pth")

class PhysicsCanvas(QWidget):
    """60 FPS smooth physics rendering of cart, track, wheels, and inverted pole."""

    def __init__(self, size_w: int = 560, size_h: int = 340, parent=None):
        super().__init__(parent)
        self.setMinimumSize(size_w, size_h)

        self.cart_x = 0.0
        self.pole_angle = 0.0
        self.pole_length = 0.5
        self.last_action = 0
        self.action_active = False

    def update_physics(self, state: np.ndarray, action: int, pole_len: float):
        self.cart_x = float(state[0])
        self.pole_angle = float(state[2])
        self.pole_length = pole_len
        self.last_action = action
        self.action_active = True
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()

        painter.fillRect(0, 0, w, h, QColor(COLOR_BG_DARK))

        # Ground Track Line
        track_y = h * 0.72
        painter.setPen(QPen(QColor(COLOR_BORDER), 3))
        painter.drawLine(30, int(track_y), w - 30, int(track_y))

        # Track tick markers
        for x in range(50, w - 30, 60):
            painter.setPen(QPen(QColor("#334155"), 1))
            painter.drawLine(x, int(track_y), x, int(track_y + 8))

        # Position mapping: x is in [-2.4, 2.4] meters
        scale_x = (w - 140) / 4.8
        cart_pixel_x = (w / 2) + self.cart_x * scale_x
        cart_pixel_y = track_y - 20

        # Draw Force Impulse Arrows (Left / Right)
        if self.action_active:
            if self.last_action == 0:  # Push Left
                painter.setPen(QPen(QColor(COLOR_ACCENT_AMBER), 2))
                painter.drawLine(int(cart_pixel_x + 50), int(cart_pixel_y + 10), int(cart_pixel_x + 25), int(cart_pixel_y + 10))
                painter.drawLine(int(cart_pixel_x + 30), int(cart_pixel_y + 5), int(cart_pixel_x + 25), int(cart_pixel_y + 10))
                painter.drawLine(int(cart_pixel_x + 30), int(cart_pixel_y + 15), int(cart_pixel_x + 25), int(cart_pixel_y + 10))
            else:  # Push Right
                painter.setPen(QPen(QColor(COLOR_ACCENT_CYAN), 2))
                painter.drawLine(int(cart_pixel_x - 50), int(cart_pixel_y + 10), int(cart_pixel_x - 25), int(cart_pixel_y + 10))
                painter.drawLine(int(cart_pixel_x - 30), int(cart_pixel_y + 5), int(cart_pixel_x - 25), int(cart_pixel_y + 10))
                painter.drawLine(int(cart_pixel_x - 30), int(cart_pixel_y + 15), int(cart_pixel_x - 25), int(cart_pixel_y + 10))

        # Draw Cart Body
        cart_w, cart_h = 64, 30
        painter.setBrush(QBrush(QColor(COLOR_BG_SURFACE)))
        painter.setPen(QPen(QColor(COLOR_ACCENT_CYAN), 2))
        painter.drawRoundedRect(int(cart_pixel_x - cart_w / 2), int(cart_pixel_y - cart_h / 2), cart_w, cart_h, 6, 6)

        # Draw Wheels
        wheel_r = 7
        painter.setBrush(QBrush(QColor("#0F172A")))
        painter.setPen(QPen(QColor(COLOR_TEXT_MUTED), 2))
        painter.drawEllipse(int(cart_pixel_x - 20 - wheel_r), int(track_y - wheel_r * 2), wheel_r * 2, wheel_r * 2)
        painter.drawEllipse(int(cart_pixel_x + 20 - wheel_r), int(track_y - wheel_r * 2), wheel_r * 2, wheel_r * 2)

        # Draw Inverted Pendulum Pole
        pixel_pole_len = self.pole_length * 200.0  # visual scaling
        # Inverted pendulum: angle 0 is vertical UP
        pole_tip_x = cart_pixel_x + pixel_pole_len * np.sin(self.pole_angle)
        pole_tip_y = cart_pixel_y - pixel_pole_len * np.cos(self.pole_angle)

        # Color pole according to tilt hazard
        abs_deg = np.degrees(abs(self.pole_angle))
        if abs_deg < 5.0:
            pole_color = QColor(COLOR_ACCENT_GREEN)
        elif abs_deg < 10.0:
            pole_color = QColor(COLOR_ACCENT_AMBER)
        else:
            pole_color = QColor(COLOR_ACCENT_RED)

        painter.setPen(QPen(pole_color, 6, Qt.PenStyle.SolidLine, Qt.PenCapStyle.RoundCap))
        painter.drawLine(int(cart_pixel_x), int(cart_pixel_y), int(pole_tip_x), int(pole_tip_y))

        # Axle Pin Joint
        painter.setBrush(QBrush(QColor("#FFFFFF")))
        painter.setPen(QPen(QColor("#000000"), 1))
        painter.drawEllipse(int(cart_pixel_x - 4), int(cart_pixel_y - 4), 8, 8)

        # Tip Ball
        painter.setBrush(QBrush(pole_color))
        painter.drawEllipse(int(pole_tip_x - 6), int(pole_tip_y - 6), 12, 12)

        # Danger zone angle limit threshold arcs
        max_rad = 12 * 2 * np.pi / 360
        left_arc_x = cart_pixel_x + pixel_pole_len * np.sin(-max_rad)
        left_arc_y = cart_pixel_y - pixel_pole_len * np.cos(-max_rad)
        right_arc_x = cart_pixel_x + pixel_pole_len * np.sin(max_rad)
        right_arc_y = cart_pixel_y - pixel_pole_len * np.cos(max_rad)

        painter.setPen(QPen(QColor(239, 68, 68, 60), 1, Qt.PenStyle.DashLine))
        painter.drawLine(int(cart_pixel_x), int(cart_pixel_y), int(left_arc_x), int(left_arc_y))
        painter.drawLine(int(cart_pixel_x), int(cart_pixel_y), int(right_arc_x), int(right_arc_y))

        # Canvas Border
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor(COLOR_BORDER), 2))
        painter.drawRoundedRect(0, 0, w, h, 8, 8)


class CartPoleView(QWidget):
    """Main Reinforcement Learning Cart-Pole Balancer View."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.device = get_device()
        self.env = CartPoleEnv()
        self.agent = DQNAgent(state_dim=4, action_dim=2, lr=0.002, epsilon_decay=0.99)

        # Load pre-trained solved policy weights
        if os.path.exists(WEIGHTS_PATH):
            try:
                self.agent.q_net.load_state_dict(torch.load(WEIGHTS_PATH, map_location=self.device))
                self.agent.target_net.load_state_dict(self.agent.q_net.state_dict())
                self.agent.epsilon = 0.01  # Greedy evaluation
            except Exception as e:
                print("Error loading DQN weights:", e)

        self.mode = "pretrained"  # "pretrained", "train_scratch", "manual"
        self.current_score = 0
        self.high_score = 0
        self.episodes_done = 0
        self.external_disturbance = 0.0

        self._init_ui()

        # Physics simulation timer (50 Hz = 20ms)
        self.sim_timer = QTimer(self)
        self.sim_timer.timeout.connect(self._sim_step)
        self.sim_timer.start(20)

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(16)

        # Header Title
        hdr_layout = QHBoxLayout()
        title_box = QVBoxLayout()
        title = QLabel("REINFORCEMENT LEARNING CART-POLE BALANCER (DQN)")
        title.setStyleSheet("font-size: 20px; font-weight: 800; color: #FFFFFF;")
        subtitle = QLabel("Deep Q-Network with Experience Replay, Newtonian Physics & Interactive Disturbance Perturbations")
        subtitle.setStyleSheet(f"font-size: 12px; color: {COLOR_TEXT_SECONDARY};")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        hdr_layout.addLayout(title_box)
        hdr_layout.addStretch()

        # Kick buttons for instant interactive disturbance
        lbl_disturb = QLabel("PHYSICAL NUDGE:")
        lbl_disturb.setStyleSheet(f"font-size: 11px; font-weight: 700; color: {COLOR_TEXT_MUTED};")
        hdr_layout.addWidget(lbl_disturb)

        self.btn_nudge_left = QPushButton("⬅ Push Left")
        self.btn_nudge_left.clicked.connect(lambda: self._apply_disturbance(-16.0))
        hdr_layout.addWidget(self.btn_nudge_left)

        self.btn_nudge_right = QPushButton("Push Right ➡")
        self.btn_nudge_right.clicked.connect(lambda: self._apply_disturbance(16.0))
        hdr_layout.addWidget(self.btn_nudge_right)

        self.btn_reset = QPushButton("↺ Reset Env")
        self.btn_reset.clicked.connect(self._reset_environment)
        hdr_layout.addWidget(self.btn_reset)

        main_layout.addLayout(hdr_layout)

        # Content Splitter (Left: Physics Canvas, Right: Q-Values & Controls)
        content_layout = QHBoxLayout()
        content_layout.setSpacing(20)

        # Left: Physics Canvas Card
        canvas_card = Card("NEWTONIAN DYNAMICS SIMULATOR (RUNGE-KUTTA 4TH ORDER)")
        self.canvas = PhysicsCanvas(580, 360)
        canvas_card.addWidget(self.canvas)
        content_layout.addWidget(canvas_card, stretch=2)

        # Right: Q-Values & Telemetry Panel
        right_panel = QVBoxLayout()
        right_panel.setSpacing(12)

        # Telemetry Metrics Card
        metrics_card = Card("EPISODIC PERFORMANCE")
        m_row = QHBoxLayout()
        self.badge_score = MetricBadge("SURVIVAL SCORE", "0", "steps", COLOR_ACCENT_GREEN)
        self.badge_high = MetricBadge("RECORD", "0", "steps", COLOR_ACCENT_CYAN)
        self.badge_eps = MetricBadge("EPSILON (ε)", "0.01", "", COLOR_ACCENT_AMBER)
        m_row.addWidget(self.badge_score)
        m_row.addWidget(self.badge_high)
        m_row.addWidget(self.badge_eps)
        metrics_card.addLayout(m_row)
        right_panel.addWidget(metrics_card)

        # Live Q-Value Decision Engine Card
        q_card = Card("DEEP Q-NETWORK ACTION VALUE ESTIMATION Q(s, a)")

        # Q Left
        row_ql = QHBoxLayout()
        lbl_ql = QLabel("Q(s, Push Left):")
        lbl_ql.setFixedWidth(130)
        lbl_ql.setStyleSheet("font-weight: 700; font-size: 11px;")
        row_ql.addWidget(lbl_ql)

        self.bar_ql = QLabel("0.00")
        self.bar_ql.setStyleSheet(f"font-size: 14px; font-weight: 800; color: {COLOR_ACCENT_AMBER};")
        row_ql.addWidget(self.bar_ql)
        row_ql.addStretch()
        q_card.addLayout(row_ql)

        # Q Right
        row_qr = QHBoxLayout()
        lbl_qr = QLabel("Q(s, Push Right):")
        lbl_qr.setFixedWidth(130)
        lbl_qr.setStyleSheet("font-weight: 700; font-size: 11px;")
        row_qr.addWidget(lbl_qr)

        self.bar_qr = QLabel("0.00")
        self.bar_qr.setStyleSheet(f"font-size: 14px; font-weight: 800; color: {COLOR_ACCENT_CYAN};")
        row_qr.addWidget(self.bar_qr)
        row_qr.addStretch()
        q_card.addLayout(row_qr)

        # State Telemetry HUD
        state_hud = QHBoxLayout()
        self.lbl_hud_x = QLabel("x: 0.0m")
        self.lbl_hud_theta = QLabel("θ: 0.0°")
        self.lbl_hud_x.setStyleSheet(f"color: {COLOR_TEXT_MUTED}; font-size: 11px;")
        self.lbl_hud_theta.setStyleSheet(f"color: {COLOR_TEXT_MUTED}; font-size: 11px;")
        state_hud.addWidget(self.lbl_hud_x)
        state_hud.addWidget(self.lbl_hud_theta)
        q_card.addLayout(state_hud)
        right_panel.addWidget(q_card)

        # Operational Mode Card
        mode_card = Card("AGENT EXECUTION MODE")
        row_mode = QHBoxLayout()
        lbl_mode = QLabel("Policy Mode:")
        lbl_mode.setStyleSheet(f"font-weight: 600; color: {COLOR_TEXT_SECONDARY};")
        row_mode.addWidget(lbl_mode)

        self.combo_mode = QComboBox()
        self.combo_mode.addItems(["Pre-trained Master Policy (Flawless Balance)", "Live Training from Scratch (DQN Replay)", "Manual Human Control"])
        self.combo_mode.currentIndexChanged.connect(self._on_mode_changed)
        row_mode.addWidget(self.combo_mode, stretch=1)
        mode_card.addLayout(row_mode)

        # Physics Customization Sliders
        self.slider_gravity = LabeledSlider("Planetary Gravity (m/s²)", 1.62, 24.8, 9.8, step=0.5, is_float=True)
        self.slider_gravity.valueChanged.connect(lambda g: setattr(self.env, "gravity", g))
        mode_card.addWidget(self.slider_gravity)

        self.slider_pole = LabeledSlider("Pole Length (m)", 0.2, 1.2, 0.5, step=0.05, is_float=True)
        self.slider_pole.valueChanged.connect(lambda l: setattr(self.env, "length", l))
        mode_card.addWidget(self.slider_pole)

        right_panel.addWidget(mode_card)
        content_layout.addLayout(right_panel, stretch=1)
        main_layout.addLayout(content_layout)

        # Educational Panel
        edu = EducationalPanel(
            "Deep Q-Networks (DQN) & Reinforcement Learning",
            "**Core Reinforcement Learning Mechanics:**\n\n"
            "- **The Bellman Optimality Equation:** The neural network approximates $Q^*(s, a)$, the maximum cumulative expected future reward discounted by $\\gamma=0.99$:\n"
            "$$Q(s, a) \\leftarrow r + \\gamma \\max_{a'} Q_{\\text{target}}(s', a')$$\n"
            "- **Experience Replay Buffer:** Storing past experiences $(s, a, r, s', d)$ and sampling random mini-batches breaks temporal correlations between consecutive simulation frames, preventing catastrophic forgetting.\n"
            "- **Target Network Freezing:** Using a separate target network $Q_{\\text{target}}$ prevents the moving-target instability inherent to bootstrapping values."
        )
        main_layout.addWidget(edu)

    def _apply_disturbance(self, force: float):
        self.external_disturbance += force

    def _reset_environment(self):
        self.env.reset()
        self.current_score = 0
        self.badge_score.set_value("0")

    def _on_mode_changed(self, idx: int):
        if idx == 0:  # Pre-trained
            self.mode = "pretrained"
            if os.path.exists(WEIGHTS_PATH):
                self.agent.q_net.load_state_dict(torch.load(WEIGHTS_PATH, map_location=self.device))
            self.agent.epsilon = 0.01
        elif idx == 1:  # Train scratch
            self.mode = "train_scratch"
            self.agent = DQNAgent(state_dim=4, action_dim=2, lr=0.002, epsilon_decay=0.99)
            self.agent.epsilon = 1.0
        else:
            self.mode = "manual"

        self._reset_environment()

    def _sim_step(self):
        state = self.env.state.copy()

        # Action selection
        eval_mode = (self.mode == "pretrained")
        action, q_values = self.agent.select_action(state, evaluate=eval_mode)

        # Apply disturbance
        disturb = self.external_disturbance
        self.external_disturbance *= 0.5  # rapid decay

        # Physics step
        next_state, reward, done, _ = self.env.step(action, external_disturbance=disturb)

        # Live training step if active
        if self.mode == "train_scratch":
            shaped_reward = reward - 0.5 * abs(state[0]) - 1.0 * abs(state[2])
            self.agent.remember(state, action, shaped_reward, next_state, done)
            self.agent.train_step()

        self.current_score += 1
        if self.current_score > self.high_score:
            self.high_score = self.current_score

        # Update telemetry
        self.badge_score.set_value(str(self.current_score))
        self.badge_high.set_value(str(self.high_score))
        self.badge_eps.set_value(f"{self.agent.epsilon:.2f}")

        self.bar_ql.setText(f"{q_values[0]:.2f}")
        self.bar_qr.setText(f"{q_values[1]:.2f}")

        # HUD
        self.lbl_hud_x.setText(f"Position: {state[0]:.2f}m")
        self.lbl_hud_theta.setText(f"Tilt: {np.degrees(state[2]):.1f}°")

        # Refresh canvas
        self.canvas.update_physics(state, action, self.env.length)

        # Auto-reset upon failure
        if done or self.current_score > 500:
            if self.mode == "train_scratch":
                self.episodes_done += 1
                if self.episodes_done % 5 == 0:
                    self.agent.update_target_network()
            self._reset_environment()
