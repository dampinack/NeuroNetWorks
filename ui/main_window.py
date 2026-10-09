"""Master Application Window for NeuroNetWorks."""

import sys
import torch
import numpy as np

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QLabel,
    QPushButton, QStackedWidget, QStatusBar, QFrame, QButtonGroup
)
from PyQt6.QtCore import Qt, QSize
from PyQt6.QtGui import QIcon, QFont

from ui.theme import (
    GLOBAL_STYLESHEET, COLOR_BG_DARK, COLOR_BG_CARD, COLOR_BG_SURFACE,
    COLOR_BORDER, COLOR_ACCENT_CYAN, COLOR_ACCENT_PURPLE, COLOR_ACCENT_GREEN,
    COLOR_TEXT_PRIMARY, COLOR_TEXT_SECONDARY, COLOR_TEXT_MUTED
)
from ui.custom_widgets import MetricBadge
from ui.auxiliary_window import AuxiliaryNeuralWindow

from ui.modules.mnist_view import MNISTView
from ui.modules.boundary_view import BoundaryView
from ui.modules.autoencoder_view import AutoencoderView
from ui.modules.approximator_view import ApproximatorView
from ui.modules.signal_view import SignalView
from ui.modules.cartpole_view import CartPoleView
from ui.modules.transformer_view import TransformerView
from ui.modules.adversarial_view import AdversarialView
from ui.modules.diffusion_view import DiffusionView
from core.device import get_device_name

class MainWindow(QMainWindow):
    """The central shell orchestrating all 6 interactive neural network modules."""

    def __init__(self):
        super().__init__()
        self.setWindowTitle("NeuroNetWorks - Deep Learning Interactive Educational Suite")
        self.resize(1340, 880)
        self.setStyleSheet(GLOBAL_STYLESHEET)

        # Auxiliary Neural Activity Inspector window (can float or dock)
        self.aux_window = AuxiliaryNeuralWindow()

        self._init_ui()

    def _init_ui(self):
        central_widget = QWidget()
        self.setCentralWidget(central_widget)
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # 1. Top Header Application Banner
        top_bar = self._create_top_bar()
        main_layout.addWidget(top_bar)

        # 2. Body Area (Sidebar Navigation + Stacked View Area)
        body_layout = QHBoxLayout()
        body_layout.setContentsMargins(0, 0, 0, 0)
        body_layout.setSpacing(0)

        # Navigation Sidebar
        sidebar = self._create_sidebar()
        body_layout.addWidget(sidebar)

        # Stacked Views Container
        self.stack = QStackedWidget()
        self.stack.setStyleSheet(f"background-color: {COLOR_BG_DARK};")

        # Instantiate Module Views
        self.view_mnist = MNISTView()
        self.view_boundary = BoundaryView()
        self.view_autoencoder = AutoencoderView()
        self.view_approximator = ApproximatorView()
        self.view_signal = SignalView()
        self.view_cartpole = CartPoleView()
        self.view_transformer = TransformerView()
        self.view_adversarial = AdversarialView()
        self.view_diffusion = DiffusionView()

        # Wire MNIST telemetry to Auxiliary Inspector
        self.view_mnist.telemetry_ready.connect(self.aux_window.update_telemetry)
        self.view_mnist.open_auxiliary_requested.connect(self._open_auxiliary_window)

        self.stack.addWidget(self.view_mnist)         # Index 0
        self.stack.addWidget(self.view_boundary)      # Index 1
        self.stack.addWidget(self.view_autoencoder)   # Index 2
        self.stack.addWidget(self.view_approximator)  # Index 3
        self.stack.addWidget(self.view_signal)        # Index 4
        self.stack.addWidget(self.view_cartpole)      # Index 5
        self.stack.addWidget(self.view_transformer)   # Index 6
        self.stack.addWidget(self.view_adversarial)   # Index 7
        self.stack.addWidget(self.view_diffusion)     # Index 8

        body_layout.addWidget(self.stack, stretch=1)
        main_layout.addLayout(body_layout, stretch=1)

        # 3. Status Bar
        self.status = QStatusBar()
        self.setStatusBar(self.status)
        self._setup_status_bar()

    def _create_top_bar(self) -> QWidget:
        bar = QFrame()
        bar.setFixedHeight(64)
        bar.setStyleSheet(f"""
            QFrame {{
                background-color: #060911;
                border-bottom: 1px solid {COLOR_BORDER};
            }}
        """)
        layout = QHBoxLayout(bar)
        layout.setContentsMargins(20, 0, 20, 0)
        layout.setSpacing(16)

        # App Brand & Logo
        logo_box = QHBoxLayout()
        logo_icon = QLabel("🧠")
        logo_icon.setStyleSheet("font-size: 26px;")
        logo_box.addWidget(logo_icon)

        title_box = QVBoxLayout()
        lbl_brand = QLabel("NEURONETWORKS")
        lbl_brand.setStyleSheet(f"""
            font-size: 17px;
            font-weight: 900;
            color: #FFFFFF;
            letter-spacing: 1.5px;
        """)
        lbl_sub = QLabel("Cutting-Edge Deep Learning Interactive Suite")
        lbl_sub.setStyleSheet(f"font-size: 11px; color: {COLOR_ACCENT_CYAN}; font-weight: 600;")
        title_box.addWidget(lbl_brand)
        title_box.addWidget(lbl_sub)
        logo_box.addLayout(title_box)
        layout.addLayout(logo_box)

        layout.addStretch()

        # Auxiliary Window Quick Toggle
        self.btn_top_aux = QPushButton("🔍 Neural Activity Inspector")
        self.btn_top_aux.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLOR_BG_SURFACE};
                border: 1px solid {COLOR_ACCENT_CYAN};
                color: {COLOR_ACCENT_CYAN};
                font-weight: 700;
                padding: 8px 16px;
                border-radius: 8px;
            }}
            QPushButton:hover {{
                background-color: {COLOR_ACCENT_CYAN};
                color: #000000;
            }}
        """)
        self.btn_top_aux.clicked.connect(self._open_auxiliary_window)
        layout.addWidget(self.btn_top_aux)

        # Hardware Badge
        dev_name = get_device_name()
        badge_dev = QLabel(f"⚡ {dev_name}")
        badge_dev.setStyleSheet(f"""
            background-color: #111827;
            color: {COLOR_TEXT_SECONDARY};
            border: 1px solid {COLOR_BORDER};
            border-radius: 6px;
            padding: 6px 12px;
            font-size: 11px;
            font-weight: 600;
        """)
        layout.addWidget(badge_dev)

        return bar

    def _create_sidebar(self) -> QWidget:
        sidebar = QFrame()
        sidebar.setFixedWidth(240)
        sidebar.setStyleSheet(f"""
            QFrame {{
                background-color: #0A0E17;
                border-right: 1px solid {COLOR_BORDER};
            }}
        """)
        layout = QVBoxLayout(sidebar)
        layout.setContentsMargins(10, 16, 10, 16)
        layout.setSpacing(6)

        lbl_nav = QLabel("DEMO SUITE MODULES")
        lbl_nav.setStyleSheet(f"font-size: 10px; font-weight: 800; color: {COLOR_TEXT_MUTED}; letter-spacing: 1px; padding-left: 8px;")
        layout.addWidget(lbl_nav)

        self.nav_buttons = []
        modules = [
            ("✍️", "MNIST Digit Recognizer", "CNN & Saliency Map"),
            ("🌐", "2D Decision Boundaries", "Multi-Layer Perceptron"),
            ("🌌", "Autoencoder Explorer", "2D Latent Manifold"),
            ("📈", "Universal Approximator", "1D MLP Regression"),
            ("📡", "Signal Denoising", "1D-CNN Oscilloscope"),
            ("🕹️", "RL Cart-Pole Balancer", "Deep Q-Network (DQN)"),
            ("👁️", "Vision Transformer", "Self-Attention (ViT)"),
            ("🎯", "Adversarial Studio", "FGSM & PGD Attacks"),
            ("🌀", "2D Toy Diffusion", "Flow Matching Generative"),
        ]

        self.btn_group = QButtonGroup(self)
        self.btn_group.setExclusive(True)

        for idx, (icon, title, desc) in enumerate(modules):
            btn = QPushButton(f"{icon}  {title}")
            btn.setCheckable(True)
            btn.setStyleSheet(f"""
                QPushButton {{
                    text-align: left;
                    background-color: transparent;
                    color: {COLOR_TEXT_SECONDARY};
                    border: 1px solid transparent;
                    border-radius: 8px;
                    padding: 8px 12px;
                    font-size: 11px;
                    font-weight: 600;
                }}
                QPushButton:hover {{
                    background-color: #161F30;
                    color: #FFFFFF;
                }}
                QPushButton:checked {{
                    background-color: #111D30;
                    color: {COLOR_ACCENT_CYAN};
                    border: 1px solid {COLOR_ACCENT_CYAN};
                    font-weight: 700;
                }}
            """)
            btn.clicked.connect(lambda checked, i=idx: self._switch_module(i))
            self.btn_group.addButton(btn, idx)
            layout.addWidget(btn)
            self.nav_buttons.append(btn)

        self.nav_buttons[0].setChecked(True)
        layout.addStretch()

        # Educational Suite Footnote
        foot = QLabel("PyTorch & NumPy\nReal-Time Computation")
        foot.setStyleSheet(f"color: {COLOR_TEXT_MUTED}; font-size: 10px; padding-left: 8px;")
        layout.addWidget(foot)

        return sidebar

    def _setup_status_bar(self):
        lbl_tech = QLabel(f"PyTorch v{torch.__version__} | NumPy v{np.__version__} | PyQt6 GUI Framework")
        lbl_tech.setStyleSheet(f"color: {COLOR_TEXT_MUTED}; padding-right: 12px; font-size: 11px;")
        self.status.addPermanentWidget(lbl_tech)
        self.status.showMessage("Ready. Select a module from the left navigation sidebar.")

    def _switch_module(self, idx: int):
        self.stack.setCurrentIndex(idx)
        names = [
            "MNIST Handwriting Recognizer (CNN)",
            "2D Decision Boundary Playground (MLP)",
            "Autoencoder 2D Latent Explorer & Denoising",
            "Universal Function Approximator (1D Regression)",
            "Real-Time Signal & Audio Waveform Denoising (1D-CNN)",
            "Reinforcement Learning Cart-Pole Balancer (DQN)",
            "Vision Transformer (ViT) & Multi-Head Self-Attention",
            "Adversarial Attack Studio (FGSM & PGD Robustness)",
            "2D Toy Diffusion Model & Flow Matching Generative Playground"
        ]
        self.status.showMessage(f"Active Module: {names[idx]}")

    def _open_auxiliary_window(self):
        """Shows and focuses the dedicated Auxiliary Neural Activity Inspector window."""
        self.aux_window.show()
        self.aux_window.raise_()
        self.aux_window.activateWindow()
