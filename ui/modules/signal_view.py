"""Module 5: Real-Time Signal & Audio Waveform Denoising (1D-CNN)."""

import os
import numpy as np
import torch

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QFrame, QSplitter
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal, QPointF
from PyQt6.QtGui import QPainter, QColor, QPen, QBrush

from core.signal_models import (
    SignalDenoising1DCNN, generate_synthetic_signal, inject_noise, compute_snr_gain
)
from core.device import get_device
from ui.theme import (
    COLOR_BG_DARK, COLOR_BG_CARD, COLOR_BG_SURFACE, COLOR_BORDER,
    COLOR_ACCENT_CYAN, COLOR_ACCENT_PURPLE, COLOR_ACCENT_GREEN,
    COLOR_ACCENT_AMBER, COLOR_ACCENT_RED, COLOR_TEXT_PRIMARY,
    COLOR_TEXT_SECONDARY, COLOR_TEXT_MUTED
)
from ui.custom_widgets import Card, MetricBadge, LabeledSlider, EducationalPanel

WEIGHTS_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), "models_weights", "signal_denoiser.pth")

class OscilloscopeCanvas(QWidget):
    """Real-time 60 FPS oscilloscope displaying Time-Domain waveforms and FFT Spectrum."""

    def __init__(self, size_w: int = 560, size_h: int = 360, parent=None):
        super().__init__(parent)
        self.setMinimumSize(size_w, size_h)

        self.clean_sig = None
        self.noisy_sig = None
        self.denoised_sig = None
        self.show_noisy = True
        self.show_clean = True
        self.show_denoised = True

    def set_waveforms(self, clean: np.ndarray, noisy: np.ndarray, denoised: np.ndarray):
        self.clean_sig = clean
        self.noisy_sig = noisy
        self.denoised_sig = denoised
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        w, h = self.width(), self.height()

        painter.fillRect(0, 0, w, h, QColor(COLOR_BG_DARK))

        # Split into Top (Time Domain Oscilloscope) and Bottom (FFT Spectrum)
        h_time = int(h * 0.62)
        h_fft = h - h_time

        # ---------------- 1. TIME DOMAIN OSCILLOSCOPE ----------------
        # Grid lines
        painter.setPen(QPen(QColor("#1E293B"), 1, Qt.PenStyle.DashLine))
        center_y = h_time / 2
        painter.drawLine(20, int(center_y), w - 20, int(center_y))
        for x in range(40, w - 20, 80):
            painter.drawLine(x, 15, x, h_time - 10)

        if self.clean_sig is not None:
            n_pts = len(self.clean_sig)
            dx = (w - 40) / max(1, n_pts - 1)

            def to_pix(idx, val):
                px = 20 + idx * dx
                # Val range roughly [-2.0, 2.0]
                py = center_y - (val / 2.0) * (center_y - 20)
                return QPointF(px, py)

            # Draw Noisy Signal (Red/Amber)
            if self.show_noisy and self.noisy_sig is not None:
                painter.setPen(QPen(QColor(239, 68, 68, 140), 1.2))
                for i in range(n_pts - 1):
                    painter.drawLine(to_pix(i, self.noisy_sig[i]), to_pix(i + 1, self.noisy_sig[i + 1]))

            # Draw Clean Signal (Subtle Green)
            if self.show_clean:
                painter.setPen(QPen(QColor(16, 185, 129, 180), 1.5, Qt.PenStyle.DashLine))
                for i in range(n_pts - 1):
                    painter.drawLine(to_pix(i, self.clean_sig[i]), to_pix(i + 1, self.clean_sig[i + 1]))

            # Draw 1D-CNN Denoised Output (Vivid Cyan)
            if self.show_denoised and self.denoised_sig is not None:
                painter.setPen(QPen(QColor(COLOR_ACCENT_CYAN), 2.5))
                for i in range(n_pts - 1):
                    painter.drawLine(to_pix(i, self.denoised_sig[i]), to_pix(i + 1, self.denoised_sig[i + 1]))

        # Time-domain labels
        painter.setPen(QColor(COLOR_TEXT_MUTED))
        painter.setFont(painter.font())
        painter.drawText(30, 24, "TIME DOMAIN OSCILLOSCOPE")

        painter.setPen(QColor(COLOR_ACCENT_RED))
        painter.drawText(220, 24, "━ Noisy Input")
        painter.setPen(QColor(COLOR_ACCENT_GREEN))
        painter.drawText(330, 24, "--- Clean Ground Truth")
        painter.setPen(QColor(COLOR_ACCENT_CYAN))
        painter.drawText(500, 24, "━ 1D-CNN Denoised")

        # ---------------- 2. FFT SPECTRUM ----------------
        painter.setPen(QPen(QColor(COLOR_BORDER), 1))
        painter.drawLine(0, h_time, w, h_time)

        painter.setPen(QColor(COLOR_TEXT_MUTED))
        painter.drawText(30, h_time + 20, "FFT FREQUENCY SPECTRUM (0 - Fs/2)")

        if self.noisy_sig is not None and self.denoised_sig is not None:
            # Compute real FFT magnitudes
            fft_noisy = np.abs(np.fft.rfft(self.noisy_sig))[:64]
            fft_clean = np.abs(np.fft.rfft(self.denoised_sig))[:64]

            max_mag = max(np.max(fft_noisy), 1.0)
            n_bins = len(fft_noisy)
            bin_w = (w - 60) / max(1, n_bins)
            base_y = h - 15

            for bi in range(n_bins):
                bx = 30 + bi * bin_w
                # Noisy bar (dim red)
                bh_noisy = (fft_noisy[bi] / max_mag) * (h_fft - 35)
                painter.fillRect(int(bx), int(base_y - bh_noisy), int(max(1, bin_w - 2)), int(bh_noisy), QColor(239, 68, 68, 80))

                # Denoised bar (cyan)
                bh_denoised = (fft_clean[bi] / max_mag) * (h_fft - 35)
                painter.fillRect(int(bx), int(base_y - bh_denoised), int(max(1, bin_w - 2)), int(bh_denoised), QColor(COLOR_ACCENT_CYAN))

        # Border
        painter.setBrush(Qt.BrushStyle.NoBrush)
        painter.setPen(QPen(QColor(COLOR_BORDER), 2))
        painter.drawRoundedRect(0, 0, w, h, 8, 8)


class SignalView(QWidget):
    """Main Real-Time Signal & Audio Waveform Denoising View."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.device = get_device()
        self.model = SignalDenoising1DCNN().to(self.device)

        if os.path.exists(WEIGHTS_PATH):
            try:
                self.model.load_state_dict(torch.load(WEIGHTS_PATH, map_location=self.device))
                self.model.eval()
            except Exception as e:
                print("Error loading signal denoiser weights:", e)

        # Signal parameters
        self.time_offset = 0.0
        self.sample_rate = 512
        self.time_window = np.linspace(0, 1.0, self.sample_rate, endpoint=False)
        self.signal_type = "harmonics"
        self.signal_freq = 5.0
        self.noise_mode = "gaussian"
        self.noise_intensity = 0.35
        self.is_streaming = True

        self._init_ui()

        # Stream update timer (30 updates per second)
        self.stream_timer = QTimer(self)
        self.stream_timer.timeout.connect(self._stream_step)
        self.stream_timer.start(33)

    def _init_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(20, 20, 20, 20)
        main_layout.setSpacing(16)

        # Header Title
        hdr_layout = QHBoxLayout()
        title_box = QVBoxLayout()
        title = QLabel("REAL-TIME SIGNAL & AUDIO WAVEFORM DENOISING (1D-CNN)")
        title.setStyleSheet("font-size: 20px; font-weight: 800; color: #FFFFFF;")
        subtitle = QLabel("High-Throughput Deep 1D Convolutional Neural Filter with Real-Time Oscilloscope & Spectral FFT")
        subtitle.setStyleSheet(f"font-size: 12px; color: {COLOR_TEXT_SECONDARY};")
        title_box.addWidget(title)
        title_box.addWidget(subtitle)
        hdr_layout.addLayout(title_box)
        hdr_layout.addStretch()

        self.btn_stream = QPushButton("⏸ Pause Stream")
        self.btn_stream.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLOR_ACCENT_AMBER};
                color: #000;
                font-weight: 800;
                padding: 10px 18px;
                border-radius: 8px;
            }}
        """)
        self.btn_stream.clicked.connect(self._toggle_stream)
        hdr_layout.addWidget(self.btn_stream)
        main_layout.addLayout(hdr_layout)

        # Main Content Layout
        content_layout = QHBoxLayout()
        content_layout.setSpacing(20)

        # Left: Oscilloscope Canvas Card
        canvas_card = Card("DUAL-DOMAIN WAVEFORM & SPECTRAL ANALYZER")
        self.canvas = OscilloscopeCanvas(580, 380)
        canvas_card.addWidget(self.canvas)
        content_layout.addWidget(canvas_card, stretch=2)

        # Right: Signal Controls & Metrics
        right_panel = QVBoxLayout()
        right_panel.setSpacing(12)

        # Telemetry Metrics Card
        metrics_card = Card("SIGNAL-TO-NOISE RATIO (SNR) TELEMETRY")
        m_row = QHBoxLayout()
        self.badge_snr_in = MetricBadge("INPUT SNR", "--", "dB", COLOR_ACCENT_AMBER)
        self.badge_snr_out = MetricBadge("OUTPUT SNR", "--", "dB", COLOR_ACCENT_CYAN)
        self.badge_gain = MetricBadge("NET RECOVERY GAIN", "--", "dB", COLOR_ACCENT_GREEN)
        m_row.addWidget(self.badge_snr_in)
        m_row.addWidget(self.badge_snr_out)
        m_row.addWidget(self.badge_gain)
        metrics_card.addLayout(m_row)
        right_panel.addWidget(metrics_card)

        # Signal Synthesis Card
        sig_card = Card("PHYSICAL / BIOLOGICAL WAVEFORM SYNTHESIZER")
        row_sig = QHBoxLayout()
        lbl_stype = QLabel("Waveform:")
        lbl_stype.setStyleSheet(f"font-weight: 600; color: {COLOR_TEXT_SECONDARY};")
        row_sig.addWidget(lbl_stype)

        self.combo_signal = QComboBox()
        self.combo_signal.addItems(["Acoustic Harmonics / Chord", "ECG Electrocardiogram Pulse", "Linear Chirp Sweep (FM)", "Band-Limited Square Wave"])
        self.combo_signal.currentIndexChanged.connect(self._on_signal_changed)
        row_sig.addWidget(self.combo_signal, stretch=1)
        sig_card.addLayout(row_sig)

        self.slider_freq = LabeledSlider("Carrier Frequency (Hz)", 1.0, 15.0, 5.0, step=0.5, is_float=True)
        self.slider_freq.valueChanged.connect(lambda v: setattr(self, "signal_freq", v))
        sig_card.addWidget(self.slider_freq)
        right_panel.addWidget(sig_card)

        # Noise Injection Card
        noise_card = Card("NOISE & DISTORTION INJECTION")
        row_noise = QHBoxLayout()
        lbl_ntype = QLabel("Noise Mode:")
        lbl_ntype.setStyleSheet(f"font-weight: 600; color: {COLOR_TEXT_SECONDARY};")
        row_noise.addWidget(lbl_ntype)

        self.combo_noise = QComboBox()
        self.combo_noise.addItems(["Gaussian White Noise", "Impulsive Spikes & Clicks", "60Hz Powerline AC Hum"])
        self.combo_noise.currentIndexChanged.connect(self._on_noise_mode_changed)
        row_noise.addWidget(self.combo_noise, stretch=1)
        noise_card.addLayout(row_noise)

        self.slider_noise = LabeledSlider("Noise Amplitude (σ)", 0.05, 0.8, 0.35, step=0.05, is_float=True)
        self.slider_noise.valueChanged.connect(lambda v: setattr(self, "noise_intensity", v))
        noise_card.addWidget(self.slider_noise)
        right_panel.addWidget(noise_card)

        content_layout.addLayout(right_panel, stretch=1)
        main_layout.addLayout(content_layout)

        # Educational Panel
        edu = EducationalPanel(
            "1D Deep Convolutional Temporal Filters vs Classic DSP",
            "**Why 1D-CNNs Outperform Classic FIR/IIR Filters:**\n\n"
            "- **Non-Stationary Noise Filtering:** Traditional Fourier bandpass filters assume noise and signal occupy disjoint frequency bands. When wideband Gaussian noise or transient clicks corrupt the signal, classic filters either blur sharp features (like ECG QRS spikes) or let noise pass through.\n"
            "- **Receptive Field Hierarchy:** 1D Conv layers learn hierarchical temporal motifs: early layers detect sharp zero-crossings and micro-slopes, while deeper layers capture harmonic periodicity and structural envelopes.\n"
            "- **Real-Time Zero-Phase Inference:** The fully-convolutional architecture processes streaming audio blocks with sub-millisecond PyTorch tensor operations."
        )
        main_layout.addWidget(edu)

    def _on_signal_changed(self, idx: int):
        types = ["harmonics", "ecg", "chirp", "square"]
        self.signal_type = types[idx]

    def _on_noise_mode_changed(self, idx: int):
        modes = ["gaussian", "spikes", "hum"]
        self.noise_mode = modes[idx]

    def _toggle_stream(self):
        if self.is_streaming:
            self.is_streaming = False
            self.stream_timer.stop()
            self.btn_stream.setText("▶ Resume Stream")
            self.btn_stream.setStyleSheet(f"background-color: {COLOR_ACCENT_GREEN}; color: #000; font-weight: 800; padding: 10px 18px; border-radius: 8px;")
        else:
            self.is_streaming = True
            self.stream_timer.start(33)
            self.btn_stream.setText("⏸ Pause Stream")
            self.btn_stream.setStyleSheet(f"background-color: {COLOR_ACCENT_AMBER}; color: #000; font-weight: 800; padding: 10px 18px; border-radius: 8px;")

    def _stream_step(self):
        # Advance time window
        self.time_offset += 0.03
        t = self.time_window + self.time_offset

        # 1. Generate clean signal
        clean = generate_synthetic_signal(self.signal_type, t, freq=self.signal_freq)

        # 2. Inject noise
        noisy, _ = inject_noise(
            clean,
            noise_level=self.noise_intensity,
            noise_mode=self.noise_mode,
            hum_freq=60.0,
            time_points=t
        )

        # 3. 1D-CNN Neural Denoising Forward Pass
        self.model.eval()
        with torch.no_grad():
            noisy_t = torch.from_numpy(noisy.astype(np.float32)).unsqueeze(0).unsqueeze(0).to(self.device)
            denoised_t = self.model(noisy_t)
            denoised = denoised_t.squeeze().cpu().numpy()

        # 4. Telemetry metrics
        metrics = compute_snr_gain(clean, noisy, denoised)
        self.badge_snr_in.set_value(f"{metrics['snr_in']:.1f}")
        self.badge_snr_out.set_value(f"{metrics['snr_out']:.1f}")
        gain = metrics['gain']
        self.badge_gain.set_value(f"+{gain:.1f}" if gain >= 0 else f"{gain:.1f}")

        # 5. Update Oscilloscope Canvas
        self.canvas.set_waveforms(clean, noisy, denoised)
