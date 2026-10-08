"""Reusable custom GUI components for the NeuroNetWorks suite."""

from PyQt6.QtWidgets import (
    QWidget, QFrame, QVBoxLayout, QHBoxLayout, QLabel, QSlider,
    QPushButton, QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal, QPropertyAnimation, QEasingCurve, pyqtProperty
from PyQt6.QtGui import QColor, QPainter, QBrush, QPen, QFont
from ui.theme import (
    COLOR_BG_CARD, COLOR_BG_SURFACE, COLOR_BORDER, COLOR_ACCENT_CYAN,
    COLOR_ACCENT_GREEN, COLOR_TEXT_PRIMARY, COLOR_TEXT_SECONDARY, COLOR_TEXT_MUTED
)

class Card(QFrame):
    """A sleek card panel with rounded corners and border."""
    def __init__(self, title: str = "", parent=None):
        super().__init__(parent)
        self.setObjectName("CardFrame")
        self.setStyleSheet(f"""
            QFrame#CardFrame {{
                background-color: {COLOR_BG_CARD};
                border: 1px solid {COLOR_BORDER};
                border-radius: 12px;
            }}
        """)
        self.layout_ = QVBoxLayout(self)
        self.layout_.setContentsMargins(16, 16, 16, 16)
        self.layout_.setSpacing(12)

        if title:
            self.header_label = QLabel(title)
            self.header_label.setStyleSheet(f"""
                font-size: 14px;
                font-weight: 700;
                color: #FFFFFF;
                letter-spacing: 0.5px;
            """)
            self.layout_.addWidget(self.header_label)

    def addWidget(self, widget: QWidget):
        self.layout_.addWidget(widget)

    def addLayout(self, layout):
        self.layout_.addLayout(layout)


class MetricBadge(QFrame):
    """High-contrast telemetry badge for key metrics (accuracy, loss, epoch, etc.)."""
    def __init__(self, label: str, value: str = "--", unit: str = "", accent_color: str = COLOR_ACCENT_CYAN, parent=None):
        super().__init__(parent)
        self.accent_color = accent_color
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {COLOR_BG_SURFACE};
                border: 1px solid {COLOR_BORDER};
                border-radius: 8px;
                padding: 6px 12px;
            }}
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(8, 6, 8, 6)
        layout.setSpacing(2)

        self.label_widget = QLabel(label.upper())
        self.label_widget.setStyleSheet(f"""
            font-size: 10px;
            font-weight: 700;
            color: {COLOR_TEXT_MUTED};
            letter-spacing: 1px;
        """)
        layout.addWidget(self.label_widget)

        self.val_layout = QHBoxLayout()
        self.val_layout.setContentsMargins(0, 0, 0, 0)
        self.val_layout.setSpacing(4)

        self.val_widget = QLabel(value)
        self.val_widget.setStyleSheet(f"""
            font-size: 18px;
            font-weight: 800;
            color: {self.accent_color};
        """)
        self.val_layout.addWidget(self.val_widget)

        if unit:
            self.unit_widget = QLabel(unit)
            self.unit_widget.setStyleSheet(f"""
                font-size: 11px;
                font-weight: 600;
                color: {COLOR_TEXT_SECONDARY};
                padding-top: 4px;
            """)
            self.val_layout.addWidget(self.unit_widget)

        self.val_layout.addStretch()
        layout.addLayout(self.val_layout)

    def set_value(self, val_str: str):
        self.val_widget.setText(val_str)


class LabeledSlider(QWidget):
    """Horizontal slider with real-time value indicator badge."""
    valueChanged = pyqtSignal(float)

    def __init__(
        self,
        label: str,
        min_val: float,
        max_val: float,
        default_val: float,
        step: float = 1.0,
        is_float: bool = False,
        format_str: str = "{:.2f}",
        parent=None
    ):
        super().__init__(parent)
        self.min_val = min_val
        self.max_val = max_val
        self.step = step
        self.is_float = is_float
        self.format_str = format_str

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 4, 0, 4)
        layout.setSpacing(4)

        # Header with label and current value
        hdr = QHBoxLayout()
        self.lbl = QLabel(label)
        self.lbl.setStyleSheet(f"font-size: 12px; font-weight: 600; color: {COLOR_TEXT_SECONDARY};")
        hdr.addWidget(self.lbl)
        hdr.addStretch()

        self.val_lbl = QLabel(self._format_value(default_val))
        self.val_lbl.setStyleSheet(f"""
            font-size: 12px;
            font-weight: 700;
            color: {COLOR_ACCENT_CYAN};
            background-color: {COLOR_BG_SURFACE};
            padding: 2px 8px;
            border-radius: 4px;
        """)
        hdr.addWidget(self.val_lbl)
        layout.addLayout(hdr)

        # Slider
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.multiplier = 1000 if is_float else 1
        self.slider.setMinimum(int(min_val * self.multiplier))
        self.slider.setMaximum(int(max_val * self.multiplier))
        self.slider.setValue(int(default_val * self.multiplier))
        self.slider.valueChanged.connect(self._on_change)
        layout.addWidget(self.slider)

    def _format_value(self, val: float) -> str:
        if self.is_float:
            return self.format_str.format(val)
        return str(int(val))

    def _on_change(self, int_val: int):
        val = int_val / self.multiplier if self.is_float else float(int_val)
        self.val_lbl.setText(self._format_value(val))
        self.valueChanged.emit(val)

    def value(self) -> float:
        int_val = self.slider.value()
        return int_val / self.multiplier if self.is_float else float(int_val)

    def setValue(self, val: float):
        self.slider.setValue(int(val * self.multiplier))


class PulseIndicator(QWidget):
    """Glowing LED indicator circle showing real-time network activity."""
    def __init__(self, size: int = 16, color: str = COLOR_ACCENT_GREEN, parent=None):
        super().__init__(parent)
        self.setFixedSize(size, size)
        self.color = QColor(color)
        self.active = True

    def set_active(self, active: bool):
        self.active = active
        self.update()

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        w, h = self.width(), self.height()
        if self.active:
            # Outer glow
            glow = QColor(self.color)
            glow.setAlpha(60)
            painter.setBrush(QBrush(glow))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(1, 1, w - 2, h - 2)

            # Core light
            painter.setBrush(QBrush(self.color))
            painter.drawEllipse(w // 4, h // 4, w // 2, h // 2)
        else:
            painter.setBrush(QBrush(QColor(COLOR_TEXT_MUTED)))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(w // 4, h // 4, w // 2, h // 2)


class EducationalPanel(QFrame):
    """Expandable drawer providing interactive deep learning mathematical intuition."""
    def __init__(self, title: str, content_markdown: str, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: #0A0E17;
                border: 1px solid #1E293B;
                border-radius: 10px;
            }}
        """)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(12, 10, 12, 10)
        layout.setSpacing(8)

        # Header toggle button
        self.toggle_btn = QPushButton(f"🧠 Educational Insights: {title}  ▼")
        self.toggle_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                border: none;
                text-align: left;
                font-weight: 700;
                font-size: 12px;
                color: {COLOR_ACCENT_CYAN};
                padding: 0;
            }}
            QPushButton:hover {{
                color: #FFFFFF;
            }}
        """)
        self.toggle_btn.clicked.connect(self._toggle)
        layout.addWidget(self.toggle_btn)

        self.text_widget = QLabel()
        self.text_widget.setTextFormat(Qt.TextFormat.MarkdownText)
        self.text_widget.setText(content_markdown)
        self.text_widget.setWordWrap(True)
        self.text_widget.setStyleSheet(f"""
            color: {COLOR_TEXT_SECONDARY};
            font-size: 12px;
            line-height: 1.5;
            padding-top: 4px;
        """)
        self.text_widget.setVisible(False)
        layout.addWidget(self.text_widget)

        self.is_expanded = False

    def _toggle(self):
        self.is_expanded = not self.is_expanded
        self.text_widget.setVisible(self.is_expanded)
        arrow = "▲" if self.is_expanded else "▼"
        self.toggle_btn.setText(f"🧠 Educational Insights  {arrow}")
