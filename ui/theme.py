"""Futuristic Cyber-Lab Dark Stylesheet and Design Tokens for NeuroNetWorks."""

# Color Palette Constants
COLOR_BG_DARK = "#090D16"
COLOR_BG_CARD = "#111827"
COLOR_BG_SURFACE = "#1E293B"
COLOR_BG_HOVER = "#334155"

COLOR_BORDER = "#334155"
COLOR_BORDER_LIGHT = "#475569"

COLOR_ACCENT_CYAN = "#00E5FF"
COLOR_ACCENT_PURPLE = "#A855F7"
COLOR_ACCENT_GREEN = "#10B981"
COLOR_ACCENT_AMBER = "#F59E0B"
COLOR_ACCENT_RED = "#EF4444"

COLOR_TEXT_PRIMARY = "#F8FAFC"
COLOR_TEXT_SECONDARY = "#94A3B8"
COLOR_TEXT_MUTED = "#64748B"

GLOBAL_STYLESHEET = f"""
/* Universal Window & Base Widget Styles */
QWidget {{
    background-color: {COLOR_BG_DARK};
    color: {COLOR_TEXT_PRIMARY};
    font-family: 'Inter', 'Segoe UI', 'Roboto', -apple-system, sans-serif;
    font-size: 13px;
    selection-background-color: {COLOR_ACCENT_CYAN};
    selection-color: #000000;
}}

/* Main Window */
QMainWindow {{
    background-color: {COLOR_BG_DARK};
}}

/* Panels and Cards */
QFrame.CardFrame, QWidget.CardWidget {{
    background-color: {COLOR_BG_CARD};
    border: 1px solid {COLOR_BORDER};
    border-radius: 12px;
}}

QFrame.CardFrame:hover {{
    border: 1px solid {COLOR_BORDER_LIGHT};
}}

/* Section Headers */
QLabel.HeaderTitle {{
    font-size: 20px;
    font-weight: 700;
    color: #FFFFFF;
    letter-spacing: 0.5px;
}}

QLabel.SubTitle {{
    font-size: 12px;
    color: {COLOR_TEXT_SECONDARY};
}}

QLabel.MetricValue {{
    font-size: 24px;
    font-weight: 800;
    color: {COLOR_ACCENT_CYAN};
}}

QLabel.MetricLabel {{
    font-size: 11px;
    font-weight: 600;
    color: {COLOR_TEXT_MUTED};
    text-transform: uppercase;
    letter-spacing: 1px;
}}

/* Push Buttons */
QPushButton {{
    background-color: {COLOR_BG_SURFACE};
    color: {COLOR_TEXT_PRIMARY};
    border: 1px solid {COLOR_BORDER};
    border-radius: 8px;
    padding: 8px 16px;
    font-weight: 600;
    font-size: 12px;
}}

QPushButton:hover {{
    background-color: {COLOR_BG_HOVER};
    border-color: {COLOR_ACCENT_CYAN};
    color: {COLOR_ACCENT_CYAN};
}}

QPushButton:pressed {{
    background-color: #0F172A;
}}

QPushButton:disabled {{
    background-color: #0F172A;
    color: {COLOR_TEXT_MUTED};
    border-color: #1E293B;
}}

/* Primary Action Buttons */
QPushButton.PrimaryBtn {{
    background-color: #0284C7;
    color: #FFFFFF;
    border: 1px solid #38BDF8;
}}

QPushButton.PrimaryBtn:hover {{
    background-color: #0369A1;
    border-color: {COLOR_ACCENT_CYAN};
}}

/* Success Action Buttons */
QPushButton.SuccessBtn {{
    background-color: #059669;
    color: #FFFFFF;
    border: 1px solid #34D399;
}}

QPushButton.SuccessBtn:hover {{
    background-color: #047857;
    border-color: #6EE7B7;
}}

/* Danger Action Buttons */
QPushButton.DangerBtn {{
    background-color: #DC2626;
    color: #FFFFFF;
    border: 1px solid #F87171;
}}

QPushButton.DangerBtn:hover {{
    background-color: #B91C1C;
}}

/* Sliders */
QSlider::groove:horizontal {{
    height: 6px;
    background: #1E293B;
    border-radius: 3px;
}}

QSlider::sub-page:horizontal {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {COLOR_ACCENT_PURPLE}, stop:1 {COLOR_ACCENT_CYAN});
    border-radius: 3px;
}}

QSlider::handle:horizontal {{
    background: #FFFFFF;
    border: 2px solid {COLOR_ACCENT_CYAN};
    width: 16px;
    margin-top: -5px;
    margin-bottom: -5px;
    border-radius: 8px;
}}

QSlider::handle:horizontal:hover {{
    background: {COLOR_ACCENT_CYAN};
    border-color: #FFFFFF;
    transform: scale(1.2);
}}

/* ComboBoxes */
QComboBox {{
    background-color: {COLOR_BG_CARD};
    color: {COLOR_TEXT_PRIMARY};
    border: 1px solid {COLOR_BORDER};
    border-radius: 8px;
    padding: 6px 12px;
    font-weight: 500;
}}

QComboBox:hover {{
    border-color: {COLOR_ACCENT_CYAN};
}}

QComboBox::drop-down {{
    border: none;
    width: 24px;
}}

QComboBox QAbstractItemView {{
    background-color: {COLOR_BG_CARD};
    color: {COLOR_TEXT_PRIMARY};
    border: 1px solid {COLOR_BORDER};
    selection-background-color: #1E293B;
    selection-color: {COLOR_ACCENT_CYAN};
    border-radius: 6px;
    padding: 4px;
}}

/* SpinBoxes */
QSpinBox, QDoubleSpinBox {{
    background-color: {COLOR_BG_CARD};
    color: {COLOR_TEXT_PRIMARY};
    border: 1px solid {COLOR_BORDER};
    border-radius: 8px;
    padding: 6px 10px;
}}

QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: {COLOR_ACCENT_CYAN};
}}

/* Progress Bar */
QProgressBar {{
    background-color: #1E293B;
    border: 1px solid {COLOR_BORDER};
    border-radius: 6px;
    height: 14px;
    text-align: center;
    color: #FFFFFF;
    font-size: 10px;
    font-weight: 700;
}}

QProgressBar::chunk {{
    background: qlineargradient(x1:0, y1:0, x2:1, y2:0, stop:0 {COLOR_ACCENT_PURPLE}, stop:1 {COLOR_ACCENT_CYAN});
    border-radius: 5px;
}}

/* Tab Widget */
QTabWidget::pane {{
    border: 1px solid {COLOR_BORDER};
    border-radius: 12px;
    background-color: {COLOR_BG_CARD};
    top: -1px;
}}

QTabBar::tab {{
    background-color: transparent;
    color: {COLOR_TEXT_SECONDARY};
    padding: 10px 18px;
    font-weight: 600;
    border-bottom: 2px solid transparent;
    margin-right: 4px;
}}

QTabBar::tab:selected {{
    color: {COLOR_ACCENT_CYAN};
    border-bottom: 2px solid {COLOR_ACCENT_CYAN};
}}

QTabBar::tab:hover {{
    color: #FFFFFF;
}}

/* ScrollBars */
QScrollBar:vertical {{
    background: transparent;
    width: 8px;
    margin: 0;
}}

QScrollBar::handle:vertical {{
    background: #334155;
    min-height: 24px;
    border-radius: 4px;
}}

QScrollBar::handle:vertical:hover {{
    background: #475569;
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}

/* Status Bar */
QStatusBar {{
    background-color: #030712;
    border-top: 1px solid {COLOR_BORDER};
    color: {COLOR_TEXT_MUTED};
    font-size: 11px;
}}

/* GroupBox */
QGroupBox {{
    border: 1px solid {COLOR_BORDER};
    border-radius: 10px;
    margin-top: 18px;
    padding-top: 14px;
    font-weight: 700;
    color: {COLOR_TEXT_SECONDARY};
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    left: 12px;
    padding: 0 6px;
    background-color: {COLOR_BG_CARD};
    color: {COLOR_ACCENT_CYAN};
}}
"""
