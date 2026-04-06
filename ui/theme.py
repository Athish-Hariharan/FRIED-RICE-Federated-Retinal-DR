DARK_THEME = """
QMainWindow, QWidget {
    background-color: #0b0f14;
    color: #e2e8f0;
    font-family: 'Segoe UI', Arial, sans-serif;
    font-size: 13px;
}

QGroupBox {
    border: 1px solid #1e2d3d;
    border-radius: 6px;
    margin-top: 12px;
    padding-top: 12px;
    color: #00d4ff;
    font-size: 11px;
    font-weight: bold;
    letter-spacing: 1px;
}
QGroupBox::title {
    subcontrol-origin: margin;
    left: 10px;
    padding: 0 4px;
}

QLabel {
    color: #e2e8f0;
    font-size: 13px;
}

QPushButton {
    background-color: transparent;
    color: #00d4ff;
    border: 1px solid #00d4ff;
    border-radius: 3px;
    padding: 7px 14px;
    font-size: 11px;
    letter-spacing: 1px;
}
QPushButton:hover {
    background-color: rgba(0, 212, 255, 0.1);
}
QPushButton:pressed {
    background-color: rgba(0, 212, 255, 0.2);
}
QPushButton:disabled {
    color: #2a4058;
    border-color: #1e2d3d;
}
QPushButton#stop_btn {
    color: #ff6b6b;
    border-color: #ff6b6b;
}
QPushButton#stop_btn:hover {
    background-color: rgba(255, 107, 107, 0.1);
}
QPushButton#export_btn {
    color: #ffd93d;
    border-color: #ffd93d;
}
QPushButton#export_btn:hover {
    background-color: rgba(255, 217, 61, 0.1);
}
QPushButton#export_btn:disabled {
    color: #2a4058;
    border-color: #1e2d3d;
}

QSlider::groove:horizontal {
    height: 4px;
    background: #1e2d3d;
    border-radius: 2px;
}
QSlider::handle:horizontal {
    background: #00d4ff;
    width: 14px;
    height: 14px;
    margin: -5px 0;
    border-radius: 7px;
}
QSlider::sub-page:horizontal {
    background: #00d4ff;
    border-radius: 2px;
}

QComboBox {
    background-color: #111720;
    color: #e2e8f0;
    border: 1px solid #1e2d3d;
    border-radius: 3px;
    padding: 4px 8px;
}
QComboBox:hover { border-color: #00d4ff; }
QComboBox QAbstractItemView {
    background-color: #111720;
    color: #e2e8f0;
    selection-background-color: #1e2d3d;
}
QComboBox::drop-down { border: none; }

QSpinBox {
    background-color: #111720;
    color: #e2e8f0;
    border: 1px solid #1e2d3d;
    border-radius: 3px;
    padding: 4px 8px;
}
QSpinBox:hover { border-color: #00d4ff; }

QScrollBar:vertical {
    background: #0b0f14;
    width: 6px;
    border-radius: 3px;
}
QScrollBar::handle:vertical {
    background: #1e2d3d;
    border-radius: 3px;
    min-height: 20px;
}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }

QTextEdit, QPlainTextEdit {
    background-color: #050a0f;
    color: #8899a6;
    border: 1px solid #1e2d3d;
    border-radius: 3px;
    font-family: 'Courier New', monospace;
    font-size: 11px;
    padding: 4px;
}

QStatusBar {
    background-color: #050a0f;
    color: #4a6278;
    font-family: 'Courier New', monospace;
    font-size: 11px;
    border-top: 1px solid #1e2d3d;
}
"""
