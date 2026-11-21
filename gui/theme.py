"""Application themes (system / light / dark / night) built from one template."""
from __future__ import annotations

_TEMPLATE = """
QWidget {{ background-color: {bg}; color: {fg}; }}
QLineEdit, QTextEdit, QPlainTextEdit, QComboBox, QSpinBox {{
    background-color: {field}; color: {fg}; border: 1px solid {border};
}}
QPushButton {{ background-color: {button}; color: {fg}; border: 1px solid {border}; }}
QPushButton:disabled {{ color: {muted}; }}
QTabWidget::pane {{ border: 1px solid {border}; }}
QMenuBar, QMenu {{ background-color: {bg}; color: {fg}; }}
QProgressBar {{
    background-color: {field}; color: {fg}; border: 1px solid {border}; text-align: center;
}}
QProgressBar::chunk {{ background-color: #0078d7; }}
"""

PALETTES = {
    "night": {"bg": "#232629", "fg": "#F0F0F0", "field": "#31363b", "button": "#31363b", "border": "#555555", "muted": "#888888"},
    "dark": {"bg": "#2d2d30", "fg": "#dddddd", "field": "#3c3c3c", "button": "#3c3c3c", "border": "#444444", "muted": "#808080"},
    "light": {"bg": "#f6f6f6", "fg": "#222222", "field": "#ffffff", "button": "#eaeaea", "border": "#cccccc", "muted": "#999999"},
}

NOTICE_STYLE = {
    "night": "color: #ffcc66;", "dark": "color: #ffcc66;", "light": "color: #9a5b00;", "system": "color: #b36b00;",
}


def stylesheet(mode: str) -> str:
    """Qt style sheet for ``mode``; empty string for ``system`` (let Qt/OS decide)."""
    palette = PALETTES.get(mode)
    return _TEMPLATE.format(**palette) if palette else ""
