"""The start-up splash shown while PyTorch/DocTR are imported in the background."""
from __future__ import annotations

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QLabel, QProgressBar, QVBoxLayout, QWidget


class StartupSplash(QWidget):
    def __init__(self) -> None:
        super().__init__(None, Qt.WindowType.SplashScreen | Qt.WindowType.FramelessWindowHint
                         | Qt.WindowType.WindowStaysOnTopHint)
        self.setObjectName("splash")
        self.setStyleSheet(
            "#splash { background-color: white; border: 3px solid #2196F3; border-radius: 15px; }"
            "QLabel { background: transparent; }"
        )
        self.resize(450, 240)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(40, 40, 40, 40)
        layout.setSpacing(18)
        title = QLabel("VisionLane OCR")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        title.setStyleSheet("color: #2196F3; font-size: 26pt; font-weight: bold;")
        self.status = QLabel("Starting...")
        self.status.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.status.setStyleSheet("color: #555555; font-size: 11pt;")
        progress = QProgressBar()
        progress.setRange(0, 0)  # indeterminate: we do not know how long the imports take
        progress.setTextVisible(False)
        progress.setFixedHeight(14)
        layout.addWidget(title)
        layout.addWidget(self.status)
        layout.addWidget(progress)

    def set_status(self, text: str) -> None:
        self.status.setText(text)

    def center_on_screen(self) -> None:
        screen = self.screen()
        if screen is not None:
            geo = screen.availableGeometry()
            self.move(geo.center().x() - self.width() // 2, geo.center().y() - self.height() // 2)
