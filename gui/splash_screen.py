# gui/splash_screen.py
from PyQt6.QtWidgets import QSplashScreen, QProgressBar, QVBoxLayout, QLabel, QWidget, QApplication
from PyQt6.QtCore import Qt, QSize, QCoreApplication
from PyQt6.QtGui import QPixmap, QPainter, QColor
import time
class SplashScreen(QSplashScreen):
    def __init__(self, app=None):
        # Get QApplication instance
        self.app = app or QApplication.instance()
        if not self.app:
            raise RuntimeError("QApplication must be created before SplashScreen")
        # Create a custom pixmap for the splash screen
        pixmap = QPixmap(QSize(400, 200))
        pixmap.fill(Qt.GlobalColor.white)
        # Initialize splash screen with the pixmap
        super().__init__(pixmap, Qt.WindowType.WindowStaysOnTopHint)
        # Create widget to hold content
        self.content = QWidget(self)
        layout = QVBoxLayout(self.content)
        layout.setContentsMargins(20, 20, 20, 20)  # Add margins for better appearance
        # Add title with updated styling
        title = QLabel("VisionLane OCR")
        title.setStyleSheet("""
            QLabel {
                color: #2C3E50;
                font-size: 24px;
                font-weight: bold;
            }
        """)
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        # Add status label with updated styling
        self.status_label = QLabel("Initializing...")
        self.status_label.setStyleSheet("""
            QLabel {
                color: #34495E;
                font-size: 12px;
            }
        """)
        self.status_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.status_label)
        # Add progress bar with updated styling
        self.progress = QProgressBar()
        self.progress.setStyleSheet("""
            QProgressBar {
                border: 2px solid #BDC3C7;
                border-radius: 5px;
                text-align: center;
                height: 20px;
            }
            QProgressBar::chunk {
                background-color: #3498DB;
            }
        """)
        self.progress.setMinimum(0)
        self.progress.setMaximum(100)
        layout.addWidget(self.progress)
        # Set fixed size for content and splash
        self.content.setFixedSize(400, 200)
        self.setFixedSize(400, 200)
        # Center on screen
        screen = self.app.primaryScreen().geometry()
        self.move(
            (screen.width() - self.width()) // 2,
            (screen.height() - self.height()) // 2
        )
        # Force content to be visible
        self.content.show()
        self.content.raise_()
        # Update layout and process events
        layout.activate()
        self.content.updateGeometry()
        QCoreApplication.processEvents()        # Prevent splash from closing automatically by keeping a reference
        self._keep_alive = True
        self._window_ready = False
    def update_status(self, message: str, progress: int = None):
        """Update status message and progress bar"""
        self.status_label.setText(message)
        if progress is not None:
            self.progress.setValue(progress)
        self.content.updateGeometry()
        self.repaint()
        QCoreApplication.processEvents()
    def set_window_ready(self):
        """Mark that the main window is ready"""
        self._window_ready = True
    def finish(self, window):
        """Smoothly transition to main window"""
        # Ensure main window is fully ready
        if hasattr(window, 'show'):
            window.show()
            window.activateWindow()
            window.raise_()
            QCoreApplication.processEvents()
        # Keep splash visible for smooth transition
        self.raise_()
        self.activateWindow()
        QCoreApplication.processEvents()
        # Longer delay for visual transition
        time.sleep(0.3)
        self._keep_alive = False  # Allow splash to close
        super().finish(window)
    def closeEvent(self, event):
        # Prevent closing if _keep_alive is True or window not ready
        if getattr(self, "_keep_alive", False) or not getattr(self, "_window_ready", False):
            event.ignore()
        else:
            super().closeEvent(event)
    def paintEvent(self, event):
        """Custom paint event to draw background and border"""
        painter = QPainter(self)
        painter.drawPixmap(0, 0, self.pixmap())
        painter.setPen(QColor("#BDC3C7"))
        painter.drawRect(0, 0, self.width() - 1, self.height() - 1)
