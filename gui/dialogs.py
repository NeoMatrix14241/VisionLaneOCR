"""Small dialogs: performance options and About."""
from __future__ import annotations

import os

from PyQt6.QtWidgets import QDialog, QDialogButtonBox, QFormLayout, QLabel, QMessageBox, QSpinBox, QWidget

ABOUT_HTML = """
<h3>VisionLane OCR</h3>
<p>OCR for scans and PDFs, built with <a href='https://github.com/mindee/doctr'>DocTR</a> and PyQt6.</p>
<ul>
  <li>Images (JPEG, PNG, TIFF incl. multi-page, BMP, GIF, HEIC if installed) and PDF input</li>
  <li>Searchable PDF and HOCR output</li>
  <li>GPU (CUDA) or CPU processing, switchable</li>
  <li>Selectable DocTR detection and recognition models (downloaded on demand)</li>
  <li>Optional JPEG compression of the page images inside the PDF</li>
  <li>Optional archiving of successfully processed inputs</li>
</ul>
<p><b>Author:</b> <a href='https://github.com/NeoMatrix14241'>NeoMatrix14241</a></p>
<p><i>Updates: <a href='https://github.com/NeoMatrix14241/VisionLaneOCR'>GitHub repository</a></i></p>
"""


def show_about(parent: QWidget) -> None:
    QMessageBox.about(parent, "About VisionLane OCR", ABOUT_HTML)


def ask_cpu_threads(parent: QWidget, current: int) -> int | None:
    """Return the new CPU thread count (0 = automatic) or ``None`` if cancelled."""
    dialog = QDialog(parent)
    dialog.setWindowTitle("Performance Options")
    layout = QFormLayout(dialog)
    spin = QSpinBox()
    spin.setRange(0, os.cpu_count() or 1)
    spin.setSpecialValueText("Automatic")
    spin.setValue(current)
    layout.addRow("CPU threads (used when processing on the CPU):", spin)
    layout.addRow(QLabel("The GPU does not use this setting."))
    buttons = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
    buttons.accepted.connect(dialog.accept)
    buttons.rejected.connect(dialog.reject)
    layout.addRow(buttons)
    return spin.value() if dialog.exec() == QDialog.DialogCode.Accepted else None
