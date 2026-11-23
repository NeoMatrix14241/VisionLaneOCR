"""One input tab (Single File / Folder / PDF): input selector, output folder, optional archive folder."""
from __future__ import annotations

from pathlib import Path
from typing import cast

from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (QCheckBox, QFileDialog, QGridLayout, QLabel, QLineEdit, QPushButton,
                             QWidget)

from core.constants import image_extensions
from .workers import Task, count_supported_files

_BTN_STYLE = "QPushButton { font-size: 10pt; padding: 1px 10px; min-height: 25px; max-height: 30px; }"


class InputTab(QWidget):
    """``kind`` is ``single``, ``folder`` or ``pdf``."""

    changed = pyqtSignal(arguments=[])

    TITLES = {"single": "Select Input File", "folder": "Select Input Folder", "pdf": "Select Input PDF"}

    def __init__(self, kind: str):
        super().__init__()  # the parent is set by QTabWidget.addTab
        self.kind = kind
        self._input: Path | None = None
        self._auto_archive = ""
        self._count_task: Task | None = None

        grid = QGridLayout(self)
        grid.setContentsMargins(15, 15, 15, 15)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(8)

        self.select_button = QPushButton(self.TITLES[kind])
        self.select_button.setStyleSheet(_BTN_STYLE)
        self.select_button.setMinimumWidth(140)
        self.select_button.setMaximumWidth(160)
        self.select_button.clicked.connect(self._select_input)
        self.input_label = QLabel("No file selected" if kind != "folder" else "No folder selected")
        self.input_label.setWordWrap(True)
        self.input_label.setMinimumHeight(32)
        self.input_label.setStyleSheet("QLabel { font-size: 10pt; padding: 5px; }")
        self.input_label.setAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)
        grid.addWidget(self.select_button, 0, 0)
        grid.addWidget(self.input_label, 0, 1, 1, 2)

        self.output_edit = QLineEdit()
        self.output_edit.setPlaceholderText("Output directory")
        self.output_edit.setMinimumWidth(220)
        browse_out = QPushButton("Browse Output")
        browse_out.setStyleSheet(_BTN_STYLE)
        browse_out.setFixedWidth(120)
        browse_out.clicked.connect(lambda: self._browse(self.output_edit))
        grid.addWidget(QLabel("Output:"), 1, 0)
        grid.addWidget(self.output_edit, 1, 1)
        grid.addWidget(browse_out, 1, 2)

        self.archive_check = QCheckBox("Archiving?")
        self.archive_edit = QLineEdit()
        self.archive_edit.setPlaceholderText("Archive directory")
        self.archive_edit.setMinimumWidth(220)
        self.archive_button = QPushButton("Browse Archive")
        self.archive_button.setStyleSheet(_BTN_STYLE)
        self.archive_button.setFixedWidth(120)
        self.archive_button.clicked.connect(lambda: self._browse(self.archive_edit))
        grid.addWidget(self.archive_check, 2, 0)
        grid.addWidget(self.archive_edit, 2, 1)
        grid.addWidget(self.archive_button, 2, 2)
        grid.setRowStretch(3, 1)

        self.archive_check.toggled.connect(self._on_archive_toggled)
        self.output_edit.textChanged.connect(self._on_output_changed)
        self._on_archive_toggled(False)

    # ------------------------------------------------------------------ state accessors
    @property
    def input_path(self) -> Path | None:
        return self._input

    def set_input(self, path: Path | str | None) -> None:
        self._input = Path(path) if path else None
        if self._input is None:
            self.input_label.setText("No folder selected" if self.kind == "folder" else "No file selected")
        elif self.kind == "folder":
            self.input_label.setText(f"Selected: {self._input}\nCounting files...")
            self.refresh_count()
        else:
            self.input_label.setText(f"Selected: {self._input.name}")
        self.changed.emit()

    def refresh_count(self) -> None:
        """Count images/PDFs in the selected folder on a background thread."""
        folder = self._input
        if self.kind != "folder" or folder is None or not folder.is_dir():
            return
        task = Task(lambda: count_supported_files(folder), "count-files")
        self._count_task = task
        task.succeeded.connect(self._on_count_done)   # bound slot: auto-disconnected if this tab is destroyed
        task.start()

    def _on_count_done(self, outcome: object) -> None:
        folder, counts = cast("tuple[Path, dict[str, int]]", outcome)
        if self._input == folder:  # ignore stale results
            self.input_label.setText(f"Selected: {folder}\nFound: {counts['images']} images, {counts['pdfs']} PDFs")

    # ------------------------------------------------------------------ dialogs
    def _select_input(self) -> None:
        if self.kind == "folder":
            chosen = QFileDialog.getExistingDirectory(self, "Select Folder Containing Images/PDFs",
                                                      str(self._input or ""))
        else:
            if self.kind == "pdf":
                flt = "PDF Files (*.pdf)"
            else:
                pats = " ".join(f"*{e}" for e in image_extensions())
                flt = f"Images and PDFs ({pats} *.pdf);;Image Files ({pats});;PDF Files (*.pdf)"
            chosen, _ = QFileDialog.getOpenFileName(self, "Select Input", str(self._input.parent if self._input else ""), flt)
        if chosen:
            self.set_input(chosen)

    def _browse(self, edit: QLineEdit) -> None:
        chosen = QFileDialog.getExistingDirectory(self, "Select Directory", edit.text())
        if chosen:
            edit.setText(chosen)

    # ------------------------------------------------------------------ archive defaults
    def _on_archive_toggled(self, enabled: bool) -> None:
        self.archive_edit.setEnabled(enabled)
        self.archive_button.setEnabled(enabled)
        if enabled and not self.archive_edit.text().strip() and self.output_edit.text().strip():
            self._auto_archive = str(Path(self.output_edit.text().strip()) / "archive")
            self.archive_edit.setText(self._auto_archive)

    def _on_output_changed(self, text: str) -> None:
        """Keep the *suggested* archive folder in step with the output folder, but never overwrite a custom one."""
        current = self.archive_edit.text().strip()
        if self.archive_check.isChecked() and text.strip() and (not current or current == self._auto_archive):
            self._auto_archive = str(Path(text.strip()) / "archive")
            self.archive_edit.setText(self._auto_archive)
