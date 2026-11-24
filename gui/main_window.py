"""Main window. Heavy work (model loading, downloads, OCR, file counting) runs on background threads
(``gui.workers``); this class only builds the UI, reacts to signals and persists settings."""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Callable

from PyQt6.QtCore import Qt, QTimer, QUrl
from PyQt6.QtGui import QAction, QDesktopServices, QIcon
from PyQt6.QtWidgets import (QCheckBox, QComboBox, QFileDialog, QHBoxLayout, QLabel, QMainWindow, QMenu, QMenuBar,
                             QMessageBox, QProgressBar, QPushButton, QSlider, QVBoxLayout, QWidget,
                             QTabWidget)

from core.archiver import archive_inputs, is_inside
from core.constants import (DEFAULT_DETECTION_MODEL, DEFAULT_RECOGNITION_MODEL, image_extensions)
from core.enhance import super_resolution_available
from core.hardware import HardwareMonitor
from core.paper import AUTO, ORIGINAL, PAGE_SIZE_CHOICES
from core.ocr_processor import OCRProcessor
from core.results import BatchResult, ProgressEvent
from utils import models as model_utils
from utils.app_config import (COMPRESSION_CHOICES, DEVICE_CHOICES, DPI_CHOICES, OUTPUT_FORMAT_CHOICES, TABS,
                              AppConfig)
from utils.app_paths import app_root
from . import theme
from .dialogs import ask_cpu_threads, show_about
from .input_tab import InputTab
from .workers import LoadEvent, LoadTask, OCRJob, Task, fetch_missing

logger = logging.getLogger(__name__)

_STAGE_TEXT = {"loading": "Loading", "ocr": "Recognising", "writing": "Writing", "merging": "Merging PDF"}


def _add_menu(menubar: QMenuBar, title: str) -> QMenu:
    menu = menubar.addMenu(title)
    assert menu is not None
    return menu


def _add_action(menu: QMenu, text: str, slot: Callable[..., object]) -> QAction:
    action = menu.addAction(text)
    assert action is not None
    action.triggered.connect(slot)
    return action


class MainWindow(QMainWindow):
    def __init__(self, config: AppConfig, config_path: Path):
        super().__init__()
        self.config = config
        self.config_path = config_path
        self.processor: OCRProcessor | None = None
        self.job: OCRJob | None = None
        self._tasks: list[Task] = []          # keep references to running background tasks
        self._busy = False                    # model (re)load in progress
        self._loaded_models = (config.detection_model, config.recognition_model)
        self._archive_request: tuple[bool, Path | None] = (False, None)
        self.monitor = HardwareMonitor()
        self.theme_actions: dict[str, QAction] = {}

        self.setWindowTitle("VisionLane OCR")
        self.setMinimumSize(800, 450)
        icon = app_root() / "icon.ico"
        if icon.exists():
            self.setWindowIcon(QIcon(str(icon)))

        self._build_ui()
        self._apply_config_to_widgets()
        self._apply_theme(config.theme_mode)

        # the only timer: refresh the hardware read-out
        self.hw_timer = QTimer(self)
        self.hw_timer.timeout.connect(self._update_hardware)
        self.hw_timer.start(1000)
        self._update_hardware()
        QTimer.singleShot(0, self._load_engine)

    # ================================================================== UI construction
    def _build_ui(self) -> None:
        central = QWidget()
        self.setCentralWidget(central)
        layout = QVBoxLayout(central)
        self._build_menu()

        self.tabs = QTabWidget()
        self.tabs.setMaximumHeight(200)
        self.input_tabs: dict[str, InputTab] = {}
        for kind, title in (("single", "Single File"), ("folder", "Folder"), ("pdf", "PDF")):
            tab = InputTab(kind)
            self.input_tabs[kind] = tab
            self.tabs.addTab(tab, title)
        layout.addWidget(self.tabs)

        # --- DPI / format / device
        row1 = QHBoxLayout()
        self.dpi_combo = QComboBox()
        self.dpi_combo.addItems(DPI_CHOICES)
        self.dpi_combo.setToolTip("Auto: use the DPI stored in the image (300 if none). "
                                  "For PDF input this is the resolution used for recognition.")
        self.format_combo = QComboBox()
        self.format_combo.addItems(OUTPUT_FORMAT_CHOICES)
        self.device_combo = QComboBox()
        for value, label in (("auto", "Auto (GPU if available)"), ("gpu", "GPU"), ("cpu", "CPU")):
            self.device_combo.addItem(label, value)
        self.device_combo.currentIndexChanged.connect(self._on_device_changed)
        for text, widget in (("DPI:", self.dpi_combo), ("Output Format:", self.format_combo),
                             ("Device:", self.device_combo)):
            row1.addWidget(QLabel(text))
            row1.addWidget(widget)
        layout.addLayout(row1)

        # --- models
        row2 = QHBoxLayout()
        self.det_combo = QComboBox()
        self.rec_combo = QComboBox()
        row2.addWidget(QLabel("Detection Model:"))
        row2.addWidget(self.det_combo)
        row2.addWidget(QLabel("Recognition Model:"))
        row2.addWidget(self.rec_combo)
        layout.addLayout(row2)
        self._populate_model_combos()
        self.det_combo.activated.connect(self._on_model_selected)
        self.rec_combo.activated.connect(self._on_model_selected)

        # --- compression
        row3 = QHBoxLayout()
        self.compress_check = QCheckBox("Compress page images in PDF")
        self.compress_check.setToolTip("Re-encodes the page images while the PDF is written. "
                                       "JPEG is lossy; Lossless only applies Flate compression.")
        self.compress_type = QComboBox()
        self.compress_type.addItems([c.upper() if c == "jpeg" else c.capitalize() for c in COMPRESSION_CHOICES])
        self.quality_slider = QSlider(Qt.Orientation.Horizontal)
        self.quality_slider.setRange(1, 100)
        self.quality_slider.setFixedWidth(120)
        self.quality_label = QLabel()
        self.quality_label.setFixedWidth(40)
        self.skip_text_check = QCheckBox("Skip PDF pages that already have text")
        row3.addWidget(self.compress_check)
        row3.addWidget(QLabel("Type:"))
        row3.addWidget(self.compress_type)
        row3.addWidget(QLabel("Quality:"))
        row3.addWidget(self.quality_slider)
        row3.addWidget(self.quality_label)
        row3.addStretch(1)
        row3.addWidget(self.skip_text_check)
        layout.addLayout(row3)

        # --- page handling
        row4 = QHBoxLayout()
        self.page_size_combo = QComboBox()
        for choice in PAGE_SIZE_CHOICES:
            label = {ORIGINAL: "Original size", AUTO: "Auto (most common paper)"}.get(choice, choice)
            self.page_size_combo.addItem(label, choice)
        self.page_size_combo.setToolTip("Give all pages of an image folder the same paper size (image scaled to fit, "
                                        "aspect ratio kept, orientation kept). Not applied to PDF input.")
        self.enhance_check = QCheckBox("Enhance low-resolution pages")
        self.enhance_check.setToolTip("Super-resolution (x2) for pages whose resolution would fall below "
                                      f"{self.config.enhance_min_dpi} dpi on the chosen paper size. "
                                      "Slow; needs the optional 'super-image' package.")
        if not super_resolution_available():
            self.enhance_check.setEnabled(False)
            self.enhance_check.setToolTip("Needs the optional 'super-image' package:  pip install super-image")
        self.straighten_check = QCheckBox("Rotated / skewed pages")
        self.straighten_check.setToolTip("Detect pages scanned sideways or upside down and skewed text lines. "
                                         "Slower; downloads two small extra models on first use.")
        self.straighten_check.toggled.connect(self._on_straighten_toggled)
        row4.addWidget(QLabel("Page size:"))
        row4.addWidget(self.page_size_combo)
        row4.addWidget(self.enhance_check)
        row4.addStretch(1)
        row4.addWidget(self.straighten_check)
        layout.addLayout(row4)
        for signal in (self.compress_check.toggled, self.compress_type.currentIndexChanged, self.quality_slider.valueChanged):
            signal.connect(self._update_compression_controls)

        # --- status
        self.current_label = QLabel("No file processing")
        layout.addWidget(self.current_label)
        prog = QHBoxLayout()
        self.progress_label = QLabel("Files Processed: 0/0")
        self.progress_bar = QProgressBar()
        self.progress_bar.setRange(0, 1000)
        self.progress_bar.setTextVisible(True)
        prog.addWidget(self.progress_label)
        prog.addWidget(self.progress_bar)
        layout.addLayout(prog)

        hw = QHBoxLayout()
        self.device_label, self.usage_label, self.memory_label = QLabel(), QLabel(), QLabel()
        for status_label in (self.device_label, self.usage_label, self.memory_label):
            status_label.setStyleSheet("padding: 5px; margin: 2px;")
            hw.addWidget(status_label)
        layout.addLayout(hw)
        self.notice_label = QLabel()
        self.notice_label.setWordWrap(True)
        self.notice_label.setVisible(False)
        layout.addWidget(self.notice_label)

        buttons = QHBoxLayout()
        self.start_button = QPushButton("Start Processing")
        self.start_button.clicked.connect(self._start_processing)
        self.cancel_button = QPushButton("Cancel")
        self.cancel_button.clicked.connect(self._cancel_processing)
        self.cancel_button.setEnabled(False)
        buttons.addWidget(self.start_button)
        buttons.addWidget(self.cancel_button)
        layout.addLayout(buttons)
        self.start_button.setEnabled(False)  # enabled once the models are loaded

    def _build_menu(self) -> None:
        menubar = self.menuBar()
        assert menubar is not None
        file_menu = _add_menu(menubar, "File")
        _add_action(file_menu, "Open...", self._open_file)
        _add_action(file_menu, "Save Settings", self._save_settings_clicked)
        file_menu.addSeparator()
        _add_action(file_menu, "Exit", self.close)
        _add_action(_add_menu(menubar, "Settings"), "Performance Options", self._performance_options)
        theme_menu = _add_menu(menubar, "Theme")
        for mode, menu_label in (("system", "System Default"), ("light", "Light Mode"), ("dark", "Dark Mode"),
                                 ("night", "Night Mode")):
            action = _add_action(theme_menu, menu_label, lambda _checked=False, m=mode: self._apply_theme(m))
            action.setCheckable(True)
            self.theme_actions[mode] = action
        _add_action(_add_menu(menubar, "Help"), "About", lambda: show_about(self))

    # ================================================================== settings <-> widgets
    def _apply_config_to_widgets(self) -> None:
        cfg = self.config
        self.dpi_combo.setCurrentText(cfg.dpi)
        self.format_combo.setCurrentText(cfg.output_format)
        self.device_combo.blockSignals(True)
        self.device_combo.setCurrentIndex(max(0, DEVICE_CHOICES.index(cfg.device)))
        self.device_combo.blockSignals(False)
        self._select_model(self.det_combo, cfg.detection_model, DEFAULT_DETECTION_MODEL)
        self._select_model(self.rec_combo, cfg.recognition_model, DEFAULT_RECOGNITION_MODEL)
        self.compress_check.setChecked(cfg.compress_enabled)
        self.compress_type.setCurrentIndex(COMPRESSION_CHOICES.index(cfg.compression_type))
        self.quality_slider.setValue(cfg.compression_quality)
        self.skip_text_check.setChecked(cfg.skip_text_pages)
        self.page_size_combo.setCurrentIndex(max(0, self.page_size_combo.findData(cfg.page_size)))
        self.enhance_check.setChecked(cfg.enhance_enabled and self.enhance_check.isEnabled())
        self.straighten_check.blockSignals(True)
        self.straighten_check.setChecked(cfg.straighten_pages)
        self.straighten_check.blockSignals(False)
        self._update_compression_controls()
        for kind in TABS:
            tab = self.input_tabs[kind]
            tab.output_edit.setText(cfg.output_paths.get(kind, ""))
            tab.archive_edit.setText(cfg.archive_paths.get(kind, ""))
            tab.archive_check.setChecked(cfg.archive_enabled.get(kind, False))
            saved = cfg.input_paths.get(kind, "")
            if saved and Path(saved).exists():
                tab.set_input(saved)

    def collect_config(self) -> AppConfig:
        cfg = self.config
        cfg.dpi = self.dpi_combo.currentText()
        cfg.output_format = self.format_combo.currentText()
        cfg.device = self.device_combo.currentData()
        cfg.detection_model = self._loaded_models[0]
        cfg.recognition_model = self._loaded_models[1]
        cfg.compress_enabled = self.compress_check.isChecked()
        cfg.compression_type = COMPRESSION_CHOICES[self.compress_type.currentIndex()]
        cfg.compression_quality = self.quality_slider.value()
        cfg.skip_text_pages = self.skip_text_check.isChecked()
        cfg.page_size = self.page_size_combo.currentData()
        cfg.enhance_enabled = self.enhance_check.isChecked()
        cfg.straighten_pages = self.straighten_check.isChecked()
        for kind, tab in self.input_tabs.items():
            cfg.input_paths[kind] = str(tab.input_path or "")
            cfg.output_paths[kind] = tab.output_edit.text().strip()
            cfg.archive_paths[kind] = tab.archive_edit.text().strip()
            cfg.archive_enabled[kind] = tab.archive_check.isChecked()
        return cfg

    def _save_settings(self) -> bool:
        try:
            self.collect_config().save(self.config_path)
            return True
        except OSError as exc:
            logger.error("Could not save settings: %s", exc)
            return False

    def _save_settings_clicked(self) -> None:
        if self._save_settings():
            QMessageBox.information(self, "Settings", f"Settings saved to\n{self.config_path}")
        else:
            QMessageBox.warning(self, "Settings", "Could not save the settings file.")

    def _apply_theme(self, mode: str) -> None:
        self.config.theme_mode = mode
        self.setStyleSheet(theme.stylesheet(mode))
        self.notice_label.setStyleSheet(theme.NOTICE_STYLE.get(mode, ""))
        for key, action in self.theme_actions.items():
            action.setChecked(key == mode)

    def _update_compression_controls(self, *_args) -> None:
        enabled = self.compress_check.isChecked()
        is_jpeg = COMPRESSION_CHOICES[self.compress_type.currentIndex()] == "jpeg"
        self.compress_type.setEnabled(enabled)
        self.quality_slider.setEnabled(enabled and is_jpeg)
        self.quality_label.setEnabled(enabled and is_jpeg)
        self.quality_label.setText(f"{self.quality_slider.value()}%")

    # ================================================================== models / engine
    def _populate_model_combos(self) -> None:
        for combo, kind in ((self.det_combo, model_utils.DETECTION), (self.rec_combo, model_utils.RECOGNITION)):
            combo.clear()
            for name in model_utils.list_models(kind):
                combo.addItem(name if model_utils.is_cached(name) else f"{name}  \u2b07", name)

    @staticmethod
    def _select_model(combo: QComboBox, wanted: str, fallback: str) -> None:
        idx = combo.findData(wanted)
        if idx < 0:
            idx = combo.findData(fallback)
        combo.setCurrentIndex(max(idx, 0))

    def _selected_models(self) -> tuple[str, str]:
        return self.det_combo.currentData(), self.rec_combo.currentData()

    def _set_busy(self, busy: bool, message: str = "") -> None:
        self._busy = busy
        for widget in (self.det_combo, self.rec_combo, self.device_combo, self.straighten_check):
            widget.setEnabled(not busy)
        self.start_button.setEnabled(not busy and self.processor is not None and self.job is None)
        if busy:
            self.progress_bar.setRange(0, 0)               # busy indicator until real progress arrives
            self.progress_label.setText("Loading models...")
        else:
            self.progress_bar.setRange(0, 1000)
            self.progress_bar.setValue(0)
            self.progress_bar.setFormat("%p%")
            self.progress_label.setText("Files Processed: 0/0")
        if message:
            self.current_label.setText(message)

    def _on_load_event(self, event: LoadEvent) -> None:
        """Model download / load progress: real byte progress while downloading, a busy bar while loading."""
        if event.stage != "download":
            self.progress_bar.setRange(0, 0)
            self.progress_label.setText("Loading models...")
            self.current_label.setText("Loading models into memory...")
            return
        title = f"Downloading model {event.index} of {event.count}: {event.name}" if event.count > 1 \
            else f"Downloading model: {event.name}"
        self.progress_label.setText(f"Download {event.index}/{event.count}")
        if event.total > 0:
            fraction = min(1.0, event.done / event.total)
            self.progress_bar.setRange(0, 1000)
            self.progress_bar.setValue(int(fraction * 1000))
            self.progress_bar.setFormat(f"{fraction:.0%}")      # same rounding as the text on the left
            self.current_label.setText(f"{title} - {event.done / 1e6:.1f} / {event.total / 1e6:.1f} MB ({fraction:.0%})")
        else:
            self.progress_bar.setRange(0, 0)
            self.current_label.setText(f"{title} - {event.done / 1e6:.1f} MB")

    def _load_engine(self) -> None:
        det, rec = self._selected_models()
        device = self.device_combo.currentData()
        threads = self.config.cpu_threads
        straighten = self.straighten_check.isChecked()
        self._set_busy(True, "Loading OCR models (the first run may download them)...")

        def build(emit) -> OCRProcessor:
            fetch_missing(model_utils.models_needed(det, rec, straighten), emit)
            return OCRProcessor(det, rec, device, cpu_threads=threads, straighten_pages=straighten)

        task = LoadTask(build, "engine-load")
        task.progress.connect(self._on_load_event)
        task.succeeded.connect(self._on_engine_ready)
        task.failed.connect(self._on_engine_load_failed)
        self._tasks = [t for t in self._tasks if t.running] + [task]
        task.start()

    def _on_engine_ready(self, processor: OCRProcessor) -> None:
        self.processor = processor
        self._loaded_models = (processor.engine.detection_model, processor.engine.recognition_model)
        self._populate_model_combos()
        self._select_model(self.det_combo, self._loaded_models[0], DEFAULT_DETECTION_MODEL)
        self._select_model(self.rec_combo, self._loaded_models[1], DEFAULT_RECOGNITION_MODEL)
        self.straighten_check.blockSignals(True)
        self.straighten_check.setChecked(processor.engine.straighten_pages)
        self.straighten_check.blockSignals(False)
        self._set_busy(False, "No file processing")
        warnings = processor.engine.report.warnings
        self.notice_label.setText("\u26a0 " + " ".join(warnings) if warnings else "")
        self.notice_label.setVisible(bool(warnings))
        self._update_hardware()

    def _on_engine_load_failed(self, message: str) -> None:
        self._on_engine_failed(message, first_load=True)

    def _on_engine_reload_failed(self, message: str) -> None:
        self._on_engine_failed(message, first_load=False)

    def _on_engine_failed(self, message: str, first_load: bool) -> None:
        logger.error("Model loading failed: %s", message)
        self._set_busy(False, "Models not loaded")
        if not first_load and self.processor is not None:
            # keep working with the previous models
            self._select_model(self.det_combo, self._loaded_models[0], DEFAULT_DETECTION_MODEL)
            self._select_model(self.rec_combo, self._loaded_models[1], DEFAULT_RECOGNITION_MODEL)
            self.device_combo.blockSignals(True)
            self.device_combo.setCurrentIndex(DEVICE_CHOICES.index(self.config.device))
            self.device_combo.blockSignals(False)
            self.straighten_check.blockSignals(True)
            self.straighten_check.setChecked(self.processor.engine.straighten_pages)
            self.straighten_check.blockSignals(False)
            self.start_button.setEnabled(True)
        QMessageBox.critical(self, "Could not load the OCR models",
                             f"{message}\n\nCheck your internet connection (models are downloaded on first use) "
                             "and the log file for details.")

    def _reload(self, action, description: str) -> None:
        """Run ``action(processor, emit)`` (model/device switch) on a background thread; ``emit`` reports progress."""
        if self.processor is None or self.job is not None:
            return
        self._set_busy(True, description)
        processor = self.processor

        def work(emit) -> OCRProcessor:
            action(processor, emit)
            return processor

        task = LoadTask(work, "engine-reload")
        task.progress.connect(self._on_load_event)
        task.succeeded.connect(self._on_engine_ready)
        task.failed.connect(self._on_engine_reload_failed)
        self._tasks = [t for t in self._tasks if t.running] + [task]
        task.start()

    def _on_model_selected(self, _index: int = 0) -> None:
        det, rec = self._selected_models()
        if (det, rec) == self._loaded_models:
            return

        def action(processor: OCRProcessor, emit) -> None:
            fetch_missing(model_utils.models_needed(det, rec, processor.engine.straighten_pages), emit)
            processor.set_models(det, rec)

        self._reload(action, f"Loading models {det} + {rec}...")

    def _on_device_changed(self, _index: int = 0) -> None:
        if self.processor is None:
            return
        preference = self.device_combo.currentData()
        self._reload(lambda p, emit: p.set_device(preference), "Switching processing device...")

    def _on_straighten_toggled(self, enabled: bool) -> None:
        if self.processor is None:
            return
        def action(processor: OCRProcessor, emit) -> None:
            engine = processor.engine
            fetch_missing(model_utils.models_needed(engine.detection_model, engine.recognition_model, enabled), emit)
            processor.set_straighten(enabled)

        self._reload(action, "Loading page-orientation models...")

    def _performance_options(self) -> None:
        threads = ask_cpu_threads(self, self.config.cpu_threads)
        if threads is not None:
            self.config.cpu_threads = threads
            if self.processor is not None:
                self.processor.engine.set_cpu_threads(threads)

    # ================================================================== hardware read-out
    def _update_hardware(self) -> None:
        proc = self.processor
        try:
            snap = self.monitor.snapshot(proc.device if proc else None, proc.engine.gpu if proc else None)
        except Exception as exc:  # pylint: disable=broad-except
            # never let a monitoring glitch escape a Qt slot (PyQt6 aborts the process on that)
            logger.debug("Hardware read-out failed: %s", exc)
            return
        self.device_label.setText(snap.device_text)
        self.usage_label.setText(snap.usage_text)
        self.memory_label.setText(snap.memory_text)

    # ================================================================== processing
    def _open_file(self) -> None:
        pats = " ".join(f"*{e}" for e in image_extensions())
        path, _ = QFileDialog.getOpenFileName(self, "Select File", "", f"Images and PDFs ({pats} *.pdf)")
        if path:
            is_pdf = path.lower().endswith(".pdf")
            self.tabs.setCurrentIndex(2 if is_pdf else 0)
            self.input_tabs["pdf" if is_pdf else "single"].set_input(path)

    def _current_kind(self) -> str:
        return TABS[self.tabs.currentIndex()]

    def _validate(self) -> tuple[str, Path, Path, bool, Path | None] | None:
        """Return (mode, input, output_dir, archive_enabled, archive_dir) or show an error and return None."""
        kind = self._current_kind()
        tab = self.input_tabs[kind]
        labels = {"single": "an image or PDF file", "folder": "a folder", "pdf": "a PDF file"}
        if tab.input_path is None or not tab.input_path.exists():
            QMessageBox.warning(self, "Input missing", f"Please select {labels[kind]}.")
            return None
        out_text = tab.output_edit.text().strip()
        if not out_text:
            QMessageBox.warning(self, "Output missing", "Please select an output directory.")
            return None
        output_dir = Path(out_text)
        if kind == "folder" and is_inside(output_dir, tab.input_path):
            QMessageBox.warning(self, "Output folder", "The output folder must not be inside the input folder "
                                "(the results would be picked up as input next time).")
            return None
        archive_dir = None
        if tab.archive_check.isChecked():
            arch_text = tab.archive_edit.text().strip()
            if not arch_text:
                QMessageBox.warning(self, "Archiving", "Archiving is enabled but no archive directory is set.")
                return None
            archive_dir = Path(arch_text)
            if kind == "folder" and is_inside(archive_dir, tab.input_path):
                QMessageBox.warning(self, "Archiving", "The archive folder must not be inside the input folder.")
                return None
            try:
                archive_dir.mkdir(parents=True, exist_ok=True)
            except OSError as exc:
                QMessageBox.critical(self, "Archiving", f"Cannot create the archive directory:\n{exc}")
                return None
        return ("folder" if kind == "folder" else "file"), tab.input_path, output_dir, archive_dir is not None, archive_dir

    def _start_processing(self) -> None:
        if self.processor is None or self.job is not None or self._busy:
            return
        checked = self._validate()
        if checked is None:
            return
        mode, input_path, output_dir, archive_on, archive_dir = checked
        formats = {"PDF": ["pdf"], "HOCR": ["hocr"], "PDF+HOCR": ["pdf", "hocr"]}[self.format_combo.currentText()]
        dpi_text = self.dpi_combo.currentText()
        self.processor.configure(
            output_formats=formats,
            dpi=None if dpi_text == "Auto" else int(dpi_text),
            compress_enabled=self.compress_check.isChecked(),
            compression_type=COMPRESSION_CHOICES[self.compress_type.currentIndex()],
            compression_quality=self.quality_slider.value(),
            skip_text_pages=self.skip_text_check.isChecked(),
            page_size=self.page_size_combo.currentData(),
            enhance_enabled=self.enhance_check.isChecked(),
            enhance_scale=self.config.enhance_scale,
            enhance_min_dpi=self.config.enhance_min_dpi,
        )
        self._archive_request = (archive_on, archive_dir)
        self._save_settings()

        self.job = OCRJob(self.processor, mode, input_path, output_dir)
        self.job.progress.connect(self._on_progress)
        self.job.succeeded.connect(self._on_job_finished)
        self.job.failed.connect(self._on_job_failed)
        self.start_button.setEnabled(False)
        self.cancel_button.setEnabled(True)
        for widget in (self.det_combo, self.rec_combo, self.device_combo):
            widget.setEnabled(False)
        self.current_label.setText("Starting processing...")
        self.progress_label.setText("Files Processed: 0/0")
        self.progress_bar.setValue(0)
        logger.info("Starting: mode=%s input=%s output=%s", mode, input_path, output_dir)
        self.job.start()

    def _cancel_processing(self) -> None:
        if self.job is not None:
            self.cancel_button.setEnabled(False)
            self.current_label.setText("Cancelling... (stopping at the next model step)")
            self.job.cancel()

    def _on_progress(self, event: ProgressEvent) -> None:
        self.progress_bar.setValue(int(event.fraction * 1000))
        self.progress_label.setText(f"Files Processed: {event.files_done}/{event.total_files}")
        if event.current is not None and event.stage not in ("done", "failed"):
            stage = _STAGE_TEXT.get(event.stage, "Processing")
            pages = f" (page {event.page + 1}/{event.page_count})" if event.page_count > 1 and event.stage == "ocr" else ""
            self.current_label.setText(f"{stage}: {event.current.name}{pages}")

    def _finish_ui(self) -> None:
        self.job = None
        self.cancel_button.setEnabled(False)
        self._set_busy(False, "No file processing")
        self.progress_label.setText("Files Processed: 0/0")
        self.progress_bar.setValue(0)
        for tab in self.input_tabs.values():
            tab.refresh_count()  # archived/processed inputs may be gone

    def _on_job_failed(self, message: str) -> None:
        self._finish_ui()
        QMessageBox.critical(self, "Processing failed", message)

    def _on_job_finished(self, result: BatchResult) -> None:
        archive_on, archive_dir = self._archive_request
        archive_note = ""
        if archive_on and archive_dir is not None and result.total:
            report = archive_inputs(result, archive_dir)
            if result.cancelled:
                archive_note = "Nothing was archived because the run was cancelled."
            else:
                archive_note = f"Archived {len(report.moved)} of {result.total} input file(s) to {archive_dir}."
                if report.errors:
                    archive_note += "\nArchive problems:\n" + "\n".join(report.errors[:10])
        self._finish_ui()

        box = QMessageBox(self)
        if result.total == 0:
            box.setIcon(QMessageBox.Icon.Information)
            box.setWindowTitle("Nothing to do")
            text = "No supported images or PDFs were found."
        elif result.cancelled:
            box.setIcon(QMessageBox.Icon.Warning)
            box.setWindowTitle("Cancelled")
            text = f"Processing was cancelled. {result.summary()}."
        elif result.ok:
            box.setIcon(QMessageBox.Icon.Information)
            box.setWindowTitle("Success")
            text = f"Processing completed successfully! ({result.summary()})"
        else:
            box.setIcon(QMessageBox.Icon.Warning)
            box.setWindowTitle("Completed with problems")
            text = f"Processing finished, but not everything worked: {result.summary()}."
            box.setDetailedText("\n".join(result.failure_lines()))
        if result.session_dir is not None and result.total:
            text += f"\n\nOutput: {result.session_dir}"
        if result.warnings:
            text += "\n\nNote:\n" + "\n".join(result.warnings)
        if archive_note:
            text += f"\n\n{archive_note}"
        box.setText(text)
        open_button = None
        if result.session_dir is not None and result.session_dir.exists():
            open_button = box.addButton("Open Output Folder", QMessageBox.ButtonRole.ActionRole)
        box.addButton(QMessageBox.StandardButton.Ok)
        box.exec()
        if open_button is not None and box.clickedButton() is open_button:
            QDesktopServices.openUrl(QUrl.fromLocalFile(str(result.session_dir)))

    # ================================================================== shutdown
    def closeEvent(self, event) -> None:  # noqa: N802 (Qt naming)
        if self.job is not None:
            answer = QMessageBox.question(self, "Processing in progress",
                                          "Processing is still running. Cancel it and exit?")
            if answer != QMessageBox.StandardButton.Yes:
                event.ignore()
                return
            self.current_label.setText("Stopping...")
            self.job.cancel()
            self.job.join(30)
            if self.job.running:
                logger.warning("OCR job still running after 30 s; exiting anyway")
        self._save_settings()
        self.hw_timer.stop()
        self.monitor.close()
        if self.processor is not None:
            self.processor.release()
        event.accept()
