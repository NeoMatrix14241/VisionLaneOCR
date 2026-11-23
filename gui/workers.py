"""Background work for the GUI. Everything slow runs here; the GUI thread only reacts to signals.

Plain daemon ``threading.Thread`` s are used (not ``QThread``/``QRunnable``): a daemon thread can never
keep the process alive or trigger "QThread destroyed while running" on shutdown. Signals emitted from
these threads are delivered to the GUI thread through Qt's queued connections.

Lifetime rules (a violation shows up as a crash, not as an exception):

* a :class:`Task` has **no Qt parent** - if a widget owned it, Qt could delete it while its thread still
  emits signals; the Python reference held by the running thread keeps it alive instead;
* connect ``succeeded`` / ``failed`` to **bound methods of a QObject**, not to lambdas: Qt disconnects those
  automatically when the receiver is destroyed, whereas a lambda would run against a deleted widget.
"""
from __future__ import annotations

import logging
import threading
from pathlib import Path
from typing import Callable

from PyQt6.QtCore import QObject, pyqtSignal

from core.archiver import ArchiveReport
from core.constants import image_extensions
from core.ocr_processor import OCRProcessor
from core.results import BatchResult

logger = logging.getLogger(__name__)


class Task(QObject):
    """Run ``fn()`` on a background thread; emit ``succeeded(result)`` or ``failed(message)``."""

    succeeded = pyqtSignal(object, arguments=["result"])
    failed = pyqtSignal(str, arguments=["message"])

    def __init__(self, fn: Callable[[], object], name: str = "task"):
        super().__init__()
        self._fn = fn
        self._name = name
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, name=self._name, daemon=True)
        self._thread.start()

    def _run(self) -> None:
        try:
            result = self._fn()
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("%s failed: %s", self._name, exc, exc_info=True)
            self.failed.emit(str(exc) or exc.__class__.__name__)
        else:
            self.succeeded.emit(result)

    @property
    def running(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def join(self, timeout: float | None = None) -> None:
        if self._thread is not None:
            self._thread.join(timeout)


class OCRJob(Task):
    """One OCR run. ``progress`` carries :class:`core.results.ProgressEvent`; ``succeeded`` carries the
    :class:`core.results.BatchResult` (also for runs with failed files - they are *results*)."""

    progress = pyqtSignal(object, arguments=["event"])

    def __init__(self, processor: OCRProcessor, mode: str, path: Path, output_dir: Path):
        super().__init__(self._execute, "ocr-job")
        self.processor = processor
        self.mode = mode
        self.path = path
        self.output_dir = output_dir

    def _execute(self) -> BatchResult:
        self.processor.progress_callback = self.progress.emit
        try:
            if self.mode == "folder":
                return self.processor.process_folder(self.path, self.output_dir)
            return self.processor.process_file(self.path, self.output_dir)
        finally:
            self.processor.progress_callback = None

    def cancel(self) -> None:
        self.processor.cancel()


def count_supported_files(folder: Path) -> tuple[Path, dict[str, int]]:
    """``(folder, {"images": n, "pdfs": m})`` below ``folder`` (run this off the GUI thread)."""
    exts = set(image_extensions())
    images = pdfs = 0
    for path in Path(folder).rglob("*"):
        if path.is_file():
            suffix = path.suffix.lower()
            if suffix in exts:
                images += 1
            elif suffix == ".pdf":
                pdfs += 1
    return Path(folder), {"images": images, "pdfs": pdfs}


__all__ = ["Task", "OCRJob", "count_supported_files", "ArchiveReport"]
