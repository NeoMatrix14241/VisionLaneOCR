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
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from PyQt6.QtCore import QObject, pyqtSignal

from core.archiver import ArchiveReport
from core.constants import image_extensions
from core.ocr_processor import OCRProcessor
from core.results import BatchResult
from utils import models as model_utils

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


@dataclass(frozen=True)
class LoadEvent:
    """Progress of loading the OCR models, for the status line and progress bar."""
    stage: str            # "download" | "load"
    name: str = ""        # model being downloaded
    index: int = 0        # 1-based position among the models that have to be downloaded
    count: int = 0
    done: int = 0         # bytes received
    total: int = 0        # bytes expected (0 = unknown)


class LoadTask(Task):
    """A :class:`Task` whose function receives an ``emit(LoadEvent)`` callback; events arrive on ``progress``."""

    progress = pyqtSignal(object, arguments=["event"])

    def __init__(self, fn: Callable[[Callable[[LoadEvent], None]], object], name: str = "load"):
        super().__init__(lambda: fn(self.progress.emit), name)


def fetch_missing(needed: list[tuple[str, str]], emit: Callable[[LoadEvent], None]) -> None:
    """Download the models in ``needed`` that are not cached yet, reporting progress, then announce the load stage."""
    missing = [(kind, name) for kind, name in needed if not model_utils.is_cached(name, kind)]
    for index, (kind, name) in enumerate(missing, 1):
        emit(LoadEvent("download", name, index, len(missing)))

        def on_progress(progress: model_utils.DownloadProgress, index: int = index, count: int = len(missing)) -> None:
            emit(LoadEvent("download", progress.name, index, count, progress.done, progress.total))

        model_utils.download(kind, name, progress=on_progress)
    emit(LoadEvent("load"))


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


__all__ = ["Task", "LoadTask", "LoadEvent", "OCRJob", "fetch_missing", "count_supported_files", "ArchiveReport"]
