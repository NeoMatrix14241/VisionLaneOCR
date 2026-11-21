"""Logging setup. Call :func:`setup_logging` exactly once, from the entry point.

Library modules only do ``logging.getLogger(__name__)``; none of them configures handlers.
"""
from __future__ import annotations

import faulthandler
import logging
import logging.handlers
import sys
from datetime import datetime
from pathlib import Path

_FORMAT = "%(asctime)s - %(levelname)s - %(name)s - %(message)s"
_DATEFMT = "%Y-%m-%d %H:%M:%S"
_fault_file = None  # kept open for faulthandler


def setup_logging(log_dir: Path, level: str = "INFO", console: bool = True) -> Path | None:
    """Configure the root logger (file + optional console) and return the log file path."""
    root = logging.getLogger()
    for handler in root.handlers[:]:
        root.removeHandler(handler)
        handler.close()
    numeric = getattr(logging, str(level).upper(), logging.INFO)
    root.setLevel(numeric)
    formatter = logging.Formatter(_FORMAT, _DATEFMT)

    log_file: Path | None = None
    try:
        log_dir.mkdir(parents=True, exist_ok=True)
        log_file = log_dir / f"visionlane_{datetime.now():%Y%m%d_%H%M%S}.log"
        file_handler = logging.FileHandler(log_file, encoding="utf-8")
        file_handler.setFormatter(formatter)
        root.addHandler(file_handler)
        _prune_old_logs(log_dir)
    except OSError as exc:  # unwritable location: keep going with console only
        print(f"Warning: cannot write log file in {log_dir}: {exc}", file=sys.stderr)
        log_file = None

    if console and sys.stderr is not None:
        stream = logging.StreamHandler(sys.stderr)
        stream.setFormatter(formatter)
        root.addHandler(stream)
    return log_file


def install_crash_handlers(log_dir: Path) -> None:
    """Log uncaught exceptions and write native crash tracebacks (faulthandler) to ``crash.log``."""
    global _fault_file  # pylint: disable=global-statement

    def _hook(exc_type, exc, tb):
        if issubclass(exc_type, KeyboardInterrupt):
            sys.__excepthook__(exc_type, exc, tb)
            return
        logging.getLogger("crash").critical("Uncaught exception", exc_info=(exc_type, exc, tb))

    sys.excepthook = _hook
    try:
        _fault_file = open(log_dir / "crash.log", "a", encoding="utf-8")  # pylint: disable=consider-using-with
        faulthandler.enable(file=_fault_file)
    except OSError:
        pass


def _prune_old_logs(log_dir: Path, keep: int = 20) -> None:
    logs = sorted(log_dir.glob("visionlane_*.log"), key=lambda p: p.stat().st_mtime, reverse=True)
    for old in logs[keep:]:
        try:
            old.unlink()
        except OSError:
            pass
