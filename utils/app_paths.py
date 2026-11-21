"""Where VisionLane OCR keeps its per-user files (config, logs).

Resolution order:

1. ``VISIONLANE_HOME`` environment variable.
2. *Portable mode*: a file named ``portable`` next to the application (``main.py`` or the
   frozen executable) -> everything lives in that folder.
3. The per-user data directory (``%LOCALAPPDATA%\\VisionLaneOCR`` on Windows,
   ``$XDG_DATA_HOME/VisionLaneOCR`` or ``~/.local/share/VisionLaneOCR`` elsewhere).

The per-user directory is always writable, unlike the install folder (``Program Files``).
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

APP_NAME = "VisionLaneOCR"


def app_root() -> Path:
    """Folder that contains ``main.py`` (or the frozen executable)."""
    if getattr(sys, "frozen", False) or "__compiled__" in globals():
        return Path(sys.argv[0]).resolve().parent
    return Path(__file__).resolve().parent.parent


def _user_data_dir() -> Path:
    if sys.platform.startswith("win"):
        base = os.environ.get("LOCALAPPDATA") or str(Path.home() / "AppData" / "Local")
    elif sys.platform == "darwin":
        base = str(Path.home() / "Library" / "Application Support")
    else:
        base = os.environ.get("XDG_DATA_HOME") or str(Path.home() / ".local" / "share")
    return Path(base) / APP_NAME


def data_dir() -> Path:
    """Directory for ``config.ini`` and ``logs/`` (created if missing)."""
    override = os.environ.get("VISIONLANE_HOME")
    if override:
        path = Path(override).expanduser()
    elif (app_root() / "portable").exists():
        path = app_root()
    else:
        path = _user_data_dir()
    path.mkdir(parents=True, exist_ok=True)
    return path


def config_path() -> Path:
    return data_dir() / "config.ini"


def log_dir() -> Path:
    path = data_dir() / "logs"
    path.mkdir(parents=True, exist_ok=True)
    return path
