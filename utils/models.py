"""DocTR model catalogue, cache detection and downloading (replaces ``utils/model_downloader.py``).

* The list of architectures comes from DocTR's own registry, so new models appear without code changes.
* Weights are stored next to the program (``<app>/doctr_cache``) instead of ``~/.cache/doctr``: call
  :func:`configure_cache` once at start-up. ``DOCTR_CACHE_DIR`` (if already set) always wins.
* Downloading simply instantiates the model with ``pretrained=True``; there is no fake progress.
"""
from __future__ import annotations

import logging
import os
import shutil
from pathlib import Path

from core.constants import FALLBACK_DETECTION_MODELS, FALLBACK_RECOGNITION_MODELS

from .app_paths import app_root, data_dir

logger = logging.getLogger(__name__)

DETECTION = "detection"
RECOGNITION = "recognition"


def _writable(folder: Path) -> bool:
    try:
        folder.mkdir(parents=True, exist_ok=True)
        probe = folder / ".write_test"
        probe.write_bytes(b"")
        probe.unlink()
        return True
    except OSError:
        return False


def configure_cache() -> Path:
    """Make DocTR store its weights beside the application. Call once, before any model is loaded.

    * ``DOCTR_CACHE_DIR`` already set  -> used as is (nothing is copied).
    * otherwise ``<app folder>/doctr_cache``; if that is not writable (e.g. ``Program Files``) the
      per-user data folder is used instead.
    * models already present in DocTR's old default location (``~/.cache/doctr/models``) are *copied*
      over once, so nothing has to be downloaded again (the old copies are left alone).

    Returns the cache base directory (weights live in its ``models`` subfolder).
    """
    explicit = os.environ.get("DOCTR_CACHE_DIR")
    if explicit:
        base = Path(explicit)
        base.mkdir(parents=True, exist_ok=True)
        return base
    base = app_root() / "doctr_cache"
    if not _writable(base):
        base = data_dir() / "doctr_cache"
        base.mkdir(parents=True, exist_ok=True)
    os.environ["DOCTR_CACHE_DIR"] = str(base)
    _import_legacy_models(base / "models")
    logger.info("DocTR model cache: %s", base / "models")
    return base


def _import_legacy_models(target: Path) -> None:
    legacy = Path.home() / ".cache" / "doctr" / "models"
    if not legacy.is_dir() or legacy.resolve() == target.resolve():
        return
    target.mkdir(parents=True, exist_ok=True)
    for source in legacy.glob("*.pt"):
        destination = target / source.name
        if source.stat().st_size == 0:  # leftover of a failed download
            continue
        if not destination.exists():
            try:
                shutil.copy2(source, destination)
                logger.info("Copied %s from the old DocTR cache", source.name)
            except OSError as exc:
                logger.warning("Could not copy %s: %s", source, exc)


def doctr_cache_dir() -> Path:
    """Folder where DocTR stores downloaded weights (``<cache>/models``)."""
    base = os.environ.get("DOCTR_CACHE_DIR") or str(Path.home() / ".cache" / "doctr")
    return Path(base) / "models"


def list_models(kind: str) -> list[str]:
    """Architecture names DocTR can build for ``kind`` (``detection`` or ``recognition``)."""
    try:
        if kind == DETECTION:
            from doctr.models.detection.zoo import ARCHS
        else:
            from doctr.models.recognition.zoo import ARCHS
        return list(ARCHS)
    except Exception as exc:  # pylint: disable=broad-except
        logger.warning("Could not read DocTR's model registry (%s); using the built-in list", exc)
        return list(FALLBACK_DETECTION_MODELS if kind == DETECTION else FALLBACK_RECOGNITION_MODELS)


def is_cached(name: str) -> bool:
    """True if weights for ``name`` are already on disk (files are named ``<name>-<hash>.pt``)."""
    folder = doctr_cache_dir()
    if not folder.is_dir():
        return False
    return any(p.suffix == ".pt" and p.name.rsplit("-", 1)[0] == name and p.stat().st_size > 0
               for p in folder.iterdir())


def download(kind: str, name: str) -> None:
    """Download weights for one model (blocking; call from a worker thread). Raises on failure."""
    import doctr.models as doctr_models
    factory = getattr(getattr(doctr_models, kind), name)
    logger.info("Downloading %s model %s", kind, name)
    factory(pretrained=True)
