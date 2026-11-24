"""DocTR model catalogue, cache detection and downloading (replaces ``utils/model_downloader.py``).

* The list of architectures comes from DocTR's own registry, so new models appear without code changes.
* Weights are stored next to the program (``<app>/doctr_cache``) instead of ``~/.cache/doctr``: call
  :func:`configure_cache` once at start-up. ``DOCTR_CACHE_DIR`` (if already set) always wins.
* Downloads use DocTR's own URL and checksum for the model (read from its ``default_cfgs``) but are done here, so they
  report **real byte progress** (:class:`DownloadProgress`), go to a ``.part`` file first and are checksum-verified before
  they appear in the cache. If a model's source cannot be determined, DocTR's own downloader is used (no progress).
"""
from __future__ import annotations

import hashlib
import logging
import os
import re
import shutil
import sys
import time
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

from core.constants import FALLBACK_DETECTION_MODELS, FALLBACK_RECOGNITION_MODELS

from .app_paths import app_root, data_dir

logger = logging.getLogger(__name__)

DETECTION = "detection"
RECOGNITION = "recognition"
CLASSIFICATION = "classification"
# The two small networks DocTR loads for rotated / skewed page handling (straighten_pages)
ORIENTATION_MODELS: tuple[tuple[str, str], ...] = (
    (CLASSIFICATION, "mobilenet_v3_small_crop_orientation"),
    (CLASSIFICATION, "mobilenet_v3_small_page_orientation"),
)

_HASH_RE = re.compile(r"-([a-f0-9]*)\.")   # same rule DocTR uses to find the checksum in a file name
_USER_AGENT = "VisionLaneOCR (+https://github.com/NeoMatrix14241/VisionLaneOCR)"
_CHUNK = 64 * 1024
_REPORT_INTERVAL = 0.1                      # seconds between progress callbacks


@dataclass(frozen=True)
class DownloadProgress:
    name: str
    done: int       # bytes received so far
    total: int      # bytes expected (0 = server did not say)

    @property
    def fraction(self) -> float:
        return min(1.0, self.done / self.total) if self.total else 0.0


ProgressCallback = Callable[[DownloadProgress], None]


@dataclass(frozen=True)
class ModelSource:
    kind: str
    name: str
    url: str
    filename: str                # name inside the cache folder, e.g. "db_resnet50-79bd7d70.pt"
    hash_prefix: str | None      # leading hex digits of the file's SHA-256


def models_needed(detection: str, recognition: str, straighten: bool = False) -> list[tuple[str, str]]:
    """Every (kind, name) the OCR engine will load for these settings."""
    needed = [(DETECTION, detection), (RECOGNITION, recognition)]
    if straighten:
        needed.extend(ORIENTATION_MODELS)
    return needed


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
    os.environ.setdefault("HF_HOME", str(base / "huggingface"))   # optional super-resolution model (Hugging Face hub)
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


def model_source(kind: str, name: str) -> ModelSource:
    """URL, file name and checksum of a pretrained model, from DocTR's own configuration. ``LookupError`` if unknown."""
    try:
        import doctr.models as doctr_models  # pylint: disable=import-outside-toplevel
        factory = getattr(getattr(doctr_models, kind), name)
        url = str(sys.modules[factory.__module__].default_cfgs[name]["url"])
    except Exception as exc:  # pylint: disable=broad-except
        raise LookupError(f"no download source known for {kind} model {name}: {exc}") from exc
    filename = url.rpartition("/")[-1].split("&")[0]
    match = _HASH_RE.search(filename)
    return ModelSource(kind, name, url, filename, match.group(1) if match else None)


def _find_source(name: str, kind: str | None = None) -> ModelSource | None:
    for candidate in ([kind] if kind else [DETECTION, RECOGNITION, CLASSIFICATION]):
        try:
            return model_source(str(candidate), name)
        except LookupError:
            continue
    return None


def is_cached(name: str, kind: str | None = None) -> bool:
    """True if the weights for ``name`` are on disk (and not empty)."""
    folder = doctr_cache_dir()
    if not folder.is_dir():
        return False
    source = _find_source(name, kind)
    if source is not None:   # the exact file name (some are not "<name>-<hash>.pt", e.g. crnn_mobilenet_v3_small_pt-...)
        path = folder / source.filename
        return path.is_file() and path.stat().st_size > 0
    return any(p.suffix == ".pt" and p.name.rsplit("-", 1)[0] == name and p.stat().st_size > 0
               for p in folder.iterdir())


def download_file(url: str, target: Path, hash_prefix: str | None = None, progress: ProgressCallback | None = None,
                  name: str = "", timeout: float = 30.0) -> None:
    """Download ``url`` to ``target`` with progress. Writes ``<target>.part`` first and only renames it after the size
    and (if given) the SHA-256 prefix are verified, so an interrupted download never leaves a plausible-looking file."""
    target.parent.mkdir(parents=True, exist_ok=True)
    part = target.with_name(target.name + ".part")
    part.unlink(missing_ok=True)
    label = name or target.name

    def report(done: int, total: int) -> None:
        if progress is not None:
            try:
                progress(DownloadProgress(label, done, total))
            except Exception:  # pylint: disable=broad-except
                logger.exception("Download progress callback raised")

    digest = hashlib.sha256()
    try:
        request = urllib.request.Request(url, headers={"User-Agent": _USER_AGENT})
        with urllib.request.urlopen(request, timeout=timeout) as response, open(part, "wb") as handle:
            total = int(response.headers.get("Content-Length") or 0)
            done, last = 0, 0.0
            report(0, total)
            while True:
                chunk = response.read(_CHUNK)
                if not chunk:
                    break
                handle.write(chunk)
                digest.update(chunk)
                done += len(chunk)
                now = time.monotonic()
                if now - last >= _REPORT_INTERVAL:
                    report(done, total)
                    last = now
        if total and done != total:
            raise OSError(f"incomplete download: received {done} of {total} bytes")
        if hash_prefix and not digest.hexdigest().startswith(hash_prefix):
            raise ValueError(f"checksum mismatch for {label} (the download is corrupt)")
        part.replace(target)
        report(done, total or done)
    except BaseException:
        part.unlink(missing_ok=True)
        raise


def download(kind: str, name: str, progress: ProgressCallback | None = None) -> None:
    """Make sure the weights for one model are in the cache (blocking; call from a worker thread). Raises on failure."""
    try:
        source = model_source(kind, name)
    except LookupError as exc:
        logger.info("%s - using DocTR's own downloader (no progress available)", exc)
        _download_with_doctr(kind, name)
        return
    target = doctr_cache_dir() / source.filename
    if target.is_file() and target.stat().st_size > 0:
        return
    logger.info("Downloading %s model %s from %s", kind, name, source.url)
    download_file(source.url, target, source.hash_prefix, progress, name)


def _download_with_doctr(kind: str, name: str) -> None:
    import doctr.models as doctr_models  # pylint: disable=import-outside-toplevel
    getattr(getattr(doctr_models, kind), name)(pretrained=True)


def console_progress() -> ProgressCallback:
    """A progress callback for command-line tools: ``parseq  45%  44.1/98.7 MB`` on one updating line."""
    state = {"last": -1}

    def show(event: DownloadProgress) -> None:
        percent = int(event.fraction * 100)
        if percent == state["last"] and event.done != event.total:
            return
        state["last"] = percent
        size = f"{event.done / 1e6:.1f}/{event.total / 1e6:.1f} MB" if event.total else f"{event.done / 1e6:.1f} MB"
        end = "\n" if event.total and event.done >= event.total else ""
        print(f"\r  {event.name}: {percent:3d}%  {size}   ", end=end, flush=True)

    return show
