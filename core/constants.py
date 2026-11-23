"""Shared constants: supported file types, defaults, model catalogues."""
from __future__ import annotations

import re
from pathlib import Path

PDF_EXTENSIONS = (".pdf",)
_BASE_IMAGE_EXTENSIONS = (".tif", ".tiff", ".jpg", ".jpeg", ".jpe", ".jfif", ".png", ".bmp", ".dib", ".gif")
_HEIF_EXTENSIONS = (".heic", ".heif")

_heif_registered: bool | None = None


def heif_available() -> bool:
    """True when ``pillow-heif`` is installed (it is registered with Pillow on first call)."""
    global _heif_registered  # pylint: disable=global-statement
    if _heif_registered is None:
        try:
            from pillow_heif import register_heif_opener  # pylint: disable=import-error
            register_heif_opener()
            _heif_registered = True
        except Exception:  # pylint: disable=broad-except
            _heif_registered = False
    return _heif_registered


def image_extensions() -> tuple[str, ...]:
    """Image extensions that can really be opened in this environment."""
    if heif_available():
        return _BASE_IMAGE_EXTENSIONS + _HEIF_EXTENSIONS
    return _BASE_IMAGE_EXTENSIONS


def is_image(path: Path) -> bool:
    return path.suffix.lower() in image_extensions()


def is_pdf(path: Path) -> bool:
    return path.suffix.lower() in PDF_EXTENSIONS


# Models offered in the GUI when doctr's own registry cannot be queried.
FALLBACK_DETECTION_MODELS = (
    "db_resnet50", "db_resnet34", "db_mobilenet_v3_large",
    "linknet_resnet18", "linknet_resnet34", "linknet_resnet50",
    "fast_tiny", "fast_small", "fast_base",
)
FALLBACK_RECOGNITION_MODELS = (
    "parseq", "crnn_vgg16_bn", "crnn_mobilenet_v3_small", "crnn_mobilenet_v3_large",
    "sar_resnet31", "master", "vitstr_small", "vitstr_base", "viptr_tiny",
)
DEFAULT_DETECTION_MODEL = "db_resnet50"
DEFAULT_RECOGNITION_MODEL = "parseq"

# Hardware guidance (see README): below this much VRAM out-of-memory errors become likely.
RECOMMENDED_VRAM_GB = 8.0

DEFAULT_DPI = 300           # used when an image carries no (or implausible) DPI metadata
MIN_PLAUSIBLE_DPI = 30
MAX_PLAUSIBLE_DPI = 4800
PDF_RENDER_MAX_PIXELS = 40_000_000  # cap when rasterising PDF pages for OCR
MIN_TEXT_CHARS_FOR_SKIP = 20        # a PDF page with at least this many characters "has text"

_NATURAL_SPLIT = re.compile(r"(\d+)")


def natural_key(path: Path | str) -> list:
    """Sort key so that ``img2`` sorts before ``img10`` (what file managers show)."""
    name = path.name if isinstance(path, Path) else str(path)
    return [int(tok) if tok.isdigit() else tok.lower() for tok in _NATURAL_SPLIT.split(name)]
