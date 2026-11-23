"""Optional image enhancement: super-resolution for low-resolution scans (roadmap item "super-image").

Used together with *uniform page size*: when a page has to be enlarged to fill the chosen paper size and its
effective resolution would drop below a threshold, it is upscaled by a super-resolution network first, so the
OCR (and the PDF) see a sharper page instead of a blurry enlargement.

The network comes from the optional ``super-image`` package (``pip install super-image``; the default model
``eugenesiow/edsr-base`` is downloaded from the Hugging Face hub on first use). Large pages are processed in
overlapping tiles so that a 6 GB GPU does not run out of memory; on an out-of-memory error the tile size is halved.
"""
from __future__ import annotations

import importlib.util
import logging
from typing import Any, Callable, Protocol

import numpy as np
from PIL import Image

logger = logging.getLogger(__name__)

DEFAULT_MODEL = "eugenesiow/edsr-base"
MIN_TILE = 64


class Upscaler(Protocol):
    """What :class:`core.ocr_processor.OCRProcessor` needs from an enhancer."""

    scale: int

    def upscale(self, image: Image.Image) -> Image.Image:  # pragma: no cover - protocol
        ...


def super_resolution_available() -> bool:
    """True when the optional ``super-image`` package is installed."""
    return importlib.util.find_spec("super_image") is not None


def upscale_tiled(array: np.ndarray, fn: Callable[[np.ndarray], np.ndarray], scale: int, tile: int = 192,
                  overlap: int = 16) -> np.ndarray:
    """Apply ``fn`` (``HxWx3 uint8`` -> ``(H*scale)x(W*scale)x3 uint8``) tile by tile and stitch the result.

    Each tile is processed with ``overlap`` extra pixels on every side that are cropped away again, which hides
    the seams. For a pure per-pixel-neighbourhood ``fn`` the stitched output equals ``fn(array)``.
    """
    height, width, channels = array.shape
    out = np.zeros((height * scale, width * scale, channels), dtype=np.uint8)
    for y0 in range(0, height, tile):
        for x0 in range(0, width, tile):
            y1, x1 = min(y0 + tile, height), min(x0 + tile, width)
            py0, py1 = max(0, y0 - overlap), min(height, y1 + overlap)
            px0, px1 = max(0, x0 - overlap), min(width, x1 + overlap)
            up = fn(array[py0:py1, px0:px1])
            if up.shape[0] != (py1 - py0) * scale or up.shape[1] != (px1 - px0) * scale:
                raise ValueError(f"upscaler returned {up.shape[:2]} for a {(py1 - py0, px1 - px0)} tile at x{scale}")
            out[y0 * scale:y1 * scale, x0 * scale:x1 * scale] = up[(y0 - py0) * scale:(y1 - py0) * scale,
                                                                   (x0 - px0) * scale:(x1 - px0) * scale]
    return out


def _is_oom(exc: BaseException) -> bool:
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


class SuperResolution:
    """Tiled super-resolution with a ``super-image`` model (EDSR by default). ``model`` can be injected."""

    def __init__(self, scale: int = 2, model_name: str = DEFAULT_MODEL, device: str = "cpu", tile: int = 192,
                 overlap: int = 16, model: object | None = None):
        if scale not in (2, 3, 4):
            raise ValueError("scale must be 2, 3 or 4")
        self.scale = scale
        self.model_name = model_name
        self.device = device
        self.tile = tile
        self.overlap = overlap
        self._model: Any = model

    def _ensure_model(self) -> None:
        if self._model is None:
            if not super_resolution_available():
                raise RuntimeError("the 'super-image' package is not installed (pip install super-image)")
            from super_image import EdsrModel  # pylint: disable=import-error
            logger.info("Loading super-resolution model %s (x%d) on %s", self.model_name, self.scale, self.device)
            self._model = EdsrModel.from_pretrained(self.model_name, scale=self.scale).to(self.device).eval()

    def _run_tile(self, tile: np.ndarray) -> np.ndarray:
        import torch  # pylint: disable=import-outside-toplevel
        tensor = torch.from_numpy(np.array(tile)).permute(2, 0, 1).unsqueeze(0).float().div(255.0)
        with torch.inference_mode():
            result = self._model(tensor.to(self.device))
        return result.squeeze(0).permute(1, 2, 0).clamp(0, 1).mul(255).round().byte().cpu().numpy()

    def upscale(self, image: Image.Image) -> Image.Image:
        self._ensure_model()
        rgb = np.asarray(image.convert("RGB"))
        while True:
            try:
                result = upscale_tiled(rgb, self._run_tile, self.scale, self.tile, self.overlap)
                break
            except RuntimeError as exc:
                if not _is_oom(exc) or self.tile <= MIN_TILE:
                    raise
                self.tile = max(MIN_TILE, self.tile // 2)
                logger.warning("Out of memory during super-resolution; retrying with %d px tiles", self.tile)
                self._free_cuda()
        out = Image.fromarray(result)
        return out.convert("L") if image.mode in ("L", "1") else out

    @staticmethod
    def _free_cuda() -> None:
        import gc
        import sys
        gc.collect()
        torch = sys.modules.get("torch")
        if torch is not None and torch.cuda.is_available():
            torch.cuda.empty_cache()
