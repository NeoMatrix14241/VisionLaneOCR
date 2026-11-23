"""The DocTR model wrapper: loading, device handling and out-of-memory recovery.

Deliberately small and free of monkey-patching: it uses stock PyTorch/DocTR APIs only.
"""
from __future__ import annotations

import gc
import logging
from typing import Any, Callable

import numpy as np
import torch

from .errors import Cancelled
from .hardware import DeviceReport, resolve_device

logger = logging.getLogger(__name__)

_OOM_TYPES: tuple[type[BaseException], ...] = (
    getattr(torch, "OutOfMemoryError", torch.cuda.OutOfMemoryError),
)


def _is_oom(exc: BaseException) -> bool:
    if isinstance(exc, _OOM_TYPES):
        return True
    return isinstance(exc, RuntimeError) and "out of memory" in str(exc).lower()


class OCREngine:
    """Detection + recognition predictor on the best available device."""

    def __init__(self, detection_model: str, recognition_model: str, device_preference: str = "auto", *,
                 pretrained: bool = True, straighten_pages: bool = False, cpu_threads: int = 0,
                 predictor_kwargs: dict | None = None):
        """``straighten_pages``: handle rotated (90/180/270 degrees) and skewed pages. Slower, and DocTR loads
        two extra small models (page and crop orientation) which are downloaded on first use.
        ``predictor_kwargs`` are passed on to ``ocr_predictor`` (tests use them to disable the extra models)."""
        self.detection_model = detection_model
        self.recognition_model = recognition_model
        self.device_preference = device_preference
        self.pretrained = pretrained
        self.straighten_pages = straighten_pages
        self.predictor_kwargs = dict(predictor_kwargs or {})
        # Polled before every model forward pass; when it returns True the pass is aborted with ``Cancelled``.
        self.cancel_check: Callable[[], bool] | None = None
        self.report: DeviceReport = DeviceReport()
        self.device = "cpu"
        self.model: Any = None
        self.oom_fallbacks = 0
        self.set_cpu_threads(cpu_threads)
        self.load()

    @staticmethod
    def set_cpu_threads(count: int) -> None:
        """Number of CPU threads PyTorch may use (0 = leave PyTorch's default)."""
        if count and count > 0:
            torch.set_num_threads(int(count))

    # ------------------------------------------------------------------ loading
    def load(self) -> None:
        """(Re)build the predictor for the current model names and device preference."""
        from doctr.models import ocr_predictor

        self.report = resolve_device(self.device_preference)
        self.device = self.report.device
        if self.report.reason:
            logger.info("Using CPU: %s", self.report.reason)
        for warning in self.report.warnings:
            logger.warning(warning)
        self.model = None
        gc.collect()
        logger.info("Loading OCR models: det=%s reco=%s device=%s", self.detection_model,
                    self.recognition_model, self.device)
        options: dict = {"assume_straight_pages": True}
        if self.straighten_pages:
            # Pages are rotated/deskewed internally for recognition and the word boxes are mapped back onto the
            # ORIGINAL page as 4-point polygons, so the text layer lines up with the untouched scan.
            options = {"assume_straight_pages": False, "straighten_pages": True, "preserve_original_coords": True}
        options.update(self.predictor_kwargs)
        model = ocr_predictor(det_arch=self.detection_model, reco_arch=self.recognition_model,
                              pretrained=self.pretrained,
                              # pretrained=False must be fully offline (DocTR would still fetch backbone weights)
                              pretrained_backbone=self.pretrained, **options)
        self.model = model.to(self.device).eval()
        self._install_cancel_hooks()

    def _cancel_hook(self, _module, _args) -> None:
        if self.cancel_check is not None and self.cancel_check():
            raise Cancelled()

    def _install_cancel_hooks(self) -> None:
        """Make inference interruptible: DocTR calls its detection / recognition / orientation networks
        once per stage or batch, so a pre-hook that raises lets Cancel act within a page instead of after it."""
        for name in ("det_predictor", "reco_predictor", "crop_orientation_predictor", "page_orientation_predictor"):
            network = getattr(getattr(self.model, name, None), "model", None)
            if isinstance(network, torch.nn.Module):
                network.register_forward_pre_hook(self._cancel_hook)

    def set_models(self, detection_model: str, recognition_model: str) -> bool:
        """Switch models; returns True if the predictor was rebuilt."""
        if (detection_model, recognition_model) == (self.detection_model, self.recognition_model):
            return False
        self.detection_model, self.recognition_model = detection_model, recognition_model
        self.load()
        return True

    def set_straighten(self, enabled: bool) -> bool:
        """Turn rotated/skewed page handling on or off (rebuilds the predictor)."""
        if enabled == self.straighten_pages:
            return False
        self.straighten_pages = enabled
        self.load()
        return True

    def set_device_preference(self, preference: str) -> bool:
        if preference == self.device_preference:
            return False
        self.device_preference = preference
        self.load()
        return True

    @property
    def gpu(self):
        """``GpuInfo`` of the GPU the model runs on (``None`` on the CPU)."""
        return self.report.primary if self.device == "cuda" else None

    @property
    def gpu_name(self) -> str:
        gpu = self.gpu
        return gpu.name if gpu else ""

    # ------------------------------------------------------------------ inference
    def predict(self, images: list[np.ndarray]):
        """Run OCR on RGB ``uint8`` arrays. Recovers from CUDA out-of-memory (see :meth:`_run`)."""
        try:
            return self._run(images)
        except Cancelled:
            raise
        except Exception as exc:  # pylint: disable=broad-except
            if self.device != "cuda" or not _is_oom(exc):
                if self.device == "cuda" and isinstance(exc, RuntimeError) and "CUDA" in str(exc):
                    # Not an OOM: the CUDA context is probably unusable. Stay on the CPU for this session.
                    logger.error("CUDA error (%s); switching to CPU for the rest of the session", exc)
                    self._move("cpu")
                    return self._run(images)
                raise
        # --- CUDA out of memory: free memory, retry once on the GPU, then do this one input on the CPU
        self.oom_fallbacks += 1
        logger.warning("CUDA out of memory; clearing the cache and retrying")
        self._free_cuda()
        try:
            return self._run(images)
        except Exception as exc:  # pylint: disable=broad-except
            if not _is_oom(exc):
                raise
        logger.warning("Still out of memory; processing this page on the CPU (the model stays on the GPU)")
        self._free_cuda()
        self._move("cpu")
        try:
            return self._run(images)
        finally:
            self._move("cuda")  # per-image fallback, not a permanent latch

    def _run(self, images: list[np.ndarray]):
        with torch.inference_mode():
            return self.model(images)

    def _move(self, device: str) -> None:
        self.model = self.model.to(device)
        self.device = device
        if device == "cpu":
            self._free_cuda()

    @staticmethod
    def _free_cuda() -> None:
        gc.collect()
        if torch.cuda.is_available():
            try:
                torch.cuda.empty_cache()
            except Exception:  # pylint: disable=broad-except
                pass

    def release(self) -> None:
        self.model = None
        self._free_cuda()
