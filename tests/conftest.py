"""Shared fixtures. ``FakeEngine`` mimics the DocTR interface so file handling can be tested without
torch or model weights (a real-DocTR integration test lives in test_integration_doctr.py)."""
from __future__ import annotations

import sys
from pathlib import Path
from types import SimpleNamespace

import pytest
from PIL import Image

try:  # the real torch is used when installed; otherwise a stub lets file handling + GUI tests run
    import torch  # noqa: F401
except ImportError:
    import types
    stub = types.ModuleType("torch")
    stub.cuda = types.SimpleNamespace(is_available=lambda: False, empty_cache=lambda: None,
                                      OutOfMemoryError=type("OutOfMemoryError", (RuntimeError,), {}))
    stub.version = types.SimpleNamespace(cuda=None)
    stub.inference_mode = lambda: __import__("contextlib").nullcontext()
    stub.set_num_threads = lambda n: None
    stub.OutOfMemoryError = stub.cuda.OutOfMemoryError
    sys.modules["torch"] = stub

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.ocr_processor import OCRProcessor  # noqa: E402


class FakeDocument:
    def __init__(self, text: str):
        word = SimpleNamespace(value=text, geometry=((0.1, 0.1), (0.4, 0.15)))
        line = SimpleNamespace(words=[word, SimpleNamespace(value="second", geometry=((0.5, 0.1), (0.8, 0.15)))])
        self.pages = [SimpleNamespace(blocks=[SimpleNamespace(lines=[line])])]

    def export_as_xml(self):
        return [(b"<html><body>hocr</body></html>", None)]


class FakeEngine:
    device = "cpu"
    gpu = None
    gpu_name = ""
    detection_model = "db_resnet50"
    recognition_model = "parseq"
    straighten_pages = False
    report = SimpleNamespace(warnings=[])

    def __init__(self, fail_heights: tuple[int, ...] = ()):
        self.fail_heights = fail_heights
        self.calls: list[tuple[int, int]] = []
        self.on_call = None

    def predict(self, images):
        h, w = images[0].shape[:2]
        self.calls.append((w, h))
        if self.on_call:
            self.on_call(len(self.calls))
        if h in self.fail_heights:
            raise RuntimeError("synthetic OCR failure")
        return FakeDocument(f"page{len(self.calls)}")

    def set_models(self, det, reco):
        changed = (det, reco) != (self.detection_model, self.recognition_model)
        self.detection_model, self.recognition_model = det, reco
        return changed

    def set_cpu_threads(self, *_):
        pass

    def set_straighten(self, enabled):
        changed = enabled != self.straighten_pages
        self.straighten_pages = enabled
        return changed

    def set_device_preference(self, *_):
        return False

    def release(self):
        pass


@pytest.fixture
def make_image():
    def _make(path: Path, size=(200, 100), color="white", fmt=None, **kw) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", size, color).save(path, fmt, **kw)
        return path
    return _make


@pytest.fixture
def processor():
    return OCRProcessor(engine=FakeEngine())
