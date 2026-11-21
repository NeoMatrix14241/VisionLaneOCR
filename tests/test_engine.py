"""OOM / device-fallback logic of OCREngine, exercised with a fake model (no GPU or DocTR needed)."""
import pytest

from core.ocr_engine import OCREngine, _OOM_TYPES


class FakeModel:
    def __init__(self, behaviour):
        self.behaviour = behaviour      # list of outcomes consumed per call: "ok" | "oom" | "cuda_error"
        self.device = "cuda"
        self.moves = []

    def to(self, device):
        self.moves.append(device)
        self.device = device
        return self

    def __call__(self, images):
        outcome = self.behaviour.pop(0) if self.behaviour else "ok"
        if outcome == "oom":
            raise _OOM_TYPES[0]("CUDA out of memory")
        if outcome == "cuda_error":
            raise RuntimeError("CUDA error: device-side assert triggered")
        return f"result-on-{self.device}"


def make_engine(behaviour, device="cuda"):
    engine = OCREngine.__new__(OCREngine)     # skip __init__: no real model / torch device needed
    engine.model = FakeModel(list(behaviour))
    engine.device = device
    engine.oom_fallbacks = 0
    engine.model.device = device
    return engine


def test_normal_call_untouched():
    engine = make_engine(["ok"])
    assert engine.predict([None]) == "result-on-cuda"
    assert engine.oom_fallbacks == 0


def test_oom_is_retried_on_the_gpu_first():
    engine = make_engine(["oom", "ok"])
    assert engine.predict([None]) == "result-on-cuda"
    assert engine.oom_fallbacks == 1 and engine.device == "cuda" and engine.model.moves == []


def test_second_oom_falls_back_to_cpu_for_this_call_only():
    engine = make_engine(["oom", "oom", "ok"])
    assert engine.predict([None]) == "result-on-cpu"
    assert engine.device == "cuda"                          # back on the GPU: not a permanent latch
    assert engine.model.moves == ["cpu", "cuda"]
    assert engine.predict([None]) == "result-on-cuda"       # next page uses the GPU again


def test_non_oom_cuda_error_switches_to_cpu_for_the_session():
    engine = make_engine(["cuda_error", "ok"])
    assert engine.predict([None]) == "result-on-cpu"
    assert engine.device == "cpu"


def test_other_errors_propagate():
    engine = make_engine([], device="cpu")
    engine.model.behaviour = ["cuda_error"]
    with pytest.raises(RuntimeError):
        engine.predict([None])
