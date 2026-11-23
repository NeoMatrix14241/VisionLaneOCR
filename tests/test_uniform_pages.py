"""Uniform page size and optional super-resolution (VL-067)."""
import numpy as np
import pytest
from PIL import Image
from pypdf import PdfReader

from core.ocr_processor import OCRProcessor
from core.paper import paper_size_pt
from tests.conftest import FakeEngine

A4 = paper_size_pt("A4")


class NearestUpscaler:
    """Stand-in for the super-resolution network: x2 nearest neighbour."""

    def __init__(self, scale=2):
        self.scale = scale
        self.calls = 0

    def upscale(self, image):
        self.calls += 1
        w, h = image.size
        return image.resize((w * self.scale, h * self.scale), Image.NEAREST)


def page_sizes(result):
    pdf = next((result.session_dir / "pdf").rglob("*.pdf"))
    return [(round(float(p.mediabox.width), 2), round(float(p.mediabox.height), 2)) for p in PdfReader(str(pdf)).pages]


def make(tmp_path, make_image, sizes):
    for i, size in enumerate(sizes):
        make_image(tmp_path / "in" / f"{i}.png", size)
    return tmp_path / "in"


def processor(**cfg):
    p = OCRProcessor(engine=FakeEngine(), upscaler_factory=cfg.pop("factory", None))
    p.configure(dpi=cfg.pop("dpi", 200), **cfg)
    return p


def test_original_keeps_every_page_size(tmp_path, make_image):
    folder = make(tmp_path, make_image, [(1654, 2339), (1700, 2200), (800, 600)])
    sizes = page_sizes(processor().process_folder(folder, tmp_path / "out"))
    assert len(set(sizes)) == 3


def test_named_paper_makes_all_pages_identical_and_keeps_orientation(tmp_path, make_image):
    folder = make(tmp_path, make_image, [(1654, 2339), (1700, 2200), (2339, 1654)])   # A4, Letter, A4 landscape
    sizes = page_sizes(processor(page_size="a4").process_folder(folder, tmp_path / "out"))
    assert sizes == [(595.44, 841.68), (595.44, 841.68), (841.68, 595.44)]


def test_auto_picks_the_paper_most_pages_are_closest_to(tmp_path, make_image):
    folder = make(tmp_path, make_image, [(1654, 2339), (1650, 2330), (1700, 2200)])   # two A4, one Letter
    sizes = page_sizes(processor(page_size="auto").process_folder(folder, tmp_path / "out"))
    assert set(sizes) == {(595.44, 841.68)}


def test_auto_leaves_odd_sizes_alone(tmp_path, make_image):
    folder = make(tmp_path, make_image, [(100, 50), (400, 100)])       # nowhere near a paper size
    sizes = page_sizes(processor(page_size="auto").process_folder(folder, tmp_path / "out"))
    assert len(set(sizes)) == 2


def test_unknown_page_size_means_original():
    p = OCRProcessor(engine=FakeEngine())
    p.configure(page_size="nonsense")
    assert p.page_size == "original"


# ------------------------------------------------------------------------------------ enhancement
def test_only_low_resolution_pages_are_upscaled(tmp_path, make_image):
    folder = make(tmp_path, make_image, [(800, 1100), (1654, 2339)])   # 100 dpi page, 200 dpi page (both ~A4 sized)
    up = NearestUpscaler()
    engine = FakeEngine()
    p = OCRProcessor(engine=engine, upscaler_factory=lambda scale, device: up)
    p.configure(dpi=100, page_size="A4", enhance_enabled=True, enhance_min_dpi=150)
    result = p.process_folder(folder, tmp_path / "out")
    assert result.ok and not result.warnings
    # page 1 (8x11 in at 100 dpi) is enlarged to A4 -> ~97 dpi -> upscaled; page 2 (16.5x23.4 in) is *shrunk*
    # to A4 -> its effective resolution rises to 200 dpi -> left alone
    assert up.calls == 1
    assert engine.calls == [(1600, 2200), (1654, 2339)]           # OCR saw the enhanced page 1
    assert page_sizes(result) == [(595.44, 841.68)] * 2


def test_high_resolution_pages_are_not_touched(tmp_path, make_image):
    folder = make(tmp_path, make_image, [(2480, 3508)])
    up = NearestUpscaler()
    engine = FakeEngine()
    p = OCRProcessor(engine=engine, upscaler_factory=lambda scale, device: up)
    p.configure(dpi=300, page_size="A4", enhance_enabled=True, enhance_min_dpi=150)
    p.process_folder(folder, tmp_path / "out")
    assert up.calls == 0 and engine.calls == [(2480, 3508)]


def test_enhancement_keeps_the_physical_page_size_without_a_paper_target(tmp_path, make_image):
    folder = make(tmp_path, make_image, [(400, 500)])
    p = OCRProcessor(engine=FakeEngine(), upscaler_factory=lambda scale, device: NearestUpscaler())
    p.configure(dpi=100, enhance_enabled=True, enhance_min_dpi=150)
    result = p.process_folder(folder, tmp_path / "out")
    assert page_sizes(result) == [(288.0, 360.0)]                  # still 4 x 5 inches, now with 2x the pixels


def test_failing_enhancer_is_reported_but_pages_still_succeed(tmp_path, make_image):
    folder = make(tmp_path, make_image, [(800, 1100), (800, 1100)])

    def broken(scale, device):
        raise RuntimeError("model download failed")

    p = OCRProcessor(engine=FakeEngine(), upscaler_factory=broken)
    p.configure(dpi=100, enhance_enabled=True)
    result = p.process_folder(folder, tmp_path / "out")
    assert result.ok and result.succeeded == 2
    assert result.warnings == ["Image enhancement was skipped: model download failed"]    # once, not per page


def test_enhancer_error_on_a_page_switches_it_off_for_the_run(tmp_path, make_image):
    folder = make(tmp_path, make_image, [(800, 1100), (800, 1100)])

    class Exploding(NearestUpscaler):
        def upscale(self, image):
            self.calls += 1
            raise RuntimeError("boom")

    up = Exploding()
    p = OCRProcessor(engine=FakeEngine(), upscaler_factory=lambda s, d: up)
    p.configure(dpi=100, enhance_enabled=True)
    result = p.process_folder(folder, tmp_path / "out")
    assert result.ok and up.calls == 1 and "boom" in result.warnings[0]


def test_missing_super_image_package_gives_a_clear_warning(tmp_path, make_image, monkeypatch):
    import core.ocr_processor as module
    monkeypatch.setattr(module, "super_resolution_available", lambda: False)
    folder = make(tmp_path, make_image, [(800, 1100)])
    p = OCRProcessor(engine=FakeEngine())
    p.configure(dpi=100, enhance_enabled=True)
    result = p.process_folder(folder, tmp_path / "out")
    assert result.ok and "pip install super-image" in result.warnings[0]


def test_enhance_scale_is_validated():
    with pytest.raises(ValueError):
        OCRProcessor(engine=FakeEngine()).configure(enhance_scale=5)


def test_pdf_inputs_ignore_page_size(tmp_path):
    from reportlab.pdfgen.canvas import Canvas
    src = tmp_path / "in" / "d.pdf"
    src.parent.mkdir()
    c = Canvas(str(src), pagesize=(300, 400)); c.rect(10, 10, 50, 50, fill=1); c.showPage(); c.save()
    result = processor(page_size="A4").process_file(src, tmp_path / "out")
    assert page_sizes(result) == [(300.0, 400.0)]


# ------------------------------------------------------------------------------------ tiling
@pytest.mark.parametrize("shape,tile,overlap", [((50, 70), 16, 4), ((33, 33), 64, 8), ((10, 200), 32, 0), ((1, 1), 8, 2)])
def test_tiled_upscaling_equals_whole_image_upscaling(shape, tile, overlap):
    from core.enhance import upscale_tiled
    rng = np.random.default_rng(0)
    array = rng.integers(0, 256, size=(*shape, 3), dtype=np.uint8)

    def nearest(a):
        return a.repeat(2, axis=0).repeat(2, axis=1)

    assert np.array_equal(upscale_tiled(array, nearest, 2, tile, overlap), nearest(array))


def test_tiled_upscaling_rejects_a_wrong_output_size():
    from core.enhance import upscale_tiled
    with pytest.raises(ValueError):
        upscale_tiled(np.zeros((8, 8, 3), dtype=np.uint8), lambda a: a, 2, 4, 1)


def test_super_resolution_class_with_a_fake_model_and_oom_halving():
    torch = pytest.importorskip("torch")
    if not hasattr(torch, "__version__"):
        pytest.skip("real PyTorch required")
    from core.enhance import SuperResolution, super_resolution_available

    class Model:
        def __init__(self):
            self.max_pixels_seen = 0
            self.limit = None

        def __call__(self, x):
            pixels = x.shape[-1] * x.shape[-2]
            if self.limit is not None and pixels > self.limit:
                raise RuntimeError("CUDA out of memory. Tried to allocate ...")
            return x.repeat_interleave(2, dim=2).repeat_interleave(2, dim=3)

    model = Model()
    model.limit = 6000                                                # 96 px tiles fail, 64 px tiles fit
    sr = SuperResolution(scale=2, tile=96, overlap=4, model=model)
    image = Image.fromarray(np.random.default_rng(1).integers(0, 256, (90, 120, 3), dtype=np.uint8))
    out = sr.upscale(image)
    assert out.size == (240, 180) and sr.tile < 96                    # tiles were halved after the OOM
    assert np.array_equal(np.asarray(out), np.asarray(image).repeat(2, 0).repeat(2, 1))
    assert Image.new("L", (4, 4)).mode == SuperResolution(model=Model()).upscale(Image.new("L", (4, 4))).mode
    assert isinstance(super_resolution_available(), bool)
