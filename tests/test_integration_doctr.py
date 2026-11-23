"""Runs the real DocTR stack end-to-end (random weights, so no download). Skipped when DocTR/torch are missing."""
import pytest

pytest.importorskip("doctr")
torch = pytest.importorskip("torch")
if not hasattr(torch, "__version__"):   # the conftest torch stub is not enough here
    pytest.skip("real PyTorch required", allow_module_level=True)

from pypdf import PdfReader  # noqa: E402
from core.ocr_processor import OCRProcessor  # noqa: E402


def test_real_doctr_pipeline_produces_pdf_and_hocr(tmp_path, make_image):
    img = make_image(tmp_path / "in" / "page.png", (640, 480))
    processor = OCRProcessor("db_mobilenet_v3_large", "crnn_mobilenet_v3_small", "cpu", pretrained=False)
    processor.configure(output_formats=["pdf", "hocr"])
    result = processor.process_file(img, tmp_path / "out")
    assert result.ok, result.failure_lines()
    assert len(PdfReader(str(result.session_dir / "pdf" / "page.pdf")).pages) == 1
    assert (result.session_dir / "hocr" / "page.hocr").read_bytes().startswith(b"<")


def test_cancel_interrupts_inference_inside_a_page():
    import numpy as np
    from core.errors import Cancelled
    from core.ocr_engine import OCREngine

    engine = OCREngine("db_mobilenet_v3_large", "crnn_mobilenet_v3_small", "cpu", pretrained=False)
    postprocessor = engine.model.det_predictor.model.postprocessor
    postprocessor.bin_thresh = postprocessor.box_thresh = 0.0       # random weights: make it find something
    page = np.full((480, 640, 3), 255, dtype=np.uint8)

    passes = []
    engine.cancel_check = lambda: passes.append(1) and False        # count forward passes, never cancel
    engine.predict([page])
    assert len(passes) >= 1

    engine.cancel_check = lambda: True                              # cancelled before the first pass
    with pytest.raises(Cancelled):
        engine.predict([page])

    if len(passes) >= 2:                                            # cancelled between detection and recognition
        seen = []
        engine.cancel_check = lambda: (seen.append(1), len(seen) >= 2)[1]
        with pytest.raises(Cancelled):
            engine.predict([page])
        assert len(seen) == 2


def test_straighten_mode_gives_polygons_and_still_exports_hocr(tmp_path, make_image):
    """Rotated-page mode, with DocTR's two extra orientation models disabled so no download is needed."""
    from core.ocr_engine import OCREngine
    from core.pdf_writer import ocr_page_from_doctr
    import numpy as np

    engine = OCREngine("db_mobilenet_v3_large", "crnn_mobilenet_v3_small", "cpu", pretrained=False,
                       straighten_pages=True,
                       predictor_kwargs={"disable_crop_orientation": True, "disable_page_orientation": True})
    postprocessor = engine.model.det_predictor.model.postprocessor
    postprocessor.bin_thresh = postprocessor.box_thresh = 0.0
    document = engine.predict([np.full((480, 640, 3), 255, dtype=np.uint8)])
    assert ocr_page_from_doctr(document.pages[0], 640, 480).lines[0][0].quad is not None

    processor = OCRProcessor(engine=engine)
    processor.configure(output_formats=["pdf", "hocr"])
    result = processor.process_file(make_image(tmp_path / "a.png", (640, 480)), tmp_path / "out")
    assert result.ok, result.failure_lines()
    assert list((result.session_dir / "hocr").glob("*.hocr")) and list((result.session_dir / "pdf").glob("*.pdf"))
