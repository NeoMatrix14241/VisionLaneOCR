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
