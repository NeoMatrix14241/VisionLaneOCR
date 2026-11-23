from pathlib import Path

import pytest
from pypdf import PdfReader
from reportlab.pdfgen.canvas import Canvas
from PIL import Image

from core.ocr_processor import OCRProcessor
from core.results import Status
from tests.conftest import FakeEngine


def pdf_pages(path: Path) -> int:
    return len(PdfReader(str(path)).pages)


def test_folder_becomes_one_pdf_in_natural_order(tmp_path, make_image):
    src = tmp_path / "in" / "Box 1"
    # widths encode the page order we expect: img2 < img10
    make_image(src / "img10.png", (300, 100))
    make_image(src / "img2.png", (200, 100))
    make_image(src / "img1.png", (100, 100))
    p = OCRProcessor(engine=FakeEngine())
    result = p.process_folder(tmp_path / "in", tmp_path / "out")
    assert result.ok and result.succeeded == 3
    pdfs = list((result.session_dir / "pdf").rglob("*.pdf"))
    assert [x.name for x in pdfs] == ["Box 1.pdf"]            # no extra subfolder next to the PDF
    reader = PdfReader(str(pdfs[0]))
    widths = [round(float(pg.mediabox.width) * 300 / 72) for pg in reader.pages]
    assert widths == [100, 200, 300]


def test_root_level_images_use_root_folder_name(tmp_path, make_image):
    make_image(tmp_path / "scans" / "a.png")
    result = OCRProcessor(engine=FakeEngine()).process_folder(tmp_path / "scans", tmp_path / "out")
    assert (result.session_dir / "pdf" / "scans.pdf").exists()


def test_failed_page_is_reported_and_not_silent(tmp_path, make_image):
    make_image(tmp_path / "in" / "a.png", (100, 40))
    make_image(tmp_path / "in" / "b.png", (100, 33))   # height 33 -> engine raises
    make_image(tmp_path / "in" / "c.png", (100, 50))
    p = OCRProcessor(engine=FakeEngine(fail_heights=(33,)))
    result = p.process_folder(tmp_path / "in", tmp_path / "out")
    by_name = {f.source.name: f for f in result.files}
    assert by_name["b.png"].status == Status.FAILED and "synthetic" in by_name["b.png"].error
    assert by_name["a.png"].status == Status.SUCCESS and by_name["c.png"].status == Status.SUCCESS
    assert not result.ok and result.failed == 1
    assert {s.name for s in result.confirmed_sources()} == {"a.png", "c.png"}
    pdf = next((result.session_dir / "pdf").rglob("*.pdf"))
    assert pdf_pages(pdf) == 2                            # failed page absent, others present
    assert result.failure_lines() and "b.png" in result.failure_lines()[0]


def test_all_pages_failing_gives_no_blank_pdf(tmp_path, make_image):
    make_image(tmp_path / "in" / "x.png", (100, 33))
    result = OCRProcessor(engine=FakeEngine(fail_heights=(33,))).process_folder(tmp_path / "in", tmp_path / "out")
    assert result.files[0].status == Status.FAILED
    assert not list((result.session_dir / "pdf").rglob("*.pdf"))


def test_corrupt_image_fails_cleanly(tmp_path):
    bad = tmp_path / "in" / "bad.png"
    bad.parent.mkdir()
    bad.write_bytes(b"not an image")
    result = OCRProcessor(engine=FakeEngine()).process_folder(tmp_path / "in", tmp_path / "out")
    assert result.files[0].status == Status.FAILED and result.files[0].error


def test_hocr_only_writes_no_pdf_and_no_wait(tmp_path, make_image):
    make_image(tmp_path / "in" / "sub" / "a.png")
    make_image(tmp_path / "in" / "sub" / "b.png")
    p = OCRProcessor(engine=FakeEngine())
    p.configure(output_formats=["hocr"])
    result = p.process_folder(tmp_path / "in", tmp_path / "out")
    assert result.ok
    assert sorted(x.name for x in (result.session_dir / "hocr").rglob("*.hocr")) == ["a.hocr", "b.hocr"]
    assert not (result.session_dir / "pdf").exists()


def test_same_stem_different_extension_does_not_collide(tmp_path, make_image):
    make_image(tmp_path / "in" / "x.png")
    make_image(tmp_path / "in" / "x.jpg")
    p = OCRProcessor(engine=FakeEngine())
    p.configure(output_formats=["hocr"])
    result = p.process_folder(tmp_path / "in", tmp_path / "out")
    assert len(list((result.session_dir / "hocr").rglob("*.hocr"))) == 2


def test_single_image_mode(tmp_path, make_image):
    img = make_image(tmp_path / "scan.jpg", fmt="JPEG")
    p = OCRProcessor(engine=FakeEngine())
    p.configure(output_formats=["pdf", "hocr"])
    result = p.process_file(img, tmp_path / "out")
    assert result.ok
    assert (result.session_dir / "pdf" / "scan.pdf").exists()
    assert (result.session_dir / "hocr" / "scan.hocr").exists()
    assert result.files[0].source == img.resolve()


def test_multiframe_tiff_keeps_all_pages(tmp_path):
    tiff = tmp_path / "in" / "multi.tif"
    tiff.parent.mkdir()
    frames = [Image.new("RGB", (120, 80 + i), "white") for i in range(3)]
    frames[0].save(tiff, save_all=True, append_images=frames[1:])
    result = OCRProcessor(engine=FakeEngine()).process_folder(tmp_path / "in", tmp_path / "out")
    assert result.ok and result.files[0].pages_total == 3
    assert pdf_pages(next((result.session_dir / "pdf").rglob("*.pdf"))) == 3


def test_dpi_controls_page_size(tmp_path, make_image):
    img = make_image(tmp_path / "a.png", (600, 300))
    p = OCRProcessor(engine=FakeEngine())
    p.configure(dpi=150)
    result = p.process_file(img, tmp_path / "out")
    page = PdfReader(str(next((result.session_dir / "pdf").rglob("*.pdf")))).pages[0]
    assert round(float(page.mediabox.width)) == 288      # 600 px / 150 dpi * 72


def test_cancel_stops_and_marks_rest_cancelled(tmp_path, make_image):
    for i in range(5):
        make_image(tmp_path / "in" / f"{i}.png")
    engine = FakeEngine()
    p = OCRProcessor(engine=engine)
    engine.on_call = lambda n: p.cancel() if n == 2 else None
    result = p.process_folder(tmp_path / "in", tmp_path / "out")
    assert result.cancelled and not result.ok
    assert result.count(Status.CANCELLED) >= 3
    assert not list((result.session_dir / "pdf").rglob("*.pdf"))   # no half-finished PDF
    p.process_folder(tmp_path / "in", tmp_path / "out2")           # flag is reset for the next run


def test_progress_events_count_finished_files(tmp_path, make_image):
    for i in range(3):
        make_image(tmp_path / "in" / f"{i}.png")
    p = OCRProcessor(engine=FakeEngine())
    events = []
    p.progress_callback = events.append
    p.process_folder(tmp_path / "in", tmp_path / "out")
    assert events[0].files_done == 0 and events[-1].files_done == 3
    assert all(0.0 <= e.fraction <= 1.0 for e in events)
    assert [e.files_done for e in events] == sorted(e.files_done for e in events)  # never runs backwards


def test_no_files_creates_no_session_dir(tmp_path):
    (tmp_path / "empty").mkdir()
    result = OCRProcessor(engine=FakeEngine()).process_folder(tmp_path / "empty", tmp_path / "out")
    assert result.total == 0 and result.session_dir is None
    assert not (tmp_path / "out").exists()


def test_temp_files_are_private_and_cleaned(tmp_path, make_image):
    import tempfile
    before = set(Path(tempfile.gettempdir()).glob("VisionLaneOCR_*"))
    make_image(tmp_path / "in" / "a.png")
    OCRProcessor(engine=FakeEngine()).process_folder(tmp_path / "in", tmp_path / "out")
    assert set(Path(tempfile.gettempdir()).glob("VisionLaneOCR_*")) == before


def make_text_pdf(path: Path, pages: list[str | None]) -> None:
    c = Canvas(str(path), pagesize=(300, 400))
    for text in pages:
        if text:
            c.drawString(20, 350, text)
        else:
            c.rect(20, 20, 100, 100, fill=1)       # image-only-like page: no text
        c.showPage()
    c.save()


def test_pdf_input_keeps_original_pages_and_adds_text(tmp_path):
    src = tmp_path / "in" / "doc.pdf"
    src.parent.mkdir()
    make_text_pdf(src, [None, "This page already has plenty of text in it", None])
    engine = FakeEngine()
    p = OCRProcessor(engine=engine)
    result = p.process_file(src, tmp_path / "out")
    f = result.files[0]
    assert result.ok and f.pages_total == 3 and f.skipped_pages == [2]
    assert len(engine.calls) == 2                      # page 2 already had text: not re-OCR'd
    out = result.session_dir / "pdf" / "doc.pdf"       # no "_ocr" suffix
    reader = PdfReader(str(out))
    assert len(reader.pages) == 3
    assert "page1" in reader.pages[0].extract_text()   # text layer added to page 1
    assert "already has plenty" in reader.pages[1].extract_text()


def test_pdf_input_can_force_ocr_of_all_pages(tmp_path):
    src = tmp_path / "doc.pdf"
    make_text_pdf(src, ["Some existing text that is long enough", None])
    engine = FakeEngine()
    p = OCRProcessor(engine=engine)
    p.configure(skip_text_pages=False)
    p.process_file(src, tmp_path / "out")
    assert len(engine.calls) == 2


def test_pdf_in_folder_mode_and_encrypted_pdf_failure(tmp_path):
    src = tmp_path / "in" / "sub" / "doc.pdf"
    src.parent.mkdir(parents=True)
    make_text_pdf(src, [None])
    broken = tmp_path / "in" / "broken.pdf"
    broken.write_bytes(b"%PDF-1.4 garbage")
    result = OCRProcessor(engine=FakeEngine()).process_folder(tmp_path / "in", tmp_path / "out")
    status = {f.source.name: f.status for f in result.files}
    assert status == {"doc.pdf": Status.SUCCESS, "broken.pdf": Status.FAILED}
    assert (result.session_dir / "pdf" / "sub" / "doc.pdf").exists()


def test_configure_validates_formats():
    p = OCRProcessor(engine=FakeEngine())
    with pytest.raises(ValueError):
        p.configure(output_formats=["docx"])


def test_pillow_decompression_bomb_guard_is_still_on():
    from PIL import Image
    assert Image.MAX_IMAGE_PIXELS is not None


def test_exif_rotated_jpeg_gets_upright_page_geometry(tmp_path):
    from PIL import Image
    src = tmp_path / "in" / "rot.jpg"
    src.parent.mkdir()
    img = Image.new("RGB", (200, 100), "white")
    exif = Image.Exif()
    exif[0x0112] = 6                      # "rotate 90 degrees clockwise to display"
    img.save(src, exif=exif)
    engine = FakeEngine()
    result = OCRProcessor(engine=engine).process_file(src, tmp_path / "out")
    assert result.ok
    assert engine.calls == [(100, 200)]   # OCR saw the upright (portrait) image ...
    page = PdfReader(str(next((result.session_dir / "pdf").rglob("*.pdf")))).pages[0]
    assert float(page.mediabox.height) > float(page.mediabox.width)   # ... and so does the PDF page


def test_rotated_words_survive_into_the_pdf_and_hocr_export_uses_straight_boxes(tmp_path, make_image):
    """FakeEngine returns polygon geometry (as DocTR does in rotated-page mode)."""
    from types import SimpleNamespace
    from tests.conftest import FakeDocument

    class RotatedDocument(FakeDocument):
        def __init__(self, text):
            super().__init__(text)
            quad = ((0.2, 0.8), (0.2, 0.4), (0.3, 0.4), (0.3, 0.8))          # text reading upwards
            word = SimpleNamespace(value="UP", geometry=quad)
            line = SimpleNamespace(words=[word], geometry=quad)
            self.pages = [SimpleNamespace(blocks=[SimpleNamespace(lines=[line], geometry=quad, artefacts=[])])]
            self.seen_geometries = []

        def export_as_xml(self):
            word = self.pages[0].blocks[0].lines[0].words[0]
            self.seen_geometries.append(word.geometry)
            return [(b"<html/>", None)]

    class RotatedEngine(FakeEngine):
        def predict(self, images):
            self.document = RotatedDocument("x")
            return self.document

    engine = RotatedEngine()
    p = OCRProcessor(engine=engine)
    p.configure(output_formats=["pdf", "hocr"])
    result = p.process_file(make_image(tmp_path / "a.png", (200, 100)), tmp_path / "out")
    assert result.ok, result.failure_lines()
    assert engine.document.seen_geometries == [((0.2, 0.4), (0.3, 0.8))]     # hOCR got straight boxes
    runs = []
    PdfReader(str(next((result.session_dir / "pdf").glob("*.pdf")))).pages[0].extract_text(
        visitor_text=lambda text, cm, tm, font, size: runs.append(tm) if text.strip() else None)
    assert round(float(runs[0][1]), 3) == 1.0 and round(float(runs[0][0]), 3) == 0.0     # PDF text is rotated 90
