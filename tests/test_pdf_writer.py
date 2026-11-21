from types import SimpleNamespace

from PIL import Image
from pypdf import PdfReader

from core.pdf_writer import (OcrPage, OcrWord, ocr_page_from_doctr, resolve_dpi, write_image_page_pdf)


def noisy(size):
    import random
    random.seed(1)
    img = Image.new("RGB", size)
    img.putdata([(random.randrange(256), random.randrange(256), random.randrange(256)) for _ in range(size[0] * size[1])])
    return img


def test_jpeg_compression_actually_shrinks_the_pdf(tmp_path):
    img = noisy((400, 300))
    page = OcrPage(400, 300, [[OcrWord("hello", 10, 10, 90, 30)]])
    write_image_page_pdf(tmp_path / "lossless.pdf", img, page, 300)
    write_image_page_pdf(tmp_path / "q90.pdf", img, page, 300, compress_type="jpeg", quality=90)
    write_image_page_pdf(tmp_path / "q20.pdf", img, page, 300, compress_type="jpeg", quality=20)
    sizes = [(tmp_path / n).stat().st_size for n in ("lossless.pdf", "q90.pdf", "q20.pdf")]
    assert sizes[0] > sizes[1] > sizes[2]                      # quality slider has a real effect
    assert "hello" in PdfReader(str(tmp_path / "q20.pdf")).pages[0].extract_text()


def test_jpeg_source_is_embedded_untouched_without_compression(tmp_path):
    src = tmp_path / "s.jpg"
    noisy((200, 200)).save(src, quality=75)
    write_image_page_pdf(tmp_path / "o.pdf", Image.open(src), None, 300, passthrough=src)
    assert (tmp_path / "o.pdf").stat().st_size < src.stat().st_size * 1.2   # DCT data copied, not Flate'd


def test_modes_and_odd_characters_do_not_crash(tmp_path):
    for mode in ("1", "L", "P", "RGBA", "LA", "CMYK"):
        img = Image.new(mode, (50, 50))
        write_image_page_pdf(tmp_path / f"{mode}.pdf", img, OcrPage(50, 50, [[OcrWord("Ж日本€é", 1, 1, 40, 10)]]), 100)
        assert PdfReader(str(tmp_path / f"{mode}.pdf")).pages


def test_ocr_page_from_doctr_handles_straight_and_polygon_boxes():
    straight = SimpleNamespace(value="a", geometry=((0.1, 0.2), (0.3, 0.4)))
    polygon = SimpleNamespace(value="b", geometry=((0.5, 0.5), (0.7, 0.5), (0.7, 0.6), (0.5, 0.6)))
    empty = SimpleNamespace(value="  ", geometry=((0, 0), (1, 1)))
    page = SimpleNamespace(blocks=[SimpleNamespace(lines=[SimpleNamespace(words=[straight, empty]),
                                                          SimpleNamespace(words=[polygon])])])
    out = ocr_page_from_doctr(page, 1000, 500)
    assert out.word_count == 2
    assert (out.lines[0][0].x0, out.lines[0][0].y1) == (100, 200)
    assert (out.lines[1][0].x1, out.lines[1][0].y0) == (700, 250)


def test_resolve_dpi_prefers_user_then_metadata_then_default():
    img = Image.new("RGB", (10, 10))
    assert resolve_dpi(200, img) == 200
    img.info["dpi"] = (150.0, 150.0)
    assert resolve_dpi(None, img) == 150
    img.info["dpi"] = (1, 1)           # implausible metadata
    assert resolve_dpi(None, img) == 300
    assert resolve_dpi(None, None) == 300
