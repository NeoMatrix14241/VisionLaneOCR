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


def extract(path):
    """(text, tm) tuples for every text run, via pypdf's visitor."""
    runs = []
    PdfReader(str(path)).pages[0].extract_text(
        visitor_text=lambda text, cm, tm, font, size: runs.append((text, tm, font.get("/BaseFont") if font else None)))
    return [r for r in runs if r[0].strip()]


# ----------------------------------------------------------------------------- VL-062: Unicode text layer
def test_non_latin_text_is_searchable_and_latin_only_pages_embed_no_font(tmp_path):
    words = [OcrWord("Привет", 10, 10, 120, 30), OcrWord("Ελληνικά", 130, 10, 260, 30),
             OcrWord("Việt", 10, 40, 80, 60), OcrWord("日本", 90, 40, 140, 60)]
    write_image_page_pdf(tmp_path / "u.pdf", Image.new("RGB", (300, 100), "white"), OcrPage(300, 100, [words]), 150)
    text = " ".join(t for t, _tm, _f in extract(tmp_path / "u.pdf"))
    assert "Привет" in text and "Ελληνικά" in text and "Việt" in text
    assert "日本" not in text and "?" in text            # CJK is not in DejaVu Sans -> explicit '?'
    assert b"DejaVuSans" in (tmp_path / "u.pdf").read_bytes()

    write_image_page_pdf(tmp_path / "l.pdf", Image.new("RGB", (300, 100), "white"),
                         OcrPage(300, 100, [[OcrWord("Café déjà vu €", 10, 10, 120, 30)]]), 150)
    assert b"DejaVuSans" not in (tmp_path / "l.pdf").read_bytes()    # WinAnsi page: nothing embedded
    assert "Café" in extract(tmp_path / "l.pdf")[0][0]


# ----------------------------------------------------------------------------- VL-061: rotated words
def test_rotated_word_gets_a_rotated_text_matrix(tmp_path):
    # Text reading upwards (rotated 90 degrees counter-clockwise): corners TL, TR, BR, BL of the *text*
    quad = ((30.0, 250.0), (30.0, 150.0), (50.0, 150.0), (50.0, 250.0))
    word = OcrWord("UP", 30, 150, 50, 250, quad)
    write_image_page_pdf(tmp_path / "r.pdf", Image.new("RGB", (200, 300), "white"), OcrPage(200, 300, [[word]]), 72)
    (text, tm, _font), = extract(tmp_path / "r.pdf")
    assert text.strip() == "UP"
    assert [round(float(v), 3) for v in tm] == [0.0, 1.0, -1.0, 0.0, 46.0, 50.0]


def test_upside_down_word_points_the_other_way(tmp_path):
    # rotated 180 degrees: reading right-to-left, text "down" is up the page
    quad = ((100.0, 80.0), (20.0, 80.0), (20.0, 60.0), (100.0, 60.0))
    write_image_page_pdf(tmp_path / "d.pdf", Image.new("RGB", (200, 100), "white"),
                         OcrPage(200, 100, [[OcrWord("DOWN", 20, 60, 100, 80, quad)]]), 72)
    (_text, tm, _font), = extract(tmp_path / "d.pdf")
    assert round(float(tm[0]), 3) == -1.0 and round(float(tm[3]), 3) == -1.0     # rotated by 180 degrees


def test_ocr_page_from_doctr_keeps_polygon_corners():
    polygon = SimpleNamespace(value="b", geometry=((0.5, 0.5), (0.7, 0.5), (0.7, 0.6), (0.5, 0.6)))
    straight = SimpleNamespace(value="a", geometry=((0.1, 0.2), (0.3, 0.4)))
    page = SimpleNamespace(blocks=[SimpleNamespace(lines=[SimpleNamespace(words=[polygon, straight])])])
    words = ocr_page_from_doctr(page, 1000, 500).lines[0]
    assert words[0].quad == ((500.0, 250.0), (700.0, 250.0), (700.0, 300.0), (500.0, 300.0))
    assert words[1].quad is None


# ----------------------------------------------------------------------------- VL-067: fit to paper
def test_image_is_fitted_and_centred_on_the_paper_and_text_follows(tmp_path):
    from core.paper import paper_size_pt
    paper = paper_size_pt("A4")                                       # 595.44 x 841.68 pt
    word = OcrWord("hi", 100, 100, 200, 150)
    write_image_page_pdf(tmp_path / "p.pdf", Image.new("RGB", (1000, 500), "white"),
                         OcrPage(1000, 500, [[word]]), 100, paper_pt=paper)   # image = 720 x 360 pt
    page = PdfReader(str(tmp_path / "p.pdf")).pages[0]
    assert [round(float(v), 2) for v in (page.mediabox.width, page.mediabox.height)] == [595.44, 841.68]
    fit = min(paper[0] / 720, paper[1] / 360)                         # width-limited
    draw_w, draw_h = 720 * fit, 360 * fit
    top = (paper[1] - draw_h) / 2
    (_text, tm, _font), = extract(tmp_path / "p.pdf")
    sx = draw_w / 1000
    assert abs(float(tm[4]) - 100 * sx) < 0.01
    assert abs(float(tm[5]) - (paper[1] - (top + 150 * sx) + 0.2 * 50 * sx)) < 0.01
