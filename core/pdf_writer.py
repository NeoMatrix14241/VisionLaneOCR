"""Searchable-PDF output: page image + invisible text layer, built with reportlab.

Replaces ``utils/hocr_to_pdf.py`` (HOCR parsing with an integer-only regex, base-14 font without
horizontal scaling) and the Ghostscript based ``utils/pypdfcompressor.py``. Words come straight from
DocTR's result (:func:`ocr_page_from_doctr`), so no HOCR round trip is needed for PDF output, and image
compression happens *while embedding* the page, so the PDF is small from the start.
"""
from __future__ import annotations

import io
import logging
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Sequence

from PIL import Image
from reportlab import rl_config
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.pdfgen.canvas import Canvas

from .constants import DEFAULT_DPI, MAX_PLAUSIBLE_DPI, MIN_PLAUSIBLE_DPI

logger = logging.getLogger(__name__)

# reportlab defaults to ASCII85-wrapping image/content streams, which inflates every PDF by ~25 %.
rl_config.useA85 = 0

PDF_POINTS_PER_INCH = 72.0
TEXT_FONT = "Helvetica"                      # built-in, not embedded: used whenever the page is WinAnsi-only
UNICODE_FONT = "DejaVuSans"                  # embedded (subset) only for pages that need more than WinAnsi
UNICODE_FONT_FILE = Path(__file__).resolve().parent.parent / "assets" / "fonts" / "DejaVuSans.ttf"
_unicode_font_ready: bool | None = None
Quad = tuple[tuple[float, float], tuple[float, float], tuple[float, float], tuple[float, float]]


@dataclass
class OcrWord:
    text: str
    x0: float  # axis-aligned bounding box, pixel coordinates, origin top-left
    y0: float
    x1: float
    y1: float
    # Rotated words: corners in reading order (top-left, top-right, bottom-right, bottom-left of the *text*),
    # pixel coordinates. ``None`` for straight words (the bounding box is then exact).
    quad: Quad | None = None


@dataclass
class OcrPage:
    width: int   # page image size in pixels
    height: int
    lines: list[list[OcrWord]] = field(default_factory=list)

    @property
    def word_count(self) -> int:
        return sum(len(line) for line in self.lines)


def ocr_page_from_doctr(page, width: int, height: int) -> OcrPage:
    """Convert a ``doctr.io.elements.Page`` into :class:`OcrPage` (pixel coordinates)."""
    result = OcrPage(width=width, height=height)
    for block in page.blocks:
        for line in block.lines:
            words: list[OcrWord] = []
            for word in line.words:
                text = (word.value or "").strip()
                if not text:
                    continue
                geometry = word.geometry
                quad: Quad | None = None
                try:  # straight page: ((xmin, ymin), (xmax, ymax)); rotated page: 4 corner points
                    xs: Sequence[float]
                    ys: Sequence[float]
                    if len(geometry) == 2 and len(geometry[0]) == 2 and not hasattr(geometry[0][0], "__len__"):
                        (xa, ya), (xb, yb) = geometry
                        xs, ys = (xa, xb), (ya, yb)
                    else:
                        xs = [pt[0] for pt in geometry]
                        ys = [pt[1] for pt in geometry]
                        if len(geometry) == 4:
                            p0, p1, p2, p3 = ((float(pt[0]) * width, float(pt[1]) * height) for pt in geometry)
                            quad = (p0, p1, p2, p3)
                except (TypeError, IndexError, ValueError):
                    continue
                words.append(OcrWord(text, min(xs) * width, min(ys) * height, max(xs) * width, max(ys) * height, quad))
            if words:
                result.lines.append(words)
    return result


def resolve_dpi(requested: int | None, image: Image.Image | None = None) -> int:
    """DPI to use for page geometry: the user's choice, else image metadata, else 300."""
    if requested:
        return int(requested)
    if image is not None:
        meta = image.info.get("dpi")
        try:
            value = float(meta[0]) if meta else 0.0
        except (TypeError, ValueError, IndexError):
            value = 0.0
        if MIN_PLAUSIBLE_DPI <= value <= MAX_PLAUSIBLE_DPI:
            return int(round(value))
    return DEFAULT_DPI


def unicode_font() -> str | None:
    """Register the bundled Unicode font once; ``None`` if it cannot be loaded (Helvetica is used then)."""
    global _unicode_font_ready  # pylint: disable=global-statement
    if _unicode_font_ready is None:
        try:
            pdfmetrics.registerFont(TTFont(UNICODE_FONT, str(UNICODE_FONT_FILE)))
            _unicode_font_ready = True
        except Exception as exc:  # pylint: disable=broad-except
            logger.warning("Unicode font unavailable (%s); non-Latin text will appear as '?' in PDFs", exc)
            _unicode_font_ready = False
    return UNICODE_FONT if _unicode_font_ready else None


def _is_winansi(text: str) -> bool:
    try:
        text.encode("cp1252")
        return True
    except UnicodeEncodeError:
        return False


def page_font(page: OcrPage) -> str:
    """Helvetica when every word is WinAnsi (the usual case, nothing is embedded); otherwise the Unicode font."""
    if all(_is_winansi(word.text) for line in page.lines for word in line):
        return TEXT_FONT
    return unicode_font() or TEXT_FONT


def _prepare_text(text: str, font: str) -> str:
    """Replace characters the font cannot show with '?' (reportlab would otherwise fail or draw .notdef)."""
    if font == TEXT_FONT:
        return text if _is_winansi(text) else text.encode("cp1252", errors="replace").decode("cp1252")
    glyphs = pdfmetrics.getFont(font).face.charToGlyph  # type: ignore[attr-defined]
    return "".join(ch if ord(ch) in glyphs else "?" for ch in text)


def _draw_straight_word(canvas: Canvas, word: OcrWord, text: str, font: str, trailing: str,
                        scale_x: float, scale_y: float, page_h: float, origin: tuple[float, float]) -> bool:
    width_pt = (word.x1 - word.x0) * scale_x
    height_pt = (word.y1 - word.y0) * scale_y
    if width_pt <= 0 or height_pt <= 0:
        return False
    font_size = max(1.0, height_pt * 0.85)
    natural = stringWidth(text, font, font_size)
    if natural <= 0:
        return False
    obj = canvas.beginText()
    obj.setTextRenderMode(3)  # invisible
    obj.setFont(font, font_size)
    obj.setHorizScale(max(10.0, min(400.0, 100.0 * width_pt / natural)))
    obj.setTextOrigin(origin[0] + word.x0 * scale_x,
                      page_h - (origin[1] + word.y1 * scale_y) + height_pt * 0.2)  # baseline a little above the box bottom
    obj.textOut(text + trailing)
    canvas.drawText(obj)
    return True


def _draw_rotated_word(canvas: Canvas, quad: Quad, text: str, font: str, trailing: str,
                       scale_x: float, scale_y: float, page_h: float, origin: tuple[float, float]) -> bool:
    """Draw a word whose corners are (top-left, top-right, bottom-right, bottom-left) in *text* orientation."""
    p0, p1, _p2, p3 = ((origin[0] + x * scale_x, page_h - (origin[1] + y * scale_y)) for x, y in quad)
    width_pt = math.hypot(p1[0] - p0[0], p1[1] - p0[1])
    down_x, down_y = p3[0] - p0[0], p3[1] - p0[1]          # from the top edge to the bottom edge of the text
    height_pt = math.hypot(down_x, down_y)
    if width_pt <= 0 or height_pt <= 0:
        return False
    font_size = max(1.0, height_pt * 0.85)
    natural = stringWidth(text, font, font_size)
    if natural <= 0:
        return False
    cos_t, sin_t = (p1[0] - p0[0]) / width_pt, (p1[1] - p0[1]) / width_pt    # reading direction
    origin_x = p3[0] - down_x / height_pt * 0.2 * height_pt    # bottom-left corner moved "up" the text by 20 %
    origin_y = p3[1] - down_y / height_pt * 0.2 * height_pt
    obj = canvas.beginText()
    obj.setTextRenderMode(3)
    obj.setFont(font, font_size)
    obj.setHorizScale(max(10.0, min(400.0, 100.0 * width_pt / natural)))
    obj.setTextTransform(cos_t, sin_t, -sin_t, cos_t, origin_x, origin_y)
    obj.textOut(text + trailing)
    canvas.drawText(obj)
    return True


def draw_text_layer(canvas: Canvas, page: OcrPage, scale_x: float, scale_y: float, page_height_pt: float,
                    origin: tuple[float, float] = (0.0, 0.0)) -> int:
    """Draw ``page`` as invisible text and return the number of words drawn.

    ``scale_*`` convert OCR pixels to PDF points; ``origin`` is the top-left corner of the OCR image on the
    PDF page in points (x from the left, y from the top) - non-zero when the image is centred on a larger page.
    """
    font = page_font(page)
    drawn = 0
    for line in page.lines:
        for index, word in enumerate(line):
            text = _prepare_text(word.text, font)
            trailing = " " if index < len(line) - 1 else ""
            if word.quad is not None:
                ok = _draw_rotated_word(canvas, word.quad, text, font, trailing, scale_x, scale_y, page_height_pt, origin)
            else:
                ok = _draw_straight_word(canvas, word, text, font, trailing, scale_x, scale_y, page_height_pt, origin)
            drawn += int(ok)
    return drawn


def normalize_for_pdf(image: Image.Image) -> Image.Image:
    """Return an 'L' or 'RGB' image (transparency is composited on white)."""
    if image.mode in ("L", "RGB"):
        return image
    if image.mode == "1":
        return image.convert("L")
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        background = Image.new("RGB", rgba.size, (255, 255, 255))
        background.paste(rgba, mask=rgba.split()[3])
        return background
    if image.mode.startswith("I;16") or image.mode in ("I", "F"):
        return image.convert("L")
    return image.convert("RGB")


def _embeddable_image(image: Image.Image, compress_type: str | None, quality: int,
                      passthrough: Path | None) -> ImageReader:
    if compress_type is None and passthrough is not None:
        return ImageReader(str(passthrough))  # JPEG source embedded untouched (reportlab keeps DCT data)
    if compress_type == "jpeg":
        buf = io.BytesIO()
        image.save(buf, "JPEG", quality=max(1, min(100, int(quality))), optimize=True)
        buf.seek(0)
        return ImageReader(buf)
    return ImageReader(image)  # lossless (Flate)


def effective_dpi(width_px: int, height_px: int, dpi: int, paper_pt: tuple[float, float] | None) -> float:
    """Resolution of the image as it will appear on the page (drops when a small scan is enlarged to fit the paper)."""
    if paper_pt is None:
        return float(dpi)
    image_w = width_px * PDF_POINTS_PER_INCH / dpi
    image_h = height_px * PDF_POINTS_PER_INCH / dpi
    fit = min(paper_pt[0] / image_w, paper_pt[1] / image_h)
    return dpi / fit


def write_image_page_pdf(out_path: Path, image: Image.Image, ocr: OcrPage | None, dpi: int, *,
                         compress_type: str | None = None, quality: int = 85,
                         passthrough: Path | None = None,
                         paper_pt: tuple[float, float] | None = None) -> None:
    """Write a one-page PDF: the scan with an invisible text layer on top.

    The page is ``pixels x 72 / dpi`` points, or - with ``paper_pt`` - exactly that paper size, with the image
    scaled to fit (aspect ratio kept) and centred.
    ``compress_type``: ``None`` = lossless, ``"jpeg"`` = JPEG at ``quality``.
    ``passthrough``: original JPEG file to embed unmodified when no recompression was requested.
    """
    image = normalize_for_pdf(image)
    width_px, height_px = image.size
    image_w = width_px * PDF_POINTS_PER_INCH / dpi
    image_h = height_px * PDF_POINTS_PER_INCH / dpi
    if paper_pt is None:
        page_w, page_h, draw_w, draw_h = image_w, image_h, image_w, image_h
    else:
        page_w, page_h = paper_pt
        fit = min(page_w / image_w, page_h / image_h)
        draw_w, draw_h = image_w * fit, image_h * fit
    left, top = (page_w - draw_w) / 2.0, (page_h - draw_h) / 2.0
    canvas = Canvas(str(out_path), pagesize=(page_w, page_h), pageCompression=1)
    canvas.drawImage(_embeddable_image(image, compress_type, quality, passthrough), left, page_h - top - draw_h,
                     width=draw_w, height=draw_h)
    if ocr is not None:
        draw_text_layer(canvas, ocr, draw_w / width_px, draw_h / height_px, page_h, origin=(left, top))
    canvas.showPage()
    canvas.save()


def write_text_overlay_pdf(out_path: Path, ocr: OcrPage, width_pt: float, height_pt: float) -> None:
    """Write a one-page PDF containing only the invisible text layer (for overlaying an existing page)."""
    canvas = Canvas(str(out_path), pagesize=(width_pt, height_pt), pageCompression=1)
    draw_text_layer(canvas, ocr, width_pt / ocr.width, height_pt / ocr.height, height_pt)
    canvas.showPage()
    canvas.save()


def merge_pdfs(page_pdfs: Iterable[Path], out_path: Path) -> int:
    """Merge single-page PDFs (in order) into ``out_path`` atomically. Returns the page count."""
    from pypdf import PdfWriter
    paths = list(page_pdfs)
    tmp = out_path.with_name(out_path.name + ".part")
    writer = PdfWriter()
    try:
        for path in paths:
            writer.append(str(path))
        out_path.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp, "wb") as handle:
            writer.write(handle)
        tmp.replace(out_path)
    finally:
        writer.close()
        if tmp.exists():
            tmp.unlink(missing_ok=True)
    return len(paths)
