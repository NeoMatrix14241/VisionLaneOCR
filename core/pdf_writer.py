"""Searchable-PDF output: page image + invisible text layer, built with reportlab.

Replaces ``utils/hocr_to_pdf.py`` (HOCR parsing with an integer-only regex, base-14 font without
horizontal scaling) and the Ghostscript based ``utils/pypdfcompressor.py``. Words come straight from
DocTR's result (:func:`ocr_page_from_doctr`), so no HOCR round trip is needed for PDF output, and image
compression happens *while embedding* the page, so the PDF is small from the start.
"""
from __future__ import annotations

import io
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable

from PIL import Image
from reportlab import rl_config
from reportlab.lib.utils import ImageReader
from reportlab.pdfbase.pdfmetrics import stringWidth
from reportlab.pdfgen.canvas import Canvas

from .constants import DEFAULT_DPI, MAX_PLAUSIBLE_DPI, MIN_PLAUSIBLE_DPI

logger = logging.getLogger(__name__)

# reportlab defaults to ASCII85-wrapping image/content streams, which inflates every PDF by ~25 %.
rl_config.useA85 = 0

PDF_POINTS_PER_INCH = 72.0
TEXT_FONT = "Helvetica"


@dataclass
class OcrWord:
    text: str
    x0: float  # pixel coordinates, origin top-left
    y0: float
    x1: float
    y1: float


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
                try:  # straight page: ((xmin, ymin), (xmax, ymax)); rotated page: 4 corner points
                    if len(geometry) == 2 and len(geometry[0]) == 2 and not hasattr(geometry[0][0], "__len__"):
                        (xa, ya), (xb, yb) = geometry
                        xs, ys = (xa, xb), (ya, yb)
                    else:
                        xs = [pt[0] for pt in geometry]
                        ys = [pt[1] for pt in geometry]
                except (TypeError, IndexError, ValueError):
                    continue
                words.append(OcrWord(text, min(xs) * width, min(ys) * height, max(xs) * width, max(ys) * height))
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


def _safe_text(text: str) -> str:
    """Helvetica can only encode WinAnsi; other characters become '?' instead of raising."""
    try:
        text.encode("cp1252")
        return text
    except UnicodeEncodeError:
        return text.encode("cp1252", errors="replace").decode("cp1252")


def draw_text_layer(canvas: Canvas, page: OcrPage, scale_x: float, scale_y: float, page_height_pt: float,
                    ) -> int:
    """Draw ``page`` as invisible text. ``scale_*`` convert OCR pixels to PDF points. Returns words drawn."""
    drawn = 0
    for line in page.lines:
        for index, word in enumerate(line):
            width_pt = (word.x1 - word.x0) * scale_x
            height_pt = (word.y1 - word.y0) * scale_y
            if width_pt <= 0 or height_pt <= 0:
                continue
            text = _safe_text(word.text)
            font_size = max(1.0, height_pt * 0.85)
            natural = stringWidth(text, TEXT_FONT, font_size)
            if natural <= 0:
                continue
            horiz = max(10.0, min(400.0, 100.0 * width_pt / natural))
            x = word.x0 * scale_x
            y = page_height_pt - word.y1 * scale_y + height_pt * 0.2  # baseline a little above the box bottom
            obj = canvas.beginText()
            obj.setTextRenderMode(3)  # invisible
            obj.setFont(TEXT_FONT, font_size)
            obj.setHorizScale(horiz)
            obj.setTextOrigin(x, y)
            obj.textOut(text + (" " if index < len(line) - 1 else ""))
            canvas.drawText(obj)
            drawn += 1
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


def write_image_page_pdf(out_path: Path, image: Image.Image, ocr: OcrPage | None, dpi: int, *,
                         compress_type: str | None = None, quality: int = 85,
                         passthrough: Path | None = None) -> None:
    """Write a one-page PDF: the scan at ``dpi`` with an invisible text layer on top.

    ``compress_type``: ``None`` = lossless, ``"jpeg"`` = JPEG at ``quality``.
    ``passthrough``: original JPEG file to embed unmodified when no recompression was requested.
    """
    image = normalize_for_pdf(image)
    width_px, height_px = image.size
    page_w = width_px * PDF_POINTS_PER_INCH / dpi
    page_h = height_px * PDF_POINTS_PER_INCH / dpi
    canvas = Canvas(str(out_path), pagesize=(page_w, page_h), pageCompression=1)
    canvas.drawImage(_embeddable_image(image, compress_type, quality, passthrough), 0, 0, width=page_w, height=page_h)
    if ocr is not None:
        draw_text_layer(canvas, ocr, page_w / width_px, page_h / height_px, page_h)
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
