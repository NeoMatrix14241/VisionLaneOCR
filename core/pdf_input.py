"""PDF input: rasterise pages for OCR with ``pypdfium2`` and put the recognised text back onto the
*original* pages with ``pypdf``.

Compared with the old Ghostscript/pdf2image route this keeps vector content and any existing text, does
not need Ghostscript or Poppler, honours the DPI setting, and does not re-OCR pages that already have text.
"""
from __future__ import annotations

import logging
import tempfile
from pathlib import Path

from PIL import Image

from .constants import DEFAULT_DPI, MIN_TEXT_CHARS_FOR_SKIP, PDF_RENDER_MAX_PIXELS
from .pdf_writer import OcrPage, write_text_overlay_pdf

logger = logging.getLogger(__name__)


class PdfSource:
    """Read-only view of a PDF used for rasterising pages."""

    def __init__(self, path: Path):
        import pypdfium2 as pdfium
        self.path = Path(path)
        self._doc = pdfium.PdfDocument(str(self.path))

    def __enter__(self) -> "PdfSource":
        return self

    def __exit__(self, *exc) -> None:
        self.close()

    def close(self) -> None:
        if self._doc is not None:
            self._doc.close()
            self._doc = None

    @property
    def page_count(self) -> int:
        return len(self._doc)

    def has_text(self, index: int) -> bool:
        page = self._doc[index]
        textpage = page.get_textpage()
        try:
            return textpage.count_chars() >= MIN_TEXT_CHARS_FOR_SKIP
        finally:
            textpage.close()
            page.close()

    def render(self, index: int, dpi: int | None) -> Image.Image:
        """Render page ``index`` (0-based) as RGB at ``dpi`` (default 300), capped to a sane pixel count."""
        page = self._doc[index]
        try:
            width_pt, height_pt = page.get_size()
            scale = (dpi or DEFAULT_DPI) / 72.0
            pixels = width_pt * scale * height_pt * scale
            if pixels > PDF_RENDER_MAX_PIXELS:
                scale *= (PDF_RENDER_MAX_PIXELS / pixels) ** 0.5
                logger.info("Page %d of %s is very large; rendering at %.0f dpi", index + 1, self.path.name,
                            scale * 72)
            bitmap = page.render(scale=scale)
            image = bitmap.to_pil().convert("RGB")
            bitmap.close()
            return image
        finally:
            page.close()


def apply_text_layers(src_pdf: Path, out_pdf: Path, layers: dict[int, OcrPage]) -> None:
    """Copy ``src_pdf`` to ``out_pdf`` adding an invisible text layer to the pages in ``layers``.

    ``layers`` maps 0-based page index -> OCR result for the page *as rendered* (upright).
    Pages without an entry are copied unchanged, so the output always has every page of the input.
    """
    from pypdf import PdfReader, PdfWriter, Transformation

    reader = PdfReader(str(src_pdf))
    if reader.is_encrypted:
        raise ValueError("The PDF is password protected")
    writer = PdfWriter(clone_from=reader)
    tmp_out = out_pdf.with_name(out_pdf.name + ".part")
    try:
        with tempfile.TemporaryDirectory(prefix="vl_overlay_") as tmp:
            for index, ocr in layers.items():
                page = writer.pages[index]
                if page.rotation % 360:
                    page.transfer_rotation_to_content()  # the OCR'd image is upright; make the page upright too
                box = page.cropbox
                width, height = float(box.width), float(box.height)
                overlay_path = Path(tmp) / f"overlay_{index}.pdf"
                write_text_overlay_pdf(overlay_path, ocr, width, height)
                overlay_page = PdfReader(str(overlay_path)).pages[0]
                page.merge_transformed_page(overlay_page,
                                            Transformation().translate(float(box.left), float(box.bottom)))
        out_pdf.parent.mkdir(parents=True, exist_ok=True)
        with open(tmp_out, "wb") as handle:
            writer.write(handle)
        tmp_out.replace(out_pdf)
    finally:
        writer.close()
        tmp_out.unlink(missing_ok=True)
