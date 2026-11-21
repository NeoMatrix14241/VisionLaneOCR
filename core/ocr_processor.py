"""OCR orchestration: discover inputs, run the engine, write PDF/HOCR, report results.

Public API (used by the GUI and ``run_ocr.py``)::

    processor = OCRProcessor(detection_model=..., recognition_model=..., device="auto")
    processor.configure(output_formats=["pdf", "hocr"], dpi=None, compress_enabled=True, ...)
    processor.progress_callback = lambda event: ...      # core.results.ProgressEvent
    result = processor.process_folder(folder, output_dir)  # -> core.results.BatchResult
    result = processor.process_file(path, output_dir)      # image or PDF
    processor.cancel()                                      # from any thread

Output layout (inside ``<output_dir>/OCR_Session_<timestamp>/``)::

    pdf/<relative folder>/<folder name>.pdf     one PDF per image folder (pages in natural name order)
    pdf/<relative folder>/<pdf name>.pdf        PDF inputs: the original pages + a text layer
    hocr/<relative folder>/<image name>.hocr    one HOCR per image (multi-frame TIFF: <name>_p0002.hocr, ...)
    hocr/<relative folder>/<pdf name>/<pdf name>_page_0001.hocr
"""
from __future__ import annotations

import gc
import logging
import sys
import tempfile
import threading
from collections import defaultdict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import TYPE_CHECKING, Callable, Iterator, Sequence

import numpy as np
from PIL import Image, ImageOps

from . import pdf_input
from .constants import (DEFAULT_DETECTION_MODEL, DEFAULT_RECOGNITION_MODEL, image_extensions, is_image, is_pdf,
                        natural_key)
from .pdf_writer import (OcrPage, merge_pdfs, normalize_for_pdf, ocr_page_from_doctr, resolve_dpi,
                         write_image_page_pdf)
from .results import BatchResult, FileResult, ProgressEvent, Status

if TYPE_CHECKING:  # the engine (and torch) is imported lazily so that file handling can be used/tested without it
    from .ocr_engine import OCREngine

# Pillow's decompression-bomb guard stays ON. Raise the limit moderately: big (but legitimate) scans pass,
# absurd headers still raise ``DecompressionBombError`` -> the file is reported as failed.
Image.MAX_IMAGE_PIXELS = 400_000_000

logger = logging.getLogger(__name__)

ProgressCallback = Callable[[ProgressEvent], None]


class Cancelled(Exception):
    """Raised internally to unwind when :meth:`OCRProcessor.cancel` was called."""


@dataclass
class _Frame:
    index: int            # 1-based frame number within the file
    count: int
    image: Image.Image    # upright PIL image (EXIF orientation applied)
    dpi: int
    passthrough: Path | None  # original JPEG file when it can be embedded untouched


def _open_frames(path: Path, requested_dpi: int | None) -> Iterator[_Frame]:
    """Yield the pages of an image file (multi-frame TIFFs yield every frame)."""
    with Image.open(path) as src:
        is_tiff = (src.format or "").upper() == "TIFF"
        count = getattr(src, "n_frames", 1) if is_tiff else 1
        is_jpeg = (src.format or "").upper() == "JPEG"
        for frame_no in range(count):
            if count > 1:
                src.seek(frame_no)
            frame = src.copy() if count > 1 else src
            orientation = frame.getexif().get(0x0112, 1)
            image = ImageOps.exif_transpose(frame) or frame
            image.load()
            dpi = resolve_dpi(requested_dpi, frame)
            passthrough = path if (is_jpeg and orientation in (0, 1) and image.mode in ("L", "RGB")) else None
            yield _Frame(frame_no + 1, count, image, dpi, passthrough)


def _empty_cuda_cache() -> None:
    torch = sys.modules.get("torch")  # only if the engine loaded it
    if torch is not None and torch.cuda.is_available():
        torch.cuda.empty_cache()


def _to_rgb_array(image: Image.Image) -> np.ndarray:
    return np.asarray(normalize_for_pdf(image).convert("RGB"))


class OCRProcessor:
    """Runs OCR over images, folders and PDFs. One instance = one loaded model."""

    def __init__(self, detection_model: str = DEFAULT_DETECTION_MODEL,
                 recognition_model: str = DEFAULT_RECOGNITION_MODEL, device: str = "auto", *,
                 pretrained: bool = True, cpu_threads: int = 0, engine: OCREngine | None = None):
        if engine is None:
            from .ocr_engine import OCREngine  # pylint: disable=import-outside-toplevel
            engine = OCREngine(detection_model, recognition_model, device, pretrained=pretrained,
                               cpu_threads=cpu_threads)
        self.engine = engine
        # settings (see configure())
        self.output_formats: list[str] = ["pdf"]
        self.dpi: int | None = None
        self.compress_enabled = False
        self.compression_type = "jpeg"
        self.compression_quality = 85
        self.skip_text_pages = True
        self.progress_callback: ProgressCallback | None = None
        self._cancel = threading.Event()
        self._claimed: set[Path] = set()

    # ------------------------------------------------------------------ settings / control
    def configure(self, *, output_formats: Sequence[str] | None = None, dpi: int | None | str = "keep",
                  compress_enabled: bool | None = None, compression_type: str | None = None,
                  compression_quality: int | None = None, skip_text_pages: bool | None = None) -> None:
        if output_formats is not None:
            formats = [f.lower() for f in output_formats]
            if not formats or any(f not in ("pdf", "hocr") for f in formats):
                raise ValueError("output_formats must contain 'pdf' and/or 'hocr'")
            self.output_formats = formats
        if dpi != "keep":
            self.dpi = int(dpi) if dpi else None
        if compress_enabled is not None:
            self.compress_enabled = bool(compress_enabled)
        if compression_type is not None:
            self.compression_type = compression_type
        if compression_quality is not None:
            self.compression_quality = max(1, min(100, int(compression_quality)))
        if skip_text_pages is not None:
            self.skip_text_pages = bool(skip_text_pages)

    @property
    def device(self) -> str:
        return self.engine.device

    def set_models(self, detection_model: str, recognition_model: str) -> bool:
        return self.engine.set_models(detection_model, recognition_model)

    def set_device(self, preference: str) -> bool:
        return self.engine.set_device_preference(preference)

    def cancel(self) -> None:
        """Request cancellation. Takes effect between pages (a page already in inference finishes first)."""
        self._cancel.set()

    @property
    def is_cancelled(self) -> bool:
        return self._cancel.is_set()

    # ------------------------------------------------------------------ public processing API
    def discover_inputs(self, folder: Path) -> list[Path]:
        """All supported images and PDFs below ``folder``: folder by folder, natural name order."""
        exts = set(image_extensions()) | {".pdf"}
        found = [p for p in Path(folder).rglob("*") if p.is_file() and p.suffix.lower() in exts]
        return sorted(found, key=lambda p: ([natural_key(part) for part in p.parent.relative_to(folder).parts],
                                            natural_key(p)))

    def process_folder(self, folder: Path | str, output_dir: Path | str) -> BatchResult:
        folder = Path(folder).resolve()
        if not folder.is_dir():
            raise ValueError(f"Not a folder: {folder}")
        return self._run(self.discover_inputs(folder), Path(output_dir), folder)

    def process_file(self, path: Path | str, output_dir: Path | str) -> BatchResult:
        """Process one image or one PDF."""
        path = Path(path).resolve()
        if not path.is_file():
            raise ValueError(f"Not a file: {path}")
        if not (is_image(path) or is_pdf(path)):
            raise ValueError(f"Unsupported file type: {path.suffix}")
        return self._run([path], Path(output_dir), None)

    # ------------------------------------------------------------------ batch driver
    def _emit(self, done: int, total: int, current: Path | None, stage: str, page: int = 0, count: int = 0) -> None:
        if self.progress_callback:
            try:
                self.progress_callback(ProgressEvent(done, total, current, stage, page, count))
            except Exception:  # pylint: disable=broad-except
                logger.exception("Progress callback raised")

    def _check_cancel(self) -> None:
        if self._cancel.is_set():
            raise Cancelled()

    def _run(self, inputs: list[Path], output_dir: Path, root: Path | None) -> BatchResult:
        self._cancel.clear()
        self._claimed = set()
        result = BatchResult(input_root=root)
        total = len(inputs)
        if not total:
            logger.warning("No supported files found")
            return result
        session = self._create_session_dir(output_dir)
        result.session_dir = session
        logger.info("Processing %d file(s); output: %s", total, session)

        # group images by folder (a folder becomes one PDF); PDFs are processed on their own
        groups: dict[Path, list[Path]] = defaultdict(list)
        pdfs: list[Path] = []
        for path in inputs:
            if is_pdf(path):
                pdfs.append(path)
            else:
                groups[path.parent].append(path)
        work: list[tuple[str, Path, list[Path]]] = [("images", folder, files) for folder, files in groups.items()]
        work += [("pdf", p.parent, [p]) for p in pdfs]

        done = 0
        with tempfile.TemporaryDirectory(prefix="VisionLaneOCR_") as tmp:
            tmp_dir = Path(tmp)
            try:
                for kind, folder, files in work:
                    self._check_cancel()
                    if kind == "images":
                        file_results = self._process_image_group(files, folder, root, session, tmp_dir, done, total)
                    else:
                        file_results = [self._process_pdf(files[0], root, session, tmp_dir, done, total)]
                    result.files.extend(file_results)
                    done += len(file_results)
                    self._emit(done, total, None, "done")
                    gc.collect()
                    _empty_cuda_cache()
            except Cancelled:
                result.cancelled = True
                logger.info("Processing cancelled")
        # anything not processed (because of cancellation) is reported as cancelled
        seen = {f.source for f in result.files}
        for path in inputs:
            if path not in seen:
                result.files.append(FileResult(path, Status.CANCELLED))
        logger.info("Finished: %s", result.summary())
        return result

    @staticmethod
    def _create_session_dir(output_dir: Path) -> Path:
        output_dir.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        session = output_dir / f"OCR_Session_{stamp}"
        counter = 1
        while session.exists():
            counter += 1
            session = output_dir / f"OCR_Session_{stamp}_{counter}"
        session.mkdir()
        return session

    # ------------------------------------------------------------------ output paths
    def _unique(self, path: Path) -> Path:
        """Reserve ``path``; adds `` (2)``, `` (3)`` ... if it is already taken in this session."""
        candidate, n = path, 1
        while candidate in self._claimed or candidate.exists():
            n += 1
            candidate = path.with_name(f"{path.stem} ({n}){path.suffix}")
        self._claimed.add(candidate)
        return candidate

    @staticmethod
    def _relative_dir(folder: Path, root: Path | None) -> Path:
        if root is None:
            return Path()
        try:
            return folder.relative_to(root)
        except ValueError:
            return Path()

    def _group_pdf_path(self, folder: Path, files: list[Path], root: Path | None, session: Path) -> Path:
        rel = self._relative_dir(folder, root)
        if root is None:  # single image: named after the file
            return self._unique(session / "pdf" / f"{files[0].stem}.pdf")
        if rel == Path():
            return self._unique(session / "pdf" / f"{root.name}.pdf")
        return self._unique(session / "pdf" / rel.parent / f"{rel.name}.pdf")

    # ------------------------------------------------------------------ images
    def _ocr_page(self, image: Image.Image) -> tuple[OcrPage, bytes | None]:
        array = _to_rgb_array(image)
        document = self.engine.predict([array])
        page = document.pages[0]
        height, width = array.shape[:2]
        hocr = None
        if "hocr" in self.output_formats:
            hocr = document.export_as_xml()[0][0]
        return ocr_page_from_doctr(page, width, height), hocr

    def _process_image_group(self, files: list[Path], folder: Path, root: Path | None, session: Path,
                             tmp_dir: Path, done: int, total: int) -> list[FileResult]:
        want_pdf = "pdf" in self.output_formats
        want_hocr = "hocr" in self.output_formats
        rel = self._relative_dir(folder, root)
        pdf_path = self._group_pdf_path(folder, files, root, session) if want_pdf else None
        page_pdfs: list[Path] = []
        results: list[FileResult] = []
        compress = self.compression_type if self.compress_enabled else None

        for offset, path in enumerate(files):
            self._check_cancel()
            res = FileResult(path)
            results.append(res)
            self._emit(done + offset, total, path, "loading")
            try:
                for frame in _open_frames(path, self.dpi):
                    self._check_cancel()
                    res.pages_total += 1
                    self._emit(done + offset, total, path, "ocr", frame.index - 1, frame.count)
                    try:
                        ocr, hocr = self._ocr_page(frame.image)
                        if want_hocr and hocr is not None:
                            res.outputs.append(self._write_hocr(hocr, session, rel, path, frame, files))
                        if want_pdf:
                            self._emit(done + offset, total, path, "writing", frame.index, frame.count)
                            page_pdf = tmp_dir / f"{len(page_pdfs):06d}.pdf"
                            write_image_page_pdf(page_pdf, frame.image, ocr, frame.dpi, compress_type=compress,
                                                 quality=self.compression_quality, passthrough=frame.passthrough)
                            page_pdfs.append(page_pdf)
                        res.pages_ok += 1
                    except Cancelled:
                        raise
                    except Exception as exc:  # pylint: disable=broad-except
                        logger.error("Page %d of %s failed: %s", frame.index, path, exc, exc_info=True)
                        res.failed_pages.append(frame.index)
                        res.error = f"page {frame.index}: {exc}"
                    finally:
                        frame.image.close()
            except Cancelled:
                raise
            except Exception as exc:  # pylint: disable=broad-except
                # (unreadable / corrupt file, decompression bomb, ...)
                logger.error("Cannot process %s: %s", path, exc, exc_info=True)
                res.error = str(exc)
            if res.pages_ok == 0:
                res.status = Status.FAILED
                res.error = res.error or "no pages could be processed"
            elif res.failed_pages:
                res.status = Status.PARTIAL
            else:
                res.status = Status.SUCCESS
            self._emit(done + offset + 1, total, path, "failed" if res.status == Status.FAILED else "done")

        if want_pdf:  # merge once the whole folder is finished (no polling, no trigger-by-last-file)
            self._emit(done + len(files), total, folder, "merging")
            if page_pdfs:
                try:
                    merge_pdfs(page_pdfs, pdf_path)
                    for res in results:
                        if res.pages_ok:
                            res.outputs.append(pdf_path)
                    logger.info("Created %s (%d pages)", pdf_path, len(page_pdfs))
                except Exception as exc:  # pylint: disable=broad-except
                    logger.error("Merging %s failed: %s", pdf_path, exc, exc_info=True)
                    for res in results:
                        res.status = Status.FAILED
                        res.error = f"could not write {pdf_path.name}: {exc}"
            else:
                for res in results:
                    res.status = Status.FAILED
        return results

    def _write_hocr(self, data: bytes, session: Path, rel: Path, path: Path, frame: _Frame,
                    siblings: list[Path]) -> Path:
        name = path.stem if frame.count == 1 else f"{path.stem}_p{frame.index:04d}"
        # x.tif and x.png in one folder would both want x.hocr: keep the extension for the later one
        clash = [s for s in siblings if s.stem == path.stem and s != path]
        if clash and sorted(siblings, key=natural_key).index(path) > min(
                sorted(siblings, key=natural_key).index(c) for c in clash):
            name = f"{name}_{path.suffix.lstrip('.').lower()}"
        target = self._unique(session / "hocr" / rel / f"{name}.hocr")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(data)
        return target

    # ------------------------------------------------------------------ PDFs
    def _process_pdf(self, path: Path, root: Path | None, session: Path, tmp_dir: Path, done: int,
                     total: int) -> FileResult:
        res = FileResult(path)
        want_pdf = "pdf" in self.output_formats
        want_hocr = "hocr" in self.output_formats
        rel = self._relative_dir(path.parent, root)
        layers: dict[int, OcrPage] = {}
        hocr_files: list[Path] = []
        self._emit(done, total, path, "loading")
        try:
            with pdf_input.PdfSource(path) as source:
                res.pages_total = source.page_count
                if not res.pages_total:
                    raise ValueError("The PDF has no pages")
                for index in range(source.page_count):
                    self._check_cancel()
                    number = index + 1
                    self._emit(done, total, path, "ocr", index, res.pages_total)
                    try:
                        if self.skip_text_pages and not want_hocr and source.has_text(index):
                            res.skipped_pages.append(number)
                            res.pages_ok += 1
                            continue
                        image = source.render(index, self.dpi)
                        ocr, hocr = self._ocr_page(image)
                        layers[index] = ocr
                        if want_hocr and hocr is not None:
                            target = self._unique(session / "hocr" / rel / path.stem / f"{path.stem}_page_{number:04d}.hocr")
                            target.parent.mkdir(parents=True, exist_ok=True)
                            target.write_bytes(hocr)
                            hocr_files.append(target)
                        res.pages_ok += 1
                    except Cancelled:
                        raise
                    except Exception as exc:  # pylint: disable=broad-except
                        logger.error("Page %d of %s failed: %s", number, path, exc, exc_info=True)
                        res.failed_pages.append(number)
                        res.error = f"page {number}: {exc}"
            res.outputs.extend(hocr_files)
            if want_pdf and res.pages_ok:
                self._emit(done, total, path, "writing")
                target = self._unique(session / "pdf" / rel / f"{path.stem}.pdf")
                pdf_input.apply_text_layers(path, target, layers)
                res.outputs.append(target)
        except Cancelled:
            raise
        except Exception as exc:  # pylint: disable=broad-except
            logger.error("Cannot process PDF %s: %s", path, exc, exc_info=True)
            res.error = str(exc)
            res.pages_ok = 0
        if res.pages_ok == 0:
            res.status = Status.FAILED
            res.error = res.error or "no pages could be processed"
        elif res.failed_pages:
            res.status = Status.PARTIAL
        else:
            res.status = Status.SUCCESS
        self._emit(done + 1, total, path, "failed" if res.status == Status.FAILED else "done")
        return res

    def release(self) -> None:
        self.engine.release()
