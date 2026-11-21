"""Command line interface.

    python run_ocr.py INPUT OUTPUT_DIR [--formats pdf hocr] [--dpi 300] [--compress jpeg --quality 80]
                      [--device auto|gpu|cpu] [--det db_resnet50] [--reco parseq] [--archive DIR]

INPUT may be a folder (searched recursively), an image or a PDF.
Exit code: 0 = everything OK, 1 = some files failed, 2 = cancelled / bad arguments.
"""
from __future__ import annotations

import argparse
import logging
import sys
from pathlib import Path

from core.constants import DEFAULT_DETECTION_MODEL, DEFAULT_RECOGNITION_MODEL


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="VisionLane OCR (command line)")
    p.add_argument("input", type=Path, help="folder, image or PDF")
    p.add_argument("output", type=Path, help="output directory (an OCR_Session_<time> folder is created inside)")
    p.add_argument("--formats", nargs="+", choices=["pdf", "hocr"], default=["pdf"])
    p.add_argument("--dpi", type=int, default=None, help="default: from the image metadata, else 300")
    p.add_argument("--compress", choices=["jpeg", "lossless"], default=None, help="recompress page images in the PDF")
    p.add_argument("--quality", type=int, default=85, help="JPEG quality 1-100 (with --compress jpeg)")
    p.add_argument("--device", choices=["auto", "gpu", "cpu"], default="auto")
    p.add_argument("--det", default=DEFAULT_DETECTION_MODEL, help="detection model")
    p.add_argument("--reco", default=DEFAULT_RECOGNITION_MODEL, help="recognition model")
    p.add_argument("--no-skip-text", action="store_true", help="OCR PDF pages even if they already contain text")
    p.add_argument("--archive", type=Path, default=None, help="move successfully processed inputs here")
    p.add_argument("--verbose", action="store_true")
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s - %(levelname)s - %(message)s")
    if not args.input.exists():
        print(f"Input not found: {args.input}", file=sys.stderr)
        return 2

    from core.archiver import archive_inputs
    from core.ocr_processor import OCRProcessor
    from utils import models

    models.configure_cache()
    for kind, name in ((models.DETECTION, args.det), (models.RECOGNITION, args.reco)):
        if not models.is_cached(name):
            models.download(kind, name)

    processor = OCRProcessor(args.det, args.reco, args.device)
    processor.configure(output_formats=args.formats, dpi=args.dpi, compress_enabled=args.compress is not None,
                        compression_type=args.compress or "jpeg", compression_quality=args.quality,
                        skip_text_pages=not args.no_skip_text)
    processor.progress_callback = lambda e: None
    try:
        result = (processor.process_folder(args.input, args.output) if args.input.is_dir()
                  else processor.process_file(args.input, args.output))
    except KeyboardInterrupt:
        processor.cancel()
        print("Cancelled", file=sys.stderr)
        return 2

    print(result.summary())
    for line in result.failure_lines():
        print("FAILED:", line, file=sys.stderr)
    if result.session_dir:
        print("Output:", result.session_dir)
    if args.archive and result.total:
        report = archive_inputs(result, args.archive)
        print(f"Archived {len(report.moved)} file(s) to {args.archive}")
        for err in report.errors:
            print("ARCHIVE:", err, file=sys.stderr)
    return 0 if result.ok else (2 if result.cancelled else 1)


if __name__ == "__main__":
    sys.exit(main())
