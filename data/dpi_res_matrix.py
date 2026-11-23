"""Paper size -> pixel dimension table generator.

The paper-size data itself lives in ``core/paper.py`` (it is used by the "uniform page size" feature).
Run this script to (re)generate a lookup table on demand:

    python data/dpi_res_matrix.py --output paper_pixels.csv [--max-dpi 1200]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from core.paper import PAPER_SIZES_IN, nearest_paper_size, pixels  # noqa: E402  pylint: disable=wrong-import-position

__all__ = ["PAPER_SIZES_IN", "nearest_paper_size", "pixels", "write_csv"]


def write_csv(path: str, max_dpi: int = 1200) -> int:
    rows = 0
    with open(path, "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["Paper Size", "Width (in)", "Height (in)", "DPI", "Width (px)", "Height (px)"])
        for name, (width_in, height_in) in PAPER_SIZES_IN.items():
            for dpi in range(1, max_dpi + 1):
                writer.writerow([name, width_in, height_in, dpi, *pixels(name, dpi)])
                rows += 1
    return rows


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", default="paper_pixels.csv")
    parser.add_argument("--max-dpi", type=int, default=1200)
    args = parser.parse_args()
    print(f"Wrote {write_csv(args.output, args.max_dpi)} rows to {args.output}")
