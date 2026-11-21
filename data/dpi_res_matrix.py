"""Paper size -> pixel dimension table (groundwork for the "uniform paper size" roadmap item).

Nothing in the application imports this module. Run it to (re)generate the table on demand:

    python data/dpi_res_matrix.py --output paper_pixels.csv [--max-dpi 1200]
"""
from __future__ import annotations

import argparse
import csv

# name: (width_in, height_in) - portrait
PAPER_SIZES_IN: dict[str, tuple[float, float]] = {
    "A0": (33.11, 46.81), "A1": (23.39, 33.11), "A2": (16.54, 23.39), "A3": (11.69, 16.54),
    "A4": (8.27, 11.69), "A5": (5.83, 8.27), "A6": (4.13, 5.83),
    "B4": (9.84, 13.9), "B5": (6.93, 9.84), "B6": (4.92, 6.93),
    "C4": (9.02, 12.76), "C5": (6.38, 9.02), "DL": (4.33, 8.66),
    "Letter": (8.5, 11.0), "Legal": (8.5, 14.0), "Tabloid": (11.0, 17.0), "Executive": (7.25, 10.5),
}


def pixels(size: str, dpi: int) -> tuple[int, int]:
    width_in, height_in = PAPER_SIZES_IN[size]
    return round(width_in * dpi), round(height_in * dpi)


def nearest_paper_size(width_px: int, height_px: int, dpi: int) -> tuple[str, float]:
    """Closest named paper size (orientation-agnostic) and the relative size error."""
    w, h = sorted((width_px, height_px))
    best = ("", float("inf"))
    for name in PAPER_SIZES_IN:
        pw, ph = sorted(pixels(name, dpi))
        error = abs(pw - w) / pw + abs(ph - h) / ph
        if error < best[1]:
            best = (name, error)
    return best


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
