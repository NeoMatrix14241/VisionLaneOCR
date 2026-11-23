"""Paper sizes: used to give all pages of a PDF the same size ("uniform page size")."""
from __future__ import annotations

PAPER_SIZES_IN: dict[str, tuple[float, float]] = {  # name: (width, height) in inches, portrait
    "A0": (33.11, 46.81), "A1": (23.39, 33.11), "A2": (16.54, 23.39), "A3": (11.69, 16.54),
    "A4": (8.27, 11.69), "A5": (5.83, 8.27), "A6": (4.13, 5.83),
    "B4": (9.84, 13.9), "B5": (6.93, 9.84), "B6": (4.92, 6.93),
    "C4": (9.02, 12.76), "C5": (6.38, 9.02), "DL": (4.33, 8.66),
    "Letter": (8.5, 11.0), "Legal": (8.5, 14.0), "Tabloid": (11.0, 17.0), "Executive": (7.25, 10.5),
}

ORIGINAL = "original"   # keep every page at its own size
AUTO = "auto"           # snap to the paper size most pages of the folder are closest to
PAGE_SIZE_CHOICES: tuple[str, ...] = (ORIGINAL, AUTO, *PAPER_SIZES_IN)
AUTO_MAX_ERROR = 0.25   # "auto" ignores pages that are further than this (relative error) from any paper size


def pixels(size: str, dpi: int) -> tuple[int, int]:
    width_in, height_in = PAPER_SIZES_IN[size]
    return round(width_in * dpi), round(height_in * dpi)


def paper_size_pt(size: str, landscape: bool = False) -> tuple[float, float]:
    """(width, height) in PDF points; ``landscape`` swaps them."""
    width_in, height_in = PAPER_SIZES_IN[size]
    width, height = width_in * 72.0, height_in * 72.0
    return (height, width) if landscape else (width, height)


def nearest_paper_size(width_px: int, height_px: int, dpi: int) -> tuple[str, float]:
    """Closest named paper size (orientation-agnostic) and the summed relative size error."""
    short, long_ = sorted((width_px, height_px))
    best_name, best_error = "", float("inf")
    for name in PAPER_SIZES_IN:
        paper_short, paper_long = sorted(pixels(name, dpi))
        error = abs(paper_short - short) / paper_short + abs(paper_long - long_) / paper_long
        if error < best_error:
            best_name, best_error = name, error
    return best_name, best_error


def normalize_page_size(value: str) -> str:
    """Case-insensitive lookup of a page-size choice; unknown values mean ``original``."""
    lookup = {choice.lower(): choice for choice in PAGE_SIZE_CHOICES}
    return lookup.get((value or "").strip().lower(), ORIGINAL)
