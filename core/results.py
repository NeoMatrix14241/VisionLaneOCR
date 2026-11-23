"""Result and progress types returned by :class:`core.ocr_processor.OCRProcessor`.

Failures are *values*, not log lines: every input file ends up in a :class:`FileResult`, and the GUI
(and the archiver) decide what to do from those.
"""
from __future__ import annotations

import enum
from dataclasses import dataclass, field
from pathlib import Path


class Status(str, enum.Enum):
    SUCCESS = "success"
    PARTIAL = "partial"      # some pages failed; the output contains the pages that worked
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class FileResult:
    source: Path
    status: Status = Status.FAILED
    outputs: list[Path] = field(default_factory=list)
    pages_total: int = 0
    pages_ok: int = 0
    failed_pages: list[int] = field(default_factory=list)   # 1-based
    skipped_pages: list[int] = field(default_factory=list)  # PDF pages that already had text
    error: str | None = None


@dataclass
class BatchResult:
    session_dir: Path | None = None
    files: list[FileResult] = field(default_factory=list)
    cancelled: bool = False
    input_root: Path | None = None
    warnings: list[str] = field(default_factory=list)   # non-fatal problems (e.g. an optional feature was unavailable)

    def count(self, status: Status) -> int:
        return sum(1 for f in self.files if f.status == status)

    @property
    def total(self) -> int:
        return len(self.files)

    @property
    def succeeded(self) -> int:
        return self.count(Status.SUCCESS)

    @property
    def failed(self) -> int:
        return self.count(Status.FAILED) + self.count(Status.PARTIAL)

    @property
    def ok(self) -> bool:
        """True only if nothing failed and the run was not cancelled."""
        return not self.cancelled and self.total > 0 and self.failed == 0 and self.count(Status.CANCELLED) == 0

    def confirmed_sources(self) -> list[Path]:
        """Inputs whose complete output was written - the only files that may be archived."""
        return [f.source for f in self.files if f.status == Status.SUCCESS]

    def failure_lines(self) -> list[str]:
        lines = []
        for f in self.files:
            if f.status in (Status.FAILED, Status.PARTIAL):
                detail = f.error or (f"pages failed: {f.failed_pages}" if f.failed_pages else "unknown error")
                lines.append(f"{f.source}: {detail}")
        return lines

    def summary(self) -> str:
        parts = [f"{self.succeeded}/{self.total} files OK"]
        if self.count(Status.PARTIAL):
            parts.append(f"{self.count(Status.PARTIAL)} partial")
        if self.count(Status.FAILED):
            parts.append(f"{self.count(Status.FAILED)} failed")
        if self.count(Status.CANCELLED):
            parts.append(f"{self.count(Status.CANCELLED)} cancelled")
        return ", ".join(parts)


@dataclass(frozen=True)
class ProgressEvent:
    """The single progress message. ``files_done`` counts finished files (never "started" ones)."""
    files_done: int
    total_files: int
    current: Path | None = None
    stage: str = ""            # "loading" | "ocr" | "writing" | "merging" | "done" | "failed"
    page: int = 0
    page_count: int = 0

    @property
    def fraction(self) -> float:
        """Overall progress in [0, 1], including the fraction of the current file."""
        if self.total_files <= 0:
            return 0.0
        within = (self.page / self.page_count) if self.page_count else 0.0
        return min(1.0, (self.files_done + (within if self.stage not in ("done", "failed") else 0.0)) / self.total_files)
