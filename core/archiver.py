"""Archiving: move *confirmed* inputs into an archive folder, keeping the folder structure.

Rules (these replace the old behaviour that moved every file of the input tree regardless of outcome):

* only files whose output was completely written (``Status.SUCCESS``) are moved;
* nothing is moved after a cancelled run;
* name collisions get `` (2)``, `` (3)`` ... instead of overwriting / raising;
* directories emptied by the move are removed (the input root itself is kept).
"""
from __future__ import annotations

import logging
import shutil
from dataclasses import dataclass, field
from pathlib import Path

from .results import BatchResult

logger = logging.getLogger(__name__)


@dataclass
class ArchiveReport:
    moved: list[tuple[Path, Path]] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)


def is_inside(child: Path, parent: Path) -> bool:
    try:
        child.resolve().relative_to(parent.resolve())
        return True
    except ValueError:
        return False


def _free_name(path: Path) -> Path:
    candidate, n = path, 1
    while candidate.exists():
        n += 1
        candidate = path.with_name(f"{path.stem} ({n}){path.suffix}")
    return candidate


def archive_inputs(result: BatchResult, archive_dir: Path) -> ArchiveReport:
    report = ArchiveReport()
    if result.cancelled:
        logger.info("Run was cancelled: nothing archived")
        return report
    root = result.input_root
    if root is not None and is_inside(archive_dir, root):
        report.errors.append("The archive folder must not be inside the input folder.")
        return report
    for source in result.confirmed_sources():
        try:
            rel = source.relative_to(root) if root is not None else Path(source.name)
            target = _free_name(archive_dir / rel)
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.move(str(source), str(target))
            report.moved.append((source, target))
        except (OSError, ValueError) as exc:
            logger.error("Archiving %s failed: %s", source, exc)
            report.errors.append(f"{source}: {exc}")
    if root is not None:
        _prune_empty_dirs(root)
    return report


def _prune_empty_dirs(root: Path) -> None:
    for directory in sorted((d for d in root.rglob("*") if d.is_dir()), key=lambda d: len(d.parts), reverse=True):
        try:
            directory.rmdir()  # only succeeds when empty
        except OSError:
            pass
