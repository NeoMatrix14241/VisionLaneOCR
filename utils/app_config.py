"""Single reader/writer for ``config.ini``.

Previously three components wrote this file with different defaults. Everything now goes through
:class:`AppConfig`: ``AppConfig.load(path)`` / ``config.save(path)``. Unknown keys and the legacy
``[Startup]`` section are ignored on load and dropped on save.
"""
from __future__ import annotations

import configparser
import logging
import os
import tempfile
from dataclasses import dataclass, field, fields
from pathlib import Path

from core.paper import normalize_page_size

logger = logging.getLogger(__name__)

DPI_CHOICES = ("Auto", "72", "96", "150", "200", "240", "250", "300", "350", "400", "450", "500",
               "600", "800", "900", "1200")
OUTPUT_FORMAT_CHOICES = ("PDF", "HOCR", "PDF+HOCR")
THEME_CHOICES = ("system", "light", "dark", "night")
DEVICE_CHOICES = ("auto", "gpu", "cpu")
COMPRESSION_CHOICES = ("jpeg", "lossless")
TABS = ("single", "folder", "pdf")


def normalize_compression_type(value: str) -> str:
    """Map legacy values (``jpeg2000``, ``lzw``, ``png``) onto what the PDF writer supports."""
    value = (value or "").strip().lower()
    if value in ("jpeg", "jpg", "jpeg2000", "jp2", "jpx"):
        return "jpeg"
    if value in ("lossless", "lzw", "png", "flate", "zip"):
        return "lossless"
    return "jpeg"


@dataclass
class AppConfig:
    # [General]
    dpi: str = "Auto"
    output_format: str = "PDF"
    theme_mode: str = "system"
    detection_model: str = "db_resnet50"
    recognition_model: str = "parseq"
    device: str = "auto"
    compress_enabled: bool = False
    compression_type: str = "jpeg"
    compression_quality: int = 85
    skip_text_pages: bool = True
    straighten_pages: bool = False      # handle rotated / skewed pages (slower, 2 extra models)
    page_size: str = "original"         # "original", "auto" or a paper name (A4, Letter, ...)
    enhance_enabled: bool = False       # super-resolution for low-resolution pages (needs `super-image`)
    enhance_scale: int = 2
    enhance_min_dpi: int = 150
    # [Paths] - one entry per tab
    input_paths: dict[str, str] = field(default_factory=lambda: {t: "" for t in TABS})
    output_paths: dict[str, str] = field(default_factory=lambda: {t: "" for t in TABS})
    archive_paths: dict[str, str] = field(default_factory=lambda: {t: "" for t in TABS})
    archive_enabled: dict[str, bool] = field(default_factory=lambda: {t: False for t in TABS})
    # [Performance]
    cpu_threads: int = 0  # 0 = let PyTorch decide
    # [Logging]
    log_level: str = "INFO"

    # ------------------------------------------------------------------ load
    @classmethod
    def load(cls, path: Path | str | None) -> "AppConfig":
        cfg = cls()
        if not path or not Path(path).exists():
            return cfg
        parser = configparser.ConfigParser(interpolation=None)
        try:
            parser.read(path, encoding="utf-8")
        except (configparser.Error, OSError, UnicodeDecodeError) as exc:
            logger.warning("Could not read %s (%s); using defaults", path, exc)
            return cfg

        def get(section: str, key: str, default: str) -> str:
            return parser.get(section, key, fallback=default).strip()

        def get_bool(section: str, key: str, default: bool) -> bool:
            try:
                return parser.getboolean(section, key, fallback=default)
            except ValueError:
                return default

        def get_int(section: str, key: str, default: int) -> int:
            try:
                return parser.getint(section, key, fallback=default)
            except ValueError:
                return default

        dpi = get("General", "dpi", cfg.dpi)
        cfg.dpi = dpi if dpi in DPI_CHOICES else "Auto"
        fmt = get("General", "output_format", cfg.output_format).upper()
        cfg.output_format = fmt if fmt in OUTPUT_FORMAT_CHOICES else "PDF"
        theme = get("General", "theme_mode", cfg.theme_mode).lower()
        cfg.theme_mode = theme if theme in THEME_CHOICES else "system"
        cfg.detection_model = get("General", "detection_model", cfg.detection_model) or cfg.detection_model
        cfg.recognition_model = get("General", "recognition_model", cfg.recognition_model) or cfg.recognition_model
        device = get("General", "device", cfg.device).lower()
        cfg.device = device if device in DEVICE_CHOICES else "auto"
        cfg.compress_enabled = get_bool("General", "compress_enabled", cfg.compress_enabled)
        cfg.compression_type = normalize_compression_type(get("General", "compression_type", cfg.compression_type))
        cfg.compression_quality = min(100, max(1, get_int("General", "compression_quality", cfg.compression_quality)))
        cfg.skip_text_pages = get_bool("General", "skip_text_pages", cfg.skip_text_pages)
        cfg.straighten_pages = get_bool("General", "straighten_pages", cfg.straighten_pages)
        cfg.page_size = normalize_page_size(get("General", "page_size", cfg.page_size))
        cfg.enhance_enabled = get_bool("General", "enhance_enabled", cfg.enhance_enabled)
        scale = get_int("General", "enhance_scale", cfg.enhance_scale)
        cfg.enhance_scale = scale if scale in (2, 3, 4) else 2
        cfg.enhance_min_dpi = min(1200, max(30, get_int("General", "enhance_min_dpi", cfg.enhance_min_dpi)))

        legacy_archive = get_bool("General", "archive_enabled", False)  # one flag for all tabs (old format)
        for tab in TABS:
            cfg.input_paths[tab] = get("Paths", tab, "")
            cfg.output_paths[tab] = get("Paths", f"output_{tab}", "")
            cfg.archive_paths[tab] = get("Paths", f"archive_{tab}", "")
            cfg.archive_enabled[tab] = get_bool("General", f"archive_enabled_{tab}", legacy_archive)

        threads = get_int("Performance", "cpu_threads", get_int("Performance", "thread_count", 0))
        cfg.cpu_threads = max(0, min(threads, os.cpu_count() or 1))
        level = get("Logging", "level", cfg.log_level).upper()
        cfg.log_level = level if level in ("DEBUG", "INFO", "WARNING", "ERROR") else "INFO"
        return cfg

    # ------------------------------------------------------------------ save
    def save(self, path: Path | str) -> None:
        parser = configparser.ConfigParser(interpolation=None)
        parser["General"] = {
            "dpi": self.dpi,
            "output_format": self.output_format,
            "theme_mode": self.theme_mode,
            "detection_model": self.detection_model,
            "recognition_model": self.recognition_model,
            "device": self.device,
            "compress_enabled": str(self.compress_enabled),
            "compression_type": self.compression_type,
            "compression_quality": str(self.compression_quality),
            "skip_text_pages": str(self.skip_text_pages),
            "straighten_pages": str(self.straighten_pages),
            "page_size": self.page_size,
            "enhance_enabled": str(self.enhance_enabled),
            "enhance_scale": str(self.enhance_scale),
            "enhance_min_dpi": str(self.enhance_min_dpi),
        }
        for tab in TABS:
            parser["General"][f"archive_enabled_{tab}"] = str(self.archive_enabled.get(tab, False))
        parser["Paths"] = {}
        for tab in TABS:
            parser["Paths"][tab] = self.input_paths.get(tab, "")
            parser["Paths"][f"output_{tab}"] = self.output_paths.get(tab, "")
            parser["Paths"][f"archive_{tab}"] = self.archive_paths.get(tab, "")
        parser["Performance"] = {"cpu_threads": str(self.cpu_threads)}
        parser["Logging"] = {"level": self.log_level}

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        fd, tmp_name = tempfile.mkstemp(prefix=".config-", suffix=".tmp", dir=str(path.parent))
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                handle.write("# VisionLane OCR settings - written by the application.\n")
                parser.write(handle)
            os.replace(tmp_name, path)
        except OSError:
            try:
                os.unlink(tmp_name)
            except OSError:
                pass
            raise


# Keep ``fields`` importable for tests that want to enumerate the schema.
CONFIG_FIELDS = tuple(f.name for f in fields(AppConfig))
