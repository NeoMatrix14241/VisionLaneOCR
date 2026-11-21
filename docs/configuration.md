# Configuration (`config.ini`)

**Location** (first match wins):

1. `%VISIONLANE_HOME%` environment variable, if set;
2. the program folder, if an (empty) file named `portable` exists next to `main.py` / the executable;
3. `%LOCALAPPDATA%\VisionLaneOCR` (Windows), `~/.local/share/VisionLaneOCR` (Linux), `~/Library/Application Support/VisionLaneOCR` (macOS).

The file is created on first run and is git-ignored. The single reader/writer is `utils/app_config.AppConfig`
(atomic write; unknown keys and the old `[Startup]` section are ignored on load and dropped on save; invalid values
fall back to defaults). Settings are saved when you click *Start Processing*, *File -> Save Settings* and on exit.

## `[General]`

| Key | Values | Default | Effect |
|-----|--------|---------|--------|
| `dpi` | `Auto`, 72, 96, 150, 200, 240, 250, 300, 350, 400, 450, 500, 600, 800, 900, 1200 | `Auto` | PDF page size = pixels x 72 / dpi. `Auto` = image metadata (30-4800), else 300. For PDF input it is the render resolution (default 300). |
| `output_format` | `PDF`, `HOCR`, `PDF+HOCR` | `PDF` | |
| `theme_mode` | `system`, `light`, `dark`, `night` | `system` | |
| `detection_model` | any DocTR detection architecture | `db_resnet50` | |
| `recognition_model` | any DocTR recognition architecture | `parseq` | |
| `device` | `auto`, `gpu`, `cpu` | `auto` | `auto`/`gpu` use CUDA when a working device exists; otherwise the CPU is used and the reason is logged |
| `compress_enabled` | bool | `False` | re-encode page images while writing the PDF |
| `compression_type` | `jpeg`, `lossless` | `jpeg` | legacy `jpeg2000` -> `jpeg`; `lzw`, `png` -> `lossless` |
| `compression_quality` | 1-100 | `85` | JPEG quality (ignored for `lossless`) |
| `skip_text_pages` | bool | `True` | PDF input: do not re-OCR pages that already contain text (ignored when HOCR output is requested) |
| `archive_enabled_single` / `_folder` / `_pdf` | bool | `False` | per tab. A legacy single `archive_enabled` applies to all three tabs |

## `[Paths]`
`single`, `folder`, `pdf` (last inputs), `output_single`, `output_folder`, `output_pdf`, `archive_single`, `archive_folder`,
`archive_pdf`. Empty by default; a saved input path is restored only if it still exists.

## `[Performance]`
| Key | Default | Effect |
|-----|---------|--------|
| `cpu_threads` | `0` (automatic) | `torch.set_num_threads(n)` - affects CPU inference only. (Legacy `thread_count` is read as a fallback; `operation_timeout` / `chunk_timeout` were removed because they were never enforced.) |

## `[Logging]`
| Key | Default | Effect |
|-----|---------|--------|
| `level` | `INFO` | `DEBUG`, `INFO`, `WARNING`, `ERROR`. `python main.py --debug` forces `DEBUG` |

## Files written

| Path | Content |
|------|---------|
| `<data dir>/config.ini` | settings (see above) |
| `<data dir>/logs/visionlane_<timestamp>.log` | one file per session; the newest 20 are kept |
| `<data dir>/logs/crash.log` | native crash tracebacks (`faulthandler`) |
| `%TEMP%\VisionLaneOCR_*` | private scratch folder of a running job; removed when the job ends |
| `<app folder>/doctr_cache/models/*.pt` | DocTR weights (`<name>-<hash>.pt`); git-ignored. Falls back to `<data dir>/doctr_cache` if the app folder is not writable, or to `$DOCTR_CACHE_DIR/models` if that is set |

## Environment variables
| Variable | Used for |
|----------|----------|
| `VISIONLANE_HOME` | overrides the config/log folder |
| `DOCTR_CACHE_DIR` | where DocTR stores model weights. If you do not set it, `utils.models.configure_cache()` sets it to `<app folder>/doctr_cache` at start-up (and copies models found in the old `~/.cache/doctr/models` once). If you set it, your value wins and nothing is copied |
| `QT_QPA_PLATFORM=offscreen` | headless GUI tests |

The application sets **no** environment variables itself.
