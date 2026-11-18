# Configuration (`config.ini`)

File location: repository root (next to `main.py`; bundled as a data file in builds).
Writers: `utils/startup_config.StartupConfig` (creates/repairs defaults at startup) and
`gui/main_window.MainWindow._save_config` (on close and *File → Save Settings*). Both rewrite the whole file and
drop comments. Defaults differ between them (VL-043).

## `[General]`

| Key | Values | Default (StartupConfig) | Used by |
|-----|--------|-------------------------|---------|
| `dpi` | `Auto`, 72, 96, 150, 200, 240, 250, 300, 350, 400, 450, 500, 600, 800, 900, 1200 | `Auto` | GUI → `ocr.dpi` (`None` = from image metadata, else 300). Ignored for PDF input (always 300). |
| `output_format` | `PDF`, `HOCR`, `PDF+HOCR` | `PDF` | GUI → `ocr.output_formats` |
| `theme_mode` | `system`, `light`, `dark`, `night` | `system` | `MainWindow._apply_theme` |
| `detection_model` | doctr detection arch (`db_resnet50`, `linknet_*`, `db_mobilenet_v3_large`, `fast_*`) | `db_resnet50` | `OCRProcessor` |
| `recognition_model` | doctr recognition arch (`parseq`, `crnn_*`, `sar_resnet31`, `master`, `vitstr_*`) | `parseq` | `OCRProcessor` |
| `compress_enabled` | bool | `False` (repo file: `True`) | per-page Ghostscript step (VL-014) |
| `compression_type` | `jpeg`, `jpeg2000`, `lzw`, `png` | `jpeg` | `pypdfcompressor` |
| `compression_quality` | 0–100 | `100` | `pypdfcompressor` |
| `archive_enabled` | bool (true if any tab's checkbox is ticked) | `False` | GUI (VL-003) |

## `[Paths]`

`single`, `folder`, `pdf` (last inputs), `output_single`, `output_folder`, `output_pdf` (last outputs),
`archive_single`, `archive_folder`, `archive_pdf`. Empty by default. The committed file contains the developer's
`D:` paths (VL-045).

## `[Performance]`

| Key | Default | Effect |
|-----|---------|--------|
| `thread_count` | logical CPUs | `QThreadPool.setMaxThreadCount` only (no real effect, VL-037) |
| `operation_timeout` | 300 (GUI fallback 600) | stored on `OCRProcessor`, **never enforced** |
| `chunk_timeout` | 60 | stored, **never enforced** |

## `[Startup]` ("experimental – do not edit")

| Key | Default | Actually used? |
|-----|---------|----------------|
| `enable_parallel_loading` | True | only printed in the log header |
| `show_detailed_progress` | True | yes — selects file/console log levels (`logging_config.py`) |
| `cache_validation_results` | True | no |
| `skip_doctr_setup_check` | False | no |
| `skip_model_validation` | False | no |
| `auto_download_models` | True | constructs `EnhancedModelManager` only (downloads nothing) |
| `use_minimal_diagnostics` | False | no |
| `startup_timeout` | 120 | no |
| `max_parallel_workers` | logical CPUs | no |
| `cache_expiry_hours` | 24 | no (cache TTLs are hard-coded in `startup_cache.py`) |
| `skip_system_diagnostics` | False | yes (`main.py:226`) |
| `detailed_logging`, `log_level` | True / INFO | written by `StartupConfig` only; never read |

## Other persistent state

| Path | Content |
|------|---------|
| `~/.cache/visionlane_ocr/*.json` | `StartupCache` (DocTR setup 24 h, models 7 d, system 1 h, config hash) — only written by the unused `setup_doctr_with_progress` |
| `~/.cache/doctr/models/*.pt` (or `$DOCTR_CACHE_DIR/models`) | DocTR weights |
| `<app>/logs/` | `ocr_process_*.log`, `ocr_gui_*.log`, `debug_*.log`, `memory_*.log`, `crash_*.log` |
| `%TEMP%\VisionLaneOCR_temp` | scratch directory (deleted/recreated on start, VL-027) |

## Environment variables touched by the code

`USE_TORCH`, `DOCTR_BACKEND` (obsolete in doctr ≥ 1.0), `CUDA_LAUNCH_BLOCKING`, `PYTORCH_CUDA_ALLOC_CONF`,
`CUDA_VISIBLE_DEVICES` (set to empty!, VL-008), `CUDA_MEMORY_FRACTION`, `CUDA_CACHE_DISABLE`, `CUDA_FORCE_PTX_JIT`,
`CUDA_DISABLE_PTX_JIT`, `CUDA_DEVICE_ORDER`, `PYTHONWARNINGS`, `PYTORCH_ENABLE_MPS_FALLBACK`,
`PYTORCH_MPS_HIGH_WATERMARK_RATIO`, `CUDA_DRIVER_LIBRARY_PATH`, `CUDA_TOOLKIT_ROOT_DIR`, `PATH` (adds NVSMI/System32).
Honoured (read) by DocTR: `DOCTR_CACHE_DIR`.
