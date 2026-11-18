# `utils/` reference

| Module | Contents | Used by |
|--------|----------|---------|
| `__init__.py` | exports `ProcessManager`, `ThreadKiller`, `setup_logging`, `SafeLogHandler` | – |
| `hocr_to_pdf.py` | `CustomHOCRTransform(hocr_file, image_file, dpi)`, `convert_hocr_to_pdf()`, `hocr_to_pdf()` | `ocr_processor` |
| `pypdfcompressor.py` | `PDFProcessor.compress_pdf/process_directory`, module-level `compress_pdf()`, CLI `main()` | `ocr_processor` |
| `thread_killer.py` | `ThreadKiller.terminate_thread/_all_threads/_thread_pool/safe_kill_processes/get_current_thread_stack` | `ocr_processor.cancel_processing` |
| `process_manager.py` | `ProcessManager.track_thread/track_process/force_exit/is_running` | `MainWindow`, `OCRWorker` |
| `logging_config.py` | `setup_logging(base_dir, startup_config)` → file+console handlers on root logger | `main.py`, `run_ocr.py` |
| `safe_logger.py` | `SafeLogHandler` (weakref to a widget with `.append`) | base class of `LogHandler` |
| `debug_helper.py` | `CrashHandler` (excepthooks → `logs/crash_*.log`), `DebugLogger` (debug + memory logs, 10 s monitor thread) | `main.py`, `debug_run.py` |
| `system_diagnostics.py` | `SystemDiagnostics.run_diagnostics(quick)`; checks platform, torch/CUDA, deps, doctr, connectivity | `main.py` (quick mode only) |
| `startup_config.py` | `StartupConfig` – creates/repairs `config.ini`, typed getters for `[Startup]` | `main.py`, `logging_config` |
| `startup_cache.py` | `StartupCache` + global `startup_cache` (JSON in `~/.cache/visionlane_ocr`) | `doctr_torch_setup.setup_doctr_with_progress` (unused path) |
| `parallel_loader.py` | `LoadingTask`, `ParallelLoader`, `StartupLoader` | demo only |
| `model_downloader.py` | `ModelDownloadProgress` (fake progress), `EnhancedModelManager.model_exists/download_model_if_needed/get_model_info` | `main.py` (constructed only), `parallel_loader` |
| `image_processor.py` | `ImageProcessor(Thread)`, `ensure_rgb_format()` | `OCRWorker` (idle thread) |

## `hocr_to_pdf.py` in detail
`hocr_to_pdf()` first tries `ocrmypdf.hocrtransform.HocrTransform` (always fails or is absent → logged as warning), then
`CustomHOCRTransform`:
1. Read image size; parse hOCR (`lxml.html`): page bbox from `div.ocr_page`, words from `span.ocrx_word` (`bbox x1 y1 x2 y2`, integers only).
2. Flip y (hOCR top-left → PDF bottom-left).
3. `Canvas(pagesize = pixels/dpi inches)`; draw the image full-page; for each word set Helvetica at 0.8 × box height, text render mode 3 (invisible), write at box bottom-left.
Limitations: VL-025.

## `pypdfcompressor.py` in detail
Builds a Ghostscript `pdfwrite` command string; two regimes — PDF < 1 MB ("conservative": no image downsampling) and ≥ 1 MB
(downsampling by quality tier 72/150/300/600 dpi, per-type image filters `/DCTEncode`, `/JPXEncode`, `/LZWEncode`,
`/FlateEncode`). Returns the original copy if the result grew > 5 % (< 2 MB) or at all (≥ 2 MB). Standalone CLI:
`python -m utils.pypdfcompressor <input file|dir> <output dir> --quality 0-100`. Known defects: VL-014, VL-021.

## `startup_config.py` / `startup_cache.py`
`StartupConfig.get_startup_option(key, default)` coerces `true/false`, ints and floats. See which keys matter in
[configuration.md](../configuration.md). `StartupCache` TTLs: DocTR setup 24 h, models 7 d, system 1 h; invalidates by
MD5 of `config.ini` (never called).

## `thread_killer.py` / `process_manager.py`
Brute-force termination helpers (async exception injection via `PyThreadState_SetAsyncExc`, `Thread._stop()`, `SIGKILL`);
see VL-022 for why these are unsafe and partly non-functional on Windows.
