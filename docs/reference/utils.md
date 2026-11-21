# `utils/` reference

| Module | Contents | Used by |
|--------|----------|---------|
| `app_paths.py` | `data_dir()`, `config_path()`, `log_dir()`, `app_root()`; honours `VISIONLANE_HOME` and a `portable` marker file | `main.py` |
| `app_config.py` | `AppConfig` dataclass: `load(path)` / `save(path)` (atomic), validation, legacy-key migration, choice constants | `main.py`, `gui/main_window.py` |
| `logging_config.py` | `setup_logging(log_dir, level)` (call once), `install_crash_handlers(log_dir)` | `main.py` |
| `models.py` | `configure_cache()` (weights beside the app, one-time copy from `~/.cache/doctr`), `list_models(kind)`, `is_cached(name)`, `doctr_cache_dir()`, `download(kind, name)` | `main.py`, GUI, `run_ocr.py`, `verify_models.py`, diagnostics |
| `diagnostics.py` | `python -m utils.diagnostics` - environment report (no network access) | support |

Removed in the refactor: `hocr_to_pdf`, `pypdfcompressor`, `thread_killer`, `process_manager`, `safe_logger`,
`debug_helper`, `startup_cache`, `startup_config`, `parallel_loader`, `system_diagnostics`, `model_downloader`, `image_processor`.
