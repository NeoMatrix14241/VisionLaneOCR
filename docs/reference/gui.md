# `gui/` reference

## `gui/main_window.py` — `MainWindow(QMainWindow)` (2064 lines)

Created in `main.launch_main_application`. Heavy initialisation is deferred to `_delayed_init` (`QTimer.singleShot(0)`).

### Construction (`__init__`)
NVML/GPUtil init → window icon → state dicts (`selected_paths`, progress/file tracking) → `QThreadPool` (4 threads) →
`sync_timer` (500 ms) and `progress_monitor` (250 ms) → theme → `_load_config()` → `_create_ui()` → save config if file missing.
Default output dir `<repo>/data/output`. Window title contains a debugging note (VL-036).

### UI layout (`_create_ui`)
- **Menu:** File (Open, Save Settings, Exit), Settings (Configure Paths, Performance Options), Theme (System/Light/Dark/Night), Help (About, version string `0.2.0`).
- **Tabs** (`_create_input_section`): *Single File*, *Folder*, *PDF*; each has input selector + label, output directory + browse, archive checkbox + directory (defaults to `<output>/archive`).
- **Options** (`_create_options_section`): DPI combo, output-format combo (`PDF`, `HOCR`, `PDF+HOCR`), detection/recognition combos, compression checkbox/type/quality slider, Ghostscript-missing button.
- **Status:** current-file label, "Files Processed: n/N" progress bar, device / usage / memory labels (1 s timer).
- **Buttons:** Start Processing, Cancel.

### Important methods
| Method | Role |
|--------|------|
| `_load_config` / `_restore_config_to_widgets` / `_save_config` | read/apply/write `config.ini` (see [configuration.md](../configuration.md)) |
| `_apply_theme`, `_set_theme_mode` | four hard-coded stylesheets (`night`, `dark`, `light`, system = none) |
| `_populate_model_dropdowns(download_missing)` | builds the model combos, marks missing weights with ⬇️, optionally downloads (blocking) |
| `_on_det_model_change`, `_on_rec_model_change`, `_download_model_no_dialog`, `_on_model_change` | download on selection, then `ocr.set_models()` |
| `_count_supported_files`, `_refresh_folder_label` | recursive file count (GUI thread) |
| `_get_processing_params(tab)` | configures `ocr` per tab; returns `(mode, path)` — contains VL-001/002 |
| `_start_processing` | validates, stores archive params, resets `ocr`, creates `OCRWorker`, connects signals, starts pool |
| `_update_overall_progress`, `_sync_progress`, `_check_real_progress`, `_verify_file_completion`, `_force_progress_update`, `_update_gui` | overlapping progress logic (VL-009) |
| `_cancel_processing`, `_cleanup_processing`, `_reset_processing_state`, `_on_cancelled` | cancel flow with modal "Stopping…" dialog |
| `_process_finished` (defined twice) | success dialog, **archiving**, reset UI (VL-003, VL-032) |
| `_update_hardware_info` | NVML → GPUtil → torch fallbacks; CPU/RAM when on CPU |
| `_show_path_config`, `_show_performance_options`, `_show_about`, `_on_open_file` (broken) | dialogs/menu actions |
| `closeEvent`, `_stop_all_timers`, `_cleanup_resources`, `cleanup_and_exit`, `__del__` | shutdown (VL-035) |
| `_setup_logging` / `_setup_file_logging` | removes all root log handlers and adds `logs/ocr_gui_<ts>.log` |

### Archiving behaviour (as implemented)
After a run, if the tab's archive box is ticked: *single/pdf* → move the file into the archive directory; *folder* → move
**every file** under the input folder, preserving sub-paths. Executed regardless of success (VL-003).

## `gui/processing_thread.py`
- `WorkerSignals(QObject)`: `progress(int,int,int)`, `error(str)`, `finished(bool)`, `cancelled()`, `new_log(str)` (unused).
- `LogHandler(SafeLogHandler)`: prints records to stdout; attached to logger `ocr_processor` (nothing logs there).
- `OCRWorker(QRunnable)(ocr_processor, mode, path)`:
  - `__init__`: installs SIGINT handler, wires `ocr.progress_callback`, calls `ocr.reset_state()`, creates `mp.Queue`s and starts an idle `ImageProcessor` thread.
  - `run()`: counts files (rglob per extension), emits initial progress, dispatches to `process_folder` / `process_image` / `process_pdf`, emits `finished(True/False)`, always `cleanup_temp_files(force=True)`.
  - `stop(force)`: cancels OCR, stops the image processor, clears queues; `cancelled` never emitted (VL-034).
  - `process_image_multiprocess`: unused shared-memory prototype.
  - progress callback: throttled to 1/s, uses `len(ocr._processed_files)` as the count and calls `QApplication.processEvents()` from the worker thread (VL-009).

## `gui/splash_screen.py`, `gui/log_handler.py`
`SplashScreen(QSplashScreen)` and `BufferedLogHandler` are **unused** (`main.py` builds its own splash widget).
`gui/__init__.py` exports `MainWindow`, `SplashScreen`.
