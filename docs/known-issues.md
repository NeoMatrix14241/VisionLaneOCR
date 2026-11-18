# Known Issues & Open Items

Everything below was found by reading the whole repository (and, where noted, by running small checks).
Nothing here has been fixed in code — this is the register of what is wrong, where, and how to fix it.

**Severity**

| Level | Meaning |
|-------|---------|
| P0 | Crash or data loss in a primary workflow |
| P1 | Major functional or performance problem |
| P2 | Correctness / robustness problem, workaround exists |
| P3 | Hygiene, dead code, cosmetic |

**Evidence tags**

- `[verified]` — confirmed by running code or inspecting the real upstream package in a sandbox (Python 3.12, Linux).
- `[code-read]` — derived from reading the source; logic is clear but not executed.
- `[needs-hw]` — plausible from the code but needs Windows + an NVIDIA GPU to confirm.

Line numbers refer to the repository snapshot that was analysed (`core/ocr_processor.py` = 1462 lines, `gui/main_window.py` = 2064 lines).

---

## 1. Primary workflows (P0)

### VL-001 — "Single File" mode crashes  (P0, `[verified]`)
- **Where:** `gui/main_window.py:1579-1597` (`_get_processing_params`, tab 0) → `core/ocr_processor.py:416` (`process_image`).
- **What:** Single mode never sets `ocr.input_path`, so it is `None`. `image_path.parent.relative_to(None)` raises `TypeError`, and the code only catches `ValueError`. Reproduced: `Path('/a/b').relative_to(None)` → `TypeError`.
- **Second, independent bug:** even with `input_path` set, `process_image` is built around "folder = one PDF". It only merges when the chosen file is the **last image (sorted by name) in its folder** (`ocr_processor.py:452`). Picking any other file produces a temp page PDF that is later deleted and **no output at all**.
- **Also:** `self.dpi` is only ever assigned by the GUI (`main_window.py:1596`). `OCRProcessor.__init__` accepts a `dpi` parameter but never stores it (line 148), so any non-GUI caller (e.g. `run_ocr.py`) hits `AttributeError: 'OCRProcessor' object has no attribute 'dpi'` (`ocr_processor.py:450`).
- **Fix:** split "process one image → one output page/PDF" from "merge a folder". Make single-file mode call the first directly; give `OCRProcessor.__init__` a real `self.dpi = dpi`.

### VL-002 — "PDF" mode crashes  (P0, `[verified]`)
- **Where:** `gui/main_window.py:1622` and `core/ocr_processor.py:860`.
- **What:** PDF mode only assigns `ocr.output_base_dir = Path(...)`; it never calls `set_output_directory()`, so `self.pdf_dir` stays `None` (the GUI builds `OCRProcessor()` with no output dir, `main_window.py:2042`). `None / relative_path` → `TypeError`. Folder mode works only because `process_folder` rebuilds `pdf_dir/hocr_dir/temp_dir` itself (`ocr_processor.py:1053-1057`).
- **Fix:** always go through `set_output_directory()`; make `pdf_dir`/`hocr_dir` properties derived from `output_base_dir`.

### VL-003 — Archiving moves files even when processing failed or was cancelled  (P0, `[code-read]`)
- **Where:** `gui/main_window.py:1730-1773` (the *second* `_process_finished`).
- **What:** The archive block does not look at `success` or `ocr.is_cancelled`. In folder mode it moves **every file** under the input folder (`rglob("*")`), not just the files that were OCR'd, and leaves the empty directory tree behind. Combined with VL-004 (silent page loss) this can permanently separate users from inputs that produced no/partial output. `shutil.move` onto an existing name also behaves differently on Windows (error) vs POSIX (overwrite).
- **Fix:** archive only files with a confirmed output, only on success, per file; handle name collisions; remove emptied directories.

### VL-004 — Per-image failures are swallowed; GUI reports success  (P0, `[code-read]`)
- **Where:** `core/ocr_processor.py:1422-1431` (`_process_single_image` catches everything, returns `False`), `:455-461` (`process_image` ignores the return value), `:1113-1119` (`process_folder` counts the file as completed).
- **What:** If OCR, HOCR→PDF or I/O fails for one page, the exception is logged and the page silently disappears from the merged PDF (`_merge_folder_pdfs` only logs "Missing PDFs"). The user sees "Processing completed successfully!".
- **Fix:** propagate failures (return a result object), count them in `failed`, and make the GUI show a failure summary.

### VL-005 — Blank pages are fabricated on conversion failure  (P0, `[code-read]`)
- **Where:** `ocr_processor.py:819-829` (`_direct_gs_conversion`) and `:978-998` (`process_pdf`).
- **What:** As a "last resort" an 800×1200 / 800×1100 white page is OCR'd and saved as if it were the real page. Output looks valid but the document content is gone.
- **Fix:** fail the file instead.

---

## 2. Major functional / performance problems (P1)

### VL-006 — `CUDA_LAUNCH_BLOCKING=1` set before every image  (P1, `[code-read]`)
- `ocr_processor.py:1220` forces synchronous CUDA kernels for the rest of the process (and contradicts the `'0'` set in `core/cuda_env_patch.py:23` and `core/nuitka_cuda_patch.py:12`). Expect a noticeable throughput loss. Also `torch.backends.cudnn.benchmark = True` (`:242`) re-benchmarks on every new input shape, which scanned documents of varying size trigger constantly.
- **Fix:** remove it; enable only via a debug flag.

### VL-007 — GPU→CPU fallback is a one-way, process-wide latch  (P1, `[code-read]`)
- `ocr_processor.py:1225-1229, 1237-1239, 1250-1259`: any `RuntimeError` containing "CUDA" (including plain out-of-memory) moves the model to CPU **permanently** until the app is restarted. There is no retry with `empty_cache()`, tiling, or a smaller input. The README states <6 GB cards crash often; there is also no VRAM pre-check (open README item).
- **Fix:** catch `torch.OutOfMemoryError` separately, free cache, retry once, then fall back per-image rather than forever.

### VL-008 — Packaging patches `CUDA_VISIBLE_DEVICES=''`  (P1 latent, `[verified logic]` + `[needs-hw]`)
- `core/cuda_env_patch.py:33,44-47`: the loop sets any variable that is unset **or empty**, including `CUDA_VISIBLE_DEVICES=''`. An empty value hides all GPUs. In the current import order it is harmless only because `core/__init__.py` imports `doctr_torch_setup`, which already called `torch.cuda.is_available()` (`doctr_torch_setup.py:21`) before this runs. Any reorder, or a different entry point, would silently force CPU.
- Several other variables there are not PyTorch settings at all (`CUDA_MEMORY_FRACTION`, `CUDA_FORCE_PTX_JIT`, `CUDA_DISABLE_PTX_JIT`, `PYTORCH_MPS_*`, `CUDA_DRIVER_LIBRARY_PATH`, `CUDA_TOOLKIT_ROOT_DIR`). Newer PyTorch also prefers `PYTORCH_ALLOC_CONF` over `PYTORCH_CUDA_ALLOC_CONF` (check release notes for your version).
- **Fix:** delete the module (see VL-010/011).

### VL-009 — Progress reporting has three competing sources of truth  (P1, `[code-read]`)
- Worker signals (`processing_thread.py`), a 250 ms `progress_monitor` timer (`main_window.py:2005`), a 500 ms `sync_timer` (`:1425`), plus 100 ms `progress_timer`/`update_timer` objects.
- `ocr._processed_files` is added to when a file **starts** (`ocr_processor.py:413`), so the bar runs ahead of reality.
- `progress_callback` has two meanings — `(files_done, total_files, pct)` and `(stage_pct, 100)` — separated by the hack `total_files == 100` (`processing_thread.py:79`). A folder of exactly 100 files is misinterpreted.
- `_files_to_process` (`main_window.py:1488-1499`) is sorted per extension, but `process_folder` iterates `rglob('*')` order, so the "Processing: <name>" label can name the wrong file.
- `QApplication.processEvents()` is called from the **worker thread** (`processing_thread.py:94,184`) and ~20 other places — not thread-safe and a major source of the "slow/laggy GUI".
- **Fix:** one `pyqtSignal(file_index, total, current_path, stage)` emitted from the worker; the GUI only reacts to signals; delete all timers except the hardware one.

### VL-010 — Heavy global monkey-patching of torch / subprocess / psutil  (P1, `[code-read]`)
- `core/cuda_patch_wrapper.py:76-100` overrides `torch.version.cuda = "12.9"` and makes `torch.cuda.is_available()` return `True` when an "API call is not supported" error occurs (so the app can claim a GPU that is unusable).
- `core/nuitka_cuda_patch.py:101-229` replaces `torch.cuda.*` with wrappers and, in fallback mode, returns a **fake 8 GB / sm_7.5 device**.
- `core/runtime_cuda_patch.py:123-168` replaces `torch.tensor/zeros/ones/randn/rand/empty` with Python wrappers (**module is never imported anywhere**, only listed in the Nuitka command).
- `core/hardware_monitoring_patch.py` replaces `psutil.virtual_memory` etc. with fakes returning invented numbers (8 GB RAM, 1 TB disk) on error. Its Nuitka detection (`:23-37`, requires `sys._NUITKA_BINARY_PATH` and `import __compiled__`) very likely never succeeds, so these patches are no-ops.
- `ocr_processor.py:41-56` replaces `subprocess.Popen` with a **function**, which breaks `isinstance(x, subprocess.Popen)` and subclassing in any library.
- These patches were written to work around the 12.8-vs-driver and Nuitka issues described in the build script. They mask real failures and fabricate hardware data.
- **Fix:** remove all of them; use `CREATE_NO_WINDOW` only where you call `subprocess` yourself; report real device info or none.

### VL-011 — DocTR "PyTorch detection" layer is obsolete and invasive  (P1 for upgrade, `[verified]`)
- `core/doctr_torch_setup.py:63` replaces `builtins.__import__` for the **entire process** at import time and never restores it. It also executes work at import (lines 410-422: imports doctr, builds a CUDA tensor).
- In **python-doctr 1.1.0** `doctr/file_utils.py` exports only `requires_package` and `CLASS_NAME`; `is_torch_available`, `is_tf_available`, `_TORCH_AVAILABLE`, `ENV_VARS_TRUE_VALUES` (as a list) and the `USE_TORCH`/`DOCTR_BACKEND` switches are gone because TensorFlow support was removed. Code at `doctr_torch_setup.py:74,160`, `utils/system_diagnostics.py:160`, `main.py:193-197` therefore fails or does nothing.
- **Fix:** delete `doctr_patch.py`, `doctr_torch_setup.py` and the corresponding `main.py` steps; see [modernization.md](modernization.md).

### VL-012 — `ocrmypdf` import breaks on ocrmypdf ≥ 17, and the integration never worked  (P1, `[verified]`)
- `core/ocr_processor.py:24` does `from ocrmypdf.hocrtransform import HocrTransform`. In ocrmypdf 17.14.2 the package exports `HocrParser`/`OcrElement` and a different renderer; `HocrTransform` no longer exists → `ImportError` at application start.
- `utils/hocr_to_pdf.py:161-176` passes open file objects to `HocrTransform(...)`/`to_pdf(...)`; the 16.x API takes `Path` arguments (`hocr_filename=`, `out_filename=`) `[code-read]`, so this branch raised and silently fell through to the custom implementation. The docstring even says "try only the custom fallback". The README's "fixed OCRmyPDF bugs" were fixed by bypassing it.
- **Fix:** remove ocrmypdf from requirements and drop both imports (the name is unused in `ocr_processor.py`).

### VL-013 — PyPDF2 is dead; `PdfMerger` is gone from pypdf  (P1, `[verified]`)
- `ocr_processor.py:28,145,1009` use `PyPDF2.PdfMerger`. PyPDF2 3.0.1 is the final release (project renamed to `pypdf`) and `PdfMerger` is not present in pypdf 7.1.0.
- **Fix:** `from pypdf import PdfWriter`; `w = PdfWriter(); w.append(path) ...; w.write(out)`.

### VL-014 — Compression barely does anything  (P1, `[code-read]`)
- Called **per page** on a single-page temp PDF (`ocr_processor.py:1384-1417`). Pages are almost always < 1 MB, which takes the "small PDF" branch of `utils/pypdfcompressor.py` (`:90-95, 119-128, 169-174`): `-dDownsample*Images=false`, no image filter change → essentially a re-write.
- `-dJPEGQ` (`:184,196,228`) is the quality switch of Ghostscript's *jpeg output device*; for `pdfwrite` JPEG quality is set via `-dColorImageDict`/`-dGrayImageDict` (`QFactor`). So the quality slider has no real effect `[code-read, from GS docs]`.
- `-dColorImageDepth=24` (`:187,199`) is not a valid value; `-dPDFSETTINGS` is appended up to three times (`:121,132,234,237`); `fast_mode` is ignored; one Ghostscript process is spawned per page (slow).
- Default config has `compression_quality=100` → `/prepress`.
- **Fix:** compress once, after merging, or better skip Ghostscript: JPEG-encode with Pillow (`quality=`) before embedding via reportlab `ImageReader`/`img2pdf`, so the saved PDF is small from the start.

### VL-015 — HOCR-only output waits 30 s per folder and logs errors  (P1, `[code-read]`)
- With `output_formats == ['hocr']` no page PDFs are written, but `process_image` still calls `_merge_folder_pdfs` which polls for them for `max_wait = 30` s (`ocr_processor.py:480-510`), then logs "Missing PDFs" / "No PDFs merged".
- **Fix:** skip merging when `"pdf" not in output_formats`.

### VL-016 — PDF input is rasterised and re-OCR'd at fixed 300 dpi  (P1, `[code-read]`)
- `process_pdf` hard-codes `dpi=300` (`:875,897`), ignoring the DPI control, converts each page to PNG via Ghostscript (needs `gs`) or pdf2image (needs Poppler), and rebuilds a **raster-only** PDF named `<stem>_ocr.pdf` (`:1007`). Vector content/text layer of the source is lost, size balloons, and already-searchable PDFs are re-OCR'd.
- **Fix:** rasterise with `pypdfium2` (already a doctr dependency), then overlay an invisible text layer onto the original page (pikepdf/pypdf), or skip pages that already contain text.

---

## 3. Correctness / robustness (P2)

| ID | Issue | Where | Tag |
|----|-------|-------|-----|
| VL-017 | **Folder merge is trigger-by-last-file.** Merging happens when the name-sorted last image is processed, but files are iterated in `rglob('*')` order. If order differs, the merge fires early and waits 30 s; if the last image fails it never fires and all temp page PDFs are deleted at the end (`cleanup_temp_files(force=True)`, `:1134`). | `ocr_processor.py:415-455, 479-570, 1130-1136` | code-read |
| VL-018 | **Temp-name collisions.** `folder_key` replaces `\`, `/` with `-`, so `a/b-c` and `a-b/c` collide; glob `f"{folder_key}-*.pdf"` also matches the temp files of folder `a-b` when merging folder `a`. Two inputs `x.tif`/`x.png` collide on `x.hocr`. Per-extension `glob("*.tif")` is case-sensitive on Linux but the folder scan uses `suffix.lower()`. | `ocr_processor.py:420, 485, 1297-1300` | code-read |
| VL-019 | `.heic` is accepted everywhere but no `pillow-heif` is installed/registered, so HEIC files fail to open. `.jiff` looks like a typo for `.jfif`. | `ocr_processor.py:28, 1067`; `main_window.py:832` | code-read |
| VL-020 | `Image.MAX_IMAGE_PIXELS = None` disables Pillow's decompression-bomb guard. | `ocr_processor.py:58` | code-read |
| VL-021 | Ghostscript is launched with `shell=True` and a string built by `' '.join(...)` with embedded quotes → breakage with `%`, `^`, quotes in paths and shell-injection via file names on POSIX. | `ocr_processor.py:251-260, 373-383`; `utils/pypdfcompressor.py:245-256` | code-read |
| VL-022 | **Cancellation is unsafe.** `ThreadKiller` injects `SystemExit` into every non-main thread whose name lacks "Qt" (including torch/OpenMP/logging threads); `ProcessManager` uses `thread._stop()` and `signal.SIGKILL` (does not exist on Windows → `AttributeError` swallowed by bare `except`). `cancel_processing` deletes the temp dir under the running image. After cancel `_reset_state` leaves `is_cancelled=True` until `reset_state()` is called. Cancel only takes effect between images; a running inference cannot be interrupted. | `utils/thread_killer.py`, `utils/process_manager.py`, `ocr_processor.py:329-398` | code-read |
| VL-023 | `signal.signal(SIGINT, ...)` is registered in `OCRProcessor.__init__` **and** `OCRWorker.__init__`; it raises `ValueError` if either is created off the main thread. | `ocr_processor.py:201`; `processing_thread.py:60` | code-read |
| VL-024 | **Hardware readout.** `nvidia-ml-py3==7.352.0` is an ancient third-party wrapper; modern `pynvml`/`nvidia-ml-py` return `str`, so `.decode()` raises (`ocr_processor.py:97,103`). GPU usage is shown in a label named `cpu_label`; only GPU 0 is read; `GPUtil` 1.4.0 imports `distutils` (removed in Python 3.12; works only via setuptools' shim). "Used" memory is labelled `free_mem` (`:142-143`). `min_compute_capability` is computed but never checked. | `ocr_processor.py:83-146`; `main_window.py:33-47, 1797-1860` | verified (distutils), code-read |
| VL-025 | **`hocr_to_pdf` text layer quality.** Uses base-14 Helvetica (WinAnsi only → non-Latin glyphs cannot be encoded), sets font size from box height only (no horizontal scaling, so selection boxes don't match words), one text object per word (copy/paste yields one word per line), `bbox` regex accepts only integers (float bboxes would drop words silently), page size = pixels/DPI so wrong DPI ⇒ odd paper sizes. | `utils/hocr_to_pdf.py:37-131` | code-read |
| VL-026 | `OCRProcessor` has attributes that are only set externally (`dpi`, `compress_enabled`, `compression_type`, `compression_quality`, `input_path`, `session_dir`) — see VL-001. The `output_formats=["pdf"]` default is a mutable default; the default `recognition_model="crnn_vgg16_bn"` differs from the app default `parseq`. | `ocr_processor.py:148` | code-read |
| VL-027 | `set_output_directory` `rmtree`s a shared `%TEMP%\VisionLaneOCR_temp`; a second running instance (or the CLI) deletes the first one's work. | `ocr_processor.py:276-293` | code-read |
| VL-028 | Output layout is non-obvious: **one PDF per folder** (named after the folder) under `OCR_Session_<timestamp>/pdf/<relative>/`; `.hocr` files per image. A session directory is created even if no files are found. PDF inputs become `<stem>_ocr.pdf`. | `ocr_processor.py:1051-1062, 479-570, 1007` | code-read |
| VL-029 | `assume_straight_pages=True` — rotated/skewed pages are not handled; `torch.no_grad()` should be `torch.inference_mode()`. | `ocr_processor.py:256, 1245` | code-read |
| VL-030 | Dropdown model lists are hard-coded (`main_window.py:1030-1049`); doctr 1.1 adds `db_resnet34` and `viptr_tiny`. Cache detection looks only in `~/.cache/doctr/models` but doctr honours `DOCTR_CACHE_DIR`; the README quotes yet another path (`%USERNAME%\.doctr\cache`). | `main_window.py:1051-1054`; `utils/model_downloader.py:87` | verified (doctr) |

---

## 4. GUI problems

| ID | Sev | Issue | Where |
|----|-----|-------|-------|
| VL-031 | P1 | Model download and `OCRProcessor()` construction (model load + CUDA probing) run on the **GUI thread** in `_delayed_init`; `_populate_model_dropdowns("startup")` is called three times (`:900, :1022, :2038`), each may download and swallows all errors with `except Exception: pass`. `_count_supported_files` does a full `rglob` on the GUI thread at startup and on every start/refresh. | `main_window.py` |
| VL-032 | P2 | `_process_finished` is defined twice (`:1342`, `:1716`); the later one silently wins. `_verify_file_completion` has a duplicated `except` block (`:1462-1467`). | `main_window.py` |
| VL-033 | P2 | `File → Open` raises `AttributeError` (`self.file_path_label` does not exist, `:1892`); `File → Exit` has no handler (`:528`). | `main_window.py` |
| VL-034 | P2 | `OCRWorker` starts an `ImageProcessor` thread and allocates `mp.Queue`s that are never fed (`processing_thread.py:107-118`); `process_image_multiprocess` is unused; `.stop()` checks `_batch_processing` which is never set, so the `cancelled` signal never fires (`:234`). The `LogHandler` is attached to a logger named `ocr_processor` that nothing logs to. | `processing_thread.py` |
| VL-035 | P2 | `closeEvent` ends with `sys.exit(0)` in a `finally`, and `cleanup_on_exit` (`main.py:308-342`) also `sys.exit(0)`s from `aboutToQuit`, after which `main.py:394` calls `os._exit`. Shutdown is a race of three exits. | `main_window.py:377-414`, `main.py` |
| VL-036 | P3 | Window title is a debugging note ("Can't fix GUI so slow I wanna cry…"). | `main_window.py:113` |
| VL-037 | P3 | `thread_count` only sets `QThreadPool` size, but only one worker is ever started, so it has no effect on OCR throughput. `operation_timeout`/`chunk_timeout` are stored, edited in a dialog, saved — and **never enforced** anywhere. | `main_window.py`, `ocr_processor.py:198-199` |
| VL-038 | P3 | Default output dir is `<repo>/data/output` (`main_window.py:70`) — the README lists "creates output folder in the repository" as fixed. `run_ocr.py` also writes `input/` and `output/` beside the script. | |

---

## 5. Startup, logging, configuration

| ID | Sev | Issue | Where |
|----|-----|-------|-------|
| VL-039 | P2 | **Two statements are swallowed by comments** (code merged onto a comment line): `main.py:218` (`update_status("Initializing logging system...")`) and `main.py:247` (`update_status("Loading main application...")`). | `main.py` `[verified]` |
| VL-040 | P2 | The splash is cosmetic: all loading happens on the GUI thread via `QTimer.singleShot`, interleaved with `time.sleep(0.1)` × 11 (+0.3 s), and the real cost is the import chain `core/__init__` → `ocr_processor` → torch+doctr. `gui/splash_screen.py` (a second splash) is unused. | `main.py` |
| VL-041 | P2 | Import-time side effects everywhere: `cuda_patch_wrapper`, `cuda_env_patch`, `nuitka_cuda_patch`, `runtime_cuda_patch`, `hardware_monitoring_patch`, `doctr_patch`, `doctr_torch_setup` all execute (and `print`) when imported. `ocr_processor` calls `logging.basicConfig` at import; `setup_logging`, `DebugLogger` and `MainWindow._setup_logging` each remove all root handlers, so whichever runs last wins and earlier logs go to the wrong place. | multiple |
| VL-042 | P2 | `DebugLogger` is instantiated on every start (`main.py:301`): it creates debug, crash and **memory** logs and a daemon thread that writes every 10 s forever; handlers are never closed. Logs go to `<app dir>/logs`, which is unwritable under `Program Files`. | `utils/debug_helper.py` |
| VL-043 | P2 | `config.ini` has **three writers** with different defaults: `StartupConfig.save_config`, `MainWindow._save_config` (drops `detailed_logging`, `log_level`; uses `operation_timeout=600` vs `300`), and Qt state. configparser also discards the explanatory comments. | `utils/startup_config.py`, `main_window.py:291-371` |
| VL-044 | P2 | Of the 11 `[Startup]` keys only `skip_system_diagnostics`, `auto_download_models` (which merely *constructs* a manager, it downloads nothing), and `show_detailed_progress` (log level) have any effect. `enable_parallel_loading`, `startup_timeout`, `max_parallel_workers`, `cache_expiry_hours`, `skip_doctr_setup_check`, `skip_model_validation`, `use_minimal_diagnostics`, `cache_validation_results` are read nowhere (`[verified]` by grep). `ParallelLoader`/`StartupLoader`/`startup_cache` are only used by the demo. | `utils/` |
| VL-045 | P3 | The committed `config.ini` contains developer-specific absolute paths (`D:/Desktop/OCR/...`) and non-default values (`compress_enabled=True`, `output_format=PDF+HOCR`); it is not git-ignored, so every save churns the repo. | `config.ini` |
| VL-046 | P3 | `ModelDownloadProgress` is a fake progress bar: it prints scripted messages with hard-coded sleeps (up to ~17 s) regardless of the actual download. | `utils/model_downloader.py:45-67` |
| VL-047 | P3 | `system_diagnostics` pings `8.8.8.8:53` on every full run; `pypdfcompressor.PDFProcessor.__init__` calls `os.getlogin()` which raises on POSIX without a TTY. | `utils/` |

---

## 6. Build, CI, repository hygiene

| ID | Sev | Issue |
|----|-----|-------|
| VL-048 | P1 | `build.spec`: hidden imports reference `ocr_processor` and `doctr_torch_setup` as top-level modules (they live in `core/`); data entries for non-existent `doctr_torch_setup.py` and `test_cuda.py`; copies whole `utils/` and `gui/` directories as *data* in addition to bytecode; hard-codes CUDA 11.8–12.4 DLL folders while requirements target 12.8; `upx=True` on torch/CUDA DLLs is a known source of corrupted/slow binaries; excludes `onnx.reference` though onnx is a hard doctr dependency. |
| VL-049 | P1 | `build_nuitka.bat` (888 lines): hard-coded `.venv\Lib\site-packages` paths; includes `torchaudio`, `scipy`, `cv2`, `langdetect` explicitly; `--include-data-dir=...\torch=torch` duplicates torch; copies `nvml.dll`/`nvcuda.dll` from `System32` into `dist` (driver DLLs must not be redistributed and mismatch the target driver); the Nuitka plugin `cuda_compat_plugin.py` is only `--include-package`d, never activated with `--user-plugin`; `--windows-console-mode=force` leaves a console window. |
| VL-050 | P2 | CI (`.github/workflows/pylint.yml`) runs Ubuntu only (app is Windows-only), tests Python 3.10 (incompatible with the new stack), installs packages differently from `requirements.txt`, and ends with `|| true`, so it can never fail. |
| VL-051 | P2 | `run_ocr.py` imports `datetime.UTC` (Python ≥ 3.11) although the README claims 3.10; hard-codes `User: NeoMatrix14241` in the log; calls `process_image` without `input_path` (VL-001). `debug_run.py` calls `main.main()` which does not exist. |
| VL-052 | P2 | `pylint_clean.py` **rewrites every `.py` file** (strips trailing whitespace, normalises newlines) before linting — a linter that edits sources. |
| VL-053 | P3 | No automated tests. No type checking. `pandas` is only used by `data/dpi_res_matrix.py`, a script that executes on import, comments "1 to 1200" but generates 1–9000 DPI, writes `uniform_data.csv` to the CWD, and the 5 MB / 153 000-row CSV it produced is unused by the app. |
| VL-054 | P3 | 8 MB of binaries committed: `cache_binaries.zip` (ccache 4.11.3, clcache 4.2.0) — fetch in the build script instead. |
| VL-055 | P3 | Dead code: `core/runtime_cuda_patch.py`, `core/cuda_compat_plugin.py`, `gui/splash_screen.py`, `gui/log_handler.py` (`BufferedLogHandler`), `utils/image_processor.py`, `utils/parallel_loader.py`, `OCRProcessor.image_queue/result_queue/thread_pool/max_image_size/batch_size`, `_is_last_image_in_folder`, `_track_process`, `_convert_pdf_fallback` (only reachable from a dead `ZeroDivisionError` handler), `demo_enhanced_startup.py` (894 lines). Declared-but-unused dependencies: `torchaudio`, `python-magic`, `python-magic-bin`, `colorama`, `zstandard`, `typing_extensions`, `pandas` (runtime). Used-but-undeclared: `requests`, `lxml`, `opencv` (pulled in by doctr). |
| VL-056 | P3 | `README.md` drifts from reality: states Python 3.10–3.12, PyTorch 2.0+, GPU "WILL CRASH" notes, wrong model cache path, repo URL `VisionLane` vs `VisionLaneOCR`, refers to `build.bat` and `paper_sizes_formula.xlsl` (neither exists). |

---

## 7. README roadmap vs. reality

| README item | Status in code |
|-------------|----------------|
| Fix "Single File" option | **Open** (VL-001) |
| Fix "PDF" option | **Open** (VL-002) |
| Switch GPU/CPU option | **Open** — only automatic, irreversible fallback exists (VL-007) |
| Check 8 GB VRAM and suggest CPU | **Open** — not implemented |
| super-image / uniform paper sizes | **Open** — nothing implemented; `data/` holds an unused size table |
| Enhanced PDF output pathing; remove `_ocr` suffix | **Open** (VL-016, VL-028) |
| Fix OCRmyPDF "division by zero / no output" | Closed by *bypassing* ocrmypdf (VL-012) |
| Fix PyPDFCompressor doesn't compress | **Not actually fixed** (VL-014) |
| Instant splash / parallel loading / caching ("✓") | Splash yes; parallel loading and cache are **not wired into startup** (VL-040, VL-044) |

---

## 8. Suggested order of attack

1. **Stop data loss first:** VL-003, VL-004, VL-005 (small, high value).
2. **Make the three modes work:** VL-001, VL-002, VL-015; introduce a result object so VL-004 can be reported.
3. **Upgrade dependencies and delete the patch layer:** VL-011, VL-012, VL-013, VL-010, VL-008 ([modernization.md](modernization.md)).
4. **Performance:** VL-006, VL-007, VL-014, VL-016, VL-031.
5. **Progress/cancel rewrite on a single signal path:** VL-009, VL-022, VL-034.
6. **Cleanup:** delete dead code (VL-055), unify config (VL-043/044), fix build scripts (VL-048/049/050), add tests.
