# Known Issues & Open Items

This is the register of issues in VisionLane OCR and what became of them. It started as an analysis of the whole
repository (kept, unchanged, in [known-issues-original-analysis.md](known-issues-original-analysis.md) - it has the
evidence, file/line references and reasoning for every `VL-nnn`). The code was then refactored against it.

**Verification environment.** The refactor was done and tested on Linux / Python 3.12 (CPU only) with PyQt6 on the
`offscreen` platform, using the real PyTorch 2.14.1 and python-doctr 1.1.0 (installed late in the session).

- 54 automated tests pass (`python -m pytest`), including `tests/test_integration_doctr.py`, which runs the **real** DocTR pipeline (real `ocr_predictor`, inference, `export_as_xml`, geometry conversion, PDF + HOCR output);
- that test uses randomly initialised weights: the sandbox cannot reach `doctr-static.mindee.com`, so **no pretrained model was downloaded and real recognition quality was not checked**;
- nothing was run on Windows or on a CUDA GPU (the OOM recovery is covered by a fake-model test only)

## Status legend

| Status | Meaning |
|--------|---------|
| **Fixed** | Changed in code; behaviour is covered by an automated test (named in the *Evidence* column) |
| **Fixed (unverified)** | Changed in code, but only verifiable on Windows / a GPU / with real DocTR - see the note |
| **Removed** | The code or file was deleted because it was dead, harmful or replaced |
| **Open** | Not addressed (still relevant) |

---

## 1. Resolution of the original issues

### P0 - crashes and data loss

| ID | Status | What was done | Evidence |
|----|--------|---------------|----------|
| VL-001 Single File crashes | **Fixed** | `OCRProcessor.process_file()`; the folder merge is a separate step; settings (DPI, formats, ...) are real constructor/`configure()` state, not attributes poked in by the GUI | `test_single_image_mode`, `test_dpi_controls_page_size`, GUI `test_single_and_pdf_tabs_work` |
| VL-002 PDF mode crashes | **Fixed** | Output paths are derived from the session directory inside one `_run()`; no `None` directories | `test_pdf_input_keeps_original_pages_and_adds_text`, GUI test above |
| VL-003 Archiving moves everything | **Fixed** | New `core/archiver.py`: only `Status.SUCCESS` files, never after a cancel, collisions get ` (2)`, emptied folders removed, archive folder inside input folder refused | `tests/test_archiver.py` (4 tests), GUI `test_failures_are_shown_and_inputs_kept`, `test_cancel_button` |
| VL-004 Failures swallowed | **Fixed** | Every file ends in a `FileResult` (`success / partial / failed / cancelled` + error text); the GUI shows a summary with details; failed pages are listed | `test_failed_page_is_reported_and_not_silent`, `test_corrupt_image_fails_cleanly`, GUI `test_failures_are_shown...` |
| VL-005 Blank pages fabricated | **Fixed** | The fabricated-page fallbacks no longer exist; an unreadable page fails the file | `test_all_pages_failing_gives_no_blank_pdf`, `test_pdf_in_folder_mode_and_encrypted_pdf_failure` |

### P1 - major functional / performance problems

| ID | Status | What was done | Evidence |
|----|--------|---------------|----------|
| VL-006 `CUDA_LAUNCH_BLOCKING`, `cudnn.benchmark` | **Removed** | Neither is set any more; `torch.inference_mode()` is used | code (`core/ocr_engine.py`) |
| VL-007 One-way GPU->CPU latch | **Fixed (unverified)** | `OOM -> empty_cache + retry -> this page on the CPU, model returns to the GPU`; other CUDA errors switch to the CPU for the session; GPU smoke test at start; VRAM < 8 GB shows a notice; Device selector added | `tests/test_engine.py` (5 tests, fake model). **Needs a real CUDA GPU to confirm** |
| VL-008 `CUDA_VISIBLE_DEVICES=''` | **Removed** | The whole environment-patch layer is gone | - |
| VL-009 Three progress sources | **Fixed** | One `ProgressEvent` stream from the worker; the GUI only reacts to signals; the only timer left is the 1 s hardware read-out; no `processEvents()` from threads | `test_progress_events_count_finished_files`, GUI end-to-end tests |
| VL-010 Monkey-patching of torch/subprocess/psutil | **Removed** | `cuda_patch_wrapper`, `cuda_env_patch`, `nuitka_cuda_patch`, `runtime_cuda_patch`, `cuda_compat_plugin`, `hardware_monitoring_patch` deleted; no fake hardware values anywhere | - |
| VL-011 Obsolete DocTR detection layer | **Removed** | `doctr_patch`, `doctr_torch_setup` deleted; no `builtins.__import__` replacement | - |
| VL-012 ocrmypdf | **Removed** | Dependency and code deleted | - |
| VL-013 PyPDF2 / PdfMerger | **Fixed** | `pypdf.PdfWriter.append` | `test_folder_becomes_one_pdf_in_natural_order` |
| VL-014 Compression does nothing | **Fixed** | Ghostscript step removed. Page images are JPEG-encoded (quality 1-100) or Flate-compressed while the PDF is written; JPEG sources are embedded untouched when compression is off; reportlab's default ASCII85 wrapping (+25 % size) is disabled | `test_jpeg_compression_actually_shrinks_the_pdf`, `test_jpeg_source_is_embedded_untouched...` |
| VL-015 HOCR-only waits 30 s | **Fixed** | Merging only happens when a PDF is requested, and only once per finished folder | `test_hocr_only_writes_no_pdf_and_no_wait` |
| VL-016 PDF input rasterised at 300 dpi, raster-only output | **Fixed** | `pypdfium2` renders pages (at the chosen DPI, pixel-capped); recognised text is overlaid on the **original** pages with `pypdf` (all pages kept, rotation handled); pages that already have text are skipped (option) | `test_pdf_input_*`; overlay on a `/Rotate 90` page checked manually |

### P2 - correctness / robustness

| ID | Status | What was done | Evidence |
|----|--------|---------------|----------|
| VL-017 Trigger-by-last-file merge | **Fixed** | Images are grouped per folder and merged when the group is done | `test_failed_page_is_reported...` (middle page fails, merge still happens) |
| VL-018 Temp/HOCR name collisions | **Fixed** | Page PDFs are numbered inside a private temp dir; outputs are reserved through `_unique()` (` (2)` suffix); `x.png` + `x.jpg` get distinct HOCR names | `test_same_stem_different_extension_does_not_collide` |
| VL-019 HEIC / `.jiff` | **Fixed (unverified)** | `.jiff` typo -> `.jfif`; HEIC/HEIF is offered only when `pillow-heif` is installed (optional requirement) | HEIC itself not tested |
| VL-020 Decompression-bomb guard disabled | **Fixed** | Guard re-enabled with a 400 MP limit; oversize files fail with a message | `test_pillow_decompression_bomb_guard_is_still_on` |
| VL-021 `shell=True` Ghostscript | **Removed** | No subprocesses are started any more | - |
| VL-022 Unsafe cancellation | **Fixed** | Cooperative `threading.Event`, checked between pages/files. `ThreadKiller`, `ProcessManager`, SIGKILL code deleted. Cancelled runs leave no partial PDFs and archive nothing | `test_cancel_stops_and_marks_rest_cancelled`, GUI `test_cancel_button` |
| VL-023 `signal.signal` in constructors | **Removed** | | - |
| VL-024 Hardware read-out | **Fixed** | `nvidia-ml-py` with `str` handling; real labels (GPU usage vs. memory used/total); GPUtil dropped; failures show `N/A` instead of invented values. *Limitation: see VL-063.* | GUI tests (read-out runs); NVML itself needs a GPU |
| VL-025 Text-layer quality | **Fixed** (partly open) | Float bboxes, horizontal scaling to the word width, per-line text objects with spaces, one font size per word. *Non-Latin characters still become `?` in the PDF text layer - see VL-062* | `tests/test_pdf_writer.py` |
| VL-026 Attributes set only from outside | **Fixed** | Real constructor + `configure()`; default recognition model is `parseq` everywhere | `test_configure_validates_formats` |
| VL-027 Shared temp dir deleted | **Fixed** | Each run uses its own `tempfile.TemporaryDirectory`, removed at the end | `test_temp_files_are_private_and_cleaned` |
| VL-028 Output layout | **Fixed** | Layout documented ([architecture.md](architecture.md)); the extra folder next to the PDF and the `_ocr` suffix are gone; the session folder is only created when there is something to process | `test_folder_becomes_one_pdf...`, `test_no_files_creates_no_session_dir` |
| VL-029 Rotated pages / `no_grad` | **Open** (half) | `torch.inference_mode()` done. `assume_straight_pages=True` is unchanged: rotated/skewed pages are not handled - tracked as VL-061 | - |
| VL-030 Hard-coded model lists, cache path | **Fixed** | Model lists come from DocTR's own registry (`ARCHS`; built-in fallback); models are stored in `<app>/doctr_cache` (`utils.models.configure_cache`; `DOCTR_CACHE_DIR` overrides; old `~/.cache` models are copied once) - `tests/test_model_cache.py` | checked against real doctr 1.1.0: all 18 listed architectures have a factory; cache path matches. Actual download not possible offline |

### GUI

| ID | Status | What was done | Evidence |
|----|--------|---------------|----------|
| VL-031 Model download / init on GUI thread | **Fixed** | Engine creation, downloads, model/device switches and the folder file count all run on background threads (`gui/workers.py`); errors are shown, not swallowed | GUI fixture waits for the async load |
| VL-032 Duplicate handlers | **Removed** | `main_window.py` rewritten (2064 -> ~550 lines, split into `main_window`, `input_tab`, `workers`, `dialogs`, `theme`) | - |
| VL-033 File->Open / Exit | **Fixed** | Open selects the right tab; Exit closes the window | code; menu actions not clicked in tests |
| VL-034 Dead worker machinery | **Removed** | `processing_thread.py`, `log_handler.py` deleted | - |
| VL-035 Three-way shutdown race | **Fixed** | One path: `closeEvent` (asks if a job runs, cancels, waits <= 30 s) -> `app.exec()` returns. `os._exit` only if a daemon thread is still alive afterwards | GUI `test_settings_roundtrip_on_close` |
| VL-036 Debug window title | **Fixed** | | - |
| VL-037 `thread_count`, timeouts | **Fixed** | `cpu_threads` now drives `torch.set_num_threads` (CPU inference); `operation_timeout` / `chunk_timeout` removed (a running inference cannot be interrupted; they never did anything) | `tests/test_config.py` (legacy keys read, dropped on save) |
| VL-038 Default output in repo | **Fixed** | No default output folder; the GUI requires one; `run_ocr.py` takes it as an argument | - |

### Startup, logging, configuration

| ID | Status | What was done | Evidence |
|----|--------|---------------|----------|
| VL-039 Statements swallowed by comments | **Removed** | `main.py` rewritten | - |
| VL-040 Cosmetic splash | **Fixed** | Splash shown first; PyTorch/DocTR are imported on a background thread while it animates (indeterminate bar, no sleeps); models load after the window opens | not timed on Windows |
| VL-041 Import-time side effects | **Fixed** | No module patches or prints at import; `logging.basicConfig` only in the CLI; `setup_logging()` is called once from `main.py` | - |
| VL-042 `DebugLogger`, logs in app dir | **Fixed** | Memory-logging thread removed; one log file per session (the newest 20 are kept) + `crash.log` (faulthandler); logs live in `%LOCALAPPDATA%\VisionLaneOCR\logs` (or portable/`VISIONLANE_HOME`) | `utils/app_paths.py` |
| VL-043 Three config writers | **Fixed** | `utils/app_config.AppConfig` is the only reader/writer; atomic write | `tests/test_config.py` |
| VL-044 Dead `[Startup]` keys | **Removed** | Dropped on load and not written; legacy files still load | `test_legacy_file_is_understood` |
| VL-045 Committed `config.ini` | **Fixed** | Deleted from the repo and git-ignored; created on first run in the per-user folder | - |
| VL-046 Fake download progress | **Fixed** | Downloads show an honest "loading" state, no scripted messages or sleeps | - |
| VL-047 8.8.8.8 ping, `getlogin` | **Removed** | `system_diagnostics.py` replaced by `python -m utils.diagnostics` (no network access) | - |

### Build, CI, hygiene

| ID | Status | What was done | Evidence |
|----|--------|---------------|----------|
| VL-048 `build.spec` | **Fixed (unverified)** | Rewritten (60 lines): correct module paths, no UPX, no hard-coded CUDA folders, no data copies of source | **Needs a Windows test build** |
| VL-049 `build_nuitka.bat` | **Fixed (unverified)** | Rewritten (60 lines instead of 888): no hard-coded site-packages paths, no driver DLL copying, no `torchaudio/scipy/cv2/langdetect`, console disabled; ccache optional | **Needs a Windows test build** |
| VL-050 CI | **Fixed (unverified)** | `windows-latest`, Python 3.12/3.13, installs `requirements-dev.txt`, runs pytest and pylint (`--fail-under=9.0`); no `\|\| true` | workflow not executed here |
| VL-051 `run_ocr.py`, `debug_run.py` | **Fixed** | `run_ocr.py` is an argparse CLI on the new API with meaningful exit codes; `debug_run.py` deleted (`python main.py --debug`) | argument parsing checked; not run end-to-end (needs downloaded weights) |
| VL-052 `pylint_clean.py` edits sources | **Removed** | `run_pylint.bat` runs pylint read-only | pylint 9.98/10 |
| VL-053 No tests / pandas | **Fixed** (partly open) | 54 tests; `data/dpi_res_matrix.py` rewritten without pandas and import side effects. *No static type checker yet - VL-064* | `python -m pytest` |
| VL-054 Committed binaries | **Removed** | `cache_binaries.zip` and the generated 5 MB `uniform_data.csv` deleted (the CSV is reproducible with `python data/dpi_res_matrix.py`) | - |
| VL-055 Dead code, bad dependency list | **Removed** | All listed modules/attributes deleted; `requirements.txt` trimmed (no `lxml`, `pdf2image`, `requests`, `tqdm`, `wmi`, `pandas`; added `pypdfium2`) | `pyflakes` clean |
| VL-056 README drift | **Fixed** | README rewritten | - |

---

## 2. Found and fixed during the refactor (new IDs)

| ID | Issue | Status | Evidence |
|----|-------|--------|----------|
| VL-057 | **Multi-page TIFFs lost every page after the first** (images were read as a single frame through DocTR's `DocumentFile.from_images`). | **Fixed** - every frame becomes a page (HOCR: `<name>_p0002.hocr`, ...) | `test_multiframe_tiff_keeps_all_pages` |
| VL-058 | Files were processed in plain string order, so `img10` came before `img2` and the pages of the merged PDF were in the wrong order. | **Fixed** - natural sort order | `test_folder_becomes_one_pdf_in_natural_order` |
| VL-059 | The OCR input and the image embedded in the PDF could disagree about EXIF orientation (OpenCV applies it when reading, reportlab does not) `[code-read]`. | **Fixed** - both use the same upright image | `test_exif_rotated_jpeg_gets_upright_page_geometry` |
| VL-060 | reportlab wrapped streams in ASCII85 by default, making every PDF ~25 % bigger. | **Fixed** (`rl_config.useA85 = 0`) | `test_jpeg_source_is_embedded_untouched...` |
| VL-068 | `OCRProcessor(pretrained=False)` was not offline: DocTR still downloads *backbone* weights unless `pretrained_backbone=False` is passed too (found by running the integration test). | **Fixed** in `OCREngine` | `test_real_doctr_pipeline_produces_pdf_and_hocr` |

## 3. Still open

| ID | Sev | Issue | Notes |
|----|-----|-------|-------|
| VL-061 (was part of VL-029) | P2 | **Rotated / skewed pages** are not handled (`assume_straight_pages=True`). | Needs a decision on DocTR's `straighten_pages` / `detect_orientation` (slower) and rotated text boxes in the PDF layer. Found while testing: building the predictor with `assume_straight_pages=False` makes DocTR download an extra *crop orientation* model |
| VL-062 (part of VL-025) | P2 | **Non-Latin text in the PDF text layer**: Helvetica cannot encode it, characters become `?` (HOCR output is correct). | Needs an embedded Unicode TTF font (licence/size decision) |
| VL-063 (part of VL-024) | P3 | The hardware read-out always uses GPU index 0. NVML (PCI order) and CUDA (fastest-first) can number GPUs differently on multi-GPU machines or with `CUDA_VISIBLE_DEVICES`. | Map via the device UUID |
| VL-064 (part of VL-053) | P3 | No static type checking (mypy/pyright). | |
| VL-065 | P2 | **A running inference cannot be interrupted**: Cancel (and closing the window) waits for the current page. Huge pages on the CPU can take tens of seconds. | Inherent to PyTorch; would need inference in a subprocess |
| VL-066 | P1 (unverified) | **CUDA / Windows / pretrained-weights validation is pending** - see the checklist below. (Real DocTR 1.1.0 on the CPU is now covered by the integration test.) | Everything marked *Fixed (unverified)* above |
| VL-067 | - | Roadmap: **super-image / uniform paper sizes** - not started (`data/dpi_res_matrix.py` is the only groundwork). | |

### Behaviour changes to be aware of

- PDF inputs keep their original pages and only receive a text layer; pages that already contain text are **skipped** by default (checkbox *Skip PDF pages that already have text*; with HOCR output every page is recognised so that each page gets a HOCR file).
- PDF output names no longer carry `_ocr`; the image-folder PDF is `pdf/<relative folder>/<folder name>.pdf` (no extra folder).
- Compression types are now **JPEG** and **Lossless**. `jpeg2000`, `lzw` and `png` from old config files map to JPEG / Lossless (they never produced the advertised result through Ghostscript).
- Default JPEG quality is 85 (was 100) and compression is off by default.
- `Settings -> Configure Paths` was removed (it only changed an attribute that the processing run overwrote); the per-tab output fields are the single source.
- `config.ini` and logs moved to `%LOCALAPPDATA%\VisionLaneOCR` (create an empty file named `portable` next to the program to keep them beside it).

### Verification checklist for a Windows/GPU machine (VL-066)

1. `pip install -r requirements-dev.txt` (CUDA torch first), then `python -m utils.diagnostics`.
2. `python -m pytest` - all 54 tests green, and `test_integration_doctr` **not skipped**.
3. `python main.py`: wait for "No file processing"; run a folder, a single image, a multi-page TIFF and a PDF; open the PDFs and select/search text.
4. On a GPU with < 8 GB: process large scans, confirm the notice and that out-of-memory pages still complete (`logs/` shows the retry).
5. `build_pyinstaller.bat` and `build_nuitka.bat`, then run the built executable from a clean folder.
