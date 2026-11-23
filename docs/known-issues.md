# Known Issues & Open Items

This is the register of issues in VisionLane OCR and what became of them. It started as an analysis of the whole
repository (kept, unchanged, in [known-issues-original-analysis.md](known-issues-original-analysis.md) - it has the
evidence, file/line references and reasoning for every `VL-nnn`). The code was then refactored against it.

**Verification environment.** The refactor was done and tested on Linux / Python 3.12 (CPU only) with PyQt6 on the
`offscreen` platform, using the real PyTorch 2.14.1 and python-doctr 1.1.0 (installed late in the session).

- 96 automated tests pass (`python -m pytest`) and `mypy` reports no issues, including `tests/test_integration_doctr.py`, which runs the **real** DocTR pipeline (real `ocr_predictor`, inference, `export_as_xml`, geometry conversion, PDF + HOCR output);
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
| VL-022 Unsafe cancellation | **Fixed** (granularity: VL-065) | Cooperative `threading.Event`, checked between pages/files. `ThreadKiller`, `ProcessManager`, SIGKILL code deleted. Cancelled runs leave no partial PDFs and archive nothing | `test_cancel_stops_and_marks_rest_cancelled`, GUI `test_cancel_button` |
| VL-023 `signal.signal` in constructors | **Removed** | | - |
| VL-024 Hardware read-out | **Fixed** | `nvidia-ml-py` with `str` handling; real labels (GPU usage vs. memory used/total); GPUtil dropped; failures show `N/A` instead of invented values. *GPU numbering fixed in VL-063.* | GUI tests (read-out runs); NVML itself needs a GPU |
| VL-025 Text-layer quality | **Fixed** | Float bboxes, horizontal scaling to the word width, per-line text objects with spaces, one font size per word. *Non-Latin text: VL-062.* | `tests/test_pdf_writer.py` |
| VL-026 Attributes set only from outside | **Fixed** | Real constructor + `configure()`; default recognition model is `parseq` everywhere | `test_configure_validates_formats` |
| VL-027 Shared temp dir deleted | **Fixed** | Each run uses its own `tempfile.TemporaryDirectory`, removed at the end | `test_temp_files_are_private_and_cleaned` |
| VL-028 Output layout | **Fixed** | Layout documented ([architecture.md](architecture.md)); the extra folder next to the PDF and the `_ocr` suffix are gone; the session folder is only created when there is something to process | `test_folder_becomes_one_pdf...`, `test_no_files_creates_no_session_dir` |
| VL-029 Rotated pages / `no_grad` | **Fixed** | `torch.inference_mode()`; rotated/skewed pages are now optional (VL-061) | see VL-061 |
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
| VL-053 No tests / pandas | **Fixed** | 96 tests; `data/dpi_res_matrix.py` rewritten without pandas and import side effects. *Type checking: VL-064* | `python -m pytest` |
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

| VL-069 | DocTR 1.1.0's `export_as_xml()` raises `ValueError` for rotated (polygon) word geometry, so HOCR output would have crashed as soon as rotated-page mode was on (found by running real DocTR). | **Fixed** - polygons are captured for the PDF text layer first, then replaced by their bounding boxes for the HOCR export | `test_straighten_mode_gives_polygons_and_still_exports_hocr`, `test_rotated_words_survive_into_the_pdf...` |
| VL-070 | **Crash (segfault/abort) when a background task finished after its window/tab was destroyed** (tasks were parented to widgets; callbacks were lambdas). Surfaced while adding tests. | **Fixed** - `Task` has no Qt parent, callbacks are bound slots (Qt auto-disconnects them), the hardware slot cannot raise | GUI suite run repeatedly (3x) without crashes |

## 3. The six previously open items

All six were addressed in a second pass. What "done" means for each, and what is *not* verified:

| ID | Was | Status | What was done | Evidence / not verified |
|----|-----|--------|---------------|-------------------------|
| VL-061 | Rotated / skewed pages not handled | **Fixed (unverified with pretrained models)** | Option *Rotated / skewed pages* (`straighten_pages`, CLI `--straighten`): DocTR's `straighten_pages` + `preserve_original_coords`; words come back as 4-point polygons on the **original** page and the PDF text layer draws them rotated (any angle). The image itself is never altered | Real DocTR pipeline with the two orientation models disabled (random weights, offline): `test_straighten_mode_gives_polygons...`; rotated text matrices: `test_rotated_word_gets_a_rotated_text_matrix`, `test_upside_down_word...`; visual check of 0/90/180/270/+15/-10 degrees. **Not run:** the real page/crop orientation models (extra download, first use) |
| VL-062 | Non-Latin text became `?` in the PDF text layer | **Fixed** (CJK excluded) | DejaVu Sans (`assets/fonts`, licence included) is embedded (subset) **only on pages that need more than WinAnsi**; Latin-only pages still use unembedded Helvetica, so file size is unchanged. Cyrillic, Greek, Vietnamese, ... are searchable/copyable | `test_non_latin_text_is_searchable_and_latin_only_pages_embed_no_font`. DejaVu has no CJK glyphs: those characters still become `?` (HOCR is correct) |
| VL-063 | Hardware read-out always used GPU 0 | **Fixed** | The GPU the model runs on is matched to its NVML handle by UUID, then PCI address, then index | `tests/test_hardware.py` (fake NVML with swapped numbering). Not run on a multi-GPU machine |
| VL-064 | No static type checking | **Fixed** | `mypy.ini`, `run_mypy.bat`, CI step, stub packages in `requirements-dev.txt`; code is clean (30 files) | `python -m mypy` -> no issues |
| VL-065 | Cancel waited for the whole current page | **Fixed (granularity)** | Forward pre-hooks on DocTR's detection / recognition / orientation networks raise `Cancelled` before each forward pass, so Cancel acts between model stages and recognition batches | `test_cancel_interrupts_inference_inside_a_page` (real DocTR: 2 forward passes per page, cancel honoured before the 1st and between them). **Remaining limit:** a single forward pass that has started cannot be interrupted |
| VL-067 | Super-image / uniform paper size not started | **Fixed (super-resolution unverified)** | *Page size* option (Original / Auto / A4, Letter, ...): all pages of an image folder get the same paper size - image scaled to fit, aspect ratio and orientation kept, text layer follows. *Enhance low-resolution pages*: optional x2-x4 super-resolution (`super-image` EDSR, tiled, OOM-safe) for pages whose effective resolution on the chosen paper falls below 150 dpi | Uniform sizes, Auto voting, fit/centering and the enhancement decision are tested (`tests/test_uniform_pages.py`, `test_paper.py`); tiling equals whole-image upscaling. **Not run:** the real `super-image` model (optional package, Hugging Face download) |

### Known limitations of the new features
- **Page size** applies to image inputs only; PDF inputs keep their original page sizes (their pages are annotated, not re-rendered).
- **Auto** picks the standard paper size most pages of a folder are closest to (pages further than 25 % away are ignored); with no match, pages keep their size.
- **Rotated / skewed pages** is slower and downloads two small extra DocTR models on first use (page and crop orientation). HOCR output uses the bounding boxes of rotated words.
- **Enhancement** changes the pixels that are OCR'd and embedded; HOCR coordinates then refer to the enlarged image. If the package or model is unavailable the run continues without enhancement and the summary dialog shows a note.
- Text in the PDF layer: CJK characters are shown as `?` (no glyphs in the bundled font).

## 4. Still open

| ID | Sev | Issue | Notes |
|----|-----|-------|-------|
| VL-066 | P1 (unverified) | **CUDA / Windows / pretrained-weights validation is pending** - see the checklist below. (Real DocTR 1.1.0 on the CPU is covered by the integration test.) | Everything marked *Fixed (unverified)* above |

### Behaviour changes to be aware of

- PDF inputs keep their original pages and only receive a text layer; pages that already contain text are **skipped** by default (checkbox *Skip PDF pages that already have text*; with HOCR output every page is recognised so that each page gets a HOCR file).
- PDF output names no longer carry `_ocr`; the image-folder PDF is `pdf/<relative folder>/<folder name>.pdf` (no extra folder).
- Compression types are now **JPEG** and **Lossless**. `jpeg2000`, `lzw` and `png` from old config files map to JPEG / Lossless (they never produced the advertised result through Ghostscript).
- Default JPEG quality is 85 (was 100) and compression is off by default.
- `Settings -> Configure Paths` was removed (it only changed an attribute that the processing run overwrote); the per-tab output fields are the single source.
- `config.ini` and logs moved to `%LOCALAPPDATA%\VisionLaneOCR` (create an empty file named `portable` next to the program to keep them beside it).
- DocTR models are stored in `<program folder>/doctr_cache` (old `~/.cache/doctr` models are copied once); `HF_HOME` points inside it as well (only used by the optional super-resolution model).
- The three new page options (*Page size*, *Enhance low-resolution pages*, *Rotated / skewed pages*) are **off by default**; with them off the output is the same as before.
- `OCREngine(assume_straight_pages=...)` was replaced by `straighten_pages=...`.

### Verification checklist for a Windows/GPU machine (VL-066)

1. `pip install -r requirements-dev.txt` (CUDA torch first), then `python -m utils.diagnostics`.
2. `python -m pytest` - all 96 tests green, and `test_integration_doctr` **not skipped**.
3. `python main.py`: wait for "No file processing"; run a folder, a single image, a multi-page TIFF and a PDF; open the PDFs and select/search text.
4. On a GPU with < 8 GB: process large scans, confirm the notice and that out-of-memory pages still complete (`logs/` shows the retry).
5. `build_pyinstaller.bat` and `build_nuitka.bat`, then run the built executable from a clean folder.
