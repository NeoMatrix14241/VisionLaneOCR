# `core/` reference

## `core/__init__.py`
Eagerly imports `OCRProcessor`, `doctr_patch`, `doctr_torch_setup`. Importing **anything** in `core` therefore loads
torch + doctr and runs the DocTR patch code (import order matters, VL-008/VL-041).

## `core/ocr_processor.py` (1462 lines)

### Import-time behaviour
- Applies Nuitka CUDA patches if "Nuitka" is detected (lines 6-20).
- Imports `doctr.io.DocumentFile`, `doctr.models.ocr_predictor`, `ocrmypdf.hocrtransform.HocrTransform` (unused, breaks on ocrmypdf ≥ 17), `PyPDF2.PdfMerger`, `torch`, `psutil`.
- On Windows replaces `subprocess.Popen` with a wrapper adding `CREATE_NO_WINDOW` for `nvidia-smi` (lines 41-56; VL-010).
- Sets `Image.MAX_IMAGE_PIXELS = None` (VL-020); calls `logging.basicConfig`; `_ensure_console_logging()`.

### `_check_gpu_support() -> (bool, str, list[str])`
Returns whether CUDA works, a reason string, and info lines. If CUDA is unavailable it queries NVML and WMI to explain why.
If available it lists name/compute capability/memory per device. Does **not** check VRAM size or architecture support.

### `class OCRProcessor`
`__init__(output_base_dir=None, output_formats=["pdf"], detection_model="db_resnet50", recognition_model="crnn_vgg16_bn", dpi=None)`
(`dpi` is ignored — VL-001). Steps: set models; temp dir `%TEMP%/VisionLaneOCR_temp`; validate formats; optional
`set_output_directory`; create (unused) spawn queues and a `ThreadPoolExecutor`; install SIGINT handler; init
cancellation/tracking state; wipe temp dir; clear CUDA cache; `_check_gpu_support()` → `self.device`;
cudnn settings; `_init_model()`.

| Attribute | Meaning |
|-----------|---------|
| `device` | `'cuda'` or `'cpu'` (can flip to cpu permanently, VL-007) |
| `model` | doctr `OCRPredictor` (`assume_straight_pages=True`) on `device`, eval mode |
| `output_base_dir`, `pdf_dir`, `hocr_dir`, `temp_dir`, `session_dir` | output layout; `process_folder` rewrites the last four |
| `input_path` | root of the current batch, used for relative paths (must be set externally) |
| `output_formats` | subset of `["pdf","hocr"]` |
| `dpi` | `None` or int; **only set by the GUI** |
| `compress_enabled/type/quality` | set by the GUI, read with `getattr` defaults (80/jpeg) |
| `progress_callback` | `fn(a, b, c=None) -> bool` (False = cancel); dual-meaning arguments (VL-009) |
| `is_cancelled`, `_force_stop`, `_exit_event`, `_should_exit` | cancellation flags |
| `current_file`, `_processed_files` | progress state read by the GUI timers |
| `operation_timeout`, `chunk_timeout` | stored, never enforced |

| Method | Purpose / notes |
|--------|-----------------|
| `_init_model()` | `ocr_predictor(det_arch, reco_arch, pretrained=True, assume_straight_pages=True).to(device).eval()`; logs GPU memory |
| `set_models(det, reco)` | re-creates the predictor when a name changed |
| `set_output_directory(path)` | resolves, creates `pdf/`/`hocr/` per requested format, **deletes and recreates temp dir** |
| `cleanup_temp_files(force=False)` | deletes files in temp dir; with `force` removes the directory |
| `cancel_processing()` | sets flags, wipes temp, `ThreadKiller.terminate_all_threads()`, SIGKILL of tracked pids (no-op), recreates thread pool, empties CUDA cache, `_reset_state()` |
| `_reset_state()` / `reset_state()` | the first marks everything cancelled; the second clears flags/sets for a new run, wipes temp, recreates the pool. The GUI calls `reset_state()` before each run |
| `process_image(path) -> dict` | processes one image to a numbered temp page PDF `"<folder_key>-<index:04d>.pdf"`; merges the folder when it is the last name-sorted image. Returns `{"status","folder","index","total"}` or `{"status":"cancelled"}` (VL-001, VL-017) |
| `_merge_folder_pdfs(folder_key, relative_path)` | waits ≤ 30 s for all expected page PDFs, merges them with `PdfMerger` into `pdf/<rel>/<folder>.pdf`, deletes the temps (VL-015, VL-018) |
| `_convert_pdf_to_images(pdf, out_dir, dpi=300)` | locate Ghostscript → `-sDEVICE=pngalpha` to `page_%04d.png`; fallback `pdf2image` (Poppler); returns sorted paths |
| `_direct_gs_conversion`, `_convert_pdf_fallback` | secondary Ghostscript attempts (`png16m`, then `jpeg`); last resort creates a **blank page** (VL-005) |
| `process_pdf(pdf)` | rasterise at fixed 300 dpi → `_process_single_image` per page → merge to `pdf/<rel>/<stem>_ocr.pdf` (VL-002, VL-016) |
| `process_folder(folder) -> dict` | creates `OCR_Session_<ts>/{pdf,hocr,temp}`, collects images + PDFs via `rglob`, runs them sequentially, returns `{"status": success|partial|cancelled|no_files, "processed","failed","total"}` |
| `_process_single_image(image, temp_pdf, dpi, hocr_output_folder, page_num, pdf_name)` | the per-page pipeline described in [architecture.md §3.2](../architecture.md); returns `True/False/None` which callers ignore (VL-004) |
| `_is_last_image_in_folder`, `_track_process` | unused |

Behavioural facts worth knowing:
- Image extensions: `.tif .tiff .jpg .jpeg .png .bmp .gif .dib .jpe .jiff .heic`.
- Only the first page of `export_as_xml()` is used (one image per call).
- For HOCR from PDFs: `hocr/<rel>/<pdf stem>/<pdf stem>_page_NNNN.hocr`.
- Compression is attempted on each temp page PDF and kept only if result ≤ 110 % of the original.

## `core/doctr_patch.py` / `core/doctr_torch_setup.py`
Force a PyTorch backend for DocTR < 1.0. `doctr_torch_setup` additionally: replaces `builtins.__import__`; runs
`setup_doctr_with_pytorch()` and `ensure_torch_available()` at import; builds fallback mock `doctr.file_utils`;
offers `setup_doctr_with_progress(progress_callback, use_cache, detailed_progress)` (cached via `StartupCache`; only the
unused `StartupLoader` and the demo call it). Obsolete on doctr ≥ 1.0 (VL-011).

## CUDA / hardware patch modules
See the table in [architecture.md §5](../architecture.md) and VL-008/010.

- `cuda_env_patch.CudaEnvironmentPatch.apply_environment_patches()` – sets the env vars listed in [configuration.md](../configuration.md); detects the CUDA install under `Program Files\NVIDIA GPU Computing Toolkit\CUDA`; auto-runs on import (both branches of the final `if`).
- `cuda_patch_wrapper.apply_all_cuda_patches()` – env patch → Nuitka patch (if detected) → imports `doctr_patch`/`doctr_torch_setup` → hardware patch; then `patch_torch_cuda()`. Auto-runs on import.
- `nuitka_cuda_patch.NuitkaCudaPatch` – wraps `torch.cuda.is_available/device_count/get_device_name/properties/capability/current_device/memory_*/empty_cache/synchronize/_lazy_init`; `fallback_mode` returns invented values.
- `runtime_cuda_patch.RuntimeCudaPatch` – broad torch wrappers incl. tensor factories; never imported.
- `cuda_compat_plugin.NuitkaPluginCudaCompat` – Nuitka plugin (`cuda-compat`) injecting a pre-import patch for `torch.version.cuda`; never activated.
- `hardware_monitoring_patch.HardwareMonitoringPatch` – safe wrappers for pynvml/psutil/wmi plus `get_safe_gpu_info()` / `get_safe_system_info()`; only patches when its Nuitka detection succeeds.
