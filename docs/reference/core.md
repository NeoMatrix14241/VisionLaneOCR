# `core/` reference

`core/` has no Qt dependency. `core/__init__.py` is empty on purpose: importing the package is cheap, and torch is only
imported by `core.ocr_engine` / `core.hardware`.

## `core/constants.py`
`PDF_EXTENSIONS`, `image_extensions()` (base formats, plus `.heic/.heif` when `pillow-heif` can be registered), `is_image()`, `is_pdf()`,
`natural_key()` (natural sort key), model fallback lists and defaults (`db_resnet50`, `parseq`), `RECOMMENDED_VRAM_GB = 8`,
DPI plausibility limits (30-4800, default 300), `PDF_RENDER_MAX_PIXELS` (40 MP), `MIN_TEXT_CHARS_FOR_SKIP` (20).

## `core/errors.py`
`Cancelled` - raised to unwind processing when cancellation was requested (by the processor between pages, by the engine's forward pre-hooks inside a page).

## `core/paper.py`
`PAPER_SIZES_IN`, `PAGE_SIZE_CHOICES` (`original`, `auto`, A0-A6, B4-B6, C4, C5, DL, Letter, Legal, Tabloid, Executive), `paper_size_pt(name, landscape)`,
`pixels(name, dpi)`, `nearest_paper_size(w_px, h_px, dpi)` (orientation-agnostic, returns name + relative error), `normalize_page_size()`.

## `core/enhance.py`
Optional super-resolution. `super_resolution_available()` (is `super-image` installed?), `upscale_tiled(array, fn, scale, tile, overlap)` (pure tiled stitching),
`SuperResolution(scale, model_name, device, tile, overlap, model=None)` with `upscale(PIL image)` - EDSR from `super-image`, tiles halved on out-of-memory, grey input stays grey.
`Upscaler` is the protocol `OCRProcessor` uses (`scale`, `upscale()`); tests inject fakes.

## `core/results.py`
- `Status` - `SUCCESS | PARTIAL | FAILED | CANCELLED`.
- `FileResult(source, status, outputs, pages_total, pages_ok, failed_pages, skipped_pages, error)`.
- `BatchResult(session_dir, files, cancelled, input_root, warnings)` - `warnings` are non-fatal notes (e.g. enhancement unavailable); `ok`, `succeeded`, `failed`, `count()`, `confirmed_sources()` (what may be archived), `failure_lines()`, `summary()`.
- `ProgressEvent(files_done, total_files, current, stage, page, page_count)` - `.fraction` is the overall progress in [0, 1]; `files_done` only counts **finished** files.

## `core/hardware.py`
- `resolve_device(preference)` -> `DeviceReport(device, reason, gpus, warnings)`; runs a CUDA smoke kernel; warns below 8 GB VRAM.
- `GpuInfo(index, name, total_mb, capability, uuid, pci)`; `DeviceReport.primary` is the GPU the model runs on (`current_index`).
- `HardwareMonitor.snapshot(device, gpu)` -> `HardwareSnapshot(device_text, usage_text, memory_text)` via NVML (`pynvml`), falling back to `torch.cuda.mem_get_info`; CPU mode shows psutil CPU/RAM. Unavailable values show `N/A`. The torch GPU is matched to its NVML handle **by UUID, then PCI address, then index** (the two libraries can number GPUs differently).

## `core/ocr_engine.py` - `OCREngine`
Wraps `doctr.models.ocr_predictor`. `OCREngine(det, reco, device_preference, pretrained=True, straighten_pages=False, cpu_threads=0, predictor_kwargs=None)`;
`predict(list_of_rgb_arrays)` (OOM recovery, see [architecture.md](../architecture.md) 3.3); `set_models()`, `set_device_preference()`, `set_straighten()`
(rebuild the predictor), `set_cpu_threads()`, `gpu`, `gpu_name`, `release()`.
`cancel_check` (a callable) is polled by forward pre-hooks on DocTR's networks before every forward pass; returning `True` raises `Cancelled`.

## `core/pdf_writer.py`
`OcrWord` (bounding box + optional rotated `quad`), `OcrPage`, `ocr_page_from_doctr()`, `resolve_dpi()`, `normalize_for_pdf()`,
`effective_dpi()`, `draw_text_layer()` (invisible text; straight words scaled to their box, rotated words via a rotated text matrix;
`page_font()` chooses unembedded Helvetica, or the embedded DejaVu Sans when the page has non-WinAnsi characters),
`write_image_page_pdf()` (image + text; `compress_type` = `None | "jpeg"`, `quality`, `passthrough` JPEG file, `paper_pt` to fit the image on a fixed paper),
`write_text_overlay_pdf()`, `merge_pdfs()` (atomic). Sets `reportlab.rl_config.useA85 = 0`.

## `core/pdf_input.py`
`PdfSource` (pypdfium2: `page_count`, `has_text(i)`, `render(i, dpi)`), `apply_text_layers(src, out, {page_index: OcrPage})` (pypdf overlay; handles `/Rotate` and cropbox offsets; password-protected files raise).

## `core/archiver.py`
`archive_inputs(result, archive_dir) -> ArchiveReport(moved, errors)`, `is_inside(child, parent)`. See the archiving rules in [architecture.md](../architecture.md) 3.5.

## `core/ocr_processor.py` - `OCRProcessor`
```python
p = OCRProcessor(detection_model="db_resnet50", recognition_model="parseq", device="auto",
                 pretrained=True, cpu_threads=0, engine=None)   # engine= injects a (fake) engine for tests
p.configure(output_formats=["pdf", "hocr"], dpi=None, compress_enabled=False, compression_type="jpeg",
            compression_quality=85, skip_text_pages=True,
            page_size="original", enhance_enabled=False, enhance_scale=2, enhance_min_dpi=150)
p.progress_callback = lambda event: ...        # called on the processing thread
batch = p.process_folder(folder, output_dir)   # or p.process_file(path, output_dir)
p.cancel()                                     # any thread; reset automatically at the next run
```
Other members: `discover_inputs(folder)`, `set_models()`, `set_device()`, `set_straighten()`, `device`, `is_cancelled`, `release()`. The constructor also accepts `straighten_pages=` and `upscaler_factory=` (`(scale, device) -> Upscaler`, for tests).
Input discovery is recursive, folder by folder, natural order. Images of one folder become one PDF; PDFs are processed individually.
