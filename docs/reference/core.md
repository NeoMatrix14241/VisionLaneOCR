# `core/` reference

`core/` has no Qt dependency. `core/__init__.py` is empty on purpose: importing the package is cheap, and torch is only
imported by `core.ocr_engine` / `core.hardware`.

## `core/constants.py`
`PDF_EXTENSIONS`, `image_extensions()` (base formats, plus `.heic/.heif` when `pillow-heif` can be registered), `is_image()`, `is_pdf()`,
`natural_key()` (natural sort key), model fallback lists and defaults (`db_resnet50`, `parseq`), `RECOMMENDED_VRAM_GB = 8`,
DPI plausibility limits (30-4800, default 300), `PDF_RENDER_MAX_PIXELS` (40 MP), `MIN_TEXT_CHARS_FOR_SKIP` (20).

## `core/results.py`
- `Status` - `SUCCESS | PARTIAL | FAILED | CANCELLED`.
- `FileResult(source, status, outputs, pages_total, pages_ok, failed_pages, skipped_pages, error)`.
- `BatchResult(session_dir, files, cancelled, input_root)` - `ok`, `succeeded`, `failed`, `count()`, `confirmed_sources()` (what may be archived), `failure_lines()`, `summary()`.
- `ProgressEvent(files_done, total_files, current, stage, page, page_count)` - `.fraction` is the overall progress in [0, 1]; `files_done` only counts **finished** files.

## `core/hardware.py`
- `resolve_device(preference)` -> `DeviceReport(device, reason, gpus, warnings)`; runs a CUDA smoke kernel; warns below 8 GB VRAM.
- `HardwareMonitor.snapshot(device, gpu_index, gpu_name)` -> `HardwareSnapshot(device_text, usage_text, memory_text)` via NVML (`pynvml`), falling back to `torch.cuda.mem_get_info`; CPU mode shows psutil CPU/RAM. Unavailable values show `N/A`.

## `core/ocr_engine.py` - `OCREngine`
Wraps `doctr.models.ocr_predictor`. `OCREngine(det, reco, device_preference, pretrained=True, assume_straight_pages=True, cpu_threads=0)`;
`predict(list_of_rgb_arrays)` (OOM recovery, see [architecture.md](../architecture.md) 3.3); `set_models()`, `set_device_preference()`
(rebuild the predictor), `set_cpu_threads()`, `gpu_name`, `release()`.

## `core/pdf_writer.py`
`OcrWord`, `OcrPage`, `ocr_page_from_doctr()`, `resolve_dpi()`, `normalize_for_pdf()`, `draw_text_layer()` (invisible text, horizontally
scaled to each word box), `write_image_page_pdf()` (image + text; `compress_type` = `None | "jpeg"`, `quality`, `passthrough` JPEG file),
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
            compression_quality=85, skip_text_pages=True)
p.progress_callback = lambda event: ...        # called on the processing thread
batch = p.process_folder(folder, output_dir)   # or p.process_file(path, output_dir)
p.cancel()                                     # any thread; reset automatically at the next run
```
Other members: `discover_inputs(folder)`, `set_models()`, `set_device()`, `device`, `is_cancelled`, `release()`.
Input discovery is recursive, folder by folder, natural order. Images of one folder become one PDF; PDFs are processed individually.
