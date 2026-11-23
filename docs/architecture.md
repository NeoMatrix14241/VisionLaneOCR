# Architecture

VisionLane OCR is a desktop application (PyQt6) that batch-converts scanned images and PDFs into **searchable PDFs**
and/or **hOCR** files using [DocTR](https://github.com/mindee/doctr) (PyTorch) for text detection + recognition.

```
repo
├── main.py              GUI entry point (splash -> background import -> MainWindow)
├── run_ocr.py           Command line interface (same OCRProcessor API)
├── verify_models.py     Pre-downloads the default models (used by the build scripts)
├── core/                OCR engine, orchestration, PDF reading/writing, archiving (no Qt)
├── gui/                 PyQt6 window, input tabs, background workers, theme, dialogs
├── utils/               config, paths, logging, model catalogue, diagnostics
├── assets/fonts/        DejaVu Sans (+ licence): Unicode text layer for non-WinAnsi pages
├── data/                CSV generator for the paper-size table (the table itself is core/paper.py)
├── tests/               pytest suite (headless)
└── build.spec, build_*.bat, mypy.ini, .github/workflows/pylint.yml
```

## 1. Layers

```mermaid
flowchart TD
    subgraph GUI["gui/ (PyQt6, main thread)"]
        MW[MainWindow]
        IT[InputTab x3]
        WK["workers: Task / OCRJob (daemon threads)"]
    end
    subgraph CORE["core/ (no Qt)"]
        OP[OCRProcessor]
        EN[OCREngine]
        PW[pdf_writer]
        PI[pdf_input]
        AR[archiver]
        HW[hardware]
    end
    subgraph UTIL["utils/"]
        CFG[app_config / app_paths / logging_config / models]
    end
    DOCTR[(DocTR + PyTorch)]
    MW --> IT
    MW -->|starts| WK
    WK -->|process_folder / process_file| OP
    OP --> EN --> DOCTR
    OP --> PW
    OP --> PI
    MW -->|after the run| AR
    MW --> HW
    MW --> CFG
```

The only interface between GUI and engine is: `OCRProcessor(...)`, `configure(...)`, `progress_callback`,
`process_folder()/process_file()`, `cancel()`, and the returned `BatchResult`. `core/` never imports Qt, so it can be
used from the CLI and tested without a display; `core.ocr_processor` does not even import torch unless it has to build
an engine (tests inject a fake engine).

## 2. Startup sequence (`main.py`)

1. Load `config.ini` (`AppConfig`), set up logging once (`utils/logging_config.setup_logging`), install crash handlers.
2. Create `QApplication`, show the splash immediately.
3. A **background thread** imports `torch` and `doctr.models` (the slow part) while the splash animates.
4. When the import finishes, create `MainWindow` and show it. `MainWindow` builds the UI from the config and starts an
   `engine-load` task on a background thread: download missing models, build `OCRProcessor` (device detection, CUDA
   smoke test, model load). The *Start* button is enabled when it is ready; failures are shown in a dialog.
5. Exit: `closeEvent` (asks if a job runs -> cancel -> wait up to 30 s -> save config) -> `app.exec()` returns.

There are no `sleep`s, no import-time patches and no second splash.

## 3. Processing pipeline

```mermaid
flowchart TD
    A[Start Processing] --> B[validate paths in the GUI]
    B --> C["OCRJob thread: processor.process_folder / process_file"]
    C --> D[discover inputs: natural order, folder by folder]
    D --> E[create OCR_Session_timestamp + private temp dir]
    E --> F{group}
    F -->|images of one folder| G[for each file: for each frame -> OCR -> page PDF in temp]
    G --> H["merge page PDFs once per folder -> pdf/.../folder.pdf"]
    F -->|PDF file| I["render page (pypdfium2) -> OCR -> overlay text on original page (pypdf)"]
    H --> J[BatchResult]
    I --> J
    J --> K["GUI: summary dialog, archive confirmed inputs"]
```

### 3.1 One image page (`OCRProcessor._process_image_group`)
1. `_open_frames`: Pillow opens the file; every frame of a multi-frame TIFF is a page; EXIF orientation applied; DPI = user value, else image metadata (30-4800), else 300.
2. `_prepare_frame`: **page size** - the group's paper (`_group_paper_name`: a fixed paper, or for *Auto* the paper most pages are nearest to) is chosen with the page's orientation; **enhancement** - if enabled and the page's *effective* resolution on that paper is below `enhance_min_dpi`, the image is upscaled by `core/enhance.py` (tiled super-resolution) and its DPI multiplied so the physical size is unchanged. A missing package / failing model switches enhancement off for the run and adds a warning to the result.
3. The image is converted to an RGB array and passed to `OCREngine.predict()` (`torch.inference_mode()`; CUDA OOM recovery, see 3.3).
4. The DocTR result is converted to `OcrPage` (words in pixel coordinates; rotated words keep their 4 corners) for the PDF. For HOCR, polygon geometries are first replaced by bounding boxes (DocTR's exporter only accepts those) and `export_as_xml()` is called.
5. `pdf_writer.write_image_page_pdf`: page size = pixels x 72 / dpi, **or the chosen paper** with the image scaled to fit and centred; the image is embedded (JPEG-recompressed at the chosen quality, Flate for lossless, or the original JPEG bytes if nothing changes); the text is drawn invisibly (render mode 3): straight words scaled to their box, rotated words with a rotated text matrix, in Helvetica - or in embedded DejaVu Sans on pages that contain characters Helvetica cannot encode.

### 3.2 PDF input (`core/pdf_input.py`)
`PdfSource` (pypdfium2) reports the page count, whether a page already has text (>= 20 characters) and renders a page at
the chosen DPI (default 300, capped at 40 megapixels). After OCR, `apply_text_layers` clones the source with pypdf and
merges one text-only overlay page per recognised page (the cropbox offset and `/Rotate` are handled). Pages that were
skipped, or failed, are copied unchanged, so the output always has every page of the input.

### 3.2a Rotated / skewed pages (`straighten_pages`)
`OCREngine` builds the predictor with `assume_straight_pages=False, straighten_pages=True, preserve_original_coords=True`.
DocTR rotates/deskews a copy of the page for recognition and maps every word box back onto the **original** page as a
4-point polygon (corners in reading order), so the scan itself is never modified and the text layer lines up with it.
This needs DocTR's page- and crop-orientation models (downloaded on first use) and is slower, so it is off by default.

### 3.3 Out-of-memory handling (`core/ocr_engine.py`)
`predict` -> on CUDA OOM: `empty_cache()`, retry on the GPU -> still OOM: run **this call** on the CPU and move the model
back to the GPU. Any other CUDA error switches to the CPU for the rest of the session. At start-up a one-kernel smoke
test catches GPUs the installed PyTorch cannot run, and a notice is shown when the GPU has less than 8 GB.

### 3.4 Output layout
```
<output>/OCR_Session_YYYYMMDD_HHMMSS/
    pdf/<relative folder>/<folder name>.pdf     all images of a folder, natural name order (root folder: <input folder name>.pdf)
    pdf/<relative folder>/<name>.pdf            PDF inputs: original pages + text layer
    pdf/<name>.pdf                              single image / single PDF
    hocr/<relative folder>/<image name>.hocr    per image (multi-frame TIFF: <name>_p0002.hocr; clashes: <name>_<ext>.hocr)
    hocr/<relative folder>/<pdf name>/<pdf name>_page_0001.hocr
```
Name clashes anywhere in a session get ` (2)`, ` (3)`... The session folder is only created when there are inputs.
Intermediate page PDFs live in a private `tempfile.TemporaryDirectory` that is removed at the end of every run.

### 3.5 Results and failure policy
Every input file produces a `FileResult` with a `Status`:

| Status | When | Archived? |
|--------|------|-----------|
| `SUCCESS` | all pages processed and written | yes (if archiving is on) |
| `PARTIAL` | some pages failed; the output contains the others | no |
| `FAILED` | unreadable file, no page worked, or the merge failed | no |
| `CANCELLED` | not (completely) processed because of Cancel | no (and nothing is archived after a cancel) |

A cancelled run does not write the merged PDF of the folder that was in progress.

## 4. Threading and cancellation

| Thread | What runs |
|--------|-----------|
| Main (Qt) | UI, the 1 s hardware read-out (`HardwareMonitor`), signal handlers, archiving (fast file moves) |
| `preload` (daemon) | `import torch, doctr` during start-up |
| `engine-load` / `engine-reload` (daemon) | model download, engine construction, model/device switches |
| `ocr-job` (daemon) | the whole batch, sequentially |
| `count-files` (daemon) | counting images/PDFs in a selected folder |

All are plain `threading.Thread` daemons wrapped in `gui.workers.Task`; results come back to the GUI thread as Qt signals.

**Lifetime rules** (a violation is a crash, not an exception): a `Task` has no Qt parent (a widget could otherwise delete it
under a running thread) and its signals are connected to *bound methods of QObjects*, never to lambdas (Qt disconnects those
when the receiver is destroyed); slots invoked from timers must not raise (PyQt6 aborts the process).

**Cancellation** is cooperative (`threading.Event`), checked between pages and files, and - during recognition - before
every model forward pass: `OCREngine` registers a forward pre-hook on DocTR's detection, recognition and orientation
networks that raises `Cancelled` when the flag is set, so Cancel acts between model stages and recognition batches. A single
forward pass that has already started still finishes (VL-065).
