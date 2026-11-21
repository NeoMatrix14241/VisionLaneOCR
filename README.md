# VisionLane OCR

[EXPERIMENTAL RELEASE - UPDATED REGULARLY. KNOWN ISSUES: see [docs/known-issues.md](docs/known-issues.md)]

![icon_landscape_transparent_bg2](https://github.com/user-attachments/assets/c05d5fa0-7f75-4382-b300-8884ccbe196b)

An OCR (Optical Character Recognition) application built with DocTR and PyQt6, designed for high-throughput document processing with optional GPU acceleration.

![image](https://github.com/user-attachments/assets/743cbdf4-a95b-4129-adeb-ad7cd4154c10)
![image](https://github.com/user-attachments/assets/f7f58880-f111-4671-ad9e-aafdc233e78b)

![Apache License](https://img.shields.io/badge/license-Apache-blue.svg)
![Python 3.12+](https://img.shields.io/badge/python-3.12+-blue.svg)
![PyTorch](https://img.shields.io/badge/PyTorch-2.14-red.svg)
![GPU Support](https://img.shields.io/badge/GPU-CUDA_Support-green.svg)

## Features

- 🚀 GPU (CUDA) or CPU processing - **switchable in the GUI** (Device: Auto / GPU / CPU)
- 📁 Batch processing of whole folder trees, single images and PDFs
- 📄 Input: TIFF (including multi-page), JPEG, PNG, BMP, GIF, PDF; HEIC/HEIF if `pillow-heif` is installed
- 💾 Output: searchable PDF and/or HOCR
- 📑 PDF input keeps the **original pages** and adds an invisible text layer; pages that already contain text are skipped (optional)
- 🗜️ Real PDF image compression (JPEG with adjustable quality, or lossless) - no Ghostscript needed
- 🖥️ PyQt6 GUI with real-time progress, GPU/CPU metrics and Dark / Light / Night themes
- 🧩 Selectable DocTR detection and recognition models (downloaded on demand)
- 🛟 Failures are reported, never silent: a summary lists every file that failed and why
- ⏹️ Safe cancellation (finishes the current page, leaves no half-written PDFs)
- 📦 **Archiving option:** after a run, files that were processed successfully are moved to an archive folder (structure preserved). Failed files stay where they are; nothing is archived after a cancel.
- 🔒 Settings are saved to `config.ini` and restored on the next start
- 📝 Log files for troubleshooting

## Requirements

- **Windows 10 or higher** (the code is cross-platform, but it is built and tested for Windows)
- **Python 3.12 or newer**
- **GPU (optional):** NVIDIA GPU with CUDA drivers. **8 GB VRAM is recommended.** Smaller cards work: pages that run out of memory are retried and, if needed, processed on the CPU, and the app shows a notice. You can also choose *Device: CPU*.
  The default PyPI `torch` wheel for Windows is CPU-only - see the comments in [requirements.txt](requirements.txt) for installing the CUDA build.
- Ghostscript, Poppler and Tesseract are **not** needed.

## Roadmap / open items

Items marked `[x]` are done; `[ ]` are still open. Details and evidence live in [docs/known-issues.md](docs/known-issues.md).

### 🔧 Functional improvements
- [x] Add custom selection for DocTR text detection and recognition models
- [x] Persist user settings in `config.ini`
- [x] Archiving option for processed files
- [x] Support for common image formats (not only TIFF)
- [x] Add an option to switch GPU and CPU  
  _→ Device: Auto / GPU / CPU. Auto uses the GPU when a working CUDA device exists._
- [x] Check the recommended 8 GB VRAM and suggest the CPU  
  _→ Shown as a notice in the main window; out-of-memory pages are retried and fall back to the CPU one page at a time._
- [x] Fix "Single File" option
- [x] Fix "PDF" option
- [x] PDF compression actually compresses  
  _→ Page images are JPEG-encoded while the PDF is written; the quality slider has a measurable effect (covered by a test)._
- [x] Fix OCRmyPDF-related bugs / HOCR output  
  _→ OCRmyPDF and Ghostscript are no longer used at all._
- [ ] Integrate super-image (PyTorch) for image enhancement to standardise page sizes  
  _→ Not started. `data/dpi_res_matrix.py` contains the paper-size table (inches → pixels) as groundwork._
- [ ] Rotated / skewed page handling (`assume_straight_pages` is still `True`)
- [ ] Non-Latin text layer (the invisible text uses Helvetica, so characters outside WinAnsi become `?` in the PDF text layer; HOCR output is not affected)

### 📁 File handling
- [x] Enhanced PDF output pathing  
  _→ One PDF per image folder: `pdf/<relative folder>/<folder name>.pdf` (the extra subfolder is gone)._
- [x] Remove the `_ocr` suffix for PDF inputs  
  _→ `pdf/<relative folder>/<name>.pdf`_
- [x] Multi-page TIFFs keep all their pages (previously only the first page was read)
- [x] Natural file ordering (`img2` before `img10`)

### 🛠️ Quality / tooling
- [x] Automated tests (`python -m pytest`, 54 tests, run headless)
- [x] CI that can fail (Windows, Python 3.12/3.13)
- [ ] Static type checking (mypy/pyright) is not set up yet
- [ ] Windows build scripts (`build_nuitka.bat`, `build.spec`) were rewritten without access to Windows: **do a test build before releasing**
- [ ] Validation on real CUDA hardware of the new out-of-memory recovery path (covered by unit tests with a fake model only)

## Installation

1. Clone the repository:
```bash
git clone https://github.com/NeoMatrix14241/VisionLaneOCR.git
cd VisionLaneOCR
```

2. Create the virtual environment and install the dependencies (Python 3.12+):
```bat
deploy_pyvenv.bat
```
or manually:
```bash
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
```
For GPU support install the CUDA build of PyTorch first (see the notes at the top of `requirements.txt`).

3. (Optional) Check what the app sees (versions, GPU, model cache, config/log folders):
```bash
python -m utils.diagnostics
```

## Usage

```bash
python main.py            # add --debug for verbose logs
```

1. Choose the input mode (**Single File**, **Folder** or **PDF**) and select the input.
2. Choose an output directory. Results go to `<output>/OCR_Session_<timestamp>/`.
3. (Optional) Enable **Archiving?** and choose an archive directory.
4. Set DPI, output format, device, models and compression, then click **Start Processing**.

Command line (no GUI):
```bash
python run_ocr.py INPUT OUTPUT_DIR --formats pdf hocr --dpi 300 --compress jpeg --quality 80
```

### Output layout
```
OCR_Session_<timestamp>/
  pdf/<relative folder>/<folder name>.pdf     one PDF per image folder, pages in natural name order
  pdf/<relative folder>/<name>.pdf            PDF inputs (original pages + text layer)
  hocr/<relative folder>/<image name>.hocr    one HOCR file per image
```

### Where settings and logs live
`config.ini` and `logs/` are in `%LOCALAPPDATA%\VisionLaneOCR` (or next to the program if an empty file named `portable` exists beside it; or wherever `VISIONLANE_HOME` points). Downloaded DocTR models are stored in **`doctr_cache\models` next to the program** (git-ignored). Models already present in the old default location (`%USERPROFILE%\.cache\doctr\models`) are copied over once on first start. Set `DOCTR_CACHE_DIR` to use another folder; if the program folder is not writable, the per-user data folder is used.

### Archiving
- After a run, files with a **complete** output are moved to the archive directory, preserving the folder structure; name clashes get ` (2)`, ` (3)`...
- Files that failed stay in place. After a **cancelled** run nothing is moved.
- The output and archive folders must not be inside the input folder.

## Build

Both build scripts were rewritten for the new module layout. **They have not been run on Windows yet - test the resulting executable before distributing it.**

- **Nuitka** (`build_nuitka.bat`): standalone build in `dist_nuitka\main.dist`. Needs `pip install -r requirements-dev.txt`.
- **PyInstaller** (`build_pyinstaller.bat`, uses `build.spec`): output in `dist_pyinstaller\dist\VisionLaneOCR`.

Both scripts download the default DocTR models first (`verify_models.py`).

## Development

```bash
pip install -r requirements-dev.txt
python -m pytest          # headless; uses a fake OCR engine, plus a real-DocTR test when torch/doctr are installed
run_pylint.bat            # read-only lint run
```
See [docs/](docs/README.md) for architecture, configuration and the issue register.
