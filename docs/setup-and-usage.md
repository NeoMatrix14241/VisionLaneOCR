# Setup, running, testing, building

## Requirements

- Windows 10/11 recommended (the code is portable; build scripts and CI target Windows).
- **Python 3.12+** (see [modernization.md](modernization.md)).
- NVIDIA GPU optional. 8 GB VRAM recommended; smaller cards fall back to the CPU per page when they run out of memory.
- **No Ghostscript, Poppler or Tesseract** is needed any more (PDF rendering uses `pypdfium2`, compression is done by Pillow/reportlab).

## Install and run
```bat
deploy_pyvenv.bat                 :: creates .venv, installs requirements.txt (checks for Python 3.12+)
.venv\Scripts\python main.py
```
GPU users (NVIDIA): run `deploy_pyvenv.bat gpu`, or install `requirements-torch-gpu.txt` (CUDA build from PyTorch's index) *before* `requirements.txt`.
`requirements.txt` uses ranges for torch/torchvision, so re-running it never replaces a CUDA build by the CPU wheel. Switching an existing
CPU venv to CUDA: `pip uninstall -y torch torchvision`, then `pip install -r requirements-torch-gpu.txt`.

Diagnostics (versions, GPU, model cache, config/log folders):
```bat
.venv\Scripts\python -m utils.diagnostics
```
Pre-download the default models (optional, the app also does it on first start): `python verify_models.py`.

## Using the GUI
1. Pick a tab: **Single File** (an image or a PDF), **Folder** (scanned recursively), **PDF**.
2. Choose the input and an output directory (must not be inside the input folder). Optionally enable **Archiving?** and choose an archive directory.
3. Options: DPI, output format, **Device** (Auto / GPU / CPU), detection and recognition model (a ⬇ marks models that are not downloaded yet; selecting one downloads and loads it in the background), compression (JPEG quality / lossless), skip PDF pages that already have text.
   Page handling (off by default): **Page size** (Original / Auto / A4 / Letter ...), **Enhance low-resolution pages** (greyed out unless `pip install super-image`), **Rotated / skewed pages** (reloads the models; downloads two extra models on first use).
4. **Start Processing.** The status line shows the file and page being worked on; the bar shows overall progress.
5. **Cancel** finishes the current page, writes no half-finished PDFs and archives nothing.
6. A summary dialog reports successes and lists every failed file with the reason; *Open Output Folder* jumps to the session folder.

Supported inputs: `.tif .tiff .jpg .jpeg .jpe .jfif .png .bmp .dib .gif` (multi-page TIFF supported), `.heic .heif` when
`pillow-heif` is installed, and `.pdf`.

## Command line
```bat
python run_ocr.py INPUT OUTPUT_DIR [--formats pdf hocr] [--dpi 300] [--compress jpeg|lossless] [--quality 80]
                  [--device auto|gpu|cpu] [--det db_resnet50] [--reco parseq] [--no-skip-text] [--archive DIR]
                  [--page-size original|auto|A4|Letter|...] [--enhance [--enhance-scale 2|3|4] [--enhance-min-dpi 150]]
                  [--straighten]
```
INPUT is a folder, an image or a PDF. Exit code 0 = everything OK, 1 = some files failed, 2 = cancelled / bad arguments.

## Logs
`%LOCALAPPDATA%\VisionLaneOCR\logs\` (see [configuration.md](configuration.md)). Use `python main.py --debug` for verbose output.

## Tests and lint
```bat
.venv\Scripts\pip install -r requirements-dev.txt
.venv\Scripts\python -m pytest        :: 96 tests; headless; fake engine plus real-DocTR tests
run_mypy.bat                          :: static type check (mypy.ini)
run_pylint.bat                        :: read-only; writes pylint_report.txt
```
`tests/test_integration_doctr.py` runs the real DocTR pipeline with random weights (no download) and is skipped
automatically when torch/doctr are not installed. On machines without PyQt6/torch the affected tests are skipped or use a stub.

## Build

| Script | Output | Notes |
|--------|--------|-------|
| `build_pyinstaller.bat` | `dist_pyinstaller\dist\VisionLaneOCR\` | runs `verify_models.py`, then `pyinstaller --clean build.spec` |
| `build_nuitka.bat` | `dist_nuitka\main.dist\` | standalone build; Nuitka may download MinGW64 on first use; ccache optional |
| `reset_cleanup_temp.bat` | - | deletes build output and generated reports |
| `reset_env_pycache.bat` | - | deletes `.venv` and all `__pycache__` / `.pyc` |
| `run_pylint.bat` | `pylint_report.txt` | read-only lint |
| `run_mypy.bat` | - | static type check (`mypy.ini`; needs `requirements-dev.txt`) |

**Both build scripts were rewritten without access to Windows - do a test build and run the result on a clean machine (VL-066).**

## CI
`.github/workflows/pylint.yml` runs on `windows-latest` for Python 3.12 and 3.13: installs `requirements-dev.txt`, runs
pytest (offscreen Qt), mypy and pylint with `--fail-under=9.0`. A failing step fails the build.
