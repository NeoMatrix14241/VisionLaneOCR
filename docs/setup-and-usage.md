# Setup, running, building

## Requirements

- Windows 10/11 (code has Windows-specific paths: `gswin64c.exe`, `wmi`, `CREATE_NO_WINDOW`; most of the logic is portable).
- **Python 3.12+** (see [modernization.md](modernization.md)).
- NVIDIA GPU optional (CPU fallback exists). README advises ≥ 8 GB VRAM; below 6 GB it crashes often.
- [Ghostscript](https://www.ghostscript.com/releases/gsdnld.html) — needed for PDF **input** and for the (weak) compression feature. Detected via `PATH` then `C:\Program Files[ (x86)]\gs\gs*\bin\gswin64c.exe`, highest version wins.
- Poppler only if you rely on the `pdf2image` fallback.

## Install and run
```bat
python -m venv .venv
.venv\Scripts\activate
python -m pip install --upgrade pip
:: GPU users: install torch/torchvision from the CUDA index first (see requirements.txt header)
pip install -r requirements.txt
python verify_models.py     :: optional, pre-downloads db_resnet50 + parseq
python main.py
```
`deploy_pyvenv.bat` automates the venv + install (message still says "Python 3.8+").

## Using the GUI
1. Pick a tab: **Single File** and **PDF** are currently broken (VL-001/002); use **Folder**.
2. Choose input folder (recursively scanned for images and PDFs), output directory, optional archive directory.
3. Options: DPI (Auto = from image metadata else 300), output format (PDF / HOCR / PDF+HOCR), detection and recognition model (⬇️ = not cached, downloads on selection), compression.
4. **Start Processing**. Output goes to `<output>/OCR_Session_<timestamp>/…` (see [architecture.md](architecture.md)).
5. **Cancel** stops between images and deletes temp files.

Supported inputs: `.tif .tiff .jpg .jpeg .jpe .png .bmp .gif .dib .jiff` (+ `.heic`, not actually readable, VL-019) and `.pdf`.

## Headless script
`run_ocr.py` expects `input/images` and `input/pdfs` beside the script and writes to `output/`. It currently fails
(VL-001, VL-051) until `input_path`/`dpi` handling is fixed.

## Logs
`<repo>/logs/`: `ocr_process_*.log` (startup), `ocr_gui_*.log` (GUI session), `debug_*.log`, `memory_*.log`, `crash_*.log`.

## Build

| Script | Output | Notes |
|--------|--------|-------|
| `build_pyinstaller.bat` | `dist_pyinstaller\dist\VisionLaneOCR\` | runs `verify_models.py`, then `pyinstaller --clean build.spec`; see VL-048 |
| `build_nuitka.bat` | `dist_nuitka\main.dist\` | standalone mode; detects MSVC/MinGW/clcache/ccache, writes a `USE_TORCH` marker file, copies CUDA/NVML DLLs; see VL-049 |
| `reset_cleanup_temp.bat` | – | deletes `temp_binaries`, `logs`, `dist_*`, lint temp files |
| `reset_env_pycache.bat` | – | deletes `.venv` and all `__pycache__`/`.pyc` |
| `run_pylint.bat` / `pylint_clean.py` | `pylint_cleanup.log` | **modifies source files** (VL-052) |

Both packagers reference modules and paths that changed (see VL-048/049); expect to repair them after the dependency upgrade.

## Lint / CI
`.github/workflows/pylint.yml` runs pylint on Ubuntu for Python 3.10–3.12 and never fails (`|| true`). `.pylintrc` disables most size/complexity checks.
