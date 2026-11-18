# Dependency modernization guide

`requirements.txt` now pins the newest releases available on PyPI (checked Nov 2026). The old file targeted
Python 3.10–3.12, torch 2.7, python-doctr 0.11 and ocrmypdf 16. Upgrading is **not** a pure version bump: the
code needs the small changes below. Everything marked *verified* was checked against the real wheels.

## 1. Version table

| Package | Old | New | Notes |
|---------|-----|-----|-------|
| Python | 3.10–3.12 | **≥ 3.12** | numpy 2.5.5 and scipy need 3.12; doctr/torch/Pillow need 3.11 |
| torch | 2.7.0+cu128 | 2.14.1 | newest is 2.15.0, but newest torchvision (0.29.2) is the 2.14 build; keep the pair matched |
| torchvision | 0.22.0+cu128 | 0.29.2 | |
| torchaudio | 2.7.0 | removed | never imported |
| python-doctr | 0.11.0 (`[torch]`) | 1.1.0 | torch-only; no `[torch]` extra; `onnx`, `scipy`, `opencv-python`, `pypdfium2`, `huggingface-hub` are hard deps |
| PyQt6 | 6.9.0 | 6.12.1 | |
| numpy | 2.2.6 | 2.5.5 | |
| Pillow | 11.2.1 | 13.0.0 | |
| pandas | 2.2.3 | 3.1.1 (dev only) | only `data/dpi_res_matrix.py` |
| PyPDF2 | 3.0.1 | **pypdf 7.1.0** | PyPDF2 is end-of-life |
| ocrmypdf | 16.10.2 | **removed** | 17.x removed `HocrTransform`; code never used it successfully |
| reportlab | 4.4.1 | 5.1.0 | |
| psutil | 7.0.0 | 7.2.2 | |
| nvidia-ml-py3 | 7.352.0 | **nvidia-ml-py 13.615.71** | same `import pynvml`; returns `str` not `bytes` |
| GPUtil | 1.4.0 | removed | needs `distutils`; NVML already covers it |
| wmi | 1.5.1 | 1.5.1 (Windows marker) | |
| pdf2image | 1.17.0 | 1.17.0 | optional fallback |
| tqdm | 4.67.1 | 4.70.1 | |
| requests, lxml | undeclared | 2.34.2, 6.1.3 | were used but never listed |
| pyinstaller / nuitka / pylint / zstandard | in runtime file | 6.22.3 / 4.3.5 / 4.1.4 / 0.25.0 in `requirements-dev.txt` | build tooling |
| python-magic, python-magic-bin, colorama, typing_extensions | listed | removed | never imported |

**CUDA:** PyPI's Windows torch wheel is CPU-only. Install the CUDA build from the PyTorch index first (see header of
`requirements.txt`). Recent CUDA-13 wheels no longer support Maxwell/Pascal/Volta GPUs; the README's "GTX 1000-series
maybe works" needs a CUDA 12.x build. (I could not reach `download.pytorch.org` from the analysis sandbox, so confirm
which CUDA tags exist for torch 2.14.1.)

Resolver note: I confirmed the declared constraints are mutually compatible (e.g. scipy `numpy<2.8,>=2`,
opencv-python `numpy>=2`, doctr `numpy<3`); a full `pip install` of the CUDA stack was not run in the sandbox.

## 2. Required code changes

### 2.1 Remove the ocrmypdf import — `core/ocr_processor.py:24`
```python
- from ocrmypdf.hocrtransform import HocrTransform   # unused name; ImportError on ocrmypdf >= 17
```
In `utils/hocr_to_pdf.py` delete the whole "try ocrmypdf" branch (lines 161-176) and have `hocr_to_pdf()` call
`convert_hocr_to_pdf()` directly. Also drop `ocrmypdf*` from `build.spec` and `build_nuitka.bat`.

### 2.2 PyPDF2 → pypdf — `core/ocr_processor.py:28,145-155,1009-1019`
*Verified: PdfMerger does not exist in pypdf 7.1.0; PdfWriter.append merges correctly.*
```python
from pypdf import PdfWriter
w = PdfWriter()
for pdf in temp_pdfs:
    w.append(str(pdf))
w.write(str(output_pdf))
w.close()
```
Update `build.spec` / `build_nuitka.bat` (`PyPDF2` → `pypdf`).

### 2.3 Delete the DocTR backend patching
doctr 1.1.0's `doctr/file_utils.py` only exports `requires_package` and `CLASS_NAME` (*verified*). Remove:
- `core/doctr_patch.py`, `core/doctr_torch_setup.py` and their imports in `core/__init__.py`, `core/cuda_patch_wrapper.py`;
- the `doctr_patch` / `doctr_torch_setup` steps and `ENV_VARS_TRUE_VALUES` mock code in `main.py` (lines ~175-205, 267-285);
- `from doctr.file_utils import is_torch_available, is_tf_available` in `utils/system_diagnostics.py:160`;
- `USE_TORCH` / `DOCTR_BACKEND` environment variables (harmless, but dead).

### 2.3b Delete the CUDA workaround layer
With stock PyTorch wheels, remove `core/cuda_env_patch.py`, `cuda_patch_wrapper.py`, `nuitka_cuda_patch.py`,
`runtime_cuda_patch.py`, `cuda_compat_plugin.py`, `hardware_monitoring_patch.py` and the calls in `main.py:145-173`
and `ocr_processor.py:6-20`. If a *real* driver/toolkit problem reappears in a compiled build, fix it at the
packaging level instead of faking `torch.version.cuda` (VL-008, VL-010).

### 2.4 pynvml returns `str`
`core/ocr_processor.py:97,103`: remove `.decode()` (or `if isinstance(x, bytes)`).

### 2.5 GPUtil
Remove the fallback in `gui/main_window.py:42-47,1817-1828` and `build*` lists; NVML already provides utilisation and memory.

### 2.6 DocTR model list
Optionally add `db_resnet34` (detection) and `viptr_tiny` (recognition), new in doctr 1.x. `ocr_predictor` still accepts
`det_arch`, `reco_arch`, `pretrained`, `assume_straight_pages`; `DocumentFile.from_images`, `export_as_xml` still exist (*verified*).
Weights still download to `~/.cache/doctr/models` (or `$DOCTR_CACHE_DIR/models`).

### 2.7 Python version fixes
`run_ocr.py` uses `datetime.UTC` (3.11+). The CI matrix (`.github/workflows/pylint.yml`) should become `["3.12", "3.13"]`
on `windows-latest`, and `run_pylint.bat`/README "Python 3.10" text should be updated.

### 2.8 Smoke test after upgrading
```bat
python -c "import torch, doctr, PyQt6, pypdf, reportlab, lxml, pynvml; print(torch.__version__, torch.cuda.is_available(), doctr.__version__)"
python verify_models.py
python main.py
```
Then OCR a folder with 3 images and check page count and text selectable in the output PDF.

## 3. Optional simplifications enabled by the new stack

- **PDF input without Ghostscript/Poppler:** `pypdfium2` ships with doctr; use it to rasterise and overlay the text layer on the original page (VL-016).
- **Compression without Ghostscript:** encode page images as JPEG with Pillow before embedding (VL-014).
- **Packaging:** with the patch layer gone, `build.spec` can drop the manual torch/CUDA DLL copying; PyInstaller ships maintained hooks for torch and PyQt6.
