# Dependency modernization guide

`requirements.txt` pins the newest releases available on PyPI (checked Nov 2026). The old file targeted
Python 3.10–3.12, torch 2.7, python-doctr 0.11 and ocrmypdf 16. Upgrading was **not** a pure version bump: the code
needed the changes listed in section 2, **all of which are now done** in the refactor (see
[known-issues.md](known-issues.md)). Everything marked *verified* was checked against the real wheels/sources.

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
| pandas | 2.2.3 | removed | `data/dpi_res_matrix.py` no longer needs it |
| PyPDF2 | 3.0.1 | **pypdf 7.1.0** | PyPDF2 is end-of-life |
| ocrmypdf | 16.10.2 | **removed** | 17.x removed `HocrTransform`; code never used it successfully |
| reportlab | 4.4.1 | 5.1.0 | |
| psutil | 7.0.0 | 7.2.2 | |
| nvidia-ml-py3 | 7.352.0 | **nvidia-ml-py 13.615.71** | same `import pynvml`; returns `str` not `bytes` |
| GPUtil | 1.4.0 | removed | needs `distutils`; NVML already covers it |
| wmi | 1.5.1 | removed | no longer imported |
| pdf2image | 1.17.0 | removed | replaced by `pypdfium2` (5.14.0, also a hard dependency of doctr) |
| tqdm | 4.67.1 | removed | only the deleted compressor used it |
| requests, lxml | undeclared | removed | no longer imported |
| pypdfium2 | (transitive) | 5.14.0 (declared) | PDF rendering in `core/pdf_input.py` |
| pillow-heif | - | 1.8.1 (optional, commented) | enables `.heic/.heif`; not covered by tests |
| pyinstaller / nuitka / pylint / zstandard / pytest | in runtime file | 6.22.3 / 4.3.5 / 4.1.4 / 0.25.0 / 9.2.0 in `requirements-dev.txt` | build and test tooling |
| python-magic, python-magic-bin, colorama, typing_extensions | listed | removed | never imported |

**CUDA:** PyPI's Windows torch wheel is CPU-only. Install the CUDA build from the PyTorch index first (see header of
`requirements.txt`). Recent CUDA-13 wheels no longer support Maxwell/Pascal/Volta GPUs; GTX 1000-series cards need a CUDA 12.x build. The app runs a one-kernel smoke test at start-up and falls back to the CPU (with the reason in the log) if the installed PyTorch cannot run on the GPU. (I could not reach `download.pytorch.org` from the analysis sandbox, so confirm
which CUDA tags exist for torch 2.14.1.)

Resolver note: I confirmed the declared constraints are mutually compatible (e.g. scipy `numpy<2.8,>=2`,
opencv-python `numpy>=2`, doctr `numpy<3`); a full `pip install` of the CUDA stack was not run in the sandbox.

## 2. Code changes this upgrade required (all done)

| Change | Where it ended up |
|--------|-------------------|
| Remove the `ocrmypdf` import and the "try ocrmypdf" branch | `utils/hocr_to_pdf.py` deleted; PDF writing is `core/pdf_writer.py` (reportlab only) |
| PyPDF2 `PdfMerger` -> `pypdf.PdfWriter.append` (*verified: `PdfMerger` does not exist in pypdf 7.1.0*) | `core/pdf_writer.merge_pdfs`, `core/pdf_input.apply_text_layers` |
| Delete the DocTR backend patching (*verified: `doctr/file_utils.py` in 1.1.0 only exports `requires_package` and `CLASS_NAME`*) | `core/doctr_patch.py`, `core/doctr_torch_setup.py` and the `USE_TORCH` / `DOCTR_BACKEND` handling in `main.py` deleted |
| Delete the CUDA workaround layer | all `core/*cuda*_patch*.py`, `cuda_compat_plugin.py`, `hardware_monitoring_patch.py` deleted |
| pynvml returns `str` | `core/hardware.HardwareMonitor` (no `.decode()`) |
| GPUtil removed | NVML via `nvidia-ml-py` only |
| DocTR model list | read from `doctr.models.detection.zoo.ARCHS` / `recognition.zoo.ARCHS` (`utils/models.py`), with a built-in fallback list |
| Python 3.12 baseline | `deploy_pyvenv.bat` checks it; CI matrix is 3.12 / 3.13 on Windows; `run_ocr.py` rewritten |

DocTR 1.1.0 facts the code relies on (read from the 1.1.0 source **and** confirmed by running the real DocTR 1.1.0 on the CPU with random weights; pretrained weights could not be downloaded in the sandbox):
`ocr_predictor(det_arch=, reco_arch=, pretrained=, assume_straight_pages=)`; the predictor is an `nn.Module` called with a
list of RGB `numpy` arrays; `Word.geometry` is `((xmin, ymin), (xmax, ymax))` relative to the page (4 points for rotated
pages); `Document.export_as_xml()` returns one `(bytes, ElementTree)` per page (hOCR); weights are cached in
`$DOCTR_CACHE_DIR` or `~/.cache/doctr` under `models/<name>-<hash>.pt` (the app sets `DOCTR_CACHE_DIR` to `<app>/doctr_cache` itself - see [configuration.md](configuration.md)); DocTR reads the variable at download time, so setting it before the first model load is enough. Two traps found while testing: `pretrained=False` still downloads *backbone* weights unless `pretrained_backbone=False` is also passed, and `assume_straight_pages=False` downloads an extra crop-orientation model.

### Smoke test after upgrading
```bat
python -m utils.diagnostics
python verify_models.py
python -m pytest
python main.py
```
Then OCR a folder with 3 images and check the page count and that the text is selectable in the output PDF.

## 3. Simplifications this stack enabled (done)

- **PDF input without Ghostscript/Poppler:** `pypdfium2` rasterises pages; the text layer is overlaid on the original pages (VL-016).
- **Compression without Ghostscript:** page images are JPEG-encoded with Pillow while the PDF is written (VL-014).
- **Packaging:** with the patch layer gone, `build.spec` no longer copies torch/CUDA DLLs by hand; PyInstaller and Nuitka use their maintained torch/PyQt6 support (untested on Windows - VL-066).
