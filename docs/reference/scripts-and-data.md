# Entry points, scripts, tests, data, CI

| File | What it does |
|------|--------------|
| `main.py` | GUI entry point (see [architecture.md](../architecture.md) 2); `--debug` for verbose logging |
| `run_ocr.py` | Command line interface on the same `OCRProcessor` API; exit codes 0 / 1 / 2 |
| `verify_models.py` | Downloads the default models if missing (used by both build scripts); has a `__main__` guard |
| `deploy_pyvenv.bat [gpu]` | Recreates `.venv` (requires Python 3.12+), installs `requirements.txt`; with `gpu` first installs the CUDA build of PyTorch from `requirements-torch-gpu.txt` |
| `build_pyinstaller.bat`, `build.spec` | PyInstaller build - **untested on Windows** |
| `build_nuitka.bat` | Nuitka standalone build - **untested on Windows** |
| `run_pylint.bat` | Read-only pylint run -> `pylint_report.txt` |
| `reset_cleanup_temp.bat`, `reset_env_pycache.bat` | Remove build output / `.venv` and caches |
| `run_mypy.bat`, `mypy.ini` | Static type check (`python -m mypy`) |
| `pytest.ini`, `.pylintrc` | Test and lint configuration |
| `.github/workflows/pylint.yml` | CI: pytest + mypy + pylint (fail-under 9.0) on Windows, Python 3.12 / 3.13 |
| `assets/fonts/` | `DejaVuSans.ttf` + licence: embedded for non-Latin text layers; bundled by both build scripts |

## `tests/`
| File | Covers |
|------|--------|
| `conftest.py` | `FakeEngine` (DocTR-shaped fake), image fixtures, a `torch` stub when torch is not installed |
| `test_processor.py` | folder/single/PDF workflows, ordering, failures, cancel, progress, HOCR-only, TIFF frames, EXIF, DPI, temp cleanup |
| `test_pdf_writer.py` | compression effect, JPEG passthrough, odd image modes/characters, DocTR geometry conversion, DPI resolution; Unicode text layer, rotated words (text matrices), fit-to-paper placement |
| `test_archiver.py` | archive rules (confirmed only, cancel, collisions, inside-input refusal) |
| `test_config.py` | round trip, legacy file migration, invalid values |
| `test_engine.py` | OOM retry / per-page CPU fallback / session fallback (fake model) |
| `test_hardware.py` | GPU <-> NVML matching by UUID / PCI / index (fake NVML) |
| `test_paper.py` | `core/paper.py` |
| `test_uniform_pages.py` | page-size targets (named, auto, landscape), enhancement decision, failure fallback + warnings, tiled upscaling |
| `test_gui.py` | headless MainWindow: end-to-end runs, failures, cancel, validation, settings persistence |
| `test_requirements.py` | install recipe: torch/torchvision are ranges that a CUDA build satisfies; GPU file points at PyTorch's `/whl/cuNNN` index; deploy script has the `gpu` mode |
| `test_model_cache.py` | repo-local model cache, legacy copy, `DOCTR_CACHE_DIR` override, unwritable fallback |
| `test_integration_doctr.py` | real DocTR pipeline with random weights; mid-page cancellation; rotated-page mode with the orientation models disabled (skipped without DocTR) |

## `data/`
`dpi_res_matrix.py` - CSV generator for the paper-size table (`python data/dpi_res_matrix.py --output paper_pixels.csv`).
The table itself lives in `core/paper.py` (used by the *Page size* feature, VL-067); this script is a convenience and is not
imported by the application. The old 5 MB generated CSV is no longer committed.
