# VisionLane OCR - Developer Documentation

Hobby project built on [DocTR](https://github.com/mindee/doctr) (PyTorch) with a PyQt6 front-end.

## Start here

| Document | Read it for |
|----------|-------------|
| [known-issues.md](known-issues.md) | The issue register: what was wrong (`VL-001`...`VL-056`), what was fixed and how it is verified, what is still open, and a checklist for validating on Windows/GPU |
| [known-issues-original-analysis.md](known-issues-original-analysis.md) | The original pre-refactor analysis (evidence, line references) - archive |
| [architecture.md](architecture.md) | Layers, startup sequence, processing pipeline, output layout, threading and cancellation |
| [setup-and-usage.md](setup-and-usage.md) | Install, run, GUI walkthrough, CLI, tests, build scripts |
| [configuration.md](configuration.md) | Every `config.ini` key, environment variables, where files are written |
| [modernization.md](modernization.md) | The dependency versions, why each was chosen, and which code changes they required (done) |
| [reference/core.md](reference/core.md) | `core/` modules and API |
| [reference/gui.md](reference/gui.md) | `gui/` modules |
| [reference/utils.md](reference/utils.md) | `utils/` modules |
| [reference/scripts-and-data.md](reference/scripts-and-data.md) | Root scripts, build files, CI, tests, `data/` |

## State of the project in five lines

1. **Single File, Folder and PDF** workflows all go through one `OCRProcessor` API and one result model (`BatchResult`).
2. Failures are reported per file and page; archiving only moves files whose output was fully written.
3. The CUDA/DocTR monkey-patch layer, Ghostscript, ocrmypdf and the thread-killing code are **gone**; the engine uses stock PyTorch/DocTR APIs.
4. Compression, progress reporting and cancellation are real and covered by tests (110 tests, headless; `mypy` clean).
5. **Not yet verified:** pretrained-weight recognition quality, CUDA/OOM recovery and the Windows build scripts - see `VL-066` in [known-issues.md](known-issues.md).

## Conventions
IDs `VL-nnn` are stable references used throughout the docs and tests. Status values: *Fixed*, *Fixed (unverified)*, *Removed*, *Open*.
