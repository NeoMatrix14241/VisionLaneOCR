# VisionLane OCR — Developer Documentation

Hobby project built on [DocTR](https://github.com/mindee/doctr) (PyTorch) with a PyQt6 front-end. This folder documents
the code as it is today, including everything that is broken or unfinished.

## Start here

| Document | Read it for |
|----------|-------------|
| [known-issues.md](known-issues.md) | **56 catalogued issues** with severity, file/line, evidence and fix, plus a suggested order of attack |
| [architecture.md](architecture.md) | Layers, startup sequence, processing pipeline, output layout, threading, the patch layer |
| [setup-and-usage.md](setup-and-usage.md) | Install, run, GUI walkthrough, build scripts |
| [configuration.md](configuration.md) | Every `config.ini` key, which ones actually do something, env vars, files written |
| [modernization.md](modernization.md) | The new `requirements.txt`, why each version, and the code changes needed to run on it |
| [reference/core.md](reference/core.md) | `core/` module-by-module API |
| [reference/gui.md](reference/gui.md) | `gui/` module-by-module API |
| [reference/utils.md](reference/utils.md) | `utils/` module-by-module API |
| [reference/scripts-and-data.md](reference/scripts-and-data.md) | Root scripts, build files, CI, `data/` |

## State of the project in five lines

1. **Folder mode** is the only functioning workflow; **Single File** and **PDF** crash (VL-001, VL-002).
2. Failures are swallowed and the UI still says "success"; archiving then moves inputs anyway (VL-003/004).
3. A large CUDA/DocTR monkey-patch layer exists to work around old toolchain problems; it is obsolete on current DocTR/PyTorch and risky (VL-008/010/011).
4. Compression, progress reporting and cancellation are largely ineffective or fragile (VL-009/014/022).
5. The updated `requirements.txt` needs ~5 small code edits before the app starts (modernization.md §2).

## Conventions
IDs `VL-nnn` are stable references used throughout. Evidence tags: `[verified]`, `[code-read]`, `[needs-hw]`.
The documentation was produced by static reading plus checks in a Linux sandbox; nothing was run on Windows or with a GPU.
