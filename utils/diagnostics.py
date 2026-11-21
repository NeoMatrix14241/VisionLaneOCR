"""``python -m utils.diagnostics`` - print what the OCR stack sees (versions, GPU, model cache).

Use this when something does not start or the GPU is not used; paste the output into a bug report.
"""
from __future__ import annotations

import platform
import sys
from importlib import metadata


def main() -> int:
    print(f"Python      : {sys.version.split()[0]} ({platform.platform()})")
    try:
        import torch
        print(f"PyTorch     : {torch.__version__} (built for CUDA {torch.version.cuda})")
        from core.hardware import resolve_device
        report = resolve_device("auto")
        print(f"Device      : {report.device}" + (f"  ({report.reason})" if report.reason else ""))
        for gpu in report.gpus:
            print(f"GPU {gpu.index}       : {gpu.name}, {gpu.total_mb / 1024:.1f} GB, compute {gpu.capability[0]}.{gpu.capability[1]}")
        for warning in report.warnings:
            print(f"Warning     : {warning}")
    except Exception as exc:  # pylint: disable=broad-except
        print(f"PyTorch     : not usable ({exc})")
    # (import name, distribution name) - versions come from package metadata, not from module attributes
    for module, dist in (("doctr", "python-doctr"), ("PyQt6.QtCore", "PyQt6"), ("pypdf", "pypdf"),
                         ("pypdfium2", "pypdfium2"), ("reportlab", "reportlab"), ("PIL", "pillow"),
                         ("numpy", "numpy"), ("psutil", "psutil"), ("pynvml", "nvidia-ml-py")):
        try:
            __import__(module)
            print(f"{module:<12}: {metadata.version(dist)}")
        except Exception as exc:  # pylint: disable=broad-except
            print(f"{module:<12}: MISSING ({exc})")
    from utils import models
    models.configure_cache()
    print(f"Model cache : {models.doctr_cache_dir()}")
    for name in ("db_resnet50", "parseq"):
        print(f"  {name}: {'cached' if models.is_cached(name) else 'not downloaded'}")
    from utils.app_paths import config_path, log_dir
    print(f"Config      : {config_path()}\nLogs        : {log_dir()}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
