"""Device selection and hardware monitoring (NVML via ``pynvml`` / ``nvidia-ml-py``).

Real values only: when something cannot be read the field says so instead of inventing numbers.
"""
from __future__ import annotations

import logging
from dataclasses import dataclass, field

import psutil
import torch

from .constants import RECOMMENDED_VRAM_GB

logger = logging.getLogger(__name__)


@dataclass
class GpuInfo:
    index: int
    name: str
    total_mb: float
    capability: tuple[int, int]


@dataclass
class DeviceReport:
    device: str = "cpu"                 # "cuda" or "cpu" - what will actually be used
    reason: str = ""                    # why CPU was chosen (empty when CUDA is used)
    gpus: list[GpuInfo] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def primary(self) -> GpuInfo | None:
        return self.gpus[0] if self.gpus else None


def _smoke_test_cuda() -> str | None:
    """Run one tiny kernel. Catches 'no kernel image for this GPU' before the first real page."""
    try:
        value = (torch.zeros(4, device="cuda") + 1).sum().item()
        torch.cuda.synchronize()
        return None if value == 4 else "CUDA smoke test returned a wrong value"
    except Exception as exc:  # pylint: disable=broad-except
        return f"CUDA smoke test failed: {exc}"


def resolve_device(preference: str = "auto") -> DeviceReport:
    """Decide between GPU and CPU. ``preference`` is ``auto``, ``gpu`` or ``cpu``."""
    report = DeviceReport()
    preference = (preference or "auto").lower()
    if preference == "cpu":
        report.reason = "CPU selected by the user"
        return report
    if not torch.cuda.is_available():
        built = "with" if torch.version.cuda else "without"
        report.reason = f"CUDA not available (PyTorch was built {built} CUDA support)"
        return report
    try:
        for index in range(torch.cuda.device_count()):
            props = torch.cuda.get_device_properties(index)
            report.gpus.append(GpuInfo(index, props.name, props.total_memory / 1024 ** 2,
                                       (props.major, props.minor)))
    except Exception as exc:  # pylint: disable=broad-except
        report.reason = f"Could not query the GPU: {exc}"
        return report
    problem = _smoke_test_cuda()
    if problem:
        report.reason = problem
        report.gpus.clear()
        return report
    report.device = "cuda"
    gpu = report.primary
    if gpu and gpu.total_mb / 1024 < RECOMMENDED_VRAM_GB - 0.5:
        report.warnings.append(
            f"{gpu.name} has {gpu.total_mb / 1024:.1f} GB of VRAM ({RECOMMENDED_VRAM_GB:.0f} GB recommended). "
            "Pages that do not fit are retried and, if needed, processed on the CPU; "
            "you can also choose Device: CPU."
        )
    return report


@dataclass
class HardwareSnapshot:
    device_text: str
    usage_text: str
    memory_text: str


class HardwareMonitor:
    """Reads GPU (NVML) or CPU/RAM statistics for the status bar. Cheap enough to call every second."""

    def __init__(self) -> None:
        self._nvml = None
        try:
            import pynvml  # type: ignore[import-not-found]
            pynvml.nvmlInit()
            self._nvml = pynvml
        except Exception as exc:  # pylint: disable=broad-except
            logger.info("NVML not available (%s); GPU load will be shown as N/A", exc)

    def close(self) -> None:
        if self._nvml:
            try:
                self._nvml.nvmlShutdown()
            except Exception:  # pylint: disable=broad-except
                pass
            self._nvml = None

    def snapshot(self, device: str | None, gpu_index: int = 0, gpu_name: str = "") -> HardwareSnapshot:
        if device is None:
            return HardwareSnapshot("Processing Device: loading...", "Usage: N/A", "Memory: N/A")
        if device == "cuda":
            label = f"Processing Device: GPU ({gpu_name})" if gpu_name else "Processing Device: GPU"
            return HardwareSnapshot(label, *self._gpu_text(gpu_index))
        try:
            usage = f"CPU Usage: {psutil.cpu_percent(interval=None):.0f}%"
            memory = f"Memory: {psutil.virtual_memory().percent:.0f}%"
        except Exception:  # pylint: disable=broad-except
            usage, memory = "CPU Usage: N/A", "Memory: N/A"
        return HardwareSnapshot("Processing Device: CPU", usage, memory)

    def _gpu_text(self, index: int) -> tuple[str, str]:
        if self._nvml:
            try:
                handle = self._nvml.nvmlDeviceGetHandleByIndex(index)
                util = self._nvml.nvmlDeviceGetUtilizationRates(handle)
                mem = self._nvml.nvmlDeviceGetMemoryInfo(handle)
                return (f"GPU Usage: {util.gpu}%",
                        f"GPU Memory (used/total): {mem.used / 1024 ** 2:.0f}/{mem.total / 1024 ** 2:.0f} MB")
            except Exception as exc:  # pylint: disable=broad-except
                logger.debug("NVML read failed: %s", exc)
        try:
            free, total = torch.cuda.mem_get_info(index)
            return ("GPU Usage: N/A",
                    f"GPU Memory (used/total): {(total - free) / 1024 ** 2:.0f}/{total / 1024 ** 2:.0f} MB")
        except Exception:  # pylint: disable=broad-except
            return "GPU Usage: N/A", "GPU Memory: N/A"
