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
    index: int                      # PyTorch/CUDA device index (NOT necessarily the NVML index)
    name: str
    total_mb: float
    capability: tuple[int, int]
    uuid: str = ""                  # normalised: lower case, no "gpu-" prefix
    pci: tuple[int, int, int] | None = None   # (domain, bus, device)


@dataclass
class DeviceReport:
    device: str = "cpu"                 # "cuda" or "cpu" - what will actually be used
    reason: str = ""                    # why CPU was chosen (empty when CUDA is used)
    gpus: list[GpuInfo] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    current_index: int = 0              # the CUDA device the model is placed on ("cuda" == current device)

    @property
    def primary(self) -> GpuInfo | None:
        """The GPU the model runs on."""
        for gpu in self.gpus:
            if gpu.index == self.current_index:
                return gpu
        return self.gpus[0] if self.gpus else None


def normalize_uuid(value: object) -> str:
    """``GPU-1234-ABCD`` / ``UUID('1234-abcd')`` / ``{1234-ABCD}`` -> ``1234-abcd`` (so NVML and torch compare equal)."""
    text = str(value).strip().lower()
    for junk in ("uuid(", ")", "'", '"', "{", "}"):
        text = text.replace(junk, "")
    return text[4:] if text.startswith("gpu-") else text


def parse_pci_bus_id(value: object) -> tuple[int, int, int] | None:
    """``00000000:01:00.0`` -> (domain, bus, device); ``None`` if it cannot be parsed."""
    if isinstance(value, bytes):
        value = value.decode("ascii", "ignore")
    try:
        domain, bus, rest = str(value).strip().split(":")
        return int(domain, 16), int(bus, 16), int(rest.split(".")[0], 16)
    except (ValueError, AttributeError):
        return None


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
            uuid = normalize_uuid(getattr(props, "uuid", "")) if getattr(props, "uuid", None) else ""
            pci = None
            if hasattr(props, "pci_bus_id") and hasattr(props, "pci_device_id"):
                pci = (int(getattr(props, "pci_domain_id", 0)), int(props.pci_bus_id), int(props.pci_device_id))
            report.gpus.append(GpuInfo(index, props.name, props.total_memory / 1024 ** 2,
                                       (props.major, props.minor), uuid, pci))
        report.current_index = torch.cuda.current_device()
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
            import pynvml
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

    def snapshot(self, device: str | None, gpu: GpuInfo | None = None) -> HardwareSnapshot:
        """``gpu`` is the GPU the model runs on (``DeviceReport.primary``); it is matched to its NVML handle."""
        if device is None:
            return HardwareSnapshot("Processing Device: loading...", "Usage: N/A", "Memory: N/A")
        if device == "cuda":
            label = f"Processing Device: GPU ({gpu.name})" if gpu else "Processing Device: GPU"
            return HardwareSnapshot(label, *self._gpu_text(gpu))
        try:
            usage = f"CPU Usage: {psutil.cpu_percent(interval=None):.0f}%"
            memory = f"Memory: {psutil.virtual_memory().percent:.0f}%"
        except Exception:  # pylint: disable=broad-except
            usage, memory = "CPU Usage: N/A", "Memory: N/A"
        return HardwareSnapshot("Processing Device: CPU", usage, memory)

    def _nvml_handle(self, gpu: GpuInfo | None):
        """NVML handle of the GPU PyTorch uses. NVML numbers GPUs in PCI order and CUDA by default
        'fastest first', so the indices can differ: match by UUID, then by PCI address, else by index."""
        nvml = self._nvml
        if nvml is None:
            return None
        if gpu is not None and (gpu.uuid or gpu.pci):
            for index in range(nvml.nvmlDeviceGetCount()):
                handle = nvml.nvmlDeviceGetHandleByIndex(index)
                if gpu.uuid:
                    try:
                        if normalize_uuid(nvml.nvmlDeviceGetUUID(handle)) == gpu.uuid:
                            return handle
                    except Exception:  # pylint: disable=broad-except
                        pass
                if gpu.pci:
                    try:
                        if parse_pci_bus_id(nvml.nvmlDeviceGetPciInfo(handle).busId) == gpu.pci:
                            return handle
                    except Exception:  # pylint: disable=broad-except
                        pass
        return nvml.nvmlDeviceGetHandleByIndex(gpu.index if gpu else 0)

    def _gpu_text(self, gpu: GpuInfo | None) -> tuple[str, str]:
        index = gpu.index if gpu else 0
        if self._nvml:
            try:
                handle = self._nvml_handle(gpu)
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
