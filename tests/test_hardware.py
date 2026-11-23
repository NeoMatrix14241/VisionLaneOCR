from types import SimpleNamespace

from core.hardware import GpuInfo, HardwareMonitor, normalize_uuid, parse_pci_bus_id


class FakeNvml:
    """Two GPUs. NVML lists them in PCI order; PyTorch ('fastest first') numbers them the other way round."""

    def __init__(self):
        self.devices = [  # NVML index -> (uuid, pci bus id, usage %, used MB)
            ("GPU-AAAA-1111", b"00000000:01:00.0", 11, 100),
            ("GPU-BBBB-2222", b"00000000:02:00.0", 22, 200),
        ]

    def nvmlDeviceGetCount(self):
        return len(self.devices)

    def nvmlDeviceGetHandleByIndex(self, i):
        return i

    def nvmlDeviceGetUUID(self, h):
        return self.devices[h][0]

    def nvmlDeviceGetPciInfo(self, h):
        return SimpleNamespace(busId=self.devices[h][1])

    def nvmlDeviceGetUtilizationRates(self, h):
        return SimpleNamespace(gpu=self.devices[h][2])

    def nvmlDeviceGetMemoryInfo(self, h):
        return SimpleNamespace(used=self.devices[h][3] * 1024 ** 2, total=6144 * 1024 ** 2)


def monitor():
    m = HardwareMonitor.__new__(HardwareMonitor)
    m._nvml = FakeNvml()
    return m


def test_gpu_is_matched_by_uuid_not_by_index():
    # torch calls the second NVML device "cuda:0"
    torch_gpu0 = GpuInfo(0, "GPU B", 6144, (7, 5), uuid=normalize_uuid("GPU-BBBB-2222"))
    snap = monitor().snapshot("cuda", torch_gpu0)
    assert snap.usage_text == "GPU Usage: 22%" and "200/6144" in snap.memory_text
    torch_gpu1 = GpuInfo(1, "GPU A", 6144, (7, 5), uuid=normalize_uuid("uuid('aaaa-1111')"))
    assert monitor().snapshot("cuda", torch_gpu1).usage_text == "GPU Usage: 11%"


def test_pci_address_is_the_fallback_when_no_uuid():
    gpu = GpuInfo(0, "GPU B", 6144, (7, 5), pci=(0, 2, 0))
    assert monitor().snapshot("cuda", gpu).usage_text == "GPU Usage: 22%"


def test_index_is_the_last_resort():
    assert monitor().snapshot("cuda", GpuInfo(1, "x", 1, (7, 5))).usage_text == "GPU Usage: 22%"


def test_normalisers():
    assert normalize_uuid("GPU-ABCD-12") == "abcd-12" == normalize_uuid("UUID('abcd-12')")
    assert parse_pci_bus_id(b"00000000:0A:00.0") == (0, 10, 0)
    assert parse_pci_bus_id("garbage") is None


def test_cpu_snapshot_needs_no_nvml():
    m = HardwareMonitor.__new__(HardwareMonitor)
    m._nvml = None
    assert m.snapshot("cpu").device_text == "Processing Device: CPU"
