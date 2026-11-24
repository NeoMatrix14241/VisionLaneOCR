"""Download the default DocTR models if they are missing (used by the build scripts and for first-run setup)."""
import sys

from core.constants import DEFAULT_DETECTION_MODEL, DEFAULT_RECOGNITION_MODEL
from utils import models


def main() -> int:
    print("Verifying DocTR models...")
    models.configure_cache()
    print(f"Cache folder: {models.doctr_cache_dir()}")
    ok = True
    for kind, name in ((models.DETECTION, DEFAULT_DETECTION_MODEL), (models.RECOGNITION, DEFAULT_RECOGNITION_MODEL)):
        if models.is_cached(name):
            print(f"{name} already present.")
            continue
        print(f"Downloading {name} ({kind})...")
        try:
            models.download(kind, name, progress=models.console_progress())
        except Exception as exc:  # pylint: disable=broad-except
            print(f"  FAILED: {exc}")
            ok = False
    print("All required DocTR models are present." if ok else "Some models could not be downloaded.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
