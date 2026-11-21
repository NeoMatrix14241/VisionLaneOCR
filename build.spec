# -*- mode: python ; coding: utf-8 -*-
# PyInstaller spec for VisionLane OCR.  Build with build_pyinstaller.bat (Windows).
# NOTE: not exercised in the Linux refactoring session - run a test build on Windows before releasing.

from PyInstaller.utils.hooks import collect_data_files, collect_submodules, copy_metadata

# Everything the app imports lazily or via registries (DocTR's model zoo) is collected explicitly.
hiddenimports = (
    collect_submodules("doctr")
    + collect_submodules("gui")
    + collect_submodules("core")
    + collect_submodules("utils")
    + ["pynvml", "pypdfium2", "pypdf", "reportlab.pdfbase._fontdata"]
)

datas = [("icon.ico", "."), ("LICENSE", "."), ("README.md", ".")]
datas += collect_data_files("doctr")
datas += collect_data_files("reportlab")
datas += collect_data_files("pypdfium2_raw") + collect_data_files("pypdfium2")
for package in ("torch", "python-doctr", "pypdfium2"):
    try:
        datas += copy_metadata(package)
    except Exception as exc:  # metadata is optional
        print(f"Warning: no metadata for {package}: {exc}")

a = Analysis(
    ["main.py"],
    pathex=["."],
    datas=datas,
    hiddenimports=hiddenimports,
    excludes=["tkinter", "matplotlib.tests", "torch.distributed"],
    noarchive=False,
)
pyz = PYZ(a.pure)

exe = EXE(
    pyz,
    a.scripts,
    [],
    exclude_binaries=True,
    name="VisionLaneOCR",
    debug=False,
    strip=False,
    upx=False,          # UPX corrupts / slows torch and CUDA DLLs
    console=False,
    icon="icon.ico",
)

coll = COLLECT(
    exe,
    a.binaries,
    a.datas,
    strip=False,
    upx=False,
    name="VisionLaneOCR",
)
