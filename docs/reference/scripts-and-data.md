# Entry points, scripts, data, CI

| File | What it does | State |
|------|--------------|-------|
| `main.py` | GUI entry: splash → progressive imports → `MainWindow` (see [architecture.md](../architecture.md)) | works; VL-039/040/035 |
| `run_ocr.py` | Headless batch over `input/images` (`jpg jpeg png tif tiff`) and `input/pdfs` into `output/` | broken (VL-001, VL-051) |
| `debug_run.py` | Creates `DebugLogger`, then `import main; main.main()` | broken (`main.main` missing) |
| `verify_models.py` | Downloads `db_resnet50` and `parseq` if no matching file in `~/.cache/doctr/models`; runs at import, no `__main__` guard | works; used by both build scripts |
| `demo_enhanced_startup.py` | Console + GUI demo of startup features (caching, parallel loader, diagnostics, patches, hardware monitor) | demo only; needs updating for doctr 1.x |
| `pylint_clean.py` | Strips trailing whitespace in all `.py` files, then runs pylint in parallel into `pylint_cleanup.log` | modifies sources (VL-052) |
| `run_pylint.bat` (398 lines) | Windows wrapper around pylint with colour output/summary | – |
| `deploy_pyvenv.bat` | Recreates `.venv`, upgrades pip, installs `requirements.txt` | works (text says Python 3.8+) |
| `build_pyinstaller.bat`, `build.spec` | PyInstaller one-folder build | VL-048 |
| `build_nuitka.bat` | Compiler detection, environment report, ~100 `--include-package` flags, post-build DLL copying | VL-049 |
| `reset_cleanup_temp.bat`, `reset_env_pycache.bat` | housekeeping | fine |
| `cache_binaries.zip` | ccache 4.11.3 / clcache 4.2.0 for Nuitka builds (7.7 MB) | should be downloaded, not committed |
| `icon.ico` | application icon | – |
| `LICENSE` | Apache-2.0 | – |
| `.github/workflows/pylint.yml` | pylint on push/PR, never fails | VL-050 |
| `.pylintrc` | 15 KB config; most size/complexity checks disabled | – |
| `.gitignore` | venv, caches, dist dirs, `logs/`, `input/`, `output/`, `icon/`, `temp_binaries/`… (not `config.ini`) | VL-045 |

## `data/`
- `dpi_res_matrix.py` — embeds an A0–DL paper-size table (150/300/600 dpi), then for each size generates pixel dimensions for DPI 1…9000 and writes `uniform_data.csv` into the current directory. Runs on import. Comment says 1200.
- `uniform_data.csv` — its output (153 000 rows, 5 MB). **Not read by the application.** It is the groundwork for the roadmap item "uniform paper sizes via super-image" (the referenced `paper_sizes_formula.xlsl` is not in the repo).
