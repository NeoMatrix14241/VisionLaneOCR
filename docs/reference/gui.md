# `gui/` reference

| Module | Contents |
|--------|----------|
| `main_window.py` | `MainWindow`: builds the UI, maps settings <-> widgets (`collect_config`), starts the engine load, runs jobs, shows the summary, archives, saves settings on close |
| `input_tab.py` | `InputTab(kind)` for `single` / `folder` / `pdf`: input selector, output folder, archive folder (auto-suggests `<output>/archive` without overwriting a custom value), asynchronous file count |
| `workers.py` | `Task(fn)` (daemon thread, **no Qt parent**; `succeeded` / `failed` signals - connect to bound slots, not lambdas), `OCRJob` (adds `progress`, `cancel()`), `count_supported_files()` |
| `dialogs.py` | `ask_cpu_threads()` (Settings -> Performance Options), `show_about()` |
| `theme.py` | `stylesheet(mode)` for `light` / `dark` / `night` (`system` = no style sheet), notice colours |
| `splash_screen.py` | `StartupSplash` (indeterminate progress bar) |

## `MainWindow` flow
1. `__init__`: build UI, apply config, theme, start the 1 s hardware timer, schedule `_load_engine`.
2. `_load_engine`: `Task` -> download missing models, `OCRProcessor(...)`; `_on_engine_ready` enables *Start* and shows the VRAM notice if any.
3. Model / device combos and the *Rotated / skewed pages* checkbox: `_on_model_selected`, `_on_device_changed`, `_on_straighten_toggled` -> `_reload` (background task); on failure the previous selection is restored.
   Page handling row: *Page size* combo, *Enhance low-resolution pages* (disabled unless `super-image` is installed), *Rotated / skewed pages*.
4. `_start_processing`: `_validate()` (input exists, output set and not inside the input folder, archive set and valid), `processor.configure(...)`, save settings, start `OCRJob`.
5. `_on_progress` (signals only) updates label and bar; `_on_job_finished` archives (if enabled), then shows *Success / Completed with problems / Cancelled* with the failure list as detail text, any run warnings (e.g. enhancement unavailable) and an *Open Output Folder* button.
6. `closeEvent`: confirm if a job runs, cancel, join <= 30 s, save settings, stop the timer, close NVML, release the model.

Menus: **File** (Open..., Save Settings, Exit), **Settings** (Performance Options), **Theme**, **Help** (About).
