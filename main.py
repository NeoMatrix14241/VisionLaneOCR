"""VisionLane OCR - application entry point.

Start-up order (no sleeps, no import-time patches):

1. settings + logging
2. Qt application and splash screen
3. PyTorch/DocTR are imported on a background thread while the splash animates
4. the main window opens; the OCR models are loaded on another background thread
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
import threading

logger = logging.getLogger("visionlane")


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="VisionLane OCR")
    parser.add_argument("--debug", action="store_true", help="verbose (DEBUG) logging")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)

    from utils import app_paths
    from utils.app_config import AppConfig
    from utils.logging_config import install_crash_handlers, setup_logging

    config_path = app_paths.config_path()
    config = AppConfig.load(config_path)
    log_dir = app_paths.log_dir()
    log_file = setup_logging(log_dir, "DEBUG" if args.debug else config.log_level)
    install_crash_handlers(log_dir)
    logger.info("VisionLane OCR starting (config: %s, log: %s)", config_path, log_file)

    from utils import models
    models.configure_cache()  # DocTR weights live beside the program (see utils/models.py)

    from PyQt6.QtCore import QTimer
    from PyQt6.QtGui import QIcon
    from PyQt6.QtWidgets import QApplication, QMessageBox

    app = QApplication(sys.argv[:1])
    app.setStyle("Fusion")
    icon = app_paths.app_root() / "icon.ico"
    if icon.exists():
        app.setWindowIcon(QIcon(str(icon)))

    from gui.splash_screen import StartupSplash
    splash = StartupSplash()
    splash.set_status("Loading PyTorch and DocTR...")
    splash.center_on_screen()
    splash.show()
    app.processEvents()

    outcome: dict[str, object] = {}

    def preload() -> None:
        try:
            __import__("torch")
            __import__("doctr.models")
        except Exception as exc:  # pylint: disable=broad-except
            logger.exception("Failed to import the OCR stack")
            outcome["error"] = exc

    thread = threading.Thread(target=preload, name="preload", daemon=True)
    thread.start()

    state: dict[str, object] = {}

    def poll() -> None:
        if thread.is_alive():
            return
        timer.stop()
        if "error" in outcome:
            splash.close()
            QMessageBox.critical(None, "VisionLane OCR",
                                 f"Could not load PyTorch/DocTR:\n{outcome['error']}\n\n"
                                 "Reinstall the requirements (see README) and check the log:\n" + str(log_file))
            app.exit(1)
            return
        splash.set_status("Opening window...")
        from gui.main_window import MainWindow
        window = MainWindow(config, config_path)
        state["window"] = window
        window.show()
        splash.close()

    timer = QTimer()
    timer.timeout.connect(poll)
    timer.start(50)

    code = app.exec()
    logger.info("VisionLane OCR exited with code %s", code)
    return int(code)


if __name__ == "__main__":
    exit_code = main()
    logging.shutdown()
    if threading.active_count() > 1:
        # A daemon worker (e.g. an inference still running after a forced close) must not delay shutdown.
        os._exit(exit_code)  # pylint: disable=protected-access
    sys.exit(exit_code)
