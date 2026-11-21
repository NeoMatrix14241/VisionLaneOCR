"""Headless GUI tests (Qt 'offscreen' platform). The OCR engine is replaced by FakeEngine."""
import os
import time


import pytest

os.environ.setdefault("QT_QPA_PLATFORM", "offscreen")
pytest.importorskip("PyQt6")

from PyQt6.QtWidgets import QApplication, QMessageBox  # noqa: E402

from core.ocr_processor import OCRProcessor  # noqa: E402
from tests.conftest import FakeEngine  # noqa: E402
from utils.app_config import AppConfig  # noqa: E402


@pytest.fixture(scope="module")
def qapp():
    return QApplication.instance() or QApplication([])


def wait_until(app, condition, timeout=10.0):
    end = time.time() + timeout
    while time.time() < end:
        app.processEvents()
        if condition():
            return True
        time.sleep(0.01)
    return False


@pytest.fixture
def window(qapp, tmp_path, monkeypatch):
    import gui.main_window as mw
    monkeypatch.setattr(mw, "OCRProcessor", lambda det, reco, device, cpu_threads=0: OCRProcessor(engine=FakeEngine()))
    monkeypatch.setattr(mw.model_utils, "is_cached", lambda name: True)
    shown = []
    monkeypatch.setattr(QMessageBox, "exec", lambda self: shown.append((self.windowTitle(), self.text(), self.detailedText())) or 0)
    monkeypatch.setattr(QMessageBox, "warning", staticmethod(lambda *a, **k: shown.append(("warning", a[2], "")) or 0))
    monkeypatch.setattr(QMessageBox, "critical", staticmethod(lambda *a, **k: shown.append(("critical", a[2], "")) or 0))
    monkeypatch.setattr(QMessageBox, "information", staticmethod(lambda *a, **k: shown.append(("info", a[2], "")) or 0))
    cfg_path = tmp_path / "cfg" / "config.ini"
    win = mw.MainWindow(AppConfig(), cfg_path)
    win.show()
    win.shown = shown
    assert wait_until(qapp, lambda: win.start_button.isEnabled()), "models never finished loading"
    yield win
    win.close()


def test_folder_run_end_to_end_with_archive(qapp, window, tmp_path, make_image):
    for name in ("a.png", "sub/b.png"):
        make_image(tmp_path / "in" / name)
    tab = window.input_tabs["folder"]
    window.tabs.setCurrentIndex(1)
    tab.set_input(tmp_path / "in")
    tab.output_edit.setText(str(tmp_path / "out"))
    tab.archive_check.setChecked(True)
    assert tab.archive_edit.text() == str(tmp_path / "out" / "archive")
    window.format_combo.setCurrentText("PDF+HOCR")
    seen = []
    window.start_button.click()
    window.job.progress.connect(lambda e: seen.append(e.files_done))
    assert not window.start_button.isEnabled() and window.cancel_button.isEnabled()
    assert wait_until(qapp, lambda: window.job is None), "job never finished"
    title, text, _ = window.shown[-1]
    assert title == "Success" and "2/2" in text
    session = next((tmp_path / "out").glob("OCR_Session_*"))
    assert len(list((session / "pdf").rglob("*.pdf"))) == 2 and len(list((session / "hocr").rglob("*.hocr"))) == 2
    assert (tmp_path / "out" / "archive" / "a.png").exists() and not (tmp_path / "in" / "a.png").exists()
    assert window.start_button.isEnabled() and window.progress_bar.value() == 0


def test_failures_are_shown_and_inputs_kept(qapp, window, tmp_path, make_image):
    from PIL import Image
    make_image(tmp_path / "in" / "ok.png", (100, 40))
    Image.new("RGB", (100, 33)).save(tmp_path / "in" / "bad.png")
    window.processor.engine.fail_heights = (33,)
    tab = window.input_tabs["folder"]
    window.tabs.setCurrentIndex(1)
    tab.set_input(tmp_path / "in")
    tab.output_edit.setText(str(tmp_path / "out"))
    tab.archive_check.setChecked(True)
    window.start_button.click()
    assert wait_until(qapp, lambda: window.job is None)
    title, text, detail = window.shown[-1]
    assert title == "Completed with problems" and "1/2" in text and "bad.png" in detail
    assert (tmp_path / "in" / "bad.png").exists()                       # failed input is NOT archived
    assert (tmp_path / "out" / "archive" / "ok.png").exists()


def test_single_and_pdf_tabs_work(qapp, window, tmp_path, make_image):
    from reportlab.pdfgen.canvas import Canvas
    img = make_image(tmp_path / "one.png")
    window.tabs.setCurrentIndex(0)
    window.input_tabs["single"].set_input(img)
    window.input_tabs["single"].output_edit.setText(str(tmp_path / "o1"))
    window.start_button.click()
    assert wait_until(qapp, lambda: window.job is None)
    assert window.shown[-1][0] == "Success"
    pdf = tmp_path / "x.pdf"
    c = Canvas(str(pdf), pagesize=(200, 200)); c.rect(10, 10, 50, 50, fill=1); c.showPage(); c.save()
    window.tabs.setCurrentIndex(2)
    window.input_tabs["pdf"].set_input(pdf)
    window.input_tabs["pdf"].output_edit.setText(str(tmp_path / "o2"))
    window.start_button.click()
    assert wait_until(qapp, lambda: window.job is None)
    assert window.shown[-1][0] == "Success"
    assert next((tmp_path / "o2").glob("OCR_Session_*/pdf/x.pdf")).exists()


def test_validation_messages(qapp, window, tmp_path):
    window.tabs.setCurrentIndex(1)
    window.start_button.click()
    assert window.job is None and "select a folder" in window.shown[-1][1]
    (tmp_path / "in").mkdir()
    tab = window.input_tabs["folder"]
    tab.set_input(tmp_path / "in")
    tab.output_edit.setText(str(tmp_path / "in" / "out"))
    window.start_button.click()
    assert window.job is None and "must not be inside" in window.shown[-1][1]


def test_cancel_button(qapp, window, tmp_path, make_image):
    for i in range(30):
        make_image(tmp_path / "in" / f"{i:02d}.png")
    engine = window.processor.engine
    engine.on_call = lambda n: time.sleep(0.05)
    tab = window.input_tabs["folder"]
    window.tabs.setCurrentIndex(1)
    tab.set_input(tmp_path / "in")
    tab.output_edit.setText(str(tmp_path / "out"))
    tab.archive_check.setChecked(True)
    window.start_button.click()
    assert wait_until(qapp, lambda: len(engine.calls) >= 2)
    window.cancel_button.click()
    assert wait_until(qapp, lambda: window.job is None)
    assert window.shown[-1][0] == "Cancelled"
    assert len(list((tmp_path / "in").glob("*.png"))) == 30             # nothing archived after cancel


def test_settings_roundtrip_on_close(qapp, window, tmp_path):
    window.dpi_combo.setCurrentText("300")
    window.compress_check.setChecked(True)
    window.quality_slider.setValue(60)
    window.input_tabs["folder"].output_edit.setText("D:/out")
    window.close()
    cfg = AppConfig.load(window.config_path)
    assert (cfg.dpi, cfg.compress_enabled, cfg.compression_quality, cfg.output_paths["folder"]) == ("300", True, 60, "D:/out")


def test_screenshot_for_review(qapp, window, tmp_path):
    out = os.environ.get("VL_SCREENSHOT")
    if out:
        window.resize(900, 520)
        qapp.processEvents()
        window.grab().save(out)
