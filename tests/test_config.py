from utils.app_config import AppConfig, normalize_compression_type


def test_roundtrip(tmp_path):
    cfg = AppConfig()
    cfg.dpi = "300"
    cfg.output_format = "PDF+HOCR"
    cfg.device = "cpu"
    cfg.compress_enabled = True
    cfg.compression_quality = 70
    cfg.input_paths["folder"] = r"D:\scans"
    cfg.archive_enabled["pdf"] = True
    cfg.cpu_threads = 1
    cfg.straighten_pages = True
    cfg.page_size = "A4"
    cfg.enhance_enabled = True
    cfg.enhance_scale = 3
    cfg.enhance_min_dpi = 200
    path = tmp_path / "c.ini"
    cfg.save(path)
    again = AppConfig.load(path)
    assert again == cfg


def test_legacy_file_is_understood(tmp_path):
    path = tmp_path / "config.ini"
    path.write_text("""[General]
dpi = Auto
output_format = PDF+HOCR
compress_enabled = True
compression_type = jpeg2000
compression_quality = 100
archive_enabled = True
[Paths]
folder = D:/in
output_folder = D:/out
archive_folder = D:/arch
[Performance]
thread_count = 1
operation_timeout = 300
[Startup]
enable_parallel_loading = True
""", encoding="utf-8")
    cfg = AppConfig.load(path)
    assert cfg.compression_type == "jpeg"              # jpeg2000 never worked -> mapped
    assert all(cfg.archive_enabled.values())           # old single flag applies to every tab
    assert cfg.input_paths["folder"] == "D:/in" and cfg.output_paths["folder"] == "D:/out"
    assert cfg.cpu_threads == 1
    cfg.save(path)
    assert "Startup" not in path.read_text() and "operation_timeout" not in path.read_text()


def test_garbage_values_fall_back(tmp_path):
    path = tmp_path / "c.ini"
    path.write_text("[General]\ndpi = 7\ntheme_mode = neon\ncompression_quality = abc\ndevice = tpu\n")
    cfg = AppConfig.load(path)
    assert (cfg.dpi, cfg.theme_mode, cfg.device) == ("Auto", "system", "auto")
    assert cfg.compression_quality == 85


def test_missing_file_gives_defaults(tmp_path):
    assert AppConfig.load(tmp_path / "nope.ini") == AppConfig()


def test_compression_normalization():
    assert [normalize_compression_type(x) for x in ("JPEG", "LZW", "png", "jpeg2000", "??")] == \
        ["jpeg", "lossless", "lossless", "jpeg", "jpeg"]


def test_new_page_options_are_validated(tmp_path):
    path = tmp_path / "c.ini"
    path.write_text("[General]\npage_size = a3\nenhance_scale = 7\nenhance_min_dpi = 5\nstraighten_pages = maybe\n")
    cfg = AppConfig.load(path)
    assert (cfg.page_size, cfg.enhance_scale, cfg.enhance_min_dpi, cfg.straighten_pages) == ("A3", 2, 30, False)
    path.write_text("[General]\npage_size = nonsense\n")
    assert AppConfig.load(path).page_size == "original"
