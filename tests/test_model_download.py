"""Model download with progress (utils/models.py). Uses a real HTTP server on localhost."""
import hashlib
import http.server
import os
import socket
import threading

import pytest

from utils import models

PAYLOAD = os.urandom(900_000)
DIGEST8 = hashlib.sha256(PAYLOAD).hexdigest()[:8]


@pytest.fixture
def server():
    """Serves PAYLOAD at any path. Behaviour knobs: .body (bytes sent), .extra (added to Content-Length)."""
    class Handler(http.server.BaseHTTPRequestHandler):
        def do_GET(self):  # noqa: N802
            srv.requests += 1
            body = srv.body
            self.send_response(200)
            self.send_header("Content-Length", str(len(body) + srv.extra))
            self.end_headers()
            for start in range(0, len(body), 64 * 1024):
                self.wfile.write(body[start:start + 64 * 1024])
                self.wfile.flush()

        def log_message(self, *args):
            pass

    srv = http.server.ThreadingHTTPServer(("127.0.0.1", 0), Handler)
    srv.requests, srv.body, srv.extra = 0, PAYLOAD, 0
    srv.url = f"http://127.0.0.1:{srv.server_address[1]}/models?id=v0/fake-{DIGEST8}.pt&src=0"
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    yield srv
    srv.shutdown()
    srv.server_close()


def leftovers(folder):
    return sorted(p.name for p in folder.iterdir())


def test_download_reports_real_progress_and_stores_the_verified_file(tmp_path, server):
    events = []
    target = tmp_path / "models" / f"fake-{DIGEST8}.pt"
    models.download_file(server.url, target, DIGEST8, events.append, name="fake")
    assert target.read_bytes() == PAYLOAD
    assert leftovers(target.parent) == [target.name]                      # no .part left behind
    assert events[0].done == 0 and events[0].total == len(PAYLOAD)
    assert events[-1].done == events[-1].total == len(PAYLOAD) and events[-1].fraction == 1.0
    assert [e.done for e in events] == sorted(e.done for e in events)    # never runs backwards
    assert {e.name for e in events} == {"fake"}


def test_corrupt_download_is_rejected_and_leaves_nothing(tmp_path, server):
    server.body = PAYLOAD[:-1] + bytes([PAYLOAD[-1] ^ 0xFF])              # same size, one flipped byte
    target = tmp_path / "models" / f"fake-{DIGEST8}.pt"
    with pytest.raises(ValueError, match="checksum"):
        models.download_file(server.url, target, DIGEST8)
    assert leftovers(target.parent) == []                                  # neither the file nor a .part


def test_truncated_download_is_rejected_and_leaves_nothing(tmp_path, server):
    server.extra = 5000                                                    # promises more bytes than it sends
    target = tmp_path / "models" / f"fake-{DIGEST8}.pt"
    with pytest.raises(Exception):
        models.download_file(server.url, target, DIGEST8)
    assert leftovers(target.parent) == []


def test_unreachable_server_raises_and_leaves_nothing(tmp_path):
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]                                      # a port nobody listens on
    target = tmp_path / "models" / "x-12345678.pt"
    with pytest.raises(OSError):
        models.download_file(f"http://127.0.0.1:{port}/x", target, timeout=2)
    assert leftovers(target.parent) == []


def test_a_failing_progress_callback_does_not_break_the_download(tmp_path, server):
    def boom(_event):
        raise RuntimeError("UI exploded")

    target = tmp_path / "m" / f"fake-{DIGEST8}.pt"
    models.download_file(server.url, target, DIGEST8, boom)
    assert target.read_bytes() == PAYLOAD


def test_download_uses_the_own_downloader_and_skips_cached_models(tmp_path, server, monkeypatch):
    monkeypatch.setenv("DOCTR_CACHE_DIR", str(tmp_path / "cache"))
    source = models.ModelSource(models.RECOGNITION, "fake", server.url, f"fake-{DIGEST8}.pt", DIGEST8)
    monkeypatch.setattr(models, "model_source", lambda kind, name: source)
    events = []
    models.download(models.RECOGNITION, "fake", progress=events.append)
    assert (tmp_path / "cache" / "models" / f"fake-{DIGEST8}.pt").read_bytes() == PAYLOAD
    assert events and server.requests == 1
    assert models.is_cached("fake", models.RECOGNITION)
    models.download(models.RECOGNITION, "fake", progress=events.append)    # already there: no second request
    assert server.requests == 1


def test_unknown_source_falls_back_to_doctrs_own_downloader(monkeypatch):
    def unknown(kind, name):
        raise LookupError("no source")

    called = []
    monkeypatch.setattr(models, "model_source", unknown)
    monkeypatch.setattr(models, "_download_with_doctr", lambda kind, name: called.append((kind, name)))
    models.download(models.DETECTION, "future_model")
    assert called == [(models.DETECTION, "future_model")]


def test_console_progress_prints_a_single_updating_line(capsys):
    show = models.console_progress()
    for done in (0, 250, 500, 1000):
        show(models.DownloadProgress("parseq", done, 1000))
    out = capsys.readouterr().out
    assert out.count("\n") == 1 and out.endswith("\n")                     # one line, finished once
    assert "parseq" in out and "100%" in out and "\r" in out


# --------------------------------------------------------------- against the real DocTR registry
def test_every_selectable_model_and_the_orientation_models_resolve_to_a_url_and_checksum():
    pytest.importorskip("doctr")
    kinds = {models.DETECTION: models.list_models(models.DETECTION),
             models.RECOGNITION: models.list_models(models.RECOGNITION),
             models.CLASSIFICATION: [name for _kind, name in models.ORIENTATION_MODELS]}
    for kind, names in kinds.items():
        assert names
        for name in names:
            source = models.model_source(kind, name)
            assert source.url.startswith("https://") and source.filename.endswith(".pt"), name
            assert source.hash_prefix and source.filename.rsplit("-", 1)[-1].startswith(source.hash_prefix), name


def test_cache_check_knows_models_whose_file_name_differs_from_the_model_name(tmp_path, monkeypatch):
    pytest.importorskip("doctr")
    monkeypatch.setenv("DOCTR_CACHE_DIR", str(tmp_path))
    folder = tmp_path / "models"
    folder.mkdir()
    assert not models.is_cached("crnn_mobilenet_v3_small")
    (folder / "crnn_mobilenet_v3_small_pt-3b919a02.pt").write_bytes(b"weights")   # note the "_pt" in the file name
    assert models.is_cached("crnn_mobilenet_v3_small")
    (folder / "parseq-56125471.pt").write_bytes(b"")                                # empty = failed download
    assert not models.is_cached("parseq")
