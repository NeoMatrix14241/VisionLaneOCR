import os

import pytest

from utils import models


@pytest.fixture
def clean_env(monkeypatch, tmp_path):
    saved = os.environ.pop("DOCTR_CACHE_DIR", None)
    monkeypatch.setattr(models, "app_root", lambda: tmp_path / "app")
    monkeypatch.setattr(models, "data_dir", lambda: tmp_path / "userdata")
    monkeypatch.setattr(models.Path, "home", staticmethod(lambda: tmp_path / "home"))
    yield tmp_path
    os.environ.pop("DOCTR_CACHE_DIR", None)
    if saved is not None:
        os.environ["DOCTR_CACHE_DIR"] = saved


def make_legacy(tmp_path):
    legacy = tmp_path / "home" / ".cache" / "doctr" / "models"
    legacy.mkdir(parents=True)
    (legacy / "db_resnet50-79bd7d70.pt").write_bytes(b"weights")
    (legacy / "notes.txt").write_text("not a model")
    return legacy


def test_default_is_beside_the_app_and_old_models_are_copied(clean_env):
    legacy = make_legacy(clean_env)
    base = models.configure_cache()
    assert base == clean_env / "app" / "doctr_cache"
    assert os.environ["DOCTR_CACHE_DIR"] == str(base)
    assert models.doctr_cache_dir() == base / "models"
    assert models.is_cached("db_resnet50") and not models.is_cached("parseq")
    assert not (base / "models" / "notes.txt").exists()          # only weights are copied
    assert (legacy / "db_resnet50-79bd7d70.pt").exists()          # the old copy is left alone


def test_existing_files_are_not_overwritten(clean_env):
    make_legacy(clean_env)
    target = clean_env / "app" / "doctr_cache" / "models"
    target.mkdir(parents=True)
    (target / "db_resnet50-79bd7d70.pt").write_bytes(b"newer")
    models.configure_cache()
    assert (target / "db_resnet50-79bd7d70.pt").read_bytes() == b"newer"


def test_explicit_env_var_wins_and_nothing_is_copied(clean_env):
    make_legacy(clean_env)
    custom = clean_env / "custom"
    os.environ["DOCTR_CACHE_DIR"] = str(custom)
    assert models.configure_cache() == custom
    assert not models.is_cached("db_resnet50")
    assert not (clean_env / "app").exists()


def test_unwritable_app_folder_falls_back_to_user_data(clean_env, monkeypatch):
    real = models._writable
    monkeypatch.setattr(models, "_writable", lambda folder: False if "app" in folder.parts[-2:] else real(folder))
    base = models.configure_cache()
    assert base == clean_env / "userdata" / "doctr_cache"


def test_empty_leftovers_of_failed_downloads_are_ignored(clean_env):
    legacy = make_legacy(clean_env)
    (legacy / "parseq-56125471.pt").write_bytes(b"")             # failed download
    models.configure_cache()
    assert models.is_cached("db_resnet50")
    assert not models.is_cached("parseq")
    assert not (clean_env / "app" / "doctr_cache" / "models" / "parseq-56125471.pt").exists()
