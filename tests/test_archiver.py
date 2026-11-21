from core.archiver import archive_inputs
from core.ocr_processor import OCRProcessor
from tests.conftest import FakeEngine


def run(tmp_path, make_image, heights=(), names=("a.png", "sub/b.png", "sub/c.png")):
    for i, n in enumerate(names):
        make_image(tmp_path / "in" / n, (100, 33 if (i + 1) in heights else 40))
    return OCRProcessor(engine=FakeEngine(fail_heights=(33,))).process_folder(tmp_path / "in", tmp_path / "out")


def test_only_confirmed_files_are_moved_and_structure_kept(tmp_path, make_image):
    result = run(tmp_path, make_image, heights=(2,))            # sub/b.png fails
    report = archive_inputs(result, tmp_path / "arch")
    assert not report.errors
    assert (tmp_path / "arch" / "a.png").exists()
    assert (tmp_path / "arch" / "sub" / "c.png").exists()
    assert (tmp_path / "in" / "sub" / "b.png").exists()          # failed file stays where it was
    assert not (tmp_path / "arch" / "sub" / "b.png").exists()


def test_cancelled_run_archives_nothing(tmp_path, make_image):
    result = run(tmp_path, make_image)
    result.cancelled = True
    report = archive_inputs(result, tmp_path / "arch")
    assert not report.moved and (tmp_path / "in" / "a.png").exists()


def test_collisions_get_suffix_and_empty_dirs_are_removed(tmp_path, make_image):
    (tmp_path / "arch").mkdir()
    (tmp_path / "arch" / "a.png").write_bytes(b"existing")
    result = run(tmp_path, make_image)
    archive_inputs(result, tmp_path / "arch")
    assert (tmp_path / "arch" / "a.png").read_bytes() == b"existing"
    assert (tmp_path / "arch" / "a (2).png").exists()
    assert not (tmp_path / "in" / "sub").exists()               # emptied directory pruned
    assert (tmp_path / "in").exists()                           # root itself kept


def test_archive_inside_input_is_refused(tmp_path, make_image):
    result = run(tmp_path, make_image)
    report = archive_inputs(result, tmp_path / "in" / "archive")
    assert report.errors and not report.moved and (tmp_path / "in" / "a.png").exists()
