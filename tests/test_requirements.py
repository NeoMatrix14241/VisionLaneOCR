"""Guards the install recipe: a CUDA torch must survive `pip install -r requirements.txt`."""
from pathlib import Path

from packaging.requirements import Requirement
from packaging.version import Version

ROOT = Path(__file__).resolve().parent.parent


def parse(name: str) -> tuple[list[str], dict[str, Requirement]]:
    options, requirements = [], {}
    for raw in (ROOT / name).read_text(encoding="utf-8").splitlines():
        line = raw.split("#", 1)[0].strip()
        if not line:
            continue
        if line.startswith("-"):
            options.append(line)
        else:
            requirement = Requirement(line)
            requirements[requirement.name.lower()] = requirement
    return options, requirements


def test_runtime_requirements_accept_cuda_builds_of_torch():
    _, reqs = parse("requirements.txt")
    for name, tested in (("torch", "2.15.0+cu134"), ("torch", "2.14.1"), ("torchvision", "0.29.2")):
        spec = reqs[name].specifier
        assert Version(tested) in spec, f"{name} {tested} must satisfy {spec} (exact pins would replace a CUDA build)"
        assert "==" not in str(spec), f"{name} must not be pinned exactly in requirements.txt"


def test_gpu_file_installs_torch_from_the_pytorch_cuda_index():
    options, reqs = parse("requirements-torch-gpu.txt")
    assert any(o.startswith("--index-url https://download.pytorch.org/whl/cu") for o in options)   # note: /whl/
    assert set(reqs) == {"torch", "torchvision"}


def test_deploy_script_offers_the_gpu_mode():
    script = (ROOT / "deploy_pyvenv.bat").read_text(encoding="utf-8")
    assert 'if /i "%~1"=="gpu"' in script and "requirements-torch-gpu.txt" in script
