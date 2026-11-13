"""Sentinel runs two small MiniLM models on CPU. Default PyPI torch on Linux drags in
the CUDA stack (several GB), which bloated the image until Docker ran out of disk."""
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _pyproject() -> dict:
    return tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))


def test_torch_comes_from_the_cpu_wheel_index():
    py = _pyproject()
    assert any(d.startswith("torch") for d in py["project"]["dependencies"])
    indexes = {i["name"]: i for i in py["tool"]["uv"]["index"]}
    assert indexes["pytorch-cpu"]["url"] == "https://download.pytorch.org/whl/cpu"
    assert indexes["pytorch-cpu"]["explicit"] is True
    sources = py["tool"]["uv"]["sources"]["torch"]
    assert any(s["index"] == "pytorch-cpu" for s in sources)


def test_lock_has_no_cuda_packages():
    lock = (ROOT / "uv.lock").read_text(encoding="utf-8")
    assert 'name = "nvidia-' not in lock
