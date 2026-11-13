import importlib
import tomllib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def test_app_package_importable():
    mod = importlib.import_module("app")
    assert mod is not None


def test_pyproject_declares_core_deps():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    deps = data["project"]["dependencies"]
    joined = " ".join(deps).lower()
    for pkg in ("fastapi", "uvicorn", "pydantic", "pydantic-settings",
                "asyncpg", "sse-starlette", "sentence-transformers", "httpx",
                "pyyaml"):
        assert pkg in joined, f"missing dependency: {pkg}"


def test_pyproject_registers_pytest_markers():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    markers = data["tool"]["pytest"]["ini_options"]["markers"]
    joined = " ".join(markers)
    assert "unit" in joined
    assert "integration" in joined
    assert "corpus" in joined


def test_requires_python_312():
    data = tomllib.loads((ROOT / "pyproject.toml").read_text(encoding="utf-8"))
    assert ">=3.12" in data["project"]["requires-python"]
