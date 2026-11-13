from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _read(name):
    return (ROOT / name).read_text(encoding="utf-8")


def test_dockerfile_exists():
    assert (ROOT / "Dockerfile").exists()


def test_uses_python_312_slim():
    assert "python:3.12-slim" in _read("Dockerfile")


def test_is_multistage():
    from_lines = [ln for ln in _read("Dockerfile").splitlines() if ln.strip().upper().startswith("FROM ")]
    assert len(from_lines) >= 2
    assert any(" as builder" in ln.lower() for ln in from_lines)


def test_uses_uv_sync():
    df = _read("Dockerfile")
    assert "uv" in df and "uv sync" in df


def test_runs_as_non_root():
    users = [ln.strip() for ln in _read("Dockerfile").splitlines() if ln.strip().upper().startswith("USER ")]
    assert users and users[-1].split()[1] != "root"


def test_entrypoint_uvicorn():
    df = _read("Dockerfile")
    assert "uvicorn" in df and "app.main:app" in df and "0.0.0.0" in df


def test_dockerignore_excludes():
    di = _read(".dockerignore")
    assert ".venv" in di and ".env" in di and ".git" in di
