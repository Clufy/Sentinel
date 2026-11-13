from pathlib import Path

import yaml

WORKFLOW = Path(".github/workflows/ci.yml")


def _load():
    return yaml.safe_load(WORKFLOW.read_text(encoding="utf-8"))


def test_workflow_parses_and_named():
    data = _load()
    assert data["name"]
    assert (True in data) or ("on" in data)  # PyYAML parses bare `on:` as True


def test_uses_pgvector_service():
    services = _load()["jobs"]["test"]["services"]
    assert any("pgvector" in s["image"] for s in services.values())


def test_runs_init_db_ingest_fixture_pytest_eval():
    steps = _load()["jobs"]["test"]["steps"]
    run_cmds = " ".join(s.get("run", "") for s in steps)
    assert "scripts.init_db" in run_cmds
    assert "scripts.ingest_fixture" in run_cmds
    assert "uv run pytest -q" in run_cmds
    assert "eval.run_eval" in run_cmds
    assert "app.db_migrate" not in run_cmds  # must NOT use the nonexistent migrate module


def test_pytest_step_runs_the_corpus_guards():
    # I6: the copyright guard, count invariants and mapping-integrity tests are
    # fast YAML reads; CI must not deselect them.
    runs = [s.get("run", "") for s in _load()["jobs"]["test"]["steps"]]
    pytest_runs = [r for r in runs if "pytest" in r]
    assert pytest_runs
    for r in pytest_runs:
        assert "not corpus" not in r
        assert " -m " not in f" {r} "


def test_installs_uv_and_syncs():
    steps = _load()["jobs"]["test"]["steps"]
    joined = " ".join(str(s) for s in steps)
    assert "uv" in joined
    assert "uv sync" in joined


def test_sets_env_including_fake_provider_for_eval():
    data = _load()["jobs"]["test"]
    blob = str(data.get("env", "")) + " ".join(str(s) for s in data["steps"])
    assert "DATABASE_URL" in blob
    assert "LLM_PROVIDER" in blob and "fake" in blob


def test_ingest_fixture_reruns_directly_before_eval():
    # pytest integration fixtures TRUNCATE `controls`, so the eval
    # must be preceded immediately by a fresh fixture ingest.
    runs = [s.get("run", "") for s in _load()["jobs"]["test"]["steps"]]
    eval_idx = next(i for i, r in enumerate(runs) if "eval.run_eval" in r)
    pytest_idx = next(i for i, r in enumerate(runs) if "pytest" in r)
    assert "scripts.ingest_fixture" in runs[eval_idx - 1]
    assert pytest_idx < eval_idx - 1


def test_pgvector_service_pinned_to_0_8_or_later():
    services = _load()["jobs"]["test"]["services"]
    image = next(s["image"] for s in services.values() if "pgvector" in s["image"])
    tag = image.split(":", 1)[1]
    major, minor = (int(x) for x in tag.split("-", 1)[0].split(".")[:2])
    assert (major, minor) >= (0, 8), image


def test_ruff_runs_before_pytest():
    # M11: lint gate over the same paths the review checked.
    runs = [s.get("run", "") for s in _load()["jobs"]["test"]["steps"]]
    ruff_idx = next(i for i, r in enumerate(runs) if "ruff check" in r)
    pytest_idx = next(i for i, r in enumerate(runs) if "pytest" in r)
    assert "uv run ruff check app eval scripts tests" in runs[ruff_idx]
    assert ruff_idx < pytest_idx


def test_job_has_timeout():
    # M12
    assert _load()["jobs"]["test"]["timeout-minutes"] == 30
