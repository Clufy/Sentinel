"""Schema-level validation for the hand-authored governance corpus (data files)."""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

CORPUS_DIR = Path(__file__).resolve().parents[1] / "corpus"
ISO_PATH = CORPUS_DIR / "iso27001.yaml"
CIS_PATH = CORPUS_DIR / "cis_v8.yaml"

EXPECTED_FRAMEWORK = {"iso27001.yaml": "ISO", "cis_v8.yaml": "CIS"}
REQUIRED_KEYS = {"control_id", "title", "summary"}


def load_corpus(path: Path) -> list[dict]:
    with path.open(encoding="utf-8") as fh:
        data = yaml.safe_load(fh)
    assert isinstance(data, dict), f"{path.name}: top level must be a mapping"
    assert data.get("framework") == EXPECTED_FRAMEWORK[path.name], (
        f"{path.name}: top-level 'framework' must be {EXPECTED_FRAMEWORK[path.name]!r}"
    )
    assert "controls" in data, f"{path.name}: missing top-level 'controls' key"
    controls = data["controls"]
    assert isinstance(controls, list), f"{path.name}: 'controls' must be a list"
    return controls


@pytest.mark.parametrize("path", [ISO_PATH, CIS_PATH])
def test_corpus_file_exists(path: Path) -> None:
    assert path.is_file(), f"missing corpus file: {path}"


@pytest.mark.parametrize("path", [ISO_PATH, CIS_PATH])
def test_every_entry_has_exactly_three_keys(path: Path) -> None:
    controls = load_corpus(path)
    assert controls, f"{path.name}: no controls authored yet"
    for i, entry in enumerate(controls):
        assert isinstance(entry, dict), f"{path.name}[{i}]: entry must be a mapping"
        assert set(entry.keys()) == REQUIRED_KEYS, (
            f"{path.name}[{i}] control_id={entry.get('control_id')!r}: "
            f"keys must be exactly {sorted(REQUIRED_KEYS)}, got {sorted(entry.keys())}"
        )


@pytest.mark.parametrize("path", [ISO_PATH, CIS_PATH])
def test_all_values_are_nonempty_strings(path: Path) -> None:
    for entry in load_corpus(path):
        for key in REQUIRED_KEYS:
            val = entry[key]
            assert isinstance(val, str), f"{path.name} {entry.get('control_id')!r}: {key} must be str"
            assert val.strip(), f"{path.name} {entry.get('control_id')!r}: {key} must be non-empty"


@pytest.mark.parametrize("path", [ISO_PATH, CIS_PATH])
def test_control_ids_are_unique(path: Path) -> None:
    ids = [e["control_id"] for e in load_corpus(path)]
    dupes = {cid for cid in ids if ids.count(cid) > 1}
    assert not dupes, f"{path.name}: duplicate control_id(s): {sorted(dupes)}"


def test_iso_control_ids_match_annex_a_pattern() -> None:
    pat = re.compile(r"^A\.(5|6|7|8)\.\d{1,2}$")
    for entry in load_corpus(ISO_PATH):
        assert pat.match(entry["control_id"]), f"ISO id not Annex A form: {entry['control_id']!r}"


def test_cis_control_ids_match_pattern() -> None:
    pat = re.compile(r"^CIS-\d{1,2}(\.\d{1,2})?$")
    for entry in load_corpus(CIS_PATH):
        assert pat.match(entry["control_id"]), f"CIS id not expected form: {entry['control_id']!r}"


def test_files_declare_framework_for_loader() -> None:
    from app.ingest.load_yaml import parse_yaml_file

    assert {p["framework"] for p in parse_yaml_file(ISO_PATH)} == {"ISO"}
    assert {p["framework"] for p in parse_yaml_file(CIS_PATH)} == {"CIS"}


def test_load_corpus_dir_has_no_empty_framework() -> None:
    from app.ingest.load_yaml import load_corpus_dir

    rows = load_corpus_dir(CORPUS_DIR)
    assert rows
    assert all(r["framework"] for r in rows)
