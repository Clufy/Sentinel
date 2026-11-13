"""Validate the crosswalk mapping references only real controls. Marked corpus;
runs in CI with the full suite."""
from __future__ import annotations

import re
from pathlib import Path

import pytest
import yaml

pytestmark = pytest.mark.corpus

CORPUS_DIR = Path(__file__).resolve().parents[1] / "corpus"
ISO_PATH = CORPUS_DIR / "iso27001.yaml"
CIS_PATH = CORPUS_DIR / "cis_v8.yaml"
MAP_PATH = CORPUS_DIR / "mappings.yaml"

FRAMEWORK_KEYS = {"iso", "nist", "cis"}
NIST_PAT = re.compile(r"^[A-Z]{2}-\d{1,3}$")


def _ids(path: Path) -> set[str]:
    return {e["control_id"] for e in yaml.safe_load(path.read_text(encoding="utf-8"))["controls"]}


def _crosswalk() -> list[dict]:
    data = yaml.safe_load(MAP_PATH.read_text(encoding="utf-8"))
    assert isinstance(data, dict) and "crosswalk" in data, "missing 'crosswalk' key"
    assert isinstance(data["crosswalk"], list)
    return data["crosswalk"]


def test_mappings_file_exists() -> None:
    assert MAP_PATH.is_file()


def test_every_row_has_the_three_framework_keys() -> None:
    for i, row in enumerate(_crosswalk()):
        assert set(row.keys()) == FRAMEWORK_KEYS, f"crosswalk[{i}]: keys must be {sorted(FRAMEWORK_KEYS)}"
        for key in FRAMEWORK_KEYS:
            assert isinstance(row[key], list)
            for v in row[key]:
                assert isinstance(v, str) and v.strip()


def test_every_row_links_at_least_two_frameworks() -> None:
    for i, row in enumerate(_crosswalk()):
        non_empty = [k for k in FRAMEWORK_KEYS if row[k]]
        assert len(non_empty) >= 2, f"crosswalk[{i}]: must link >= 2 frameworks, got {non_empty}"


def test_iso_references_exist() -> None:
    iso_ids = _ids(ISO_PATH)
    for i, row in enumerate(_crosswalk()):
        for cid in row["iso"]:
            assert cid in iso_ids, f"crosswalk[{i}]: unknown ISO id {cid!r}"


def test_cis_references_exist() -> None:
    cis_ids = _ids(CIS_PATH)
    for i, row in enumerate(_crosswalk()):
        for cid in row["cis"]:
            assert cid in cis_ids, f"crosswalk[{i}]: unknown CIS id {cid!r}"


def test_nist_references_match_pattern() -> None:
    for i, row in enumerate(_crosswalk()):
        for cid in row["nist"]:
            assert NIST_PAT.match(cid), f"crosswalk[{i}]: bad NIST id {cid!r}"


def test_crosswalk_has_several_rows() -> None:
    assert len(_crosswalk()) >= 5


def test_by_control_references_exist_in_corpus() -> None:
    data = yaml.safe_load(MAP_PATH.read_text(encoding="utf-8"))
    ids = {"ISO": _ids(ISO_PATH), "CIS": _ids(CIS_PATH)}
    refs = []
    for key, entries in data["by_control"].items():
        fw, cid = key.split(":", 1)
        refs.append((fw, cid))
        refs.extend((e["framework"], e["control_id"]) for e in entries)
    for fw, cid in refs:
        if fw in ids:
            assert cid in ids[fw], f"by_control references unknown {fw} id {cid!r}"
