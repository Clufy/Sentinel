# tests/test_agent_mappings.py
from app.agent import load_mappings, related_controls


def test_load_mappings_keyed_by_framework_and_id():
    m = load_mappings()
    assert "NIST:AC-2" in m
    assert {"framework": "ISO", "control_id": "A.5.15"} in m["NIST:AC-2"]


def test_load_mappings_is_cached_singleton():
    assert load_mappings() is load_mappings()


def test_related_controls_returns_cross_framework_equivalents():
    rel = related_controls("NIST", "AC-2")
    frameworks = {e["framework"] for e in rel}
    assert "ISO" in frameworks
    assert "CIS" in frameworks


def test_related_controls_unknown_returns_empty():
    assert related_controls("NIST", "ZZ-99") == []


def test_related_controls_symmetric_when_declared():
    assert {"framework": "NIST", "control_id": "AC-2"} in related_controls("ISO", "A.5.15")


def test_crosswalk_only_link_is_served():
    # NIST:AC-2 -> CIS-6 exists only in the vetted crosswalk, not by_control.
    assert {"framework": "CIS", "control_id": "CIS-6"} in related_controls("NIST", "AC-2")
    assert {"framework": "NIST", "control_id": "AC-2"} in related_controls("CIS", "CIS-6")


def test_by_control_entries_come_first_and_no_duplicates_or_self_or_same_framework():
    m = load_mappings()
    rel = m["NIST:AC-2"]
    assert rel[:3] == [
        {"framework": "ISO", "control_id": "A.5.15"},
        {"framework": "ISO", "control_id": "A.5.16"},
        {"framework": "CIS", "control_id": "CIS-5"},
    ]
    for key, entries in m.items():
        fw, cid = key.split(":", 1)
        tuples = [(e["framework"], e["control_id"]) for e in entries]
        assert len(tuples) == len(set(tuples)), f"{key}: duplicate links"
        assert (fw, cid) not in tuples, f"{key}: links to itself"
        assert all(f != fw for f, _ in tuples), f"{key}: same-framework link"
