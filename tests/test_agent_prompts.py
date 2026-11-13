# tests/test_agent_prompts.py
import pytest

from app.agent import load_prompt


def test_load_prompt_system_mentions_citation_format():
    text = load_prompt("system")
    assert text.strip()
    assert "citation" in text.lower()
    assert "[NIST" in text


def test_load_prompt_map_mentions_framework():
    assert "framework" in load_prompt("map").lower()


def test_load_prompt_is_cached():
    assert load_prompt("system") is load_prompt("system")


def test_load_prompt_unknown_raises():
    with pytest.raises(FileNotFoundError):
        load_prompt("does_not_exist")


def test_system_prompt_cis_example_matches_stored_ids():
    text = load_prompt("system")
    assert "[CIS CIS-3.1]" in text
    assert "[CIS 3.1]" not in text
