import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
LICENSE_URL = "https://creativecommons.org/licenses/by-nc-sa/4.0/"


def _sources():
    return (ROOT / "corpus" / "SOURCES.md").read_text(encoding="utf-8")


def test_exists():
    assert (ROOT / "corpus" / "SOURCES.md").exists()


def test_nist_public_domain():
    assert "NIST" in _sources() and "public domain" in _sources().lower()


def test_nist_oscal_url():
    from app.ingest.load_nist import NIST_CATALOG_URL

    assert NIST_CATALOG_URL in _sources()


def test_iso_paraphrased_no_verbatim():
    t = _sources().lower()
    assert "iso" in t and "paraphrase" in t and ("no verbatim" in t or "not verbatim" in t)


def _section(prefix):
    parts = re.split(r"(?m)^(?=## )", _sources())
    return next(p for p in parts if p.startswith("## " + prefix))


def test_cis_cc_by_nc_sa():
    t = _section("CIS")
    assert "Center for Internet Security" in t
    assert "CC BY-NC-SA 4.0" in t
    assert LICENSE_URL in t
    flat = " ".join(t.split())
    assert "adapted" in flat.lower()
    assert "may not be used for commercial purposes" in flat
    assert "same CC BY-NC-SA 4.0 licence" in flat


def test_cis_not_hand_authored():
    assert "hand-authored" not in _sources().lower()


def test_names_all_three_frameworks():
    t = _sources()
    assert "800-53" in t and "27001" in t and ("v8" in t or "Controls v8" in t)


def test_mappings_original_and_crosslink():
    assert "mappings.yaml" in _sources() and "README.md" in _sources()


CIS_CONTROLS_URL = "https://www.cisecurity.org/controls/"


def test_cis_section_links_cis_controls_for_current_guidance():
    # M6: the CIS v8 licence notice asks users to refer to this page.
    assert CIS_CONTROLS_URL in _section("CIS")


def test_cis_yaml_header_links_cis_controls():
    header = (ROOT / "corpus" / "cis_v8.yaml").read_text(encoding="utf-8").split("framework:")[0]
    assert CIS_CONTROLS_URL in header


def test_mini_cis_fixture_is_described_as_adapted_not_original():
    t = " ".join(_section("Test fixtures").split())
    assert "mini_cis.yaml" in t
    cis_sentence = next(s for s in re.split(r"(?<=\.)\s", t) if "mini_cis.yaml" in s)
    assert "adapted" in cis_sentence.lower()
    assert "CC BY-NC-SA 4.0" in cis_sentence
    assert "original" not in cis_sentence.lower()


def test_mini_cis_fixture_header_carries_licence():
    header = (ROOT / "tests" / "fixtures" / "mini_cis.yaml").read_text(
        encoding="utf-8").split("framework:")[0]
    assert "CC BY-NC-SA 4.0" in header and LICENSE_URL in header
