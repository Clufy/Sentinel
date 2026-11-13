from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _compose():
    return yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))


def test_db_service_present():
    assert "db" in _compose()["services"]


def test_db_uses_pgvector_image():
    assert "pgvector" in _compose()["services"]["db"]["image"]


def test_db_has_healthcheck_and_port():
    db = _compose()["services"]["db"]
    joined = " ".join(db["healthcheck"]["test"]) if isinstance(db["healthcheck"]["test"], list) else str(db["healthcheck"]["test"])
    assert "pg_isready" in joined
    assert any("5432" in str(p) for p in db["ports"])


def test_db_port_bound_to_loopback_only():
    # I8: a known superuser password must not be published on all interfaces.
    assert _compose()["services"]["db"]["ports"] == ["127.0.0.1:5432:5432"]


def test_db_image_pinned_to_pgvector_0_8_or_later():
    # M1: vector_search uses hnsw.iterative_scan, which needs pgvector >= 0.8.
    # The floating `pg16` tag can resolve to an older cached image.
    image = _compose()["services"]["db"]["image"]
    tag = image.split(":", 1)[1]
    major, minor = (int(x) for x in tag.split("-", 1)[0].split(".")[:2])
    assert (major, minor) >= (0, 8), image
