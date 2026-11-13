from pathlib import Path

import yaml

ROOT = Path(__file__).resolve().parents[1]


def _compose():
    return yaml.safe_load((ROOT / "docker-compose.yml").read_text(encoding="utf-8"))


def test_app_service_present_and_builds():
    app = _compose()["services"]["app"]
    assert "build" in app
    assert any("8000" in str(p) for p in app["ports"])


def test_app_depends_on_db_healthy():
    assert _compose()["services"]["app"]["depends_on"]["db"]["condition"] == "service_healthy"


def test_env_file_wired():
    services = _compose()["services"]
    assert services["app"].get("env_file") == [".env"]


def test_db_has_no_env_file():
    assert "env_file" not in _compose()["services"]["db"]


def test_app_container_overrides():
    env = _compose()["services"]["app"]["environment"]
    assert "@db:5432/" in env["DATABASE_URL"]
    assert env["OLLAMA_BASE_URL"] == "http://host.docker.internal:11434"


def test_app_extra_hosts_for_linux_host_gateway():
    app = _compose()["services"]["app"]
    assert "host.docker.internal:host-gateway" in app["extra_hosts"]


def _env_example():
    return (ROOT / ".env.example").read_text(encoding="utf-8")


def test_env_example_covers_all_settings():
    text = _env_example()
    for key in ("DATABASE_URL", "LLM_PROVIDER", "LLM_API_KEY", "LLM_MODEL",
                "EMBEDDING_MODEL", "OLLAMA_BASE_URL", "MAX_AGENT_STEPS",
                "MAX_REQUEST_TOKENS", "RETRIEVAL_TOP_K", "MIN_RELEVANCE", "MAP_MIN_SIMILARITY",
                "MAX_CHUNK_TOKENS"):
        assert f"{key}=" in text, f"{key} missing from .env.example"


def test_env_example_is_host_oriented():
    text = _env_example()
    # 127.0.0.1, not localhost: compose publishes 5432 on the IPv4 loopback only,
    # and on Windows `localhost` resolves to ::1 first, which hangs until timeout.
    assert "@127.0.0.1:5432" in text
    assert "@db:" not in text
    assert "host.docker.internal" not in text


def test_default_dsn_uses_ipv4_loopback_that_compose_binds():
    from app.config import Settings

    assert "@127.0.0.1:5432/" in Settings.model_fields["database_url"].default
    assert '"127.0.0.1:5432:5432"' in (ROOT / "docker-compose.yml").read_text(encoding="utf-8")


def test_app_port_bound_to_loopback_only():
    # I8: /ingest is unauthenticated; do not publish it on all interfaces.
    assert _compose()["services"]["app"]["ports"] == ["127.0.0.1:8000:8000"]
