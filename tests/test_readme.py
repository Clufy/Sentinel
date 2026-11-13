from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def _readme():
    return (ROOT / "README.md").read_text(encoding="utf-8")


def test_exists():
    assert (ROOT / "README.md").exists()


def test_grc_legal_note():
    assert "not legal advice" in _readme().lower()


def test_quickstart_compose_and_ingest():
    t = _readme()
    assert ("docker-compose up" in t or "docker compose up" in t)
    assert "/ingest" in t


def test_curl_query_uses_question_field():
    t = _readme()
    assert "curl" in t and "/query" in t and '"question"' in t


def test_env_table_lists_all_vars():
    t = _readme()
    for key in ("DATABASE_URL", "LLM_PROVIDER", "LLM_API_KEY", "LLM_MODEL",
                "EMBEDDING_MODEL", "OLLAMA_BASE_URL", "MAX_AGENT_STEPS",
                "MAX_REQUEST_TOKENS", "RETRIEVAL_TOP_K", "MIN_RELEVANCE", "MAP_MIN_SIMILARITY",
                "MAX_CHUNK_TOKENS"):
        assert key in t, key


def test_eval_and_hybrid_note():
    t = _readme()
    assert "eval.run_eval" in t or "run_eval" in t
    low = t.lower()
    assert "hybrid" in low and "vector-only" in low


def test_pointers():
    t = _readme()
    assert "corpus/SOURCES.md" in t
    assert "docs/architecture.md" in t


def test_readme_explains_loopback_ports_before_exposing():
    t = _readme()
    assert "127.0.0.1" in t
    assert "before exposing" in t.lower()


# --- M8 + M10: quickstart gaps -------------------------------------------
def test_readme_says_to_pull_the_ollama_model():
    assert "ollama pull llama3.1" in _readme()


def test_readme_linux_ollama_host_note():
    t = _readme()
    assert "OLLAMA_HOST=0.0.0.0" in t and "Linux" in t


def test_readme_first_ingest_downloads_and_needs_internet():
    t = " ".join(_readme().split()).lower()
    assert "first `/ingest`" in t or "first /ingest" in t
    assert "internet" in t and "nist" in t and "10 mb" in t


def test_readme_query_before_ingest_fails():
    t = " ".join(_readme().split()).lower()
    assert "before `/ingest`" in t or "before /ingest" in t


def test_readme_schema_change_needs_down_v():
    assert "docker compose down -v" in _readme()


def test_license_file_and_readme_split_code_and_cis_data():
    lic = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert lic.startswith("MIT License")
    t = _readme()
    assert "MIT" in t and "LICENSE" in t
    # The CIS-derived data is NOT MIT: both files stay CC BY-NC-SA 4.0.
    assert "corpus/cis_v8.yaml" in t and "tests/fixtures/mini_cis.yaml" in t
    assert "CC BY-NC-SA 4.0" in t


def test_license_is_unmodified_mit_text_so_github_detects_it():
    # Extra text in LICENSE makes GitHub label the repo "Other"; the CIS
    # exception lives in the README, corpus/SOURCES.md and the file headers.
    lic = (ROOT / "LICENSE").read_text(encoding="utf-8")
    assert lic.rstrip().endswith("OTHER DEALINGS IN THE\nSOFTWARE.")
