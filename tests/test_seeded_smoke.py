import pytest
import pytest_asyncio
from fastapi.testclient import TestClient

import app.main as main_mod
import app.db as db_mod
from app.llm import FakeProvider
from app.main import app
from scripts.ingest_fixture import main as ingest_fixture_main


@pytest_asyncio.fixture
async def fixture_seeded():
    await ingest_fixture_main()
    yield
    # The request handler's pool lives on the TestClient's own (now dead) loop; awaiting
    # a close from this loop is unsafe, so abandon it and reset the global for later tests.
    db_mod._pool = None


@pytest.mark.integration
@pytest.mark.usefixtures("fixture_seeded")
def test_seeded_query_returns_grounded_answer(monkeypatch):
    monkeypatch.setattr(main_mod, "get_provider", lambda: FakeProvider())
    client = TestClient(app)
    with client.stream("POST", "/query",
                       json={"question": "how do I manage user accounts?"}) as resp:
        assert resp.status_code == 200
        body = "".join(resp.iter_text())
    assert "event: token" in body
    assert "event: citations" in body
    assert "NIST AC-2" in body
