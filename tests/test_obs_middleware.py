from fastapi import FastAPI, Request
from fastapi.testclient import TestClient

from app.obs import MetricsStore, install_obs


def _app(store):
    app = FastAPI()
    install_obs(app, store)

    @app.get("/plain")
    async def plain():
        return {"ok": True}

    @app.get("/rich")
    async def rich(request: Request):
        request.state.total_tokens = 42
        request.state.cost_usd = 0.02
        request.state.refused = True
        request.state.tool_trace = [{"tool": "search_controls", "args": {}, "result_summary": "x"}]
        return {"ok": True}

    return app


def test_middleware_records_defaults():
    store = MetricsStore()
    TestClient(_app(store)).get("/plain")
    m = store.get_metrics()
    assert m["total_requests"] == 1
    assert m["total_tokens"] == 0
    assert m["refusals"] == 0
    assert store.records[-1].path == "/plain"


def test_middleware_reads_state():
    store = MetricsStore()
    TestClient(_app(store)).get("/rich")
    rec = store.records[-1]
    assert rec.total_tokens == 42
    assert rec.cost_usd == 0.02
    assert rec.refused is True
    assert rec.tool_trace[0]["tool"] == "search_controls"


def test_middleware_positive_latency():
    store = MetricsStore()
    TestClient(_app(store)).get("/plain")
    assert store.records[-1].latency_ms > 0.0


# --- I7: the per-request tool trace is visible in the log text ------------
def test_log_line_carries_trace_as_json_text(caplog):
    import json
    import logging

    store = MetricsStore()
    with caplog.at_level(logging.INFO, logger="sentinel.obs"):
        TestClient(_app(store)).get("/rich")
    rec = next(r for r in caplog.records if r.name == "sentinel.obs")
    msg = rec.getMessage()
    assert msg.startswith("request ")
    payload = json.loads(msg[len("request "):])
    assert payload["path"] == "/rich"
    assert payload["tokens"] == 42
    assert payload["refused"] is True
    assert payload["tool_trace"] == [{"tool": "search_controls", "args": {},
                                      "result_summary": "x"}]


def test_create_app_enables_sentinel_info_logging():
    import logging

    from app.main import create_app

    logging.getLogger("sentinel").setLevel(logging.NOTSET)
    create_app()
    assert logging.getLogger("sentinel.obs").isEnabledFor(logging.INFO)
    # idempotent: a second app does not stack another root handler
    before = len(logging.getLogger().handlers)
    create_app()
    assert len(logging.getLogger().handlers) == before
