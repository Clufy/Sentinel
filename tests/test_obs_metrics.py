from app.obs import COST_PER_1K, MetricsStore, RequestRecord, cost_for


def _rec(path="/query", tokens=0, latency=0.0, cost=0.0, refused=False, trace=None):
    return RequestRecord(path=path, total_tokens=tokens, latency_ms=latency,
                         cost_usd=cost, refused=refused, tool_trace=trace or [])


def test_empty_metrics_zeroed():
    assert MetricsStore().get_metrics() == {
        "total_requests": 0, "total_tokens": 0, "total_cost_usd": 0.0,
        "avg_latency_ms": 0.0, "refusals": 0,
    }


def test_metrics_aggregate():
    store = MetricsStore()
    store.record(_rec(tokens=100, latency=200.0, cost=0.01, refused=False))
    store.record(_rec(tokens=50, latency=100.0, cost=0.0, refused=True))
    m = store.get_metrics()
    assert m["total_requests"] == 2
    assert m["total_tokens"] == 150
    assert m["total_cost_usd"] == 0.01
    assert m["avg_latency_ms"] == 150.0
    assert m["refusals"] == 1


def test_reset_clears():
    store = MetricsStore()
    store.record(_rec(tokens=10, latency=5.0))
    store.reset()
    assert store.get_metrics()["total_requests"] == 0


def test_cost_for_ollama_and_fake_are_zero():
    assert cost_for("ollama", "llama3.1", 1000, 1000) == 0.0
    assert cost_for("fake", "fake", 1000, 1000) == 0.0


def test_cost_for_anthropic_uses_table():
    price = COST_PER_1K["anthropic"]
    expected = (2000 / 1000) * price["input"] + (500 / 1000) * price["output"]
    assert cost_for("anthropic", "claude-x", 2000, 500) == expected
