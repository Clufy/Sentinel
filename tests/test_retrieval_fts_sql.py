import app.retrieval as R


class _Con:
    def __init__(self, sink):
        self.sink = sink

    async def fetch(self, sql, *params):
        self.sink["sql"], self.sink["params"] = sql, params
        return []


class _Acq:
    def __init__(self, sink):
        self.sink = sink

    async def __aenter__(self):
        return _Con(self.sink)

    async def __aexit__(self, *a):
        return False


class _Pool:
    def __init__(self, sink):
        self.sink = sink

    def acquire(self):
        return _Acq(self.sink)


async def _run(monkeypatch, framework):
    sink = {}

    async def fake_pool():
        return _Pool(sink)

    monkeypatch.setattr(R, "get_pool", fake_pool)
    await R.fts_search("Which control covers account management?", framework, 5)
    return sink


async def test_fts_query_ors_lexemes_not_ands(monkeypatch):
    sql = (await _run(monkeypatch, None))["sql"]
    assert "' & ', ' | '" in sql          # lexemes OR-ed
    assert "plainto_tsquery('english', $1)" in sql  # stemming/stopwords still handled
    assert "ts_rank_cd" in sql


async def test_fts_framework_filter_kept(monkeypatch):
    sink = await _run(monkeypatch, "CIS")
    assert "framework = $3" in sink["sql"]
    assert sink["params"][-1] == "CIS"
