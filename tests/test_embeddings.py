import pytest

import app.embeddings as embed_mod


@pytest.fixture(autouse=True)
def _drop_cached_fake_model():
    # _patch() fills the lru_cache'd _model() with a 4-dim fake. monkeypatch restores
    # SentenceTransformer but not the cache, so without this every later test (e.g. the
    # pgvector integration tests) would embed with the fake and fail on dimension 384.
    yield
    embed_mod._model.cache_clear()


class _FakeST:
    def __init__(self, name):
        self.name = name

    def get_sentence_embedding_dimension(self):
        return 4

    def encode(self, texts, **kwargs):
        # deterministic: length-based vector, matches encode(list)->2D
        return [[float(len(t))] * 4 for t in texts]


def _patch(monkeypatch):
    monkeypatch.setattr(embed_mod, "SentenceTransformer", _FakeST)
    embed_mod._model.cache_clear()
    monkeypatch.setattr(embed_mod, "get_settings",
                        lambda: type("S", (), {"embedding_model": "fake/model"})())


def test_embed_texts_shapes(monkeypatch):
    _patch(monkeypatch)
    vecs = embed_mod.embed_texts(["ab", "cde"])
    assert len(vecs) == 2
    assert vecs[0] == [2.0, 2.0, 2.0, 2.0]
    assert vecs[1] == [3.0, 3.0, 3.0, 3.0]
    assert all(isinstance(x, float) for x in vecs[0])


def test_embed_empty_returns_empty(monkeypatch):
    _patch(monkeypatch)
    assert embed_mod.embed_texts([]) == []


def test_embed_query_is_single_vector(monkeypatch):
    _patch(monkeypatch)
    v = embed_mod.embed_query("abc")
    assert v == [3.0, 3.0, 3.0, 3.0]


def test_singleton_reuses_instance_and_dim(monkeypatch):
    _patch(monkeypatch)
    a = embed_mod._model()
    b = embed_mod._model()
    assert a is b
    assert embed_mod.embedding_dim() == 4


def test_fake_model_does_not_leak_to_later_tests():
    # Defined last so it runs after the tests above cached the fake; the autouse
    # teardown must have cleared it, so the cache is empty when this test starts.
    assert embed_mod._model.cache_info().currsize == 0
