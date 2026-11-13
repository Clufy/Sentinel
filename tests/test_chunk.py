from app.ingest.chunk import chunk_control


def test_short_summary_single_chunk():
    chunks = chunk_control("AC-2", "Manage system accounts.", max_chunk_tokens=512)
    assert chunks == ["Manage system accounts."]


def test_split_boundary_exact_no_split():
    words = " ".join(f"w{i}" for i in range(512))  # exactly 512 tokens
    chunks = chunk_control("X", words, max_chunk_tokens=512)
    assert len(chunks) == 1
    assert chunks[0] == words


def test_split_over_boundary():
    words = " ".join(f"w{i}" for i in range(513))  # 513 tokens -> 2 chunks
    chunks = chunk_control("X", words, max_chunk_tokens=512)
    assert len(chunks) == 2
    assert len(chunks[0].split()) == 512
    assert len(chunks[1].split()) == 1
    assert " ".join(chunks).split() == words.split()


def test_small_max_multiple_chunks():
    words = " ".join(f"t{i}" for i in range(10))
    chunks = chunk_control("X", words, max_chunk_tokens=3)
    assert [len(c.split()) for c in chunks] == [3, 3, 3, 1]
    assert " ".join(chunks).split() == words.split()


def test_empty_summary():
    assert chunk_control("X", "", max_chunk_tokens=512) == [""]
