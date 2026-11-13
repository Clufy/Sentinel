import asyncio

from app.llm import FakeProvider, Message, ProviderResponse, extract_cited_ids


def test_extract_cited_ids_parses_bracket_citations():
    text = "See [NIST AC-2] and [ISO A.5.1] and [CIS 3.1]."
    assert extract_cited_ids(text) == ["NIST AC-2", "ISO A.5.1", "CIS 3.1"]


def test_extract_cited_ids_dedupes_preserving_order():
    assert extract_cited_ids("[NIST AC-2] then [NIST AC-2] then [ISO A.5.1]") == \
        ["NIST AC-2", "ISO A.5.1"]


def test_fake_provider_is_deterministic():
    p = FakeProvider()
    msgs = [Message(role="user", content="Accounts? Context: [NIST AC-2] [ISO A.5.1]")]
    r1 = asyncio.run(p.chat(msgs))
    r2 = asyncio.run(p.chat(msgs))
    assert isinstance(r1, ProviderResponse)
    assert r1.content == r2.content
    assert r1.tool_calls == r2.tool_calls
    assert r1.input_tokens == r2.input_tokens
    assert r1.output_tokens == r2.output_tokens


def test_fake_provider_turn1_emits_search_controls_tool_call():
    """Turn 1 (no tool message yet present): emit exactly one search_controls
    tool_call whose query is the last user message."""
    p = FakeProvider()
    r = asyncio.run(p.chat([Message(role="user", content="Do accounts need MFA?")]))
    assert len(r.tool_calls) == 1
    call = r.tool_calls[0]
    assert call["name"] == "search_controls"
    assert call["arguments"]["query"] == "Do accounts need MFA?"
    assert r.content == ""


def test_fake_provider_turn2_cites_ids_from_tool_and_user_content():
    """Turn 2+ (a tool message is present): scrape citations from user/tool
    content only, and stop emitting tool_calls."""
    p = FakeProvider()
    messages = [
        Message(role="system", content="System prompt with example [NIST AC-2] ignored."),
        Message(role="user", content="Q. Context: [CIS 3.1]"),
        Message(role="assistant", content=""),
        Message(role="tool", content='{"controls": [{"citation": "[NIST AC-2]"}]}'),
    ]
    r = asyncio.run(p.chat(messages))
    assert "[NIST AC-2]" in r.content
    assert "[CIS 3.1]" in r.content
    assert r.tool_calls == []


def test_fake_provider_turn2_excludes_system_and_assistant_citations():
    """Citations that appear ONLY in system/assistant content must not be cited."""
    p = FakeProvider()
    messages = [
        Message(role="system", content="Example citations: [NIST AC-2] [ISO A.5.1] [CIS 3.1]"),
        Message(role="user", content="No real citations here."),
        Message(role="assistant", content="I will cite [ISO A.5.1] falsely."),
        Message(role="tool", content='{"controls": []}'),
    ]
    r = asyncio.run(p.chat(messages))
    assert extract_cited_ids(r.content) == []
    assert r.tool_calls == []


def test_fake_provider_no_citations_when_context_empty():
    p = FakeProvider()
    messages = [
        Message(role="user", content="No context here."),
        Message(role="tool", content='{"controls": []}'),
    ]
    r = asyncio.run(p.chat(messages))
    assert extract_cited_ids(r.content) == []
    assert r.tool_calls == []


def test_fake_provider_token_counts_length_derived():
    p = FakeProvider()
    short = asyncio.run(p.chat([Message(role="user", content="hi")]))
    long = asyncio.run(p.chat([Message(role="user", content="a longer prompt " * 20)]))
    assert short.input_tokens > 0
    assert long.input_tokens > short.input_tokens
