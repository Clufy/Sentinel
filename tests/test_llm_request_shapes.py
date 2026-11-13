"""C1: per-provider REQUEST shapes across a two-turn tool conversation.

Each test drives the real run_agent loop against an httpx.MockTransport that
records and parses every request body. Turn 1 returns a tool call, turn 2 the
final grounded answer. The assertions are on what we SEND, which the response
parsing tests in test_llm_providers.py never covered.
"""
import json

import httpx
import pytest

import app.llm as llm
from app.agent import run_agent
from app.llm import AnthropicProvider, OllamaProvider, OpenAIProvider, get_provider


async def _search(query, framework=None):
    return {"controls": [{"framework": "NIST", "control_id": "AC-2",
                          "citation": "[NIST AC-2]", "summary": "Manage accounts."}]}


TOOLS = {"search_controls": _search}


def _recording_client(responses):
    bodies: list[dict] = []
    queue = list(responses)

    def handler(request: httpx.Request) -> httpx.Response:
        bodies.append(json.loads(request.content))
        return httpx.Response(200, json=queue.pop(0))

    return httpx.AsyncClient(transport=httpx.MockTransport(handler)), bodies


async def test_anthropic_request_shape_two_turns():
    client, bodies = _recording_client([
        {"content": [{"type": "text", "text": "Searching."},
                     {"type": "tool_use", "id": "toolu_01", "name": "search_controls",
                      "input": {"query": "accounts"}}],
         "usage": {"input_tokens": 10, "output_tokens": 3}},
        {"content": [{"type": "text", "text": "Manage accounts [NIST AC-2]."}],
         "usage": {"input_tokens": 20, "output_tokens": 5}},
    ])
    provider = AnthropicProvider(client=client, api_key="sk-a", model="claude-x")
    result = await run_agent("accounts?", provider, tool_functions=TOOLS)
    assert result.refused is False
    assert len(bodies) == 2

    for body in bodies:
        # system is top level, never a message role
        assert isinstance(body["system"], str) and body["system"].strip()
        assert all(m["role"] in ("user", "assistant") for m in body["messages"])
        # tools in Anthropic's {name, description, input_schema} form
        for tool in body["tools"]:
            assert set(tool) == {"name", "description", "input_schema"}
            assert tool["input_schema"]["type"] == "object"

    first = bodies[0]["messages"]
    assert first == [{"role": "user", "content": "accounts?"}]

    second = bodies[1]["messages"]
    assert [m["role"] for m in second] == ["user", "assistant", "user"]
    assistant = second[1]
    assert {"type": "text", "text": "Searching."} in assistant["content"]
    tool_use = [b for b in assistant["content"] if b["type"] == "tool_use"]
    assert tool_use == [{"type": "tool_use", "id": "toolu_01", "name": "search_controls",
                         "input": {"query": "accounts"}}]
    results = second[2]["content"]
    assert len(results) == 1
    assert results[0]["type"] == "tool_result"
    assert results[0]["tool_use_id"] == "toolu_01"
    assert "NIST" in results[0]["content"] and "AC-2" in results[0]["content"]


async def test_anthropic_merges_consecutive_tool_results_into_one_user_message():
    client, bodies = _recording_client([
        {"content": [
            {"type": "tool_use", "id": "toolu_a", "name": "search_controls", "input": {"query": "a"}},
            {"type": "tool_use", "id": "toolu_b", "name": "search_controls", "input": {"query": "b"}},
        ], "usage": {}},
        {"content": [{"type": "text", "text": "Done [NIST AC-2]."}], "usage": {}},
    ])
    provider = AnthropicProvider(client=client, api_key="sk-a", model="claude-x")
    await run_agent("q", provider, tool_functions=TOOLS)
    second = bodies[1]["messages"]
    assert [m["role"] for m in second] == ["user", "assistant", "user"]
    # assistant turn with no text has no empty text block
    assert all(b["type"] == "tool_use" for b in second[1]["content"])
    ids = [b["tool_use_id"] for b in second[2]["content"]]
    assert ids == ["toolu_a", "toolu_b"]


async def test_openai_request_shape_two_turns():
    client, bodies = _recording_client([
        {"choices": [{"message": {"content": None, "tool_calls": [
            {"id": "call_abc", "type": "function",
             "function": {"name": "search_controls", "arguments": "{\"query\": \"accounts\"}"}}]}}],
         "usage": {"prompt_tokens": 10, "completion_tokens": 3}},
        {"choices": [{"message": {"content": "Manage accounts [NIST AC-2].", "tool_calls": []}}],
         "usage": {"prompt_tokens": 20, "completion_tokens": 5}},
    ])
    provider = OpenAIProvider(client=client, api_key="sk-o", model="gpt-x")
    result = await run_agent("accounts?", provider, tool_functions=TOOLS)
    assert result.refused is False
    assert len(bodies) == 2
    assert bodies[0]["tools"][0]["type"] == "function"

    second = bodies[1]["messages"]
    assert [m["role"] for m in second] == ["system", "user", "assistant", "tool"]
    assistant = second[2]
    assert assistant["tool_calls"] == [{
        "id": "call_abc", "type": "function",
        "function": {"name": "search_controls", "arguments": json.dumps({"query": "accounts"})},
    }]
    assert isinstance(assistant["tool_calls"][0]["function"]["arguments"], str)
    tool = second[3]
    assert tool["tool_call_id"] == "call_abc"
    assert "AC-2" in tool["content"]


async def test_ollama_request_shape_two_turns_keeps_plain_messages():
    client, bodies = _recording_client([
        {"message": {"content": "", "tool_calls": [
            {"function": {"name": "search_controls", "arguments": {"query": "accounts"}}}]},
         "prompt_eval_count": 10, "eval_count": 3},
        {"message": {"content": "Manage accounts [NIST AC-2].", "tool_calls": []},
         "prompt_eval_count": 20, "eval_count": 5},
    ])
    provider = OllamaProvider(client=client, base_url="http://ollama", model="llama3.1")
    result = await run_agent("accounts?", provider, tool_functions=TOOLS)
    assert result.refused is False
    second = bodies[1]["messages"]
    assert [m["role"] for m in second] == ["system", "user", "assistant", "tool"]
    for m in second:
        assert set(m) == {"role", "content"}


@pytest.mark.parametrize("name", ["anthropic", "openai"])
def test_get_provider_fails_fast_without_api_key(monkeypatch, name):
    monkeypatch.setattr(llm, "get_settings",
                        lambda: type("S", (), {"llm_provider": name, "llm_model": "m",
                                               "llm_api_key": "", "ollama_base_url": "u"})())
    with pytest.raises(ValueError, match="LLM_API_KEY"):
        get_provider()


@pytest.mark.parametrize("name,cls", [("anthropic", AnthropicProvider), ("openai", OpenAIProvider)])
def test_get_provider_with_api_key_builds_provider(monkeypatch, name, cls):
    monkeypatch.setattr(llm, "get_settings",
                        lambda: type("S", (), {"llm_provider": name, "llm_model": "m",
                                               "llm_api_key": "k", "ollama_base_url": "u"})())
    assert isinstance(get_provider(), cls)


async def test_ollama_sends_8k_context_window():
    # M4: the default num_ctx silently truncates system prompt + schemas + controls.
    client, bodies = _recording_client([
        {"message": {"content": "ok [NIST AC-2].", "tool_calls": []}},
    ])
    provider = OllamaProvider(client=client, base_url="http://ollama", model="llama3.1")
    from app.llm import Message
    await provider.chat([Message(role="user", content="q")])
    assert bodies[0]["options"] == {"num_ctx": 8192}
