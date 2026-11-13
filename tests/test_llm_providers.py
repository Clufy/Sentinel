import asyncio

import httpx
import pytest

import app.llm as llm
from app.llm import (
    AnthropicProvider,
    FakeProvider,
    Message,
    OllamaProvider,
    OpenAIProvider,
    get_provider,
)


def _client(handler):
    return httpx.AsyncClient(transport=httpx.MockTransport(handler))


def test_ollama_provider_maps_response():
    def handler(request):
        assert request.url.path == "/api/chat"
        return httpx.Response(200, json={
            "message": {"content": "MFA required [NIST IA-2].", "tool_calls": []},
            "prompt_eval_count": 12, "eval_count": 5,
        })

    p = OllamaProvider(client=_client(handler), base_url="http://ollama", model="llama3.1")
    r = asyncio.run(p.chat([Message(role="user", content="mfa?")]))
    assert r.content == "MFA required [NIST IA-2]."
    assert r.input_tokens == 12
    assert r.output_tokens == 5


def test_ollama_provider_maps_nested_tool_calls_to_flat_shape():
    def handler(request):
        assert request.url.path == "/api/chat"
        return httpx.Response(200, json={
            "message": {
                "content": "",
                "tool_calls": [
                    {
                        "id": "call-abc",
                        "function": {
                            "name": "search_controls",
                            "arguments": {"query": "mfa for accounts"},
                        },
                    }
                ],
            },
            "prompt_eval_count": 8, "eval_count": 0,
        })

    p = OllamaProvider(client=_client(handler), base_url="http://ollama", model="llama3.1")
    r = asyncio.run(p.chat([Message(role="user", content="mfa?")]))
    assert r.tool_calls == [
        {"id": "call-abc", "name": "search_controls", "arguments": {"query": "mfa for accounts"}}
    ]


def test_anthropic_provider_maps_usage():
    def handler(request):
        assert request.headers.get("x-api-key") == "sk-test"
        return httpx.Response(200, json={
            "content": [{"type": "text", "text": "See [NIST AC-2]."}],
            "usage": {"input_tokens": 20, "output_tokens": 7},
        })

    p = AnthropicProvider(client=_client(handler), api_key="sk-test", model="claude-x")
    r = asyncio.run(p.chat([Message(role="user", content="accounts?")]))
    assert "[NIST AC-2]" in r.content
    assert r.input_tokens == 20
    assert r.output_tokens == 7


def test_openai_provider_maps_usage():
    def handler(request):
        assert request.headers.get("authorization") == "Bearer sk-o"
        return httpx.Response(200, json={
            "choices": [{"message": {"content": "See [ISO A.5.1].", "tool_calls": []}}],
            "usage": {"prompt_tokens": 15, "completion_tokens": 4},
        })

    p = OpenAIProvider(client=_client(handler), api_key="sk-o", model="gpt-x")
    r = asyncio.run(p.chat([Message(role="user", content="policy?")]))
    assert "[ISO A.5.1]" in r.content
    assert r.input_tokens == 15
    assert r.output_tokens == 4


def test_get_provider_selects_by_settings(monkeypatch):
    monkeypatch.setattr(llm, "get_settings",
                        lambda: type("S", (), {"llm_provider": "fake", "llm_model": "m",
                                               "llm_api_key": "", "ollama_base_url": "u"})())
    assert isinstance(get_provider(), FakeProvider)


def test_get_provider_unknown_raises(monkeypatch):
    monkeypatch.setattr(llm, "get_settings",
                        lambda: type("S", (), {"llm_provider": "bogus", "llm_model": "m",
                                               "llm_api_key": "", "ollama_base_url": "u"})())
    with pytest.raises(ValueError):
        get_provider()
