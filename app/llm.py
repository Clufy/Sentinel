"""One Provider interface + all concrete providers + FakeProvider + get_provider."""
from __future__ import annotations

import json
from typing import Protocol

import httpx
from pydantic import BaseModel

from app.config import get_settings
from app.grounding import _CITATION_RE, format_citation


class Message(BaseModel):
    role: str
    content: str
    # assistant turns that called tools: [{id, name, arguments(dict)}]
    tool_calls: list[dict] | None = None
    # tool-result turns: the id of the tool call this result answers
    tool_call_id: str | None = None


class ProviderResponse(BaseModel):
    content: str
    tool_calls: list[dict] = []  # [{id, name, arguments}]
    input_tokens: int = 0
    output_tokens: int = 0


class Provider(Protocol):
    async def chat(
        self, messages: list[Message], tools: list[dict] | None = None
    ) -> ProviderResponse: ...


def extract_cited_ids(text: str) -> list[str]:
    return list(dict.fromkeys(f"{fw} {cid}" for fw, cid in _CITATION_RE.findall(text)))


class FakeProvider:
    """Deterministic, network-free Provider for eval + tests. No API key.

    Two-turn contract:
      Turn 1 (no `tool`-role message yet present): emit a single
      `search_controls` tool_call whose query is the last user message.
      Turn 2+ (a `tool`-role message is present): scrape citations from
      only the user/tool message contents and answer (grounded or refusal).
    """

    async def chat(
        self, messages: list[Message], tools: list[dict] | None = None
    ) -> ProviderResponse:
        joined = "\n".join(m.content for m in messages)
        has_tool_message = any(m.role == "tool" for m in messages)

        if not has_tool_message:
            last_user = next((m.content for m in reversed(messages) if m.role == "user"), "")
            content = ""
            return ProviderResponse(
                content=content,
                tool_calls=[
                    {
                        "id": "call-1",
                        "name": "search_controls",
                        "arguments": {"query": last_user},
                    }
                ],
                input_tokens=len(joined.split()),
                output_tokens=len(content.split()),
            )

        joined_context = "\n".join(
            m.content for m in messages if m.role in ("user", "tool")
        )
        cited = extract_cited_ids(joined_context)
        if cited:
            tags = " ".join(
                format_citation(*c.split(" ", 1)) for c in cited
            )
            content = (
                "Based on the retrieved controls, the applicable guidance is "
                f"summarized as follows. {tags}"
            )
        else:
            content = (
                "I could not find governing controls relevant to this question "
                "in the corpus."
            )
        return ProviderResponse(
            content=content,
            tool_calls=[],
            input_tokens=len(joined.split()),
            output_tokens=len(content.split()),
        )


def _messages_payload(messages: list[Message]) -> list[dict]:
    """Ollama shape: plain role/content. Ollama tolerates tool turns this way."""
    return [{"role": m.role, "content": m.content} for m in messages]


def parse_tool_arguments(arguments) -> dict:
    """Tool-call arguments as a dict: JSON strings are decoded, anything else is {}."""
    if isinstance(arguments, dict):
        return arguments
    if isinstance(arguments, str):
        try:
            parsed = json.loads(arguments)
        except json.JSONDecodeError:
            return {}
        return parsed if isinstance(parsed, dict) else {}
    return {}


def _openai_messages(messages: list[Message]) -> list[dict]:
    out: list[dict] = []
    for m in messages:
        if m.role == "assistant" and m.tool_calls:
            out.append({
                "role": "assistant",
                "content": m.content or None,
                "tool_calls": [
                    {"id": tc["id"], "type": "function",
                     "function": {"name": tc["name"],
                                  "arguments": json.dumps(parse_tool_arguments(tc.get("arguments")))}}
                    for tc in m.tool_calls
                ],
            })
        elif m.role == "tool":
            out.append({"role": "tool", "tool_call_id": m.tool_call_id or "",
                        "content": m.content})
        else:
            out.append({"role": m.role, "content": m.content})
    return out


def _anthropic_tools(tools: list[dict]) -> list[dict]:
    out: list[dict] = []
    for t in tools:
        fn = t.get("function", t)
        out.append({
            "name": fn["name"],
            "description": fn.get("description", ""),
            "input_schema": fn.get("parameters") or fn.get("input_schema")
            or {"type": "object", "properties": {}},
        })
    return out


def _anthropic_payload(messages: list[Message]) -> tuple[str, list[dict]]:
    """Split out the top-level system text and build Anthropic content blocks.

    Assistant tool turns become text + tool_use blocks; consecutive tool results
    are merged into ONE user message of tool_result blocks.
    """
    system_parts: list[str] = []
    out: list[dict] = []
    for m in messages:
        if m.role == "system":
            system_parts.append(m.content)
        elif m.role == "assistant":
            if m.tool_calls:
                blocks: list[dict] = []
                if m.content:
                    blocks.append({"type": "text", "text": m.content})
                blocks.extend(
                    {"type": "tool_use", "id": tc["id"], "name": tc["name"],
                     "input": parse_tool_arguments(tc.get("arguments"))}
                    for tc in m.tool_calls
                )
                out.append({"role": "assistant", "content": blocks})
            else:
                out.append({"role": "assistant", "content": m.content})
        elif m.role == "tool":
            block = {"type": "tool_result", "tool_use_id": m.tool_call_id or "",
                     "content": m.content}
            prev = out[-1] if out else None
            if (prev is not None and prev["role"] == "user"
                    and isinstance(prev["content"], list)
                    and all(b.get("type") == "tool_result" for b in prev["content"])):
                prev["content"].append(block)
            else:
                out.append({"role": "user", "content": [block]})
        else:
            out.append({"role": "user", "content": m.content})
    return "\n\n".join(system_parts), out


async def _post_json(client: httpx.AsyncClient | None, url: str, payload: dict,
                     headers: dict | None = None) -> dict:
    """POST JSON with the injected client, or a short-lived one; raise on HTTP errors."""
    if client is None:
        async with httpx.AsyncClient(timeout=120.0) as own:
            return await _post_json(own, url, payload, headers)
    resp = await client.post(url, json=payload, headers=headers)
    resp.raise_for_status()
    return resp.json()


class OllamaProvider:
    def __init__(self, client: httpx.AsyncClient | None = None,
                 base_url: str | None = None, model: str | None = None) -> None:
        s = get_settings()
        self._client = client
        self._base_url = (base_url or s.ollama_base_url).rstrip("/")
        self._model = model or s.llm_model

    async def chat(self, messages: list[Message], tools: list[dict] | None = None) -> ProviderResponse:
        payload: dict = {"model": self._model, "messages": _messages_payload(messages),
                         "stream": False,
                         # Ollama's default context silently truncates our prompts.
                         "options": {"num_ctx": 8192}}
        if tools:
            payload["tools"] = tools
        data = await _post_json(self._client, f"{self._base_url}/api/chat", payload)
        msg = data.get("message", {})
        raw_calls = msg.get("tool_calls") or []
        tool_calls = [
            {
                "id": tc.get("id", ""),
                "name": tc.get("function", {}).get("name", ""),
                "arguments": tc.get("function", {}).get("arguments", {}),
            }
            for tc in raw_calls
        ]
        return ProviderResponse(
            content=msg.get("content", ""),
            tool_calls=tool_calls,
            input_tokens=int(data.get("prompt_eval_count", 0)),
            output_tokens=int(data.get("eval_count", 0)),
        )


class AnthropicProvider:
    def __init__(self, client: httpx.AsyncClient | None = None,
                 api_key: str | None = None, model: str | None = None) -> None:
        s = get_settings()
        self._client = client
        self._api_key = api_key if api_key is not None else s.llm_api_key
        self._model = model or s.llm_model

    async def chat(self, messages: list[Message], tools: list[dict] | None = None) -> ProviderResponse:
        system, msgs = _anthropic_payload(messages)
        payload: dict = {"model": self._model, "max_tokens": 1024, "messages": msgs}
        if system:
            payload["system"] = system
        if tools:
            payload["tools"] = _anthropic_tools(tools)
        headers = {"x-api-key": self._api_key, "anthropic-version": "2023-06-01"}
        data = await _post_json(self._client, "https://api.anthropic.com/v1/messages",
                                payload, headers)
        text = "".join(b.get("text", "") for b in data.get("content", []) if b.get("type") == "text")
        tool_calls = [
            {"id": b.get("id", ""), "name": b.get("name", ""), "arguments": b.get("input", {})}
            for b in data.get("content", []) if b.get("type") == "tool_use"
        ]
        usage = data.get("usage", {})
        return ProviderResponse(
            content=text, tool_calls=tool_calls,
            input_tokens=int(usage.get("input_tokens", 0)),
            output_tokens=int(usage.get("output_tokens", 0)),
        )


class OpenAIProvider:
    def __init__(self, client: httpx.AsyncClient | None = None,
                 api_key: str | None = None, model: str | None = None) -> None:
        s = get_settings()
        self._client = client
        self._api_key = api_key if api_key is not None else s.llm_api_key
        self._model = model or s.llm_model

    async def chat(self, messages: list[Message], tools: list[dict] | None = None) -> ProviderResponse:
        payload: dict = {"model": self._model, "messages": _openai_messages(messages)}
        if tools:
            payload["tools"] = tools
        headers = {"Authorization": f"Bearer {self._api_key}"}
        data = await _post_json(self._client, "https://api.openai.com/v1/chat/completions",
                                payload, headers)
        choice = data["choices"][0]["message"]
        tool_calls = [
            {"id": t.get("id", ""), "name": t.get("function", {}).get("name", ""),
             "arguments": t.get("function", {}).get("arguments", {})}
            for t in (choice.get("tool_calls") or [])
        ]
        usage = data.get("usage", {})
        return ProviderResponse(
            content=choice.get("content") or "", tool_calls=tool_calls,
            input_tokens=int(usage.get("prompt_tokens", 0)),
            output_tokens=int(usage.get("completion_tokens", 0)),
        )


def get_provider() -> Provider:
    """Select the configured provider. 'fake' returns FakeProvider (eval + tests)."""
    name = get_settings().llm_provider
    if name == "ollama":
        return OllamaProvider()
    if name in ("anthropic", "openai") and not get_settings().llm_api_key:
        raise ValueError(
            f"LLM_PROVIDER={name} needs LLM_API_KEY to be set; "
            "set it in .env or use LLM_PROVIDER=ollama."
        )
    if name == "anthropic":
        return AnthropicProvider()
    if name == "openai":
        return OpenAIProvider()
    if name == "fake":
        return FakeProvider()
    raise ValueError(f"unknown LLM provider: {name!r}")
