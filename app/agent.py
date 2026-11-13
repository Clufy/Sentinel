# app/agent.py
"""Agent: prompt loader, cross-framework mappings, tools, and run_agent loop."""
from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Awaitable, Callable

import yaml
from pydantic import BaseModel

from app.config import get_settings
from app.db import get_by_control_id, get_pool
from app.grounding import (
    REFUSAL_MESSAGE,
    format_citation,
    normalize_citation,
    normalize_control_id,
)
from app.llm import Message, Provider, extract_cited_ids, parse_tool_arguments
from app.models import Chunk
from app.retrieval import retrieve, vector_search

_MAPPINGS_PATH = Path(__file__).resolve().parent.parent / "corpus" / "mappings.yaml"


@lru_cache(maxsize=1)
def load_mappings() -> dict[str, list[dict[str, str]]]:
    """Lookup map keyed by 'FRAMEWORK:control_id'.

    'by_control' entries come first, then pairwise links expanded from each vetted
    'crosswalk' row (across frameworks only), de-duplicated.
    """
    data = yaml.safe_load(_MAPPINGS_PATH.read_text(encoding="utf-8")) or {}
    out: dict[str, list[dict[str, str]]] = {}

    def add(key: str, fw: str, cid: str) -> None:
        link = {"framework": fw, "control_id": cid}
        bucket = out.setdefault(key, [])
        if link not in bucket and key != f"{fw}:{cid}":
            bucket.append(link)

    for key, entries in (data.get("by_control", {}) or {}).items():
        out.setdefault(key, [])
        for e in entries or []:
            add(key, str(e["framework"]), str(e["control_id"]))
    for row in data.get("crosswalk", []) or []:
        members = [
            (fw, str(cid))
            for fw in ("ISO", "NIST", "CIS")
            for cid in (row.get(fw.lower()) or [])
        ]
        for fw, cid in members:
            for ofw, ocid in members:
                if ofw != fw:
                    add(f"{fw}:{cid}", ofw, ocid)
    return out


def related_controls(framework: str, control_id: str) -> list[dict[str, str]]:
    """Cross-framework equivalents for one control, or [] if none declared."""
    return load_mappings().get(f"{framework}:{control_id}", [])


_PROMPTS_DIR = Path(__file__).resolve().parent / "prompts"


@lru_cache(maxsize=None)
def load_prompt(name: str) -> str:
    """Read app/prompts/<name>.txt (UTF-8). Raises FileNotFoundError if absent."""
    return (_PROMPTS_DIR / f"{name}.txt").read_text(encoding="utf-8")


_FRAMEWORKS = ("NIST", "ISO", "CIS")


def _chunk_to_dict(chunk: Chunk) -> dict:
    return {
        "framework": chunk.framework,
        "control_id": chunk.control_id,
        "title": chunk.title,
        "summary": chunk.summary,
        "source_url": chunk.source_url,
        "score": chunk.score,
        "citation": format_citation(chunk.framework, chunk.control_id),
    }


async def search_controls(query: str, framework: str | None = None) -> dict:
    settings = get_settings()
    chunks = await retrieve(query, framework, settings.retrieval_top_k)
    relevant = [c for c in chunks if c.score >= settings.min_relevance]
    return {"controls": [_chunk_to_dict(c) for c in relevant]}


async def compare_frameworks(control_id: str) -> dict:
    """Resolve the source control deterministically (get_by_control_id), then its equivalents.

    The id is normalised per framework (ac-2(1) -> AC-2.1, 5 -> CIS-5); an optional
    leading framework token ("CIS 5") restricts the lookup to that framework.
    """
    raw = str(control_id).strip()
    frameworks: tuple[str, ...] = _FRAMEWORKS
    parts = raw.split(None, 1)
    if len(parts) == 2 and parts[0].upper() in _FRAMEWORKS:
        frameworks, raw = (parts[0].upper(),), parts[1]
    pool = await get_pool()
    async with pool.acquire() as conn:
        source: Chunk | None = None
        source_fw: str | None = None
        source_id = raw
        for fw in frameworks:
            cid = normalize_control_id(fw, raw)
            found = await get_by_control_id(conn, fw, cid)
            if found is not None:
                source, source_fw, source_id = found, fw, cid
                break
        if source is None:
            return {"source": None, "related": []}

        related: list[dict] = []
        for entry in related_controls(source_fw, source_id):
            match = await get_by_control_id(conn, entry["framework"], entry["control_id"])
            if match is not None:
                related.append(_chunk_to_dict(match))
    return {"source": _chunk_to_dict(source), "related": related}


async def map_finding(text: str, framework: str | None = None) -> dict:
    """Controls most similar to an audit finding, best first, above map_min_similarity.

    Ranks by embedding cosine (vector_search), not the hybrid retrieve(): the MS-MARCO
    cross-encoder is trained on question->passage pairs and scores a declarative
    finding near 0 even against the right control, so its score can't gate /map.
    """
    settings = get_settings()
    chunks = await vector_search(text, framework, settings.retrieval_top_k)
    relevant = [c for c in chunks if c.score >= settings.map_min_similarity]
    best: dict[tuple[str, str], Chunk] = {}  # one entry per control, at its best chunk
    for c in sorted(relevant, key=lambda c: c.score, reverse=True):
        best.setdefault((c.framework, c.control_id), c)
    return {"controls": [_chunk_to_dict(c) for c in best.values()]}


TOOL_FUNCTIONS: dict[str, Callable[..., Awaitable[dict]]] = {
    "search_controls": search_controls,
    "compare_frameworks": compare_frameworks,
    "map_finding": map_finding,
}

TOOL_SCHEMAS: list[dict] = [
    {
        "type": "function",
        "function": {
            "name": "search_controls",
            "description": (
                "Search the compliance corpus for controls relevant to a query. "
                "Returns controls with title, summary, source_url, and a citation."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "query": {"type": "string", "description": "Natural-language search query."},
                    "framework": {"type": "string", "enum": ["NIST", "ISO", "CIS"],
                                  "description": "Optional framework filter. Omit to search all."},
                },
                "required": ["query"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "compare_frameworks",
            "description": (
                "Given a single control_id, return the control and its equivalent "
                "controls in the other frameworks."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "control_id": {"type": "string",
                                   "description": "Control id such as AC-2, A.5.15, or CIS-5."},
                },
                "required": ["control_id"],
            },
        },
    },
    {
        "type": "function",
        "function": {
            "name": "map_finding",
            "description": (
                "Given a plain-language audit finding, return the most relevant "
                "controls across all frameworks, ranked by relevance."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "text": {"type": "string", "description": "The audit finding text."},
                },
                "required": ["text"],
            },
        },
    },
]


class AgentResult(BaseModel):
    answer: str
    citations: list[str]
    trace: list[dict]
    steps: int
    input_tokens: int
    output_tokens: int
    stop_reason: str
    refused: bool


def _summarize(result: dict) -> str:
    if "controls" in result:
        cites = [c.get("citation", "?") for c in result["controls"]]
        return (f"{len(cites)} controls: " + ", ".join(cites))[:200]
    if "source" in result:
        src = (result.get("source") or {}).get("citation", "none")
        rel = [r.get("citation", "?") for r in result.get("related", [])]
        return (f"source {src} + {len(rel)} related: " + ", ".join(rel))[:200]
    return str(result)[:200]


# Tools whose retrieval honours a framework filter; run_agent forces it on them.
_FRAMEWORK_SCOPED_TOOLS = frozenset({"search_controls", "map_finding"})


def normalize_framework(value: object) -> str | None:
    """'iso' -> 'ISO'; '', 'ALL', 'ANY', None and anything outside the enum -> None."""
    if not isinstance(value, str):
        return None
    fw = value.strip().upper()
    return fw if fw in _FRAMEWORKS else None


_TOOL_PARAMS: dict[str, frozenset[str]] = {
    s["function"]["name"]: frozenset(s["function"]["parameters"]["properties"])
    for s in TOOL_SCHEMAS
}


def _clean_args(name: str, args: object) -> dict:
    """Keep only the tool's declared parameters and normalise `framework`."""
    if not isinstance(args, dict):
        return {}
    allowed = _TOOL_PARAMS.get(name)
    cleaned = {k: v for k, v in args.items() if allowed is None or k in allowed}
    if "framework" in cleaned:
        cleaned["framework"] = normalize_framework(cleaned["framework"])
    return cleaned


async def run_agent(
    question: str,
    provider: Provider,
    *,
    prompt_name: str = "system",
    framework: str | None = None,
    tool_functions: dict | None = None,
) -> AgentResult:
    """Hand-rolled loop capped by max_agent_steps and max_request_tokens."""
    settings = get_settings()
    tool_functions = tool_functions if tool_functions is not None else TOOL_FUNCTIONS
    scope = normalize_framework(framework)
    system = load_prompt(prompt_name)
    if scope:
        # Tell the model the scope; the tools enforce it regardless (see below).
        system += (f"\n\nSCOPE: the user limited this request to the {scope} framework. "
                   "Tool results are already restricted to it.")
    messages: list[Message] = [
        Message(role="system", content=system),
        Message(role="user", content=question),
    ]
    trace: list[dict] = []
    input_tokens = output_tokens = steps = 0
    grounded_ids: set[str] = set()

    for _ in range(settings.max_agent_steps):
        steps += 1
        resp = await provider.chat(messages, tools=TOOL_SCHEMAS)
        input_tokens += resp.input_tokens
        output_tokens += resp.output_tokens

        if not resp.tool_calls:
            citations = list(dict.fromkeys(
                normalize_citation(c) for c in extract_cited_ids(resp.content)
            ))
            refused = (not citations) or any(c not in grounded_ids for c in citations)
            if refused:
                return AgentResult(
                    answer=REFUSAL_MESSAGE, citations=[], trace=trace, steps=steps,
                    input_tokens=input_tokens, output_tokens=output_tokens,
                    stop_reason="answered", refused=True,
                )
            return AgentResult(
                answer=resp.content, citations=citations, trace=trace, steps=steps,
                input_tokens=input_tokens, output_tokens=output_tokens,
                stop_reason="answered", refused=False,
            )

        for i, call in enumerate(resp.tool_calls):
            # Ollama/FakeProvider calls may lack ids; synthesise stable ones.
            if not call.get("id"):
                call["id"] = f"call_{steps}_{i}"
            call["arguments"] = parse_tool_arguments(call.get("arguments"))
        messages.append(Message(
            role="assistant", content=resp.content or "",
            tool_calls=[{"id": c["id"], "name": c["name"], "arguments": c["arguments"]}
                        for c in resp.tool_calls],
        ))
        for call in resp.tool_calls:
            args = _clean_args(call["name"], call["arguments"])
            if scope and call["name"] in _FRAMEWORK_SCOPED_TOOLS:
                args["framework"] = scope  # forced, overrides the model
            fn = tool_functions.get(call["name"])
            if fn is None:
                result = {"error": f"unknown tool: {call['name']}"}
            else:
                try:
                    result = await fn(**args)
                except Exception as exc:  # fed back to the model, never a 500
                    result = {"error": f"{type(exc).__name__}: {exc}"}
            trace.append({"tool": call["name"], "args": args, "result_summary": _summarize(result)})
            returned = [*(result.get("controls") or []), result.get("source"),
                        *(result.get("related") or [])]
            for c in returned:
                if c and c.get("framework") and c.get("control_id"):
                    grounded_ids.add(normalize_citation(f"{c['framework']} {c['control_id']}"))
            messages.append(Message(role="tool", content=json.dumps(result),
                                    tool_call_id=call["id"]))

        if input_tokens + output_tokens >= settings.max_request_tokens:
            return AgentResult(
                answer=REFUSAL_MESSAGE, citations=[], trace=trace, steps=steps,
                input_tokens=input_tokens, output_tokens=output_tokens,
                stop_reason="max_tokens", refused=True,
            )

    return AgentResult(
        answer=REFUSAL_MESSAGE, citations=[], trace=trace, steps=steps,
        input_tokens=input_tokens, output_tokens=output_tokens,
        stop_reason="max_steps", refused=True,
    )
