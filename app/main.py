"""Sentinel FastAPI app: the create_app() factory and its routes."""
import json
import re
from collections.abc import AsyncGenerator
from pathlib import Path

from fastapi import FastAPI, Request
from fastapi.staticfiles import StaticFiles
from sse_starlette.sse import EventSourceResponse

from app.agent import load_prompt, map_finding, run_agent
from app.config import settings
from app.grounding import REFUSAL_MESSAGE, normalize_citation
from app.ingest.run import ingest_all
from app.llm import Message, get_provider
from app.models import IngestResponse, MapRequest, QueryRequest
from app.obs import configure_logging, cost_for, get_metrics, install_obs, metrics_store

_STATIC_DIR = Path(__file__).resolve().parent / "static"


async def _agent_events(result, cost_usd: float) -> AsyncGenerator[dict, None]:
    """Yield NAMED SSE events for an AgentResult (token/citations/meta/done or refusal)."""
    if result.refused:
        yield {"event": "refusal", "data": json.dumps({"message": REFUSAL_MESSAGE})}
        return
    for piece in result.answer.split(" "):
        yield {"event": "token", "data": json.dumps({"text": piece + " "})}
    yield {"event": "citations", "data": json.dumps({"citations": result.citations})}
    yield {"event": "meta", "data": json.dumps({
        "steps": result.steps,
        "input_tokens": result.input_tokens,
        "output_tokens": result.output_tokens,
        "stop_reason": result.stop_reason,
        "cost_usd": cost_usd,
        "trace": result.trace,
    })}
    yield {"event": "done", "data": json.dumps({})}


_FENCE_RE = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)


def _parse_rationales(content: str) -> dict[str, str]:
    """Model rationales keyed by canonical citation ('NIST AC-2').

    Tolerates a ```/```json fence and keys with or without brackets.
    """
    text = (content or "").strip()
    fenced = _FENCE_RE.search(text)
    if fenced:
        text = fenced.group(1)
    try:
        parsed = json.loads(text)
    except (json.JSONDecodeError, TypeError):
        return {}
    if not isinstance(parsed, dict):
        return {}
    return {normalize_citation(str(k)): str(v) for k, v in parsed.items() if v}


def create_app() -> FastAPI:
    configure_logging()
    app = FastAPI(title="Sentinel")

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok"}

    @app.post("/ingest", response_model=IngestResponse)
    async def ingest() -> IngestResponse:
        counts = await ingest_all()
        return IngestResponse(counts=counts, total=sum(counts.values()))

    @app.post("/query")
    async def query(req: QueryRequest, request: Request) -> EventSourceResponse:
        provider = get_provider()
        result = await run_agent(req.question, provider, framework=req.framework)
        cost_usd = cost_for(settings.llm_provider, settings.llm_model,
                            result.input_tokens, result.output_tokens)
        # Populate request.state for the metrics middleware.
        request.state.total_tokens = result.input_tokens + result.output_tokens
        request.state.tool_trace = result.trace
        request.state.cost_usd = cost_usd
        request.state.refused = result.refused
        return EventSourceResponse(_agent_events(result, cost_usd))

    @app.post("/map")
    async def map_route(req: MapRequest, request: Request) -> dict:
        retrieved = await map_finding(req.finding_text)
        controls = retrieved["controls"]  # already floored by map_min_similarity
        trace = [{"tool": "map_finding", "args": {"text": req.finding_text},
                  "result_summary": f'{len(controls)} controls'}]
        if not controls:
            # Nothing relevant: refuse with an empty list and skip the LLM call.
            request.state.total_tokens = 0
            request.state.tool_trace = trace
            request.state.cost_usd = 0.0
            request.state.refused = True
            return {"controls": [], "citations": [], "tool_trace": trace,
                    "input_tokens": 0, "output_tokens": 0, "cost_usd": 0.0,
                    "refused": True}

        user = "FINDING: " + req.finding_text + "\n\nCANDIDATE CONTROLS:\n" + "\n".join(
            f'{c["citation"]} {c["title"]}: {c["summary"]}' for c in controls
        ) + '\n\nReturn ONLY a JSON object mapping each citation to a one-sentence rationale.'
        provider = get_provider()
        resp = await provider.chat(
            [Message(role="system", content=load_prompt("map")),
             Message(role="user", content=user)]
        )
        rationales = _parse_rationales(resp.content)

        out_controls = [{
            "framework": c["framework"],
            "control_id": c["control_id"],
            "title": c["title"],
            "score": c["score"],
            "rationale": rationales.get(normalize_citation(c["citation"]))
            or f'No model rationale; matched by retrieval (score {c["score"]:.2f}).',
        } for c in controls]
        citations = [c["citation"].strip("[]") for c in controls]
        cost_usd = cost_for(settings.llm_provider, settings.llm_model,
                            resp.input_tokens, resp.output_tokens)

        request.state.total_tokens = resp.input_tokens + resp.output_tokens
        request.state.tool_trace = trace
        request.state.cost_usd = cost_usd
        request.state.refused = False
        return {
            "controls": out_controls,
            "citations": citations,
            "tool_trace": trace,
            "input_tokens": resp.input_tokens,
            "output_tokens": resp.output_tokens,
            "cost_usd": cost_usd,
            "refused": False,
        }

    # Known limitation: for /query the middleware's latency covers the agent run
    # but not the SSE body replay (the response body streams after call_next returns).
    install_obs(app, metrics_store)

    @app.get("/metrics")
    async def metrics() -> dict:
        return get_metrics()

    app.state.metrics_store = metrics_store
    # Mount LAST so API routes (/query, /map, /ingest, /health, /metrics) win.
    app.mount("/", StaticFiles(directory=str(_STATIC_DIR), html=True), name="static")
    return app


app = create_app()
