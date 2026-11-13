"""In-memory request metrics + per-request cost table + logging middleware."""
from __future__ import annotations

import json
import logging
import time

from pydantic import BaseModel, Field
from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import Response

logger = logging.getLogger("sentinel.obs")


def configure_logging() -> None:
    """Make sentinel.* INFO records visible without double-configuring.

    Adds a root stderr handler only when none exists (uvicorn leaves root bare),
    at WARNING so third-party INFO chatter stays quiet, and lets the `sentinel`
    logger tree through at INFO.
    """
    if not logging.getLogger().handlers:
        logging.basicConfig(level=logging.WARNING,
                            format="%(asctime)s %(levelname)s %(name)s %(message)s")
    sentinel = logging.getLogger("sentinel")
    if sentinel.level == logging.NOTSET or sentinel.level > logging.INFO:
        sentinel.setLevel(logging.INFO)

COST_PER_1K: dict = {
    "anthropic": {"input": 0.003, "output": 0.015},
    "openai": {"input": 0.0005, "output": 0.0015},
    "ollama": 0.0,
    "fake": 0.0,
}


def cost_for(provider: str, model: str, input_tokens: int, output_tokens: int) -> float:
    price = COST_PER_1K.get(provider, 0.0)
    if not isinstance(price, dict):
        return 0.0
    return (input_tokens / 1000) * price["input"] + (output_tokens / 1000) * price["output"]


class RequestRecord(BaseModel):
    path: str
    total_tokens: int = 0
    latency_ms: float = 0.0
    cost_usd: float = 0.0
    refused: bool = False
    tool_trace: list[dict] = Field(default_factory=list)


class MetricsStore:
    def __init__(self) -> None:
        self.records: list[RequestRecord] = []

    def record(self, rec: RequestRecord) -> None:
        self.records.append(rec)

    def reset(self) -> None:
        self.records.clear()

    def get_metrics(self) -> dict:
        n = len(self.records)
        avg_latency = sum(r.latency_ms for r in self.records) / n if n else 0.0
        return {
            "total_requests": n,
            "total_tokens": sum(r.total_tokens for r in self.records),
            "total_cost_usd": sum(r.cost_usd for r in self.records),
            "avg_latency_ms": avg_latency,
            "refusals": sum(1 for r in self.records if r.refused),
        }


metrics_store = MetricsStore()


def get_metrics() -> dict:
    return metrics_store.get_metrics()


class MetricsMiddleware(BaseHTTPMiddleware):
    def __init__(self, app, store: MetricsStore) -> None:
        super().__init__(app)
        self.store = store

    async def dispatch(self, request: Request, call_next) -> Response:
        start = time.perf_counter()
        response = await call_next(request)
        latency_ms = (time.perf_counter() - start) * 1000.0
        rec = RequestRecord(
            path=request.url.path,
            total_tokens=int(getattr(request.state, "total_tokens", 0)),
            latency_ms=latency_ms,
            cost_usd=float(getattr(request.state, "cost_usd", 0.0)),
            refused=bool(getattr(request.state, "refused", False)),
            tool_trace=list(getattr(request.state, "tool_trace", [])),
        )
        self.store.record(rec)
        fields = {"path": rec.path, "latency_ms": round(rec.latency_ms, 1),
                  "tokens": rec.total_tokens, "cost_usd": rec.cost_usd,
                  "refused": rec.refused, "tool_trace": rec.tool_trace}
        # The trace is in the message text (JSON) so the default formatter shows it.
        logger.info("request %s", json.dumps(fields, default=str), extra=fields)
        return response


def install_obs(app, store: MetricsStore | None = None) -> None:
    app.add_middleware(MetricsMiddleware, store=store or metrics_store)
