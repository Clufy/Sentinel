# ---- Builder: resolve + install deps into a venv using uv ----
FROM python:3.12-slim AS builder

# uv 0.10.2: uv.lock is revision 3, which older uv cannot read with --frozen
COPY --from=ghcr.io/astral-sh/uv:0.10.2 /uv /usr/local/bin/uv

ENV UV_COMPILE_BYTECODE=1 \
    UV_LINK_MODE=copy \
    UV_PYTHON_DOWNLOADS=0

WORKDIR /app
COPY pyproject.toml uv.lock ./
RUN uv sync --frozen --no-install-project --no-dev

# app/ carries schema.sql and static/; corpus/ is read at runtime (../corpus from app/agent.py)
COPY app ./app
COPY corpus ./corpus
RUN uv sync --frozen --no-dev

# ---- Runtime: slim, non-root, venv + app only ----
FROM python:3.12-slim AS runtime

RUN groupadd --system sentinel && useradd --system --gid sentinel --create-home sentinel

WORKDIR /app
COPY --from=builder --chown=sentinel:sentinel /app /app

ENV PATH="/app/.venv/bin:$PATH" \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

EXPOSE 8000
USER sentinel
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]
