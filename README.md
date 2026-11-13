# Sentinel

[![License: MIT](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)
![Python 3.12+](https://img.shields.io/badge/python-3.12%2B-blue.svg)

**Ask questions about security compliance standards and get answers that cite
the exact controls they come from.**

Sentinel is a self-hosted API that answers questions about three security
control frameworks:

- **NIST SP 800-53 rev5**
- **ISO/IEC 27001:2022 Annex A**
- **CIS Controls v8**

Every answer cites the specific controls it is based on, such as `[NIST AC-2]`.
When Sentinel finds nothing relevant, it says so instead of guessing.

> **Not legal advice.** Sentinel is a research aid for governance, risk and
> compliance (GRC) work. It finds and summarizes control text. It does not give
> legal, audit or certification advice. Always check the published standards
> before acting.

## Contents

- [Features](#features)
- [Quick start](#quick-start)
- [Usage](#usage)
- [Configuration](#configuration)
- [How it works](#how-it-works)
- [Development](#development)
- [Evaluation](#evaluation)
- [Security](#security)
- [Project structure](#project-structure)
- [Data sources](#data-sources)
- [License](#license)

## Features

- **Cited answers.** Ask a question in plain English and get a streamed answer.
  Each claim cites a control, for example `[NIST AC-2]`, `[ISO A.5.15]` or
  `[CIS CIS-5]`.
- **Refuses instead of guessing.** Sentinel refuses when nothing relevant is
  found. It also refuses when an answer cites a control it never retrieved.
- **Maps audit findings.** Paste an audit finding and get the related controls in
  all three frameworks, each with a one-sentence reason.
- **Hybrid search.** Combines keyword search and semantic search, then reranks
  the results.
- **Any LLM.** Runs on a local model through [Ollama](https://ollama.com) by
  default, or on Anthropic or OpenAI with an API key.
- **Measured quality.** Includes an evaluation suite that fails the build if
  answer quality drops.

## Quick start

### Prerequisites

- Docker with Docker Compose v2
- [Ollama](https://ollama.com), for the default local model. You don't need it if
  you use Anthropic or OpenAI.

### 1. Pull the model

```bash
ollama pull llama3.1
```

> **On Linux**, Ollama listens only on `127.0.0.1` by default, and the container
> can't reach it there. Start Ollama with `OLLAMA_HOST=0.0.0.0 ollama serve`.

### 2. Configure

```bash
cp .env.example .env
```

The defaults use Ollama. To use Anthropic or OpenAI instead, set `LLM_PROVIDER`,
`LLM_MODEL` and `LLM_API_KEY` in `.env`.

### 3. Start the stack

```bash
docker compose up --build
```

The API runs at <http://localhost:8000>.

### 4. Load the standards

```bash
curl -X POST http://localhost:8000/ingest
```

This creates the database schema and loads all three frameworks. The first
`/ingest` needs internet access and takes a few minutes. It downloads the NIST
catalog (about 10 MB) and the embedding and reranking models from Hugging Face.
Later runs reuse the model cache until the container is recreated.

Before `/ingest` runs, the database has no tables: `/query` refuses every
question and `/map` returns a 500 error.

### 5. Try it

Open <http://localhost:8000/> for the demo page, or call the API as shown below.

## Usage

### Ask a question

`POST /query` streams the answer as Server-Sent Events.

```bash
curl -N -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "What controls govern account management?"}'
```

To limit a question to one framework, add `framework` (`NIST`, `ISO` or `CIS`):

```bash
curl -N -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"question": "How should we manage privileged access?", "framework": "NIST"}'
```

The stream sends either a full answer or a single refusal:

| Event | Contents |
| --- | --- |
| `token` | The answer text, a word at a time |
| `citations` | The controls the answer cites |
| `meta` | Steps, token counts, cost and the tool-call `trace` |
| `done` | End of the answer |
| `refusal` | Sent alone, instead of all of the above, when Sentinel won't answer |

### Map an audit finding

`POST /map` returns JSON.

```bash
curl -X POST http://localhost:8000/map \
  -H "Content-Type: application/json" \
  -d '{"finding_text": "Terminated employees retained active VPN accounts for weeks."}'
```

For this finding, Sentinel returns NIST PS-4 (personnel termination), CIS 5.3
(disable dormant accounts) and ISO A.6.5 (responsibilities after leaving).

The response contains:

| Field | Contents |
| --- | --- |
| `controls` | Matching controls, each with `framework`, `control_id`, `title`, `score` and `rationale` |
| `citations` | The same controls as citation strings |
| `refused` | `true` when no control passes the `MAP_MIN_SIMILARITY` cut-off |
| `tool_trace`, `input_tokens`, `output_tokens`, `cost_usd` | Request details |

When `refused` is `true`, `controls` is empty and no LLM call is made. If the
model gives no reason for a control, its `rationale` says
`No model rationale; matched by retrieval (score 0.NN).`

### Other endpoints

| Method | Path | Purpose |
| --- | --- | --- |
| `GET` | `/` | Demo page |
| `GET` | `/health` | Liveness check, returns `{"status": "ok"}` |
| `GET` | `/metrics` | Request count, tokens, cost, average latency and refusals |
| `POST` | `/ingest` | Create the schema and load the corpus (no request body, **no authentication**) |

## Configuration

Set these in `.env`. The defaults match `app/config.py` and `.env.example`.

| Variable | Default | Purpose |
| --- | --- | --- |
| `LLM_PROVIDER` | `ollama` | `ollama`, `anthropic`, `openai` or `fake` (scripted, for tests) |
| `LLM_MODEL` | `llama3.1` | Model ID |
| `LLM_API_KEY` | *(empty)* | Required for `anthropic` and `openai` |
| `OLLAMA_BASE_URL` | `http://localhost:11434` | Ollama endpoint (compose overrides it inside the container) |
| `DATABASE_URL` | `postgresql://sentinel:sentinel@127.0.0.1:5432/sentinel` | Postgres connection string (compose overrides it inside the container) |
| `EMBEDDING_MODEL` | `sentence-transformers/all-MiniLM-L6-v2` | Embedding model (384 dimensions) |
| `RETRIEVAL_TOP_K` | `8` | Number of results retrieved per search |
| `MIN_RELEVANCE` | `0.3` | Search results scoring below this are dropped |
| `MAP_MIN_SIMILARITY` | `0.35` | `/map` drops controls less similar to the finding than this |
| `MAX_AGENT_STEPS` | `4` | Maximum tool-calling rounds per question |
| `MAX_REQUEST_TOKENS` | `8000` | Token budget per request |
| `MAX_CHUNK_TOKENS` | `512` | Controls longer than this many words are split into chunks |

## How it works

1. **Ingest.** The ingest step loads the standards, embeds each control and
   stores it in Postgres with pgvector.
2. **Search.** The search step runs a keyword search and a vector search, then
   merges them with Reciprocal Rank Fusion (k=60). A cross-encoder reranks the
   merged list, and Maximal Marginal Relevance (MMR, λ=0.7) removes near-duplicates.
3. **Answer.** An agent loop lets the LLM call search tools, then write an
   answer with citations.
4. **Check.** Sentinel refuses if the answer has no citations, or if it cites a
   control that no search returned.

For the full design, including a diagram, see
[docs/architecture.md](docs/architecture.md).

## Development

Install the dependencies (needs [uv](https://docs.astral.sh/uv/) and Python 3.12+):

```bash
uv sync --extra dev
```

Run the tests and the linter:

```bash
uv run pytest            # integration tests are skipped until the database below is set up
uv run ruff check .
```

To run the integration tests too, start the database and create the schema
first:

```bash
docker compose up -d db
uv run python -m scripts.init_db
uv run pytest
```

**Schema changes.** The schema uses `CREATE ... IF NOT EXISTS`, so it never
changes an existing table. After editing `app/schema.sql`, recreate the
database with `docker compose down -v`, then run `/ingest` again. This deletes
the `pgdata` volume.

## Evaluation

The evaluation suite asks 40 questions, 5 of which have no answer in the corpus.
It runs them through hybrid search and vector-only search and scores each on:

- recall@8
- citation correctness
- faithfulness
- refusal accuracy

It runs against a small fixture corpus of 30 controls, with a scripted LLM:

```bash
docker compose up -d db
uv run python -m scripts.init_db
uv run python -m scripts.ingest_fixture   # replaces the controls table with the fixtures
uv run python -m eval.run_eval
```

The run fails unless all of these hold:

- hybrid recall ≥ 0.7
- refusal accuracy ≥ 0.8
- citation correctness ≥ 0.6
- hybrid recall ≥ vector-only recall

Faithfulness is reported but not gated.

### Results

| Retrieval | Recall@8 | Citation correctness | Faithfulness | Refusal accuracy |
| --- | --- | --- | --- | --- |
| Hybrid | 1.000 | 0.793 | 1.000 | 0.975 |
| Vector-only | 0.857 | 0.134 | 1.000 | 0.825 |

Measured with pgvector 0.8.0 on CPU.

Vector-only scores lower on citations and refusals because raw cosine scores
aren't calibrated against `MIN_RELEVANCE`, so too many weakly related controls
get through. These results show the pipeline works on a small fixture. They
don't predict how a real LLM performs on the full corpus.

## Security

Out of the box, Sentinel is meant to run on your own machine:

- Docker Compose publishes ports only on `127.0.0.1`.
- Postgres uses the default `sentinel`/`sentinel` superuser.
- `/ingest` has no authentication.

Before exposing Sentinel beyond your machine:

1. Change the database password.
2. Keep port 5432 unpublished.
3. Put `/ingest` behind authentication, or block it at a reverse proxy.
4. Then widen the port 8000 mapping.

## Project structure

```text
app/            FastAPI app: retrieval, agent loop, LLM providers, grounding
app/ingest/     Loaders for NIST (OSCAL JSON) and the ISO/CIS YAML files
app/prompts/    Versioned system and mapping prompts
app/static/     Demo page
corpus/         ISO and CIS control summaries and cross-framework mappings
eval/           Evaluation harness and question set
scripts/        Schema setup and fixture ingest
tests/          Unit, integration and corpus tests
docs/           Architecture notes
```

## Data sources

| Framework | Source | License |
| --- | --- | --- |
| NIST SP 800-53 rev5 | Official OSCAL catalog, downloaded at ingest | Public domain |
| ISO/IEC 27001:2022 Annex A | Original paraphrased summaries; no ISO text is stored | Summaries are this project's own work |
| CIS Controls v8 | Summaries adapted from the CIS Controls | CC BY-NC-SA 4.0 |

For full details and attribution, see [corpus/SOURCES.md](corpus/SOURCES.md).

## License

The source code is released under the [MIT License](LICENSE).

The CIS-derived data is **not** under the MIT License. That covers
`corpus/cis_v8.yaml` and `tests/fixtures/mini_cis.yaml`. Both are adapted from
the CIS Critical Security Controls v8 and remain under CC BY-NC-SA 4.0
(non-commercial, share-alike).
