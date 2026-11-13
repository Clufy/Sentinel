"""NIST SP 800-53 rev5 OSCAL catalog loader."""
import re

import httpx

from app.ingest.load_yaml import build_chunks  # noqa: F401  (shared chunk+embed; re-exported)

NIST_CATALOG_URL = (
    "https://raw.githubusercontent.com/usnistgov/oscal-content/main/"
    "nist.gov/SP800-53/rev5/json/NIST_SP-800-53_rev5_catalog.json"
)


_PARAM_RE = re.compile(r"\{\{\s*insert:\s*param,\s*([^\s}]+)\s*\}\}")
_DEFAULT_PARAM = "[Assignment: organization-defined value]"


def _render_param(param: dict | None, params: dict[str, dict], depth: int) -> str:
    """Render one OSCAL parameter the way NIST's own prose rendering does."""
    if not param:
        return _DEFAULT_PARAM
    if param.get("label"):
        return f"[Assignment: {param['label']}]"
    choices = (param.get("select") or {}).get("choice") or []
    if choices:
        # Choices can themselves embed placeholders; render them too (bounded).
        rendered = [_fill_params(str(c), params, depth + 1) for c in choices]
        return "[Selection: " + "; ".join(rendered) + "]"
    return _DEFAULT_PARAM


def _fill_params(text: str, params: dict[str, dict], depth: int = 0) -> str:
    """Replace every {{ insert: param, <id> }}; unknown ids get the default form."""
    if depth > 3:
        return _PARAM_RE.sub(lambda m: _DEFAULT_PARAM, text)
    return _PARAM_RE.sub(lambda m: _render_param(params.get(m.group(1)), params, depth), text)


def _is_withdrawn(control: dict) -> bool:
    return any(
        p.get("name") == "status" and str(p.get("value", "")).strip().lower() == "withdrawn"
        for p in control.get("props", []) or []
    )


def _collect_prose(part: dict) -> list[str]:
    out: list[str] = []
    prose = part.get("prose")
    if prose:
        out.append(prose)
    for sub in part.get("parts", []) or []:
        out.extend(_collect_prose(sub))
    return out


def _statement_summary(control: dict, params: dict[str, dict]) -> str:
    pieces: list[str] = []
    for part in control.get("parts", []) or []:
        if part.get("name") == "statement":
            pieces.extend(_collect_prose(part))
    return " ".join(_fill_params(" ".join(pieces), params).split())


def _walk_control(control: dict, out: list[dict],
                  inherited: dict[str, dict] | None = None) -> None:
    """Append the control and its enhancements, skipping withdrawn ones.

    Params accumulate down the tree so an enhancement can render its parent's.
    """
    params = dict(inherited or {})
    params.update({p["id"]: p for p in control.get("params", []) or [] if p.get("id")})
    if not _is_withdrawn(control):
        out.append({
            "framework": "NIST",
            "control_id": str(control["id"]).upper(),
            "title": control.get("title", ""),
            "summary": _statement_summary(control, params),
            "source_url": NIST_CATALOG_URL,
        })
    for enh in control.get("controls", []) or []:
        _walk_control(enh, out, params)


def parse_catalog(catalog: dict) -> list[dict]:
    out: list[dict] = []
    root = catalog.get("catalog", catalog)
    for group in root.get("groups", []) or []:
        for control in group.get("controls", []) or []:
            _walk_control(control, out)
    return out


async def fetch_catalog(url: str = NIST_CATALOG_URL) -> dict:
    async with httpx.AsyncClient(timeout=60.0) as client:
        resp = await client.get(url)
        resp.raise_for_status()
        return resp.json()
