# tests/test_agent_loop.py
import json

import pytest

import app.agent as agent
from app.agent import AgentResult, run_agent
from app.llm import ProviderResponse


class Scripted:
    def __init__(self, responses):
        self._responses = list(responses)
        self.calls = []

    async def chat(self, messages, tools=None):
        self.calls.append({"messages": [(m.role, m.content) for m in messages], "tools": tools})
        return self._responses.pop(0)


def _resp(content="", tool_calls=None, it=10, ot=5):
    return ProviderResponse(content=content, tool_calls=tool_calls or [],
                            input_tokens=it, output_tokens=ot)


def _tc(id, name, arguments):
    return {"id": id, "name": name, "arguments": arguments}


@pytest.fixture
def fake_tools():
    calls = []

    async def search_controls(query, framework=None):
        calls.append(("search_controls", query, framework))
        return {"controls": [{"framework": "NIST", "control_id": "AC-2",
                              "citation": "[NIST AC-2]"}]}

    return {"search_controls": search_controls}, calls


async def test_direct_answer_no_tool_calls(fake_tools):
    """A citation is only trusted if a tool actually returned that control.
    Restructured so the fake tool runs first (grounding [NIST IA-2]) before
    the model answers citing it -- matching the new grounded-citation rule."""
    tf, _ = fake_tools

    async def search_controls(query, framework=None):
        return {"controls": [{"framework": "NIST", "control_id": "IA-2",
                              "citation": "[NIST IA-2]"}]}

    tool_functions = {"search_controls": search_controls}
    provider = Scripted([
        _resp(tool_calls=[_tc("1", "search_controls", {"query": "mfa"})]),
        _resp(content="MFA is required [NIST IA-2]."),
    ])
    result = await run_agent("does nist require mfa?", provider, tool_functions=tool_functions)
    assert isinstance(result, AgentResult)
    assert result.answer == "MFA is required [NIST IA-2]."
    assert result.stop_reason == "answered"
    assert result.citations == ["NIST IA-2"]
    assert result.refused is False
    assert len(result.trace) == 1


async def test_direct_answer_ungrounded_citation_is_refused():
    """No tool ever ran, so any citation in a direct answer is unfaithful
    and must be refused per the grounded-citation rule."""
    provider = Scripted([_resp(content="MFA is required [NIST IA-2].")])
    result = await run_agent("does nist require mfa?", provider, tool_functions={})
    assert result.refused is True
    assert result.answer == agent.REFUSAL_MESSAGE
    assert result.citations == []
    assert result.stop_reason == "answered"


async def test_tool_call_then_answer_records_trace(fake_tools):
    tf, tool_calls = fake_tools
    provider = Scripted([
        _resp(tool_calls=[_tc("1", "search_controls", {"query": "account mgmt"})]),
        _resp(content="Manage accounts [NIST AC-2]."),
    ])
    result = await run_agent("account management?", provider, tool_functions=tf)
    assert result.answer == "Manage accounts [NIST AC-2]."
    assert result.stop_reason == "answered"
    assert result.refused is False
    assert len(result.trace) == 1
    assert result.trace[0]["tool"] == "search_controls"
    assert result.trace[0]["args"] == {"query": "account mgmt"}
    assert "[NIST AC-2]" in result.trace[0]["result_summary"]
    assert tool_calls == [("search_controls", "account mgmt", None)]


async def test_step_cap_enforced(monkeypatch, fake_tools):
    monkeypatch.setattr(agent.get_settings(), "max_agent_steps", 2, raising=False)
    tf, _ = fake_tools
    forever = [_resp(tool_calls=[_tc(str(i), "search_controls", {"query": "x"})]) for i in range(10)]
    provider = Scripted(forever)
    result = await run_agent("loop forever", provider, tool_functions=tf)
    assert result.stop_reason == "max_steps"
    assert result.refused is True
    assert result.answer == agent.REFUSAL_MESSAGE
    assert len(provider.calls) == 2


async def test_token_cap_enforced(monkeypatch, fake_tools):
    s = agent.get_settings()
    monkeypatch.setattr(s, "max_agent_steps", 10, raising=False)
    monkeypatch.setattr(s, "max_request_tokens", 50, raising=False)
    tf, _ = fake_tools
    provider = Scripted([
        _resp(tool_calls=[_tc("1", "search_controls", {"query": "x"})], it=30, ot=10),
        _resp(tool_calls=[_tc("2", "search_controls", {"query": "y"})], it=30, ot=10),
        _resp(content="never reached"),
    ])
    result = await run_agent("burn tokens", provider, tool_functions=tf)
    assert result.stop_reason == "max_tokens"
    assert result.input_tokens + result.output_tokens >= 50
    assert len(provider.calls) == 2


async def test_system_prompt_is_first_message(fake_tools):
    tf, _ = fake_tools
    provider = Scripted([_resp(content="done [NIST AC-2].")])
    await run_agent("q", provider, tool_functions=tf)
    first = provider.calls[0]["messages"]
    assert first[0][0] == "system"
    assert first[1][0] == "user"
    assert first[1][1].startswith("q")


class Recording:
    """Provider that keeps deep copies of the Message objects it was sent."""

    def __init__(self, responses):
        self._responses = list(responses)
        self.sent = []

    async def chat(self, messages, tools=None):
        self.sent.append([m.model_copy(deep=True) for m in messages])
        return self._responses.pop(0)


async def test_assistant_and_tool_messages_carry_tool_calls_and_ids(fake_tools):
    tf, _ = fake_tools
    provider = Recording([
        _resp(content="looking", tool_calls=[_tc("call-x", "search_controls", {"query": "a"})]),
        _resp(content="Manage accounts [NIST AC-2]."),
    ])
    result = await run_agent("accounts?", provider, tool_functions=tf)
    assert result.refused is False
    msgs = provider.sent[1]
    assert [m.role for m in msgs] == ["system", "user", "assistant", "tool"]
    assistant, tool = msgs[2], msgs[3]
    assert assistant.content == "looking"
    assert assistant.tool_calls == [
        {"id": "call-x", "name": "search_controls", "arguments": {"query": "a"}}
    ]
    assert tool.tool_call_id == "call-x"


async def test_missing_tool_call_ids_are_synthesised_stably(fake_tools):
    tf, _ = fake_tools
    provider = Recording([
        _resp(tool_calls=[_tc("", "search_controls", {"query": "a"}),
                          {"name": "search_controls", "arguments": {"query": "b"}}]),
        _resp(content="Manage accounts [NIST AC-2]."),
    ])
    await run_agent("accounts?", provider, tool_functions=tf)
    msgs = provider.sent[1]
    ids = [c["id"] for c in msgs[2].tool_calls]
    assert ids == ["call_1_0", "call_1_1"]
    assert [m.tool_call_id for m in msgs[3:]] == ids


# --- I1/M7: citation id normalisation in the grounding check --------------
def _tool_returning(fw, cid):
    async def search_controls(query, framework=None):
        return {"controls": [{"framework": fw, "control_id": cid,
                              "citation": f"[{fw} {cid}]"}]}
    return {"search_controls": search_controls}


async def test_cis_citation_without_prefix_is_grounded():
    provider = Scripted([
        _resp(tool_calls=[_tc("1", "search_controls", {"query": "inventory"})]),
        _resp(content="Keep a data inventory [CIS 3.1]."),
    ])
    result = await run_agent("data inventory?", provider,
                             tool_functions=_tool_returning("CIS", "CIS-3.1"))
    assert result.refused is False
    assert result.citations == ["CIS CIS-3.1"]


async def test_nist_parenthesised_enhancement_is_grounded():
    provider = Scripted([
        _resp(tool_calls=[_tc("1", "search_controls", {"query": "accounts"})]),
        _resp(content="Automate it [NIST AC-2(1)] and [nist-x]."),
    ])
    result = await run_agent("automated account mgmt?", provider,
                             tool_functions=_tool_returning("NIST", "AC-2.1"))
    assert result.refused is False
    assert result.citations == ["NIST AC-2.1"]


async def test_citation_nobody_returned_is_still_refused():
    provider = Scripted([
        _resp(tool_calls=[_tc("1", "search_controls", {"query": "inventory"})]),
        _resp(content="Keep a data inventory [CIS 3.1] and [CIS 4.1]."),
    ])
    result = await run_agent("data inventory?", provider,
                             tool_functions=_tool_returning("CIS", "CIS-3.1"))
    assert result.refused is True
    assert result.citations == []


# --- I2: the framework filter is enforced, not advisory -------------------
async def test_framework_filter_is_forced_on_search_and_map():
    seen = []

    async def search_controls(query, framework=None):
        seen.append(("search_controls", query, framework))
        return {"controls": [{"framework": "ISO", "control_id": "A.5.15",
                              "citation": "[ISO A.5.15]"}]}

    async def map_finding(text, framework=None):
        seen.append(("map_finding", text, framework))
        return {"controls": []}

    provider = Scripted([
        _resp(tool_calls=[_tc("1", "search_controls", {"query": "access", "framework": "NIST"}),
                          _tc("2", "map_finding", {"text": "stale accounts"})]),
        _resp(content="Control access [ISO A.5.15]."),
    ])
    result = await run_agent("access control?", provider, framework="ISO",
                             tool_functions={"search_controls": search_controls,
                                             "map_finding": map_finding})
    assert result.refused is False
    assert seen == [("search_controls", "access", "ISO"), ("map_finding", "stale accounts", "ISO")]
    # the bare question reaches the model's user turn (and so FakeProvider's query)
    user = provider.calls[0]["messages"][1]
    assert user == ("user", "access control?")


async def test_no_framework_leaves_model_choice(fake_tools):
    tf, calls = fake_tools
    provider = Scripted([
        _resp(tool_calls=[_tc("1", "search_controls", {"query": "x", "framework": "NIST"})]),
        _resp(content="Manage accounts [NIST AC-2]."),
    ])
    await run_agent("q", provider, tool_functions=tf)
    assert calls == [("search_controls", "x", "NIST")]


# --- I3: tool arguments are validated before the call ---------------------
async def test_unexpected_tool_arg_keys_are_dropped(fake_tools):
    tf, calls = fake_tools
    provider = Scripted([
        _resp(tool_calls=[_tc("1", "search_controls",
                              {"query": "x", "top_k": 5, "frameworks": ["NIST"]})]),
        _resp(content="Manage accounts [NIST AC-2]."),
    ])
    result = await run_agent("q", provider, tool_functions=tf)
    assert result.refused is False
    assert calls == [("search_controls", "x", None)]
    assert result.trace[0]["args"] == {"query": "x"}


async def test_non_dict_tool_args_become_empty(fake_tools):
    tf, calls = fake_tools
    provider = Scripted([
        _resp(tool_calls=[_tc("1", "search_controls", "[1, 2]")]),
        _resp(content="nothing [NIST AC-2]."),
    ])
    result = await run_agent("q", provider, tool_functions=tf)
    # search_controls without a query raises TypeError -> fed back as an error result
    assert calls == []
    assert result.trace[0]["args"] == {}
    assert "error" in result.trace[0]["result_summary"]
    assert result.refused is True  # AC-2 was never returned by a tool


@pytest.mark.parametrize("given,expected", [
    ("", None), ("ALL", None), ("any", None), (None, None), ("SOC2", None),
    ("nist", "NIST"), (" iso ", "ISO"), ("Cis", "CIS"), (3, None),
])
async def test_model_framework_values_are_normalised(fake_tools, given, expected):
    tf, calls = fake_tools
    provider = Scripted([
        _resp(tool_calls=[_tc("1", "search_controls", {"query": "x", "framework": given})]),
        _resp(content="Manage accounts [NIST AC-2]."),
    ])
    await run_agent("q", provider, tool_functions=tf)
    assert calls == [("search_controls", "x", expected)]


async def test_raising_tool_is_fed_back_as_error_not_500():
    async def boom(query, framework=None):
        raise RuntimeError("db down")

    provider = Scripted([
        _resp(tool_calls=[_tc("1", "search_controls", {"query": "x"})]),
        _resp(content="Sorry, no grounded answer."),
    ])
    result = await run_agent("q", provider, tool_functions={"search_controls": boom})
    assert result.refused is True
    tool_turn = provider.calls[1]["messages"][-1]
    assert tool_turn[0] == "tool"
    assert json.loads(tool_turn[1]) == {"error": "RuntimeError: db down"}
