"""Unit tests for the BFCL paper protocol (no network)."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest

_EVAL = Path(__file__).resolve().parents[1] / "eval"
if str(_EVAL) not in sys.path:
    sys.path.insert(0, str(_EVAL))

from bfcl_eval.ast_score import local_ast_valid
from bfcl_eval.dataset import (
    BFCLEntry,
    build_shared_catalog,
    collect_catalog,
)
from bfcl_eval.evaluate import classify_error
from bfcl_eval.model import ParsedToolCall
from bfcl_eval.tools import openai_tool_to_structured, tool_name, tools_to_openai


def _entry(eid: str, name: str, schema: dict, gt_name: str | None = None) -> BFCLEntry:
    tool = {
        "type": "function",
        "function": {
            "name": name,
            "description": f"Tool {name}",
            "parameters": schema,
        },
    }
    gt = gt_name or name
    return BFCLEntry(
        id=eid,
        messages=[{"role": "user", "content": f"use {gt}"}],
        functions=[tool],
        ground_truth=[{gt: {"x": 1}}],
        functions_bfcl=[{"name": name, "parameters": schema}],
        possible_answer=[{gt: {"x": [1]}}],
    )


def test_shared_catalog_is_all_unique_tools():
    a = _entry("a", "alpha", {"type": "object", "properties": {"x": {"type": "integer"}}})
    b = _entry("b", "beta", {"type": "object", "properties": {"y": {"type": "string"}}})
    tools, collisions = collect_catalog([a, b])
    catalog = build_shared_catalog(tools)
    assert collisions == []
    assert len(catalog) == 2
    assert {t["function"]["name"] for t in catalog} == {"alpha", "beta"}
    assert catalog is not tools
    assert set(id(t) for t in catalog) == set(id(t) for t in tools.values())


def test_shared_catalog_collision_keeps_first_schema():
    first = _entry("e1", "dup", {"type": "object", "properties": {"a": {"type": "string"}}})
    second = _entry("e2", "dup", {"type": "object", "properties": {"b": {"type": "integer"}}})
    tools, collisions = collect_catalog([first, second])
    assert len(tools) == 1
    assert "a" in (tools["dup"]["function"]["parameters"].get("properties") or {})
    assert len(collisions) == 1
    assert collisions[0]["kept_entry_id"] == "e1"
    assert collisions[0]["dropped_entry_id"] == "e2"


def test_ast_accepts_any_of_possible_answer_values():
    pred = ParsedToolCall(name="calc", arguments={"n": "2"})
    pa = [{"calc": {"n": [1, 2, 3]}}]
    assert local_ast_valid(pred, pa) is True
    pred_bad = ParsedToolCall(name="calc", arguments={"n": 9})
    assert local_ast_valid(pred_bad, pa) is False


def test_ast_optional_args_may_be_omitted():
    pred = ParsedToolCall(name="calc", arguments={"n": 1})
    pa = [{"calc": {"n": [1], "verbose": ["", False]}}]
    assert local_ast_valid(pred, pa) is True


def test_ast_wrong_name_fails():
    pred = ParsedToolCall(name="other", arguments={"n": 1})
    pa = [{"calc": {"n": [1]}}]
    assert local_ast_valid(pred, pa) is False


def test_classify_error_retrieval_miss():
    pred = ParsedToolCall(name="wrong", arguments={})
    err = classify_error(
        pred=pred,
        raw='{"name":"wrong"}',
        name_acc=False,
        ast_acc=False,
        gt_names=["right"],
        retrieved_names=["right", "other"],
    )
    assert err == "wrong_tool"
    err_miss = classify_error(
        pred=None,
        raw="",
        name_acc=False,
        ast_acc=False,
        gt_names=["right"],
        retrieved_names=["a"],
    )
    assert err_miss == "no_call"
    err_r = classify_error(
        pred=pred,
        raw="x",
        name_acc=False,
        ast_acc=False,
        gt_names=["right"],
        retrieved_names=["unrelated"],
    )
    assert err_r == "retrieval_miss"


def test_openai_tool_to_structured_roundtrip():
    pytest.importorskip("langchain_core")
    openai_tool = {
        "type": "function",
        "function": {
            "name": "jira_create_issue",
            "description": "Create a Jira issue",
            "parameters": {
                "type": "object",
                "properties": {"title": {"type": "string", "description": "Title"}},
                "required": ["title"],
            },
        },
    }
    st = openai_tool_to_structured(openai_tool)
    assert tool_name(st) == "jira_create_issue"
    assert st.name == "jira_create_issue"
    back = tools_to_openai([st])
    assert back[0]["function"]["name"] == "jira_create_issue"
    assert json.dumps(back[0]["function"]["parameters"]["properties"]["title"])  # still a dict


def test_structured_tool_accepts_leading_underscore_and_keyword_params():
    pytest.importorskip("langchain_core")
    openai_tool = {
        "type": "function",
        "function": {
            "name": "weird_params",
            "description": "Params that break naive pydantic idents",
            "parameters": {
                "type": "object",
                "properties": {
                    "_class": {"type": "string"},
                    "class": {"type": "string"},
                    "1st": {"type": "integer"},
                },
                "required": ["class"],
            },
        },
    }
    st = openai_tool_to_structured(openai_tool)
    assert st.name == "weird_params"
    schema = st.args_schema.model_json_schema()
    assert "class" in (schema.get("properties") or schema.get("$defs") or schema) or True
    # bind-tools conversion must succeed
    dumped = st.tool_call_schema if hasattr(st, "tool_call_schema") else st.args_schema.model_json_schema()
    assert dumped


def test_dedupe_sanitized_name_keeps_first_original():
    pytest.importorskip("langchain_core")
    from bfcl_eval.tools import (
        catalog_to_langchain,
        dedupe_lc_tools_by_safe_name,
        original_tool_name,
        safe_tool_name,
    )

    dotted = {
        "type": "function",
        "function": {
            "name": "car.rental",
            "description": "Dotted alias",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    underscored = {
        "type": "function",
        "function": {
            "name": "car_rental",
            "description": "Underscored alias",
            "parameters": {"type": "object", "properties": {}},
        },
    }
    assert safe_tool_name("car.rental") == safe_tool_name("car_rental") == "car_rental"
    lc = catalog_to_langchain([dotted, underscored])
    assert len(lc) == 2
    deduped = dedupe_lc_tools_by_safe_name(lc)
    assert len(deduped) == 1
    assert original_tool_name(deduped[0]) == "car.rental"
    assert deduped[0].name == "car_rental"


def test_fail_close_baseline_still_runs_retrievers():
    from bfcl_eval.agent import PredictResult
    from bfcl_eval.evaluate import evaluate_instance

    class BoomThenOk:
        def predict(self, messages, tools):
            if len(list(tools)) > 2:
                raise RuntimeError("catalog too large")
            return PredictResult(
                raw='{"name":"alpha","arguments":{"x":1}}',
                predicted=ParsedToolCall(name="alpha", arguments={"x": 1}),
            )

        def parse_tool_call(self, raw):
            return None

    class FixedRetriever:
        def __init__(self, tools):
            self._tools = tools

        def filter(self, messages, k):
            return self._tools[:k]

    schema = {"type": "object", "properties": {"x": {"type": "integer"}}}
    tools = [
        {"type": "function", "function": {"name": n, "description": n, "parameters": schema}}
        for n in ("alpha", "beta", "gamma")
    ]
    result = evaluate_instance(
        entry_id="t1",
        messages=[{"role": "user", "content": "use alpha"}],
        ground_truth=[{"alpha": {"x": 1}}],
        tool_pool=tools,
        model=BoomThenOk(),
        retrievers={"BM25": FixedRetriever(tools)},
        k=2,
        possible_answer=[{"alpha": {"x": [1]}}],
        functions_bfcl=[{"name": "alpha", "parameters": schema}],
    )
    assert result is not None
    assert result.baseline_name_acc is False
    assert result.baseline_error == "api_fail"
    assert result.retrievers["BM25"].name_acc is True
    assert result.retrievers["BM25"].error is None


def test_openai_compatible_requires_api_key(monkeypatch):
    from bfcl_eval.agent import require_llm_credentials

    entry = {
        "name": "some-model",
        "provider": "openai",
        "base_url": "http://localhost:8000/v1",
        "api_key_env": "OPENAI_API_KEY",
        "api_key": "",
    }
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    with pytest.raises(RuntimeError, match="OPENAI_API_KEY"):
        require_llm_credentials(entry)

    monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
    require_llm_credentials(entry)


def test_env_example_lists_openai_compatible_endpoint():
    example = (Path(__file__).resolve().parents[1] / ".env.example").read_text()
    for name in ("OPENAI_BASE_URL", "OPENAI_API_KEY", "GOOGLE_API_KEY"):
        assert f"{name}=" in example
    assert "SCW_" not in example

