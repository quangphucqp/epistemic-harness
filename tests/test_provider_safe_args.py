"""Regression tests for provider-safe JSON probe arguments."""

from __future__ import annotations

import json
from copy import deepcopy

import pytest

from epistemic_harness.harness import EpistemicHarness, HarnessError
from epistemic_harness.schemas import PROBE_SCHEMA


def _open(harness: EpistemicHarness, case_id: str = "args-case") -> None:
    harness.model(
        {"operation": "open", "case_id": case_id, "top_unknown": "Which arguments are read?"},
        session_id="args-session",
    )


_MISSING = object()


def _commit_args(*, tool_args: dict | None = None, tool_args_json=_MISSING) -> dict:
    args = {
        "operation": "commit",
        "purpose": "learn",
        "unknown_or_goal": "Whether the ordinary tool returns the expected value",
        "tool_name": "read_file",
        "predicted_outcomes": ["The tool returns the expected value"],
        "why_this_action": "The ordinary tool is the designated check",
        "would_change_belief": "The tool returns an incompatible value",
        "authority_domain": "system_state",
    }
    if tool_args is not None:
        args["tool_args"] = tool_args
    if tool_args_json is not _MISSING:
        args["tool_args_json"] = tool_args_json
    return args


def test_probe_schema_exposes_a_provider_safe_string_route():
    properties = PROBE_SCHEMA["parameters"]["properties"]
    assert properties["tool_args_json"]["type"] == "string"
    assert "JSON object" in properties["tool_args_json"]["description"]

    try:
        from tools.schema_sanitizer import sanitize_tool_schemas
    except ImportError:
        pytest.skip("Hermes schema sanitizer unavailable")
    cleaned = sanitize_tool_schemas([{"type": "function", "function": PROBE_SCHEMA}])[0]
    cleaned_property = cleaned["function"]["parameters"]["properties"]["tool_args_json"]
    assert cleaned_property["type"] == "string"


def test_nonempty_nested_json_arguments_are_decoded_and_committed(tmp_path):
    harness = EpistemicHarness(tmp_path)
    _open(harness)
    decoded = {
        "query": "source",
        "filters": {"active": True, "limit": 3},
        "items": [{"id": "A", "weights": [1, 2.5]}],
    }
    committed = harness.probe(
        _commit_args(tool_args_json=json.dumps(decoded)),
        session_id="args-session",
    )
    assert committed["probe"]["tool_args"] == decoded
    assert committed["probe"]["tool_args_sha256"]


def test_json_arguments_preserve_redaction_and_exact_execution_commitment(tmp_path):
    harness = EpistemicHarness(tmp_path)
    _open(harness)
    decoded = {
        "query": "source",
        "token": "secret-value",
        "flags": [True, False],
        "threshold": 0.75,
        "nested": {"count": 2},
    }
    harness.probe(
        _commit_args(tool_args_json=json.dumps(decoded)),
        session_id="args-session",
    )
    captured = {}
    returned = harness.tool_execution_middleware(
        tool_name="read_file",
        args=decoded,
        session_id="args-session",
        next_call=lambda args: (captured.update({"args": args}), "ordinary result")[1],
    )
    assert returned == "ordinary result"
    assert captured["args"] == decoded
    pending = harness.store.get_active_case("args-session")["pending_probe"]
    assert pending is not None
    assert pending["execution_args_match_commitment"] is True
    assert pending["tool_args"]["token"] == "[REDACTED]"
    assert pending["tool_args"]["flags"] == [True, False]
    assert pending["tool_args"]["threshold"] == 0.75


@pytest.mark.parametrize(
    "raw",
    [
        "{\"query\":",
        "[]",
        "null",
        "{\"value\": NaN}",
        "{\"value\": Infinity}",
        "{\"query\":\"first\",\"query\":\"second\"}",
    ],
)
def test_invalid_json_arguments_are_rejected_before_probe_commit(tmp_path, raw):
    harness = EpistemicHarness(tmp_path)
    _open(harness)
    before = deepcopy(harness.store.get_active_case("args-session"))
    with pytest.raises(HarnessError, match="tool_args_json|JSON|duplicate|finite|object"):
        harness.probe(_commit_args(tool_args_json=raw), session_id="args-session")
    after = harness.store.get_active_case("args-session")
    assert after == before
    assert [event["type"] for event in harness.store.list_events("args-case")] == [
        "case_opened"
    ]


def test_json_arguments_are_authoritative_and_conflicting_legacy_args_are_rejected(tmp_path):
    harness = EpistemicHarness(tmp_path)
    _open(harness)
    decoded = {"query": "authoritative", "nested": {"value": 4}}
    before = deepcopy(harness.store.get_active_case("args-session"))
    with pytest.raises(HarnessError, match="agree|conflict|authoritative"):
        harness.probe(
            _commit_args(
                tool_args={"query": "different"},
                tool_args_json=json.dumps(decoded),
            ),
            session_id="args-session",
        )
    assert harness.store.get_active_case("args-session") == before
    assert [event["type"] for event in harness.store.list_events("args-case")] == [
        "case_opened"
    ]


def test_json_arguments_allow_provider_materialized_empty_legacy_shapes(tmp_path):
    for legacy in (None, {}):
        harness = EpistemicHarness(tmp_path / ("none" if legacy is None else "empty"))
        _open(harness)
        decoded = {"query": "authoritative", "nested": {"value": 4}}
        payload = _commit_args(tool_args=legacy, tool_args_json=json.dumps(decoded))
        payload["tool_args"] = legacy
        committed = harness.probe(
            payload,
            session_id="args-session",
        )
        assert committed["probe"]["tool_args"] == decoded


def test_legacy_dictionary_arguments_remain_supported(tmp_path):
    harness = EpistemicHarness(tmp_path)
    _open(harness)
    legacy = {"path": "legacy-source.txt", "line": 2}
    committed = harness.probe(_commit_args(tool_args=legacy), session_id="args-session")
    assert committed["probe"]["tool_args"] == legacy
