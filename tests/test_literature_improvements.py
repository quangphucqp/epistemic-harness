# AI-assisted contribution; maintained by Epistemic Harness contributors.
"""Regression tests for the literature-improvement implementation brief."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest
import yaml

from epistemic_harness.harness import EpistemicHarness, HarnessError, _closure_calibration
from epistemic_harness.store import CaseStore, StoreError


def _open_args(case_id: str = "improvement-case") -> dict:
    return {
        "operation": "open",
        "case_id": case_id,
        "decision": "Choose the account supported by the evidence",
        "stopping_condition": "The account is bounded or remains unresolved",
        "state_grounding": [
            {"id": "C1", "text": "The source supports account one", "evidence_event_ids": []}
        ],
        "mechanisms": [],
        "alternatives": [
            {"id": "A1", "text": "Task structure explains the observation"}
        ],
        "top_unknown": "Whether the source supports account one",
        "current_plan": [],
    }


def _probe_args(*, tool_name: str = "read_file", tool_args: dict | None = None) -> dict:
    return {
        "operation": "commit",
        "purpose": "learn",
        "unknown_or_goal": "Whether the designated source supports the account",
        "tool_name": tool_name,
        "tool_args": tool_args or {"path": "missing-source.txt"},
        "predicted_outcomes": [
            {"outcome": "support", "meaning": "Retain the account if the source supports it"},
            {"outcome": "contrary or unavailable", "meaning": "Keep the conclusion uncertain"},
        ],
        "why_this_action": "The designated observation can change the next judgment",
        "would_change_belief": "The result is incompatible with the account",
        "authority_domain": "external_source",
    }


def _bound_open_args(
    case_id: str = "bound-case",
    *,
    mechanism_status: str = "hypothesis",
) -> dict:
    args: dict[str, Any] = dict(_open_args(case_id))
    mechanism = {
        "id": "M1",
        "text": "Account one predicts the observation",
        "epistemic_status": mechanism_status,
    }
    args["mechanisms"] = [mechanism]
    args["current_plan"] = [
        {"id": "P1", "action": "Inspect the source", "depends_on": ["M1"]}
    ]
    if mechanism_status == "grounded":
        args["initial_evidence"] = [
            {
                "id": "I1",
                "summary": "The initial full source record",
                "source_ref": "synthetic://initial-source",
                "source_role": "primary",
                "access_scope": "full",
                "authority_domain": "external_source",
            }
        ]
        mechanism["authority_domain"] = "external_source"
        mechanism["evidence_event_ids"] = ["I1"]
    return args


def _bound_probe_args(claim_id: str = "M1") -> dict:
    args = _probe_args()
    args["claim_id"] = claim_id
    return args


def _full_evidence() -> dict:
    return {
        "summary": "The inspected source would be full text if it were available",
        "source_ref": "synthetic://full-source",
        "source_role": "primary",
        "access_scope": "full",
        "authority_domain": "external_source",
    }


def _failed_observation(harness: EpistemicHarness, session_id: str = "session-1") -> str:
    harness.probe(_probe_args(), session_id=session_id)
    returned = harness.tool_execution_middleware(
        tool_name="read_file",
        args={"path": "missing-source.txt"},
        session_id=session_id,
        next_call=lambda _args: {"error": {"code": "SOURCE_UNAVAILABLE", "details": {"retry": False}}},
    )
    assert returned["error"]["code"] == "SOURCE_UNAVAILABLE"
    observed = next(
        event
        for event in harness.store.list_events("improvement-case")
        if event["type"] == "probe_observed"
    )
    assert observed["payload"]["result_status"] == "tool_error"
    return observed["event_id"]


def _ground_claim_in_event(
    harness: EpistemicHarness,
    event_id: str,
    *,
    claim_id: str = "C1",
    session_id: str = "session-1",
) -> dict:
    return harness.model(
        {
            "operation": "revise",
            "reason": "Bind the claim only to the recorded observation",
            "updates": {
                "state_grounding": [
                    {
                        "id": claim_id,
                        "text": "The source supports account one",
                        "epistemic_status": "grounded",
                        "authority_domain": "external_source",
                        "evidence_event_ids": [event_id],
                    }
                ]
            },
        },
        session_id=session_id,
    )


def test_failed_direct_observation_cannot_support_resolved_closure(tmp_path):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="session-1")
    observation_id = _failed_observation(harness)
    harness.probe(
        {
            "operation": "compare",
            "disposition": "unresolved",
            "material": False,
            "rationale": "The source acquisition failed, so it did not test the external claim",
            "affected_claim_ids": [],
        },
        session_id="session-1",
    )

    revised = _ground_claim_in_event(harness, observation_id)
    calibration = _closure_calibration(
        revised["case"],
        events=harness.store.list_events("improvement-case"),
    )
    assert calibration["label"] == "hypothesis"
    claim = next(item for item in calibration["claims"] if item["claim_id"] == "C1")
    assert claim["label"] == "hypothesis"
    with pytest.raises(HarnessError, match="resolved closure requires"):
        harness.model(
            {
                "operation": "close",
                "outcome": "resolved",
                "summary": "The source was unavailable; the claim remains unresolved",
                "transfer": {"decision": "none"},
            },
            session_id="session-1",
        )


def test_failed_observation_cannot_gain_support_from_full_comparison_descriptor(tmp_path):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="session-1")
    _failed_observation(harness)
    compared = harness.probe(
        {
            "operation": "compare",
            "disposition": "unresolved",
            "material": False,
            "rationale": "A full-access label cannot turn an unavailable source into an observation",
            "affected_claim_ids": [],
            "evidence": _full_evidence(),
        },
        session_id="session-1",
    )
    comparison_id = compared["event"]["event_id"]
    revised = _ground_claim_in_event(harness, comparison_id)
    calibration = _closure_calibration(
        revised["case"],
        events=harness.store.list_events("improvement-case"),
    )
    assert calibration["label"] == "hypothesis"
    assert calibration["claims"][0]["label"] == "hypothesis"


def test_successful_search_lead_does_not_become_full_support_from_descriptor(tmp_path):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="session-1")
    harness.probe(
        _probe_args(tool_name="web_search", tool_args={"query": "account one"}),
        session_id="session-1",
    )
    harness.tool_execution_middleware(
        tool_name="web_search",
        args={"query": "account one"},
        session_id="session-1",
        next_call=lambda _args: "search result lead",
    )
    compared = harness.probe(
        {
            "operation": "compare",
            "disposition": "match",
            "material": False,
            "rationale": "The search lead is consistent but does not expose the source text",
            "affected_claim_ids": [],
            "evidence": _full_evidence(),
        },
        session_id="session-1",
    )
    revised = _ground_claim_in_event(harness, compared["event"]["event_id"])
    calibration = _closure_calibration(
        revised["case"],
        events=harness.store.list_events("improvement-case"),
    )
    assert calibration["label"] == "provisional"
    assert calibration["claims"][0]["label"] == "provisional"


@pytest.mark.parametrize(
    "result",
    [
        {"error": {"code": "E_NESTED", "details": {"path": ["a", "b"]}}},
        json.dumps({"error": {"code": "E_NESTED", "details": {"path": ["a", "b"]}}}),
        {"success": False, "payload": {"message": "not available"}},
        json.dumps({"success": False, "payload": {"message": "not available"}}),
    ],
)
def test_middleware_classifies_raw_and_string_error_envelopes_consistently(tmp_path, result):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="session-1")
    harness.probe(_probe_args(), session_id="session-1")
    harness.tool_execution_middleware(
        tool_name="read_file",
        args={"path": "missing-source.txt"},
        session_id="session-1",
        next_call=lambda _args: result,
    )
    case = harness.store.get_active_case("session-1")
    assert case["pending_probe"]["result_status"] == "tool_error"


def test_legitimate_payload_text_is_not_classified_as_a_tool_error(tmp_path):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="session-1")
    harness.probe(_probe_args(), session_id="session-1")
    harness.tool_execution_middleware(
        tool_name="read_file",
        args={"path": "missing-source.txt"},
        session_id="session-1",
        next_call=lambda _args: {"success": True, "message": "The word error is part of the report"},
    )
    case = harness.store.get_active_case("session-1")
    assert case["pending_probe"]["result_status"] == "ok"


def test_evidence_batch_is_prevalidated_before_any_transition(tmp_path):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="session-1")
    case_path = tmp_path / "cases" / "improvement-case" / "case.md"
    events_path = tmp_path / "cases" / "improvement-case" / "events.jsonl"
    before_case = case_path.read_bytes()
    before_events = events_path.read_bytes()
    with pytest.raises(HarnessError, match="summary and source_ref"):
        harness.model(
            {
                "operation": "record_evidence",
                "evidence": [
                    {
                        "summary": "The first item is valid",
                        "source_ref": "synthetic://first",
                        "source_role": "primary",
                        "access_scope": "full",
                        "authority_domain": "external_source",
                    },
                    {
                        "source_ref": "synthetic://malformed-later-item",
                        "source_role": "primary",
                        "access_scope": "full",
                        "authority_domain": "external_source",
                    },
                ],
            },
            session_id="session-1",
        )
    assert case_path.read_bytes() == before_case
    assert events_path.read_bytes() == before_events
    assert [event["type"] for event in harness.store.list_events("improvement-case")] == [
        "case_opened"
    ]


def test_corrupt_snapshot_cannot_be_healed_by_sibling_record_evidence(tmp_path):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="session-1")
    case_path = tmp_path / "cases" / "improvement-case" / "case.md"
    events_path = tmp_path / "cases" / "improvement-case" / "events.jsonl"
    case_path.write_text(
        case_path.read_text(encoding="utf-8").replace(
            "Whether the source supports account one", "Corrupted bounded question"
        ),
        encoding="utf-8",
    )
    corrupted_case = case_path.read_bytes()
    before_events = events_path.read_bytes()
    assert harness.replay("improvement-case")["case_snapshot"]["valid"] is False
    with pytest.raises(HarnessError, match="integrity"):
        harness.model(
            {
                "operation": "record_evidence",
                "evidence": [_full_evidence()],
            },
            session_id="session-1",
        )
    assert case_path.read_bytes() == corrupted_case
    assert events_path.read_bytes() == before_events
    assert harness.replay("improvement-case")["case_snapshot"]["valid"] is False


def test_common_persistence_boundary_rejects_corrupt_snapshot_for_update_and_transition(tmp_path):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="session-1")
    case_path = tmp_path / "cases" / "improvement-case" / "case.md"
    events_path = tmp_path / "cases" / "improvement-case" / "events.jsonl"
    case_path.write_text(
        case_path.read_text(encoding="utf-8").replace(
            "Whether the source supports account one", "Corrupted before a sibling write"
        ),
        encoding="utf-8",
    )
    corrupted_case = case_path.read_bytes()
    before_events = events_path.read_bytes()

    transition_case = harness.store.get_case("improvement-case")
    with pytest.raises(StoreError, match="integrity"):
        harness.store.transition_case(
            transition_case,
            event_type="sibling_transition",
            event_payload={"should_not": "persist"},
        )

    update_case = harness.store.get_case("improvement-case")
    update_case["version"] += 1
    update_case["model"]["top_unknown"] = "Incoming mutated copy"
    with pytest.raises(StoreError, match="integrity"):
        harness.store.update_case(
            update_case,
            event_type="sibling_update",
            event_payload={"should_not": "persist"},
        )

    assert case_path.read_bytes() == corrupted_case
    assert events_path.read_bytes() == before_events


def test_resume_rechecks_snapshot_after_outer_precheck(tmp_path, monkeypatch):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="session-1")
    harness.model(
        {"operation": "pause", "reason": "prepare integrity race"},
        session_id="session-1",
    )
    case_path = tmp_path / "cases" / "improvement-case" / "case.md"
    original_resume = harness.store.resume_paused_case

    def corrupt_then_resume(**kwargs):
        case_path.write_text(
            case_path.read_text(encoding="utf-8").replace(
                "Whether the source supports account one", "Changed after the outer replay check"
            ),
            encoding="utf-8",
        )
        return original_resume(**kwargs)

    monkeypatch.setattr(harness.store, "resume_paused_case", corrupt_then_resume)
    with pytest.raises(HarnessError, match="integrity"):
        harness.model(
            {"operation": "resume", "case_id": "improvement-case"},
            session_id="new-session",
        )
    case = harness.store.get_case("improvement-case")
    assert case["status"] == "paused"
    assert case["session_id"] == "session-1"
    assert harness.store.list_events("improvement-case")[-1]["type"] == "case_paused"


def test_manifest_and_initial_evidence_schema_are_explicitly_native():
    root = Path(__file__).parents[1]
    manifest = yaml.safe_load((root / "plugin.yaml").read_text(encoding="utf-8"))
    assert manifest["provides_middleware"] == ["tool_execution"]

    from epistemic_harness.schemas import MODEL_SCHEMA

    item = MODEL_SCHEMA["parameters"]["properties"]["initial_evidence"]["items"]
    expected = {
        "id",
        "summary",
        "source_ref",
        "source_role",
        "access_scope",
        "authority_domain",
        "locator",
        "limitation",
    }
    assert expected <= set(item["properties"])
    assert {"id", "summary", "source_ref", "authority_domain"} <= set(item["required"])

    try:
        from tools.schema_sanitizer import sanitize_tool_schemas
    except ImportError:
        pytest.skip("Hermes schema sanitizer unavailable")
    cleaned = sanitize_tool_schemas([{"type": "function", "function": MODEL_SCHEMA}])[0][
        "function"
    ]["parameters"]
    cleaned_item = cleaned["properties"]["initial_evidence"]["items"]
    assert expected <= set(cleaned_item["properties"])
    assert "id" in cleaned_item["required"]


def test_commit_snapshots_optional_claim_and_rejects_untrusted_binding_inputs(tmp_path):
    harness = EpistemicHarness(tmp_path)
    harness.model(_bound_open_args(), session_id="session-1")
    committed = harness.probe(_bound_probe_args(), session_id="session-1")
    probe = committed["probe"]
    assert probe["claim_id"] == "M1"
    assert probe["claim_snapshot"] == harness.store.get_case("bound-case")["model"]["mechanisms"][0]
    assert probe["model_version"] == harness.store.get_case("bound-case")["version"]

    harness.probe(
        {"operation": "discard", "reason": "reset the binding test"},
        session_id="session-1",
    )
    with pytest.raises(HarnessError, match="existing claim"):
        harness.probe(
            {**_bound_probe_args("P1")},
            session_id="session-1",
        )
    with pytest.raises(HarnessError, match="server-managed"):
        harness.probe(
            {
                **_bound_probe_args(),
                "claim_snapshot": {"id": "M1", "text": "caller replacement"},
            },
            session_id="session-1",
        )


def test_bound_compare_requires_separate_belief_change_and_preserves_claim_snapshot(tmp_path):
    harness = EpistemicHarness(tmp_path)
    harness.model(_bound_open_args(), session_id="session-1")
    harness.probe(_bound_probe_args(), session_id="session-1")
    harness.tool_execution_middleware(
        tool_name="read_file",
        args={"path": "missing-source.txt"},
        session_id="session-1",
        next_call=lambda _args: "the source is available",
    )
    with pytest.raises(HarnessError, match="belief_change"):
        harness.probe(
            {
                "operation": "compare",
                "disposition": "match",
                "material": False,
                "rationale": "Observation disposition alone does not state belief change",
                "evidence": _full_evidence(),
            },
            session_id="session-1",
        )
    with pytest.raises(HarnessError, match="bound claim"):
        harness.probe(
            {
                "operation": "compare",
                "disposition": "match",
                "material": False,
                "rationale": "Do not redirect a bound observation to another claim",
                "belief_change": "unchanged",
                "affected_claim_ids": ["C1"],
                "evidence": _full_evidence(),
            },
            session_id="session-1",
        )

    compared = harness.probe(
        {
            "operation": "compare",
            "disposition": "match",
            "material": False,
            "rationale": "The result was consistent, but the belief remains unchanged",
            "belief_change": "unchanged",
            "evidence": _full_evidence(),
        },
        session_id="session-1",
    )
    payload = compared["event"]["payload"]
    assert payload["affected_claim_ids"] == ["M1"]
    assert payload["belief_change"] == "unchanged"
    assert payload["claim_id"] == "M1"
    assert payload["claim_snapshot"] == compared["case"]["model"]["mechanisms"][0]
    assert payload["model_version"] == compared["case"]["version"]
    assert compared["case"]["model"]["mechanisms"][0]["epistemic_status"] == "hypothesis"


def _material_bound_mismatch(tmp_path, *, initial_status: str = "hypothesis"):
    harness = EpistemicHarness(tmp_path)
    harness.model(
        _bound_open_args(mechanism_status=initial_status),
        session_id="session-1",
    )
    harness.probe(_bound_probe_args(), session_id="session-1")
    harness.tool_execution_middleware(
        tool_name="read_file",
        args={"path": "missing-source.txt"},
        session_id="session-1",
        next_call=lambda _args: "the source contradicts account one",
    )
    compared = harness.probe(
        {
            "operation": "compare",
            "disposition": "mismatch",
            "material": True,
            "rationale": "The source contradicts the committed mechanism",
            "belief_change": "weakened",
            "evidence": _full_evidence(),
        },
        session_id="session-1",
    )
    return harness, compared["event"]["event_id"]


@pytest.mark.parametrize(
    ("initial_status", "revised_status"),
    [("hypothesis", "unresolved"), ("grounded", "hypothesis")],
)
def test_conservative_uncertainty_status_downgrade_repairs_stale_claim(
    tmp_path,
    initial_status,
    revised_status,
):
    harness, mismatch_id = _material_bound_mismatch(
        tmp_path,
        initial_status=initial_status,
    )
    revised = harness.model(
        {
            "operation": "revise",
            "reason": "Contrary evidence means the claim must remain uncertain",
            "addresses_event_ids": [mismatch_id],
            "updates": {
                "mechanisms": [
                    {
                        "id": "M1",
                        "text": "Account one predicts the observation",
                        "epistemic_status": revised_status,
                        "evidence_event_ids": [],
                    }
                ],
                "current_plan": [],
            },
        },
        session_id="session-1",
    )
    assert revised["case"]["stale"] is False
    assert revised["case"]["model"]["mechanisms"][0]["epistemic_status"] == revised_status
    assert revised["replay"]["usable"] is True

    closed = harness.model(
        {
            "operation": "close",
            "outcome": "unresolved",
            "summary": "Contrary evidence left the mechanism unresolved",
            "transfer": {"decision": "none"},
        },
        session_id="session-1",
    )
    assert closed["case"]["closure"]["outcome"] == "unresolved"


def test_status_upgrade_alone_does_not_repair_a_stale_claim(tmp_path):
    harness, mismatch_id = _material_bound_mismatch(tmp_path)
    with pytest.raises(HarnessError, match="substantive"):
        harness.model(
            {
                "operation": "revise",
                "reason": "A status label alone is not a contrary-evidence repair",
                "addresses_event_ids": [mismatch_id],
                "updates": {
                    "mechanisms": [
                        {
                            "id": "M1",
                            "text": "Account one predicts the observation",
                            "epistemic_status": "grounded",
                            "authority_domain": "external_source",
                            "evidence_event_ids": [mismatch_id],
                        }
                    ],
                    "current_plan": [],
                },
            },
            session_id="session-1",
        )


def test_argument_drift_must_be_acknowledged_at_compare_and_is_visible(tmp_path):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="session-1")
    harness.probe(
        _probe_args(tool_args={"path": "committed-source.txt"}),
        session_id="session-1",
    )
    harness.tool_execution_middleware(
        tool_name="read_file",
        args={"path": "different-source.txt"},
        session_id="session-1",
        next_call=lambda _args: "result from a different source",
    )
    with pytest.raises(HarnessError, match="argument drift"):
        harness.probe(
            {
                "operation": "compare",
                "disposition": "unresolved",
                "material": False,
                "rationale": "The result cannot be interpreted without acknowledging drift",
                "evidence": _full_evidence(),
            },
            session_id="session-1",
        )

    reason = "Used the fallback source because the committed path was unavailable"
    compared = harness.probe(
        {
            "operation": "compare",
            "disposition": "unresolved",
            "material": False,
            "rationale": "The fallback result does not settle the question",
            "argument_drift_reason": reason,
            "evidence": _full_evidence(),
        },
        session_id="session-1",
    )
    payload = compared["event"]["payload"]
    assert payload["execution_args_match_commitment"] is False
    assert payload["argument_match_status"] == "mismatched"
    assert payload["argument_drift_reason"] == reason
    assert "committed-source.txt" not in json.dumps(payload)
    assert "different-source.txt" not in json.dumps(payload)


def test_legacy_observation_status_is_unknown_not_positive_support():
    case = {
        "model": {
            "state_grounding": [
                {
                    "id": "C1",
                    "text": "A legacy observation supports the claim",
                    "epistemic_status": "grounded",
                    "authority_domain": "external_source",
                    "evidence_event_ids": ["E1"],
                }
            ],
            "mechanisms": [],
            "alternatives": [],
        }
    }
    calibration = _closure_calibration(
        case,
        events=[
            {
                "event_id": "E1",
                "type": "probe_observed",
                "payload": {
                    "authority_domain": "external_source",
                    "tool_name": "read_file",
                    "storage": "inline",
                    "result": "legacy result without a status",
                },
            }
        ],
    )
    assert calibration["label"] == "hypothesis"
    assert calibration["legacy_observation_event_ids"] == ["E1"]


@pytest.mark.parametrize("target_status", ["paused", "closed"])
def test_active_session_can_read_only_inspect_paused_or_closed_child(tmp_path, target_status):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args("parent-case"), session_id="parent-session")
    harness.model(_open_args("child-case"), session_id="child-session")
    if target_status == "paused":
        harness.model(
            {"operation": "pause", "reason": "child is waiting"},
            session_id="child-session",
        )
    else:
        harness.model(
            {
                "operation": "close",
                "outcome": "unresolved",
                "summary": "child remains unresolved",
                "transfer": {"decision": "none"},
            },
            session_id="child-session",
        )

    before = {
        str(path.relative_to(tmp_path)): path.read_bytes()
        for path in (tmp_path / "cases").rglob("*")
        if path.is_file()
    }
    shown = harness.model(
        {"operation": "show", "case_id": "child-case"},
        session_id="parent-session",
    )
    assert shown["case"]["case_id"] == "child-case"
    assert shown["case"]["status"] == target_status
    assert shown["replay"]["timeline"]["valid"] is True
    assert shown["replay"]["case_snapshot"]["valid"] is True
    assert {
        str(path.relative_to(tmp_path)): path.read_bytes()
        for path in (tmp_path / "cases").rglob("*")
        if path.is_file()
    } == before
    parent_active = harness.store.get_active_case("parent-session")
    assert parent_active is not None
    assert parent_active["case_id"] == "parent-case"
    current = harness.model({"operation": "show"}, session_id="parent-session")
    assert current["case"]["case_id"] == "parent-case"


def test_active_session_cannot_inspect_another_sessions_active_case(tmp_path):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args("parent-case"), session_id="parent-session")
    harness.model(_open_args("other-active"), session_id="other-session")
    with pytest.raises(HarnessError, match="another session"):
        harness.model(
            {"operation": "show", "case_id": "other-active"},
            session_id="parent-session",
        )


def _failed_exposure_chain(tmp_path, *, case_id="improvement-case"):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(case_id), session_id="session-1")
    harness.probe(_probe_args(), session_id="session-1")
    returned = harness.tool_execution_middleware(
        tool_name="read_file",
        args={"path": "missing-source.txt"},
        session_id="session-1",
        tool_call_id="call-exposure",
        next_call=lambda _args: json.dumps({"error": "Acquisition failed"}),
    )
    assert json.loads(returned)["error"] == "Acquisition failed"
    raw_probe = harness.store.get_active_case("session-1")["pending_probe"]
    raw_observation_id = raw_probe["observation_event_id"]
    assert raw_probe["result_status"] == "tool_error"

    exposed_ids = []
    for visible_result in (
        "The failed result was formatted for display.",
        "The formatted result received a second display wrapper.",
    ):
        rebound = harness.record_tool_result_exposure(
            tool_name="read_file",
            args={"path": "missing-source.txt"},
            result=visible_result,
            session_id="session-1",
            tool_call_id="call-exposure",
        )
        exposed_ids.append(rebound["event"]["event_id"])

    pending = harness.store.get_active_case("session-1")["pending_probe"]
    # The raw execution outcome is immutable provenance, while the current
    # result_status may continue to describe the latest model-visible form.
    assert pending.get("execution_result_status") == "tool_error"
    assert pending.get("display_result_status") == "ok"
    assert exposed_ids[1] != exposed_ids[0]
    assert harness.store.list_events(case_id)[-1]["payload"]["supersedes_event_id"] == exposed_ids[0]

    compared = harness.probe(
        {
            "operation": "compare",
            "disposition": "unresolved",
            "material": False,
            "rationale": "The failed execution and its display wrappers do not expose the source",
            "evidence": _full_evidence(),
        },
        session_id="session-1",
    )
    return harness, raw_observation_id, exposed_ids, compared["event"]["event_id"]


@pytest.mark.parametrize("citation_kind", ["direct_exposed", "comparison"])
def test_failed_execution_provenance_survives_exposure_chain_for_each_citation_path(
    tmp_path, citation_kind
):
    harness, _raw_id, exposed_ids, comparison_id = _failed_exposure_chain(tmp_path)
    cited_id = exposed_ids[-1] if citation_kind == "direct_exposed" else comparison_id
    revised = harness.model(
        {
            "operation": "revise",
            "reason": "The exposure chain records a failed acquisition, not source support",
            "updates": {
                "state_grounding": [
                    {
                        "id": "C1",
                        "text": "The source supports account one",
                        "epistemic_status": "grounded",
                        "authority_domain": "external_source",
                        "evidence_event_ids": [cited_id],
                    }
                ]
            },
        },
        session_id="session-1",
    )
    calibration = _closure_calibration(
        revised["case"],
        events=harness.store.list_events("improvement-case"),
    )
    assert calibration["label"] == "hypothesis"
    assert cited_id in calibration["failed_observation_event_ids"]
    with pytest.raises(HarnessError, match="resolved closure requires"):
        harness.model(
            {
                "operation": "close",
                "outcome": "resolved",
                "summary": "The acquisition failed and the source claim remains unresolved",
                "transfer": {"decision": "none"},
            },
            session_id="session-1",
        )


@pytest.mark.parametrize("supersedes_mode", ["omitted", "missing", "cross_probe"])
@pytest.mark.parametrize("citation_kind", ["direct_exposed", "comparison"])
def test_invalid_or_unknown_exposure_provenance_never_invents_success(
    tmp_path, supersedes_mode, citation_kind
):
    raw_event = {
        "event_id": "E1",
        "type": "probe_observed",
        "payload": {
            "probe_id": "PR1",
            "tool_name": "read_file",
            "authority_domain": "external_source",
            "result_status": "ok",
        },
    }
    events = [raw_event]
    if supersedes_mode == "cross_probe":
        events.append(
            {
                "event_id": "E2",
                "type": "probe_observed",
                "payload": {
                    "probe_id": "PR-other",
                    "tool_name": "read_file",
                    "authority_domain": "external_source",
                    "result_status": "ok",
                },
            }
        )
        supersedes = "E2"
    elif supersedes_mode == "missing":
        supersedes = "E404"
    else:
        supersedes = None
    exposed_id = "E3" if supersedes_mode == "cross_probe" else "E2"
    exposure_payload = {
        "probe_id": "PR1",
        "tool_name": "read_file",
        "authority_domain": "external_source",
        "result_status": "ok",
    }
    if supersedes is not None:
        exposure_payload["supersedes_event_id"] = supersedes
    events.append({"event_id": exposed_id, "type": "probe_exposed", "payload": exposure_payload})

    if citation_kind == "direct_exposed":
        cited_id = exposed_id
    else:
        cited_id = "E4"
        events.append(
            {
                "event_id": cited_id,
                "type": "probe_compared",
                "payload": {
                    "probe_id": "PR1",
                    "observation_event_id": exposed_id,
                    "authority_domain": "external_source",
                    "evidence": _full_evidence(),
                },
            }
        )

    case = {
        "model": {
            "state_grounding": [
                {
                    "id": "C1",
                    "text": "The external claim holds",
                    "epistemic_status": "grounded",
                    "authority_domain": "external_source",
                    "evidence_event_ids": [cited_id],
                }
            ],
            "mechanisms": [],
            "alternatives": [],
        }
    }
    calibration = _closure_calibration(case, events=events)
    assert calibration["label"] == "hypothesis"
    assert cited_id in calibration["legacy_observation_event_ids"]


def test_successful_execution_with_error_display_representation_is_not_support(tmp_path):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="session-1")
    harness.probe(_probe_args(), session_id="session-1")
    harness.tool_execution_middleware(
        tool_name="read_file",
        args={"path": "missing-source.txt"},
        session_id="session-1",
        tool_call_id="call-display-error",
        next_call=lambda _args: {"value": "successful execution"},
    )
    raw = harness.store.get_active_case("session-1")["pending_probe"]
    assert raw.get("execution_result_status") == "ok"
    rebound = harness.record_tool_result_exposure(
        tool_name="read_file",
        args={"path": "missing-source.txt"},
        result={"error": "display-only formatting failure"},
        session_id="session-1",
        tool_call_id="call-display-error",
    )
    pending = rebound["case"]["pending_probe"]
    assert pending.get("execution_result_status") == "ok"
    assert pending.get("display_result_status") == "tool_error"
    harness.probe(
        {
            "operation": "compare",
            "disposition": "unresolved",
            "material": False,
            "rationale": "An error representation does not establish the external claim",
            "evidence": _full_evidence(),
        },
        session_id="session-1",
    )
    revised = harness.model(
        {
            "operation": "revise",
            "reason": "The display error makes the observation non-supporting",
            "updates": {
                "state_grounding": [
                    {
                        "id": "C1",
                        "text": "The source supports account one",
                        "epistemic_status": "grounded",
                        "authority_domain": "external_source",
                        "evidence_event_ids": [
                            harness.store.list_events("improvement-case")[-2]["event_id"]
                        ],
                    }
                ]
            },
        },
        session_id="session-1",
    )
    assert _closure_calibration(
        revised["case"], events=harness.store.list_events("improvement-case")
    )["label"] == "hypothesis"


def _case_data_from_markdown(path):
    text = path.read_text(encoding="utf-8")
    start = text.index("<!-- EPISTEMIC-HARNESS-DATA\n") + len("<!-- EPISTEMIC-HARNESS-DATA\n")
    end = text.index("\nEND-EPISTEMIC-HARNESS-DATA -->", start)
    return json.loads(text[start:end])


def test_snapshot_substitution_is_rejected_at_read_and_resume_boundaries(tmp_path):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args("first-case"), session_id="first-owner")
    harness.model(_open_args("second-case"), session_id="second-owner")
    harness.model({"operation": "pause", "reason": "first hold"}, session_id="first-owner")
    harness.model({"operation": "pause", "reason": "second hold"}, session_id="second-owner")

    first_case_path = tmp_path / "cases" / "first-case" / "case.md"
    second_case_path = tmp_path / "cases" / "second-case" / "case.md"
    first_events_path = tmp_path / "cases" / "first-case" / "events.jsonl"
    second_events_path = tmp_path / "cases" / "second-case" / "events.jsonl"
    second_case_before = second_case_path.read_bytes()
    second_events_before = second_events_path.read_bytes()
    first_case_path.write_bytes(second_case_before)
    substituted_first = first_case_path.read_bytes()
    first_events_before = first_events_path.read_bytes()

    diagnostic = harness.replay("first-case")
    assert diagnostic["case_snapshot"]["valid"] is False
    assert diagnostic["usable"] is False
    with pytest.raises(StoreError, match="identity"):
        harness.store.get_case("first-case")
    assert "first-case" in harness.store.case_errors()
    with pytest.raises(StoreError, match="unreadable case associated"):
        harness.store.get_active_case("first-owner")
    with pytest.raises(HarnessError, match="identity"):
        harness.model(
            {"operation": "resume", "case_id": "first-case"},
            session_id="new-owner",
        )

    assert first_case_path.read_bytes() == substituted_first
    assert first_events_path.read_bytes() == first_events_before
    assert second_case_path.read_bytes() == second_case_before
    assert second_events_path.read_bytes() == second_events_before
    substituted_data = _case_data_from_markdown(first_case_path)
    assert substituted_data["case_id"] == "second-case"
    assert substituted_data["status"] == "paused"
    assert substituted_data["session_id"] == "second-owner"
    assert harness.store.get_case("second-case")["status"] == "paused"
    assert harness.store.get_active_case("new-owner") is None


def test_recovery_journal_cannot_cross_case_storage_slots(tmp_path, monkeypatch):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args("case-a"), session_id="owner-a")
    harness.model(_open_args("case-b"), session_id="owner-b")
    store = harness.store
    case_b = store.get_case("case-b")
    monkeypatch.setattr(store, "_apply_transaction_journal", lambda _journal: None)
    store.transition_case(
        case_b,
        event_type="recovery_fixture",
        event_payload={"marker": "case-b"},
    )
    pending_b = tmp_path / "cases" / "case-b" / "archive" / ".pending-transition.json"
    pending_a = tmp_path / "cases" / "case-a" / "archive" / ".pending-transition.json"
    pending_b.rename(pending_a)
    case_b_path = tmp_path / "cases" / "case-b" / "case.md"
    events_b_path = tmp_path / "cases" / "case-b" / "events.jsonl"
    case_b_before = case_b_path.read_bytes()
    events_b_before = events_b_path.read_bytes()

    with pytest.raises(StoreError, match="storage slot"):
        CaseStore(tmp_path)

    assert case_b_path.read_bytes() == case_b_before
    assert events_b_path.read_bytes() == events_b_before
    assert pending_a.exists()
