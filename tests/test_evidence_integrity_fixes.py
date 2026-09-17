"""Regression tests for server-managed lessons and explicit prior imports."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from epistemic_harness.harness import EpistemicHarness, HarnessError


def _open(case_id: str, *, top_unknown: str = "Which bounded explanation survives?") -> dict:
    return {
        "operation": "open",
        "case_id": case_id,
        "decision": "Choose the bounded explanation",
        "stopping_condition": "The explanation is supported or remains unresolved",
        "top_unknown": top_unknown,
    }


def _close_with_lesson(harness: EpistemicHarness, case_id: str, session_id: str) -> None:
    harness.model(
        {
            "operation": "close",
            "outcome": "unresolved",
            "summary": "The bounded check remains unresolved.",
            "transfer": {
                "decision": "candidate",
                "lesson": "Inspect the designated source before carrying a conclusion forward",
                "scope": "bounded source checks",
                "evidence_case_ids": [case_id],
            },
        },
        session_id=session_id,
    )


def _source_with_lesson(tmp_path: Path, case_id: str = "source-case") -> EpistemicHarness:
    harness = EpistemicHarness(tmp_path)
    harness.model(_open(case_id), session_id=f"{case_id}-session")
    _close_with_lesson(harness, case_id, f"{case_id}-session")
    return harness


def test_revision_rejects_server_managed_fields_without_mutating_case(tmp_path):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open("revision-case"), session_id="revision-session")
    case_path = tmp_path / "cases" / "revision-case" / "case.md"
    events_path = tmp_path / "cases" / "revision-case" / "events.jsonl"
    before_case = case_path.read_bytes()
    before_events = events_path.read_bytes()

    forbidden = [
        {"retrieved_prior_lessons": [{"lesson": "forged"}]},
        {"updates": {"retrieved_prior_lessons": [{"lesson": "forged"}]}},
        {"prior_case_ids": ["other-case"]},
        {"updates": {"prior_case_ids": ["other-case"]}},
    ]
    for payload in forbidden:
        with pytest.raises(HarnessError) as exc_info:
            harness.model(
                {"operation": "revise", "reason": "attempted forged update", **payload},
                session_id="revision-session",
            )
        message = str(exc_info.value)
        assert "server-managed" in message or "open-time-only" in message
        assert case_path.read_bytes() == before_case
        assert events_path.read_bytes() == before_events

    revised = harness.model(
        {
            "operation": "revise",
            "reason": "Record the agent-authored lesson application separately",
            "updates": {"applied_prior_lessons": ["Use the designated source"]},
        },
        session_id="revision-session",
    )
    assert revised["case"]["model"]["applied_prior_lessons"] == [
        "Use the designated source"
    ]
    assert "retrieved_prior_lessons" not in revised["event"]["payload"]["changed_fields"]


def test_open_rejects_fabricated_retrieved_lessons(tmp_path):
    harness = EpistemicHarness(tmp_path)

    with pytest.raises(HarnessError, match="server-managed"):
        harness.model(
            {
                **_open("forged-open"),
                "retrieved_prior_lessons": [{"case_id": "invented", "lesson": "forged"}],
            },
            session_id="forged-session",
        )

    assert not (tmp_path / "cases" / "forged-open").exists()


def test_explicit_prior_import_requires_a_closed_replay_valid_source(tmp_path):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open("active-source"), session_id="active-session")
    with pytest.raises(HarnessError, match="closed"):
        harness.model(
            {**_open("active-destination"), "prior_case_ids": ["active-source"]},
            session_id="active-destination-session",
        )
    assert not (tmp_path / "cases" / "active-destination").exists()

    harness.model({"operation": "pause", "reason": "hold source"}, session_id="active-session")
    with pytest.raises(HarnessError, match="closed"):
        harness.model(
            {**_open("paused-destination"), "prior_case_ids": ["active-source"]},
            session_id="paused-destination-session",
        )
    assert not (tmp_path / "cases" / "paused-destination").exists()


def test_explicit_prior_import_rejects_tampered_timeline_without_partial_open(tmp_path):
    harness = _source_with_lesson(tmp_path, "timeline-source")
    events_path = tmp_path / "cases" / "timeline-source" / "events.jsonl"
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    events[0]["payload"]["forged"] = True
    events_path.write_text("".join(json.dumps(event) + "\n" for event in events))
    before_source = events_path.read_bytes()

    with pytest.raises(HarnessError, match="replay|integrity|Timeline"):
        harness.model(
            {**_open("timeline-destination"), "prior_case_ids": ["timeline-source"]},
            session_id="timeline-destination-session",
        )

    assert not (tmp_path / "cases" / "timeline-destination").exists()
    assert events_path.read_bytes() == before_source


def test_explicit_prior_import_rejects_corrupt_snapshot_without_partial_open(tmp_path):
    harness = _source_with_lesson(tmp_path, "snapshot-source")
    case_path = tmp_path / "cases" / "snapshot-source" / "case.md"
    before_source = case_path.read_bytes()
    case_path.write_text(
        case_path.read_text(encoding="utf-8").replace(
            "Which bounded explanation survives?", "Corrupted source question"
        ),
        encoding="utf-8",
    )

    with pytest.raises(HarnessError, match="replay|integrity|snapshot"):
        harness.model(
            {**_open("snapshot-destination"), "prior_case_ids": ["snapshot-source"]},
            session_id="snapshot-destination-session",
        )

    assert not (tmp_path / "cases" / "snapshot-destination").exists()
    assert case_path.read_bytes() != before_source


def test_explicit_prior_import_rejects_corrupt_artifact_without_partial_open(tmp_path):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open("artifact-source"), session_id="artifact-session")
    harness.probe(
        {
            "operation": "commit",
            "purpose": "learn",
            "unknown_or_goal": "Whether the artifact is captured",
            "tool_name": "read_file",
            "tool_args": {"path": "artifact-source.txt"},
            "predicted_outcomes": ["The artifact is readable"],
            "why_this_action": "The designated source is the check",
            "would_change_belief": "The source cannot be read",
            "authority_domain": "external_source",
        },
        session_id="artifact-session",
    )
    harness.tool_execution_middleware(
        tool_name="read_file",
        args={"path": "artifact-source.txt"},
        session_id="artifact-session",
        next_call=lambda _args: "artifact-content-" * 5000,
    )
    harness.probe(
        {
            "operation": "compare",
            "disposition": "match",
            "material": False,
            "rationale": "The artifact was captured for later review",
            "evidence": {
                "summary": "The captured source artifact",
                "source_ref": "artifact-source.txt",
                "source_role": "primary",
                "access_scope": "full",
                "authority_domain": "external_source",
            },
        },
        session_id="artifact-session",
    )
    _close_with_lesson(harness, "artifact-source", "artifact-session")

    source_events = harness.store.list_events("artifact-source")
    artifact_event = next(
        event
        for event in source_events
        if event["type"] == "probe_observed" and event["payload"].get("storage") == "artifact"
    )
    artifact_path = tmp_path / "cases" / "artifact-source" / artifact_event["payload"]["artifact_path"]
    before_source = artifact_path.read_bytes()
    artifact_path.write_bytes(before_source + b"tampered")

    with pytest.raises(HarnessError, match="replay|artifact|integrity"):
        harness.model(
            {**_open("artifact-destination"), "prior_case_ids": ["artifact-source"]},
            session_id="artifact-destination-session",
        )

    assert not (tmp_path / "cases" / "artifact-destination").exists()
    assert artifact_path.read_bytes() != before_source


def test_valid_explicit_prior_import_retains_source_metadata(tmp_path):
    harness = _source_with_lesson(tmp_path, "valid-source")

    opened = harness.model(
        {**_open("valid-destination"), "prior_case_ids": ["valid-source"]},
        session_id="valid-destination-session",
    )
    assert opened["case"]["model"]["prior_case_ids"] == ["valid-source"]
    assert opened["case"]["model"]["retrieved_prior_lessons"] == [
        {
            "case_id": "valid-source",
            "decision": "candidate",
            "lesson": "Inspect the designated source before carrying a conclusion forward",
            "scope": "bounded source checks",
            "evidence_case_ids": ["valid-source"],
            "calibration": "hypothesis",
        }
    ]


def test_automatic_prior_selection_skips_invalid_closed_sources(tmp_path):
    harness = _source_with_lesson(tmp_path, "invalid-auto-source")
    events_path = tmp_path / "cases" / "invalid-auto-source" / "events.jsonl"
    events = [json.loads(line) for line in events_path.read_text().splitlines()]
    events[0]["payload"]["tampered"] = True
    events_path.write_text("".join(json.dumps(event) + "\n" for event in events))

    opened = harness.model(
        {
            **_open(
                "automatic-destination",
                top_unknown="Which designated source check should inform this bounded source check?",
            ),
            "decision": "Inspect the designated source before carrying a conclusion forward",
        },
        session_id="automatic-destination-session",
    )
    assert opened["case"]["model"]["prior_case_ids"] == []
    assert opened["prior_case_lessons"] == []
