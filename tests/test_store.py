# AI-assisted contribution; maintained by Epistemic Harness contributors.
"""Persistence contracts for the epistemic case store."""

from __future__ import annotations

import json
import multiprocessing

import pytest

from epistemic_harness.store import CaseStore, StoreError


def _concurrent_transition_worker(root, case, barrier, results, worker_id):
    store = CaseStore(root)
    barrier.wait(timeout=3)
    try:
        transition = store.transition_case(
            case,
            event_type="cross_process_transition",
            event_payload={"worker_id": worker_id},
        )
        results.put(("ok", transition["event"]["event_id"]))
    except Exception as exc:
        results.put(("error", str(exc)))


def _model() -> dict:
    return {
        "decision": "Choose the more diagnostic design",
        "stopping_condition": "A design is selected or the evidence is unresolved",
        "state_grounding": [
            {"id": "S1", "text": "Two candidate designs remain", "evidence_event_ids": []}
        ],
        "mechanisms": [
            {"id": "M1", "text": "The diagnostic design separates encoding from grounding", "evidence_event_ids": []}
        ],
        "alternatives": [
            {"id": "A1", "text": "The designs do not separate the mechanisms"}
        ],
        "top_unknown": "Whether the manipulation isolates encoding",
        "current_plan": [
            {"id": "P1", "action": "Inspect the manipulation specification", "depends_on": ["S1"]}
        ],
    }


def test_open_case_persists_model_and_hash_chained_timeline(tmp_path):
    store = CaseStore(tmp_path)

    opened = store.open_case(
        session_id="session-1",
        case_id="design-choice",
        model=_model(),
    )

    assert opened["case_id"] == "design-choice"
    assert opened["status"] == "active"
    assert opened["version"] == 1

    case_dir = tmp_path / "cases" / "design-choice"
    assert (case_dir / "case.md").exists()
    assert (case_dir / "events.jsonl").exists()
    assert (case_dir / "archive").is_dir()
    assert "Choose the more diagnostic design" in (case_dir / "case.md").read_text()

    events = [json.loads(line) for line in (case_dir / "events.jsonl").read_text().splitlines()]
    assert [event["type"] for event in events] == ["case_opened"]
    assert events[0]["prev_hash"] == "0" * 64
    assert len(events[0]["hash"]) == 64
    assert store.verify_event_chain("design-choice") == {"valid": True, "events": 1}

    fresh_store = CaseStore(tmp_path)
    shown = fresh_store.get_active_case("session-1")
    assert shown is not None
    assert shown["case_id"] == "design-choice"
    assert shown["model"]["top_unknown"] == "Whether the manipulation isolates encoding"


def test_case_portfolio_summaries_are_compact_filterable_and_read_only(tmp_path):
    store = CaseStore(tmp_path)
    store.open_case(
        session_id="session-old",
        case_id="case-a",
        model=_model(),
    )
    migrated = store.migrate_active_case(
        old_session_id="session-old",
        new_session_id="session-new",
    )
    assert migrated is not None
    case_a = migrated["case"]
    case_a["status"] = "paused"
    store.transition_case(
        case_a,
        event_type="case_paused",
        event_payload={"reason": "Review later"},
    )
    store.open_case(
        session_id="session-b",
        case_id="case-b",
        model={**_model(), "decision": "Inspect the second portfolio item"},
    )

    before = {
        path: path.read_bytes()
        for path in sorted((tmp_path / "cases").glob("*/*"))
        if path.is_file()
    }
    summaries = store.list_case_summaries()

    assert [item["case_id"] for item in summaries] == ["case-b", "case-a"]
    first = summaries[0]
    assert first == {
        "case_id": "case-b",
        "status": "active",
        "created_at": first["created_at"],
        "updated_at": first["updated_at"],
        "summary": "Inspect the second portfolio item",
        "session_id": "session-b",
        "session_ids": ["session-b"],
        "event_count": 1,
        "pending_probe_status": None,
        "epistemically_stale": False,
    }
    case_a_summary = summaries[1]
    assert case_a_summary["status"] == "paused"
    assert case_a_summary["session_id"] == "session-new"
    assert case_a_summary["session_ids"] == ["session-old", "session-new"]
    assert case_a_summary["event_count"] == 3
    assert store.list_case_summaries(status="paused") == [case_a_summary]
    assert store.list_case_summaries(session_id="session-old") == [case_a_summary]
    assert {
        path: path.read_bytes()
        for path in sorted((tmp_path / "cases").glob("*/*"))
        if path.is_file()
    } == before

    with pytest.raises(StoreError, match="status filter"):
        store.list_case_summaries(status="orphaned")
    with pytest.raises(StoreError, match="cannot be empty"):
        store.list_case_summaries(session_id="  ")


def test_timeline_escapes_unicode_separators_and_reads_legacy_raw_values(tmp_path):
    store = CaseStore(tmp_path)
    store.open_case(
        session_id="session-1",
        case_id="unicode-separators",
        model=_model(),
    )
    observation = "Prompt \u2028 Randomization \u2029 tail"
    store.append_event(
        "unicode-separators",
        event_type="probe_observed",
        session_id="session-1",
        payload={"observation": observation},
    )

    timeline = tmp_path / "cases" / "unicode-separators" / "events.jsonl"
    serialized = timeline.read_text(encoding="utf-8")
    assert "\\u2028" in serialized
    assert "\\u2029" in serialized
    assert "\u2028" not in serialized
    assert "\u2029" not in serialized

    # Version 0.10.2 wrote these valid JSON characters literally. Confirm the
    # new reader treats only LF as a JSONL record boundary and preserves the
    # existing hash chain.
    timeline.write_text(
        serialized.replace("\\u2028", "\u2028").replace("\\u2029", "\u2029"),
        encoding="utf-8",
    )
    fresh_store = CaseStore(tmp_path)
    events = fresh_store.list_events("unicode-separators")
    assert events[-1]["payload"]["observation"] == observation
    assert fresh_store.verify_event_chain("unicode-separators") == {
        "valid": True,
        "events": 2,
    }


def test_cross_process_transitions_use_file_lock_and_optimistic_token(tmp_path):
    store = CaseStore(tmp_path)
    case = store.open_case(
        session_id="session-1",
        case_id="case-a",
        model=_model(),
    )

    context = multiprocessing.get_context("fork")
    barrier = context.Barrier(2)
    results = context.Queue()
    processes = [
        context.Process(
            target=_concurrent_transition_worker,
            args=(tmp_path, case, barrier, results, worker_id),
        )
        for worker_id in (1, 2)
    ]
    for process in processes:
        process.start()
    for process in processes:
        process.join(timeout=5)
        assert process.exitcode == 0

    outcomes = sorted(results.get(timeout=1)[0] for _ in processes)
    assert outcomes == ["error", "ok"]
    events = store.list_events("case-a")
    assert [event["type"] for event in events].count("cross_process_transition") == 1
    assert store.verify_event_chain("case-a") == {"valid": True, "events": 2}
