# AI-assisted contribution; maintained by Epistemic Harness contributors.
"""Behavior contracts for the bounded epistemic harness."""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor

import pytest

import epistemic_harness.store as store_module
from epistemic_harness.harness import EpistemicHarness, HarnessError


def _open_args() -> dict:
    return {
        "operation": "open",
        "case_id": "research-case",
        "decision": "Determine which account survives the evidence",
        "stopping_condition": "One account is selected or the evidence remains unresolved",
        "state_grounding": [
            {"id": "S1", "text": "Two accounts remain", "evidence_event_ids": []}
        ],
        "mechanisms": [
            {"id": "M1", "text": "Account one predicts an observable difference", "evidence_event_ids": []}
        ],
        "alternatives": [{"id": "A1", "text": "The difference is explained by task structure"}],
        "top_unknown": "Which account predicts the observed process",
        "current_plan": [
            {"id": "P1", "action": "Inspect the process evidence", "depends_on": ["M1"]}
        ],
    }


def _close_args(transfer: dict) -> dict:
    return {
        "operation": "close",
        "outcome": "unresolved",
        "summary": "The bounded case reached its stopping condition",
        "transfer": transfer,
    }


def _external_evidence(access_scope: str = "full") -> dict:
    evidence = {
        "summary": "The inspected source bears on the bounded claim",
        "source_ref": "https://example.test/source",
        "source_role": "primary",
        "access_scope": access_scope,
        "authority_domain": "external_source",
    }
    if access_scope in {"snippet", "abstract", "partial"}:
        evidence["limitation"] = f"Only {access_scope} access was available"
    return evidence


def test_store_requires_interprocess_lock_support(tmp_path, monkeypatch):
    monkeypatch.setattr(store_module, "fcntl", None)
    with pytest.raises(store_module.StoreError, match="fcntl"):
        EpistemicHarness(tmp_path)


def test_first_use_fsyncs_parent_for_new_state_directories(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(
        store_module.CaseStore,
        "_fsync_dir",
        lambda *args: calls.append(args[-1]),
    )

    store_module.CaseStore(tmp_path / "state-root")

    assert tmp_path in calls
    assert tmp_path / "state-root" in calls


def test_directory_sync_failure_aborts_case_creation(tmp_path, monkeypatch):
    harness = EpistemicHarness(tmp_path)

    def fail_sync(_path):
        raise store_module.StoreError("simulated directory durability failure")

    monkeypatch.setattr(store_module.CaseStore, "_fsync_dir", staticmethod(fail_sync))
    with pytest.raises(HarnessError, match="durability failure"):
        harness.model(_open_args(), session_id="session-1")


def test_interrupted_first_open_does_not_permanently_poison_case_id(tmp_path):
    incomplete = tmp_path / "cases" / "research-case" / "archive"
    incomplete.mkdir(parents=True)

    harness = EpistemicHarness(tmp_path)
    opened = harness.model(_open_args(), session_id="session-1")

    assert opened["case"]["case_id"] == "research-case"
    assert (tmp_path / "cases" / "research-case" / "case.md").exists()


def test_interrupted_first_open_discards_orphaned_atomic_write_temp(tmp_path):
    archive = tmp_path / "cases" / "research-case" / "archive"
    archive.mkdir(parents=True)
    (archive / "..pending-transition.json.crash.tmp").write_text("partial")

    harness = EpistemicHarness(tmp_path)
    opened = harness.model(_open_args(), session_id="session-1")

    assert opened["case"]["case_id"] == "research-case"
    assert not list(archive.glob("..pending-transition.json.*.tmp"))


def test_list_returns_bounded_cross_session_portfolio_without_changing_focus(tmp_path):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="session-1")
    second = _open_args()
    second["case_id"] = "second-case"
    second["decision"] = "Review a second bounded question"
    harness.model(second, session_id="session-2")
    harness.model(
        {"operation": "pause", "reason": "Leave the first case for later review"},
        session_id="session-1",
    )

    portfolio = harness.model({"operation": "list"}, session_id="session-2")

    assert portfolio["returned"] == 2
    assert portfolio["total_matching"] == 2
    assert portfolio["filters"] == {
        "status": None,
        "session_id": None,
        "limit": 50,
    }
    assert [item["case_id"] for item in portfolio["cases"]] == [
        "research-case",
        "second-case",
    ]
    assert harness.store.get_active_case("session-2")["case_id"] == "second-case"

    active = harness.model(
        {"operation": "list", "status_filter": "active"},
        session_id="session-2",
    )
    assert [item["case_id"] for item in active["cases"]] == ["second-case"]
    prior_session = harness.model(
        {"operation": "list", "session_id_filter": "session-1"},
        session_id="session-2",
    )
    assert [item["case_id"] for item in prior_session["cases"]] == ["research-case"]
    limited = harness.model(
        {"operation": "list", "limit": 1},
        session_id="session-2",
    )
    assert limited["returned"] == 1
    assert limited["total_matching"] == 2

    for bad_limit in (0, 201, True, 1.5):
        with pytest.raises(HarnessError, match="list limit"):
            harness.model(
                {"operation": "list", "limit": bad_limit},
                session_id="session-2",
            )
    with pytest.raises(HarnessError, match="status_filter"):
        harness.model(
            {"operation": "list", "status_filter": "orphaned"},
            session_id="session-2",
        )


def test_model_revision_is_versioned_and_semantic_noop_is_rejected(tmp_path):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="session-1")

    revised = harness.model(
        {
            "operation": "revise",
            "updates": {"top_unknown": "Whether task structure explains the process"},
            "reason": "The first account did not distinguish the observed process",
            "addresses_event_ids": [],
        },
        session_id="session-1",
    )

    assert revised["case"]["version"] == 2
    assert revised["case"]["model"]["top_unknown"] == "Whether task structure explains the process"
    archive = tmp_path / "cases" / "research-case" / "archive"
    assert sorted(path.name for path in archive.glob("case-v*.md")) == [
        "case-v0001.md",
        "case-v0002.md",
    ]
    events = [
        json.loads(line)
        for line in (tmp_path / "cases" / "research-case" / "events.jsonl").read_text().splitlines()
    ]
    assert [event["type"] for event in events] == ["case_opened", "model_revised"]

    with pytest.raises(HarnessError, match="no material change"):
        harness.model(
            {
                "operation": "revise",
                "updates": {"top_unknown": "Whether task structure explains the process"},
                "reason": "Rewrite only",
                "addresses_event_ids": [],
            },
            session_id="session-1",
        )


def test_stale_revision_cannot_erase_a_concurrent_runtime_transition(tmp_path, monkeypatch):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="session-1")
    revision_has_read = threading.Event()
    release_revision = threading.Event()
    original_update = harness.store.update_case

    def delayed_update(*args, **kwargs):
        revision_has_read.set()
        assert release_revision.wait(timeout=5)
        return original_update(*args, **kwargs)

    monkeypatch.setattr(harness.store, "update_case", delayed_update)
    revision_errors: list[BaseException] = []

    def revise() -> None:
        try:
            harness.model(
                {
                    "operation": "revise",
                    "updates": {"top_unknown": "A stale revision"},
                    "reason": "Exercise the compare-and-swap boundary",
                    "addresses_event_ids": [],
                },
                session_id="session-1",
            )
        except BaseException as exc:
            revision_errors.append(exc)

    worker = threading.Thread(target=revise)
    worker.start()
    assert revision_has_read.wait(timeout=5)
    harness.probe(
        {
            "operation": "commit",
            "purpose": "learn",
            "unknown_or_goal": "What changed while revision was pending",
            "tool_name": "read_file",
            "tool_args": {"path": "evidence.txt"},
            "predicted_outcomes": [{"outcome": "found", "meaning": "Evidence exists"}],
            "why_this_action": "Preserve the concurrent runtime transition",
            "would_change_belief": "Evidence incompatible with the predicted outcome",
            "authority_domain": "external_source",
        },
        session_id="session-1",
    )
    release_revision.set()
    worker.join(timeout=5)

    assert not worker.is_alive()
    assert revision_errors
    assert "concurrent case transition" in str(revision_errors[0])
    case = harness.store.get_case("research-case")
    assert case["version"] == 1
    assert case["pending_probe"]["status"] == "committed"
    assert [event["type"] for event in harness.store.list_events("research-case")] == [
        "case_opened",
        "probe_committed",
    ]


def test_initial_evidence_becomes_typed_timeline_events_and_claim_citations(tmp_path):
    harness = EpistemicHarness(tmp_path)
    args = _open_args()
    args["initial_evidence"] = [
        {
            "id": "I1",
            "summary": "the user says the choice depends on preserving interpretability",
            "source_ref": "current-user-message",
            "authority_domain": "user_preference",
        }
    ]
    args["state_grounding"][0]["evidence_event_ids"] = ["I1"]
    args["state_grounding"][0]["authority_domain"] = "user_preference"

    opened = harness.model(args, session_id="session-1")

    claim = opened["case"]["model"]["state_grounding"][0]
    assert claim["evidence_event_ids"] == ["E000001"]
    assert claim["epistemic_status"] == "grounded"
    assert opened["case"]["model"]["mechanisms"][0]["epistemic_status"] == "hypothesis"
    events = harness.store.list_events("research-case")
    assert [event["type"] for event in events] == ["initial_observation", "case_opened"]
    assert events[0]["payload"]["authority_domain"] == "user_preference"
    assert events[0]["payload"]["source_ref"] == "current-user-message"
    assert opened["replay"]["usable"] is True


def test_missing_session_identity_does_not_tax_ordinary_tool_use(tmp_path):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="session-1")
    calls = []

    result = harness.tool_execution_middleware(
        tool_name="read_file",
        args={"path": "paper.txt"},
        session_id="",
        next_call=lambda args: calls.append(args) or "ordinary result",
    )

    assert result == "ordinary result"
    assert calls == [{"path": "paper.txt"}]


def test_delegate_task_cannot_be_committed_as_one_bounded_probe(tmp_path):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="session-1")

    with pytest.raises(HarnessError, match="delegation"):
        harness.probe(
            {
                "operation": "commit",
                "purpose": "learn",
                "unknown_or_goal": "What a delegated researcher finds",
                "tool_name": "delegate_task",
                "tool_args": {"goal": "Research the question and use tools as needed"},
                "predicted_outcomes": [
                    {"outcome": "report", "meaning": "Update the model from the report"}
                ],
                "why_this_action": "Delegation would otherwise hide compound actions",
                "would_change_belief": "Evidence incompatible with the predicted outcome",
                "authority_domain": "external_source",
            },
            session_id="session-1",
        )


def test_active_case_forwards_uncommitted_tool_and_captures_committed_result(tmp_path):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="session-1")
    calls: list[dict] = []

    def tool(args):
        calls.append(args)
        return json.dumps({"finding": "task structure differs"})

    ordinary = harness.tool_execution_middleware(
        tool_name="read_file",
        args={"path": "paper.txt"},
        session_id="session-1",
        next_call=tool,
    )
    assert json.loads(ordinary) == {"finding": "task structure differs"}
    assert calls == [{"path": "paper.txt"}]
    calls.clear()

    committed = harness.probe(
        {
            "operation": "commit",
            "purpose": "learn",
            "unknown_or_goal": "Whether task structure explains the process",
            "tool_name": "read_file",
            "tool_args": {"path": "paper.txt"},
            "predicted_outcomes": [
                {
                    "outcome": "task structure differs",
                    "meaning": "Increase support for the task-structure alternative",
                }
            ],
            "why_this_action": "The source directly describes the task structure",
            "would_change_belief": "Evidence incompatible with the predicted outcome",
            "authority_domain": "external_source",
        },
        session_id="session-1",
    )
    assert committed["probe"]["status"] == "committed"

    result = harness.tool_execution_middleware(
        tool_name="read_file",
        args={"path": "paper.txt"},
        session_id="session-1",
        next_call=tool,
    )
    assert json.loads(result) == {"finding": "task structure differs"}
    assert calls == [{"path": "paper.txt"}]

    active = harness.store.get_active_case("session-1")
    assert active["pending_probe"]["status"] == "observed"
    observation_id = active["pending_probe"]["observation_event_id"]
    assert observation_id.startswith("E")

    replayed = harness.replay("research-case")
    assert replayed["usable"] is False
    assert replayed["reason"] == "probe result awaits comparison"


def test_minimal_case_open_direct_revision_and_simple_close(tmp_path):
    harness = EpistemicHarness(tmp_path)
    opened = harness.model(
        {
            "operation": "open",
            "case_id": "minimal-case",
            "top_unknown": "Which explanation best fits the evidence?",
        },
        session_id="session-1",
    )

    model = opened["case"]["model"]
    assert model["decision"] == "none"
    assert model["state_grounding"] == []
    assert model["mechanisms"] == []
    assert model["alternatives"] == []
    assert model["current_plan"] == []

    revised = harness.model(
        {
            "operation": "revise",
            "top_unknown": "Which explanation survives the available evidence?",
            "reason": "Refine the question after initial inspection",
        },
        session_id="session-1",
    )
    assert revised["case"]["version"] == 2
    assert revised["case"]["model"]["top_unknown"].endswith("available evidence?")

    closed = harness.model(
        {
            "operation": "close",
            "outcome": "unresolved",
            "summary": "The evidence does not yet distinguish the explanations.",
            "transfer": {"decision": "none"},
        },
        session_id="session-1",
    )
    assert closed["case"]["status"] == "closed"
    assert closed["case"]["closure"]["transfer"] == {"decision": "none"}


def test_probe_commit_requires_an_explicit_belief_changer(tmp_path):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="session-1")

    with pytest.raises(HarnessError, match="would_change_belief"):
        harness.probe(
            {
                "operation": "commit",
                "purpose": "learn",
                "unknown_or_goal": "Whether the source supports the mechanism",
                "tool_name": "read_file",
                "tool_args": {"path": "evidence.txt"},
                "predicted_outcomes": ["The mechanism appears"],
                "why_this_action": "The source is directly relevant",
                "authority_domain": "external_source",
            },
            session_id="session-1",
        )


def test_resolved_closure_requires_evidence_linked_grounded_claim(tmp_path):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="session-1")

    with pytest.raises(HarnessError, match="evidence-linked grounded claim"):
        harness.model(
            {
                "operation": "close",
                "outcome": "resolved",
                "summary": "The account is confirmed.",
                "transfer": {"decision": "none"},
            },
            session_id="session-1",
        )


def test_resolved_closure_records_supported_calibration_for_direct_evidence(tmp_path):
    harness = EpistemicHarness(tmp_path)
    args = _open_args()
    args["initial_evidence"] = [
        {
            "id": "I1",
            "summary": "the user states which workflow they prefer",
            "source_ref": "current-user-message",
            "authority_domain": "user_preference",
        }
    ]
    args["state_grounding"] = [
        {
            "id": "S1",
            "text": "the user prefers the bounded workflow",
            "evidence_event_ids": ["I1"],
            "authority_domain": "user_preference",
        }
    ]
    harness.model(args, session_id="session-1")

    closed = harness.model(
        {
            "operation": "close",
            "outcome": "resolved",
            "summary": "The preference question is answered.",
            "transfer": {"decision": "none"},
        },
        session_id="session-1",
    )

    calibration = closed["case"]["closure"]["calibration"]
    # S1 is grounded on direct user testimony, but M1 (a mechanism claim) is
    # still an ungrounded hypothesis, so the case cannot read "supported".
    assert calibration["label"] == "mixed"
    assert calibration["grounded_claim_ids"] == ["S1"]
    assert calibration["evidence_event_ids"] == ["E000001"]


def test_search_only_grounding_is_calibrated_as_provisional(tmp_path):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="session-1")
    harness.probe(
        {
            "operation": "commit",
            "purpose": "learn",
            "unknown_or_goal": "Whether literature mentions the proposed mechanism",
            "tool_name": "web_search",
            "tool_args": {"query": "proposed mechanism"},
            "predicted_outcomes": ["A relevant result appears"],
            "would_change_belief": "No relevant result appears across the bounded search",
            "why_this_action": "The search locates candidate primary sources",
            "authority_domain": "external_source",
        },
        session_id="session-1",
    )
    harness.tool_execution_middleware(
        tool_name="web_search",
        args={"query": "proposed mechanism"},
        session_id="session-1",
        next_call=lambda _args: "A relevant result snippet",
    )
    observation_event_id = harness.store.get_active_case("session-1")[
        "pending_probe"
    ]["observation_event_id"]
    harness.probe(
        {
            "operation": "compare",
            "disposition": "match",
            "material": False,
            "rationale": "The search returned the predicted lead",
            "affected_claim_ids": ["M1"],
            "evidence": _external_evidence("snippet"),
        },
        session_id="session-1",
    )
    harness.model(
        {
            "operation": "revise",
            "reason": "Link the search lead without overstating its quality",
            "mechanisms": [
                {
                    "id": "M1",
                    "text": "The literature may contain the proposed mechanism",
                    "evidence_event_ids": [observation_event_id],
                    "authority_domain": "external_source",
                    "epistemic_status": "grounded",
                }
            ],
        },
        session_id="session-1",
    )

    closed = harness.model(
        {
            "operation": "close",
            "outcome": "resolved",
            "summary": "A literature lead was located.",
            "transfer": {"decision": "none"},
        },
        session_id="session-1",
    )
    calibration = closed["case"]["closure"]["calibration"]
    # M1 is grounded only on a snippet (per-claim provisional), and S1 in
    # state_grounding remains an ungrounded hypothesis, so the case overall
    # reads "mixed" — never stronger than its weakest asserted claim.
    assert calibration["label"] == "mixed"
    assert calibration["lead_only_event_ids"] == [observation_event_id]
    assert {
        item["claim_id"]: item["label"] for item in calibration["claims"]
    }["M1"] == "provisional"


def test_committed_probe_does_not_block_delegation_and_can_be_discarded(tmp_path):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="session-1")
    harness.probe(
        {
            "operation": "commit",
            "purpose": "learn",
            "unknown_or_goal": "What a local command would reveal",
            "tool_name": "terminal",
            "tool_args": {"command": "inspect"},
            "predicted_outcomes": ["The command returns useful evidence"],
            "why_this_action": "The command could resolve one uncertainty",
            "would_change_belief": "Evidence incompatible with the predicted outcome",
            "authority_domain": "system_state",
        },
        session_id="session-1",
    )

    delegated = harness.tool_execution_middleware(
        tool_name="delegate_task",
        args={"goal": "Research one focused question"},
        session_id="session-1",
        next_call=lambda args: {"accepted": args["goal"]},
    )
    assert delegated == {"accepted": "Research one focused question"}
    assert harness.store.get_active_case("session-1")["pending_probe"]["status"] == "committed"

    discarded = harness.probe(
        {
            "operation": "discard",
            "reason": "The planned command is no longer the right next observation",
        },
        session_id="session-1",
    )
    assert discarded["case"]["pending_probe"] is None
    assert discarded["replay"]["usable"] is True
    assert discarded["event"]["type"] == "probe_discarded"


def test_probe_argument_drift_is_recorded_instead_of_blocked(tmp_path):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="session-1")
    harness.probe(
        {
            "operation": "commit",
            "purpose": "learn",
            "unknown_or_goal": "Whether the source discusses collaboration",
            "tool_name": "web_search",
            "tool_args": {"query": "collaboration delegation"},
            "predicted_outcomes": [
                {"outcome": "Relevant work appears without exact-title matching"}
            ],
            "why_this_action": "A broad search can locate candidate sources",
            "would_change_belief": "Evidence incompatible with the predicted outcome",
            "authority_domain": "external_source",
        },
        session_id="session-1",
    )

    result = harness.tool_execution_middleware(
        tool_name="web_search",
        args={"query": "collaboration versus delegation experiments"},
        session_id="session-1",
        next_call=lambda _args: "candidate sources",
    )
    assert result == "candidate sources"
    probe = harness.store.get_active_case("session-1")["pending_probe"]
    assert probe["status"] == "observed"
    assert probe["execution_args_match_commitment"] is False

    compared = harness.probe(
        {
            "operation": "compare",
            "disposition": "match",
            "rationale": "The broader wording still tested the committed expectation",
            "argument_drift_reason": "The broader query was used to cover the same search question",
            "evidence": _external_evidence("snippet"),
        },
        session_id="session-1",
    )
    assert compared["case"]["pending_probe"] is None
    assert compared["replay"]["usable"] is True


def test_observed_probe_can_be_discarded_without_poisoning_replay(tmp_path):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="session-1")
    harness.probe(
        {
            "operation": "commit",
            "purpose": "learn",
            "unknown_or_goal": "Whether a source is accessible",
            "tool_name": "read_file",
            "tool_args": {"path": "candidate.txt"},
            "predicted_outcomes": ["The source is readable"],
            "why_this_action": "Accessibility determines the next research step",
            "would_change_belief": "Evidence incompatible with the predicted outcome",
            "authority_domain": "system_state",
        },
        session_id="session-1",
    )
    harness.tool_execution_middleware(
        tool_name="read_file",
        args={"path": "candidate.txt"},
        session_id="session-1",
        next_call=lambda _args: "tool configuration error",
    )

    discarded = harness.probe(
        {
            "operation": "discard",
            "reason": "The result reflects tool configuration, not the factual question",
        },
        session_id="session-1",
    )
    assert discarded["replay"]["usable"] is True
    assert discarded["replay"]["accounting"]["uncompared_observation_event_ids"] == []


def test_final_model_visible_result_rebinds_the_probe_observation(tmp_path):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="session-1")
    harness.probe(
        {
            "operation": "commit",
            "purpose": "learn",
            "unknown_or_goal": "What transformed evidence is exposed",
            "tool_name": "read_file",
            "tool_args": {"path": "evidence.txt"},
            "predicted_outcomes": [{"outcome": "visible", "meaning": "Compare it"}],
            "why_this_action": "The visible result is the evidence actually used",
            "would_change_belief": "Evidence incompatible with the predicted outcome",
            "authority_domain": "external_source",
        },
        session_id="session-1",
    )
    harness.tool_execution_middleware(
        tool_name="read_file",
        args={"path": "evidence.txt"},
        session_id="session-1",
        tool_call_id="call-evidence",
        next_call=lambda _args: "raw result A",
    )
    first_observation = harness.store.get_active_case("session-1")["pending_probe"][
        "observation_event_id"
    ]

    rebound = harness.record_tool_result_exposure(
        tool_name="read_file",
        args={"path": "evidence.txt"},
        result="model-visible result B",
        session_id="session-1",
        tool_call_id="call-evidence",
    )

    final_probe = rebound["case"]["pending_probe"]
    assert final_probe["observation_event_id"] != first_observation
    assert rebound["event"]["type"] == "probe_exposed"
    compared = harness.probe(
        {
            "operation": "compare",
            "disposition": "match",
            "material": False,
            "rationale": "The transformed result was the evidence actually seen",
            "affected_claim_ids": [],
            "evidence": _external_evidence(),
        },
        session_id="session-1",
    )
    assert compared["replay"]["usable"] is True


def test_unrelated_forwarded_result_does_not_rebind_legacy_observed_probe(tmp_path):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="session-1")
    harness.probe(
        {
            "operation": "commit",
            "purpose": "learn",
            "unknown_or_goal": "What evidence is exposed",
            "tool_name": "read_file",
            "tool_args": {"path": "evidence.txt"},
            "predicted_outcomes": [{"outcome": "visible", "meaning": "Compare it"}],
            "why_this_action": "The result must remain bound to its own tool call",
            "would_change_belief": "Evidence incompatible with the predicted outcome",
            "authority_domain": "external_source",
        },
        session_id="session-1",
    )
    harness.tool_execution_middleware(
        tool_name="read_file",
        args={"path": "evidence.txt"},
        session_id="session-1",
        next_call=lambda _args: "actual evidence",
    )
    before = harness.store.get_active_case("session-1")["pending_probe"]

    ordinary = harness.tool_execution_middleware(
        tool_name="read_file",
        args={"path": "another.txt"},
        session_id="session-1",
        tool_call_id="call-untracked",
        next_call=lambda _args: "untracked evidence",
    )
    assert ordinary == "untracked evidence"
    assert harness.record_tool_result_exposure(
        tool_name="read_file",
        args=None,
        result=ordinary,
        session_id="session-1",
        tool_call_id="call-untracked",
    ) is None

    after = harness.store.get_active_case("session-1")["pending_probe"]
    assert after["observation_event_id"] == before["observation_event_id"]
    assert after["result_sha256"] == before["result_sha256"]


def test_correlated_exposure_fails_closed_when_correlation_is_missing(tmp_path):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="session-1")
    harness.probe(
        {
            "operation": "commit",
            "purpose": "learn",
            "unknown_or_goal": "What evidence is exposed",
            "tool_name": "read_file",
            "tool_args": {"path": "evidence.txt"},
            "predicted_outcomes": [{"outcome": "visible", "meaning": "Compare it"}],
            "why_this_action": "Missing correlation must not weaken accounting",
            "would_change_belief": "Evidence incompatible with the predicted outcome",
            "authority_domain": "external_source",
        },
        session_id="session-1",
    )
    harness.tool_execution_middleware(
        tool_name="read_file",
        args={"path": "evidence.txt"},
        session_id="session-1",
        tool_call_id="call-evidence",
        next_call=lambda _args: "actual evidence",
    )

    with pytest.raises(HarnessError, match="missing its execution correlation"):
        harness.record_tool_result_exposure(
            tool_name="read_file",
            args={"path": "evidence.txt"},
            result="actual evidence",
            session_id="session-1",
        )


def test_material_mismatch_marks_case_stale_without_blocking_more_research(tmp_path):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="session-1")
    harness.probe(
        {
            "operation": "commit",
            "purpose": "learn",
            "unknown_or_goal": "Whether the source supports mechanism M1",
            "tool_name": "read_file",
            "tool_args": {"path": "evidence.txt"},
            "predicted_outcomes": [
                {"outcome": "M1 is supported", "meaning": "Retain mechanism M1"}
            ],
            "why_this_action": "The source contains the mechanism specification",
            "would_change_belief": "Evidence incompatible with the predicted outcome",
            "authority_domain": "external_source",
        },
        session_id="session-1",
    )
    harness.tool_execution_middleware(
        tool_name="read_file",
        args={"path": "evidence.txt"},
        session_id="session-1",
        next_call=lambda _args: "M1 is contradicted by the source",
    )

    compared = harness.probe(
        {
            "operation": "compare",
            "disposition": "mismatch",
            "material": True,
            "rationale": "The committed mechanism predicts the opposite pattern",
            "affected_claim_ids": ["M1"],
            "evidence": _external_evidence(),
        },
        session_id="session-1",
    )
    comparison_event_id = compared["event"]["event_id"]
    assert compared["case"]["stale"] is True
    assert compared["case"]["pending_probe"] is None
    assert compared["case"]["stale_claim_ids"] == ["M1", "P1"]
    assert harness.replay("research-case")["usable"] is False

    calls = []
    ordinary = harness.tool_execution_middleware(
        tool_name="web_search",
        args={"query": "more evidence"},
        session_id="session-1",
        next_call=lambda args: calls.append(args) or "more evidence",
    )
    assert ordinary == "more evidence"
    assert calls == [{"query": "more evidence"}]

    with pytest.raises(HarnessError, match="must explicitly address"):
        harness.model(
            {
                "operation": "revise",
                "updates": {"top_unknown": "A renamed unknown"},
                "reason": "Cosmetic rewrite",
                "addresses_event_ids": [],
            },
            session_id="session-1",
        )

    with pytest.raises(HarnessError, match="stale model items unchanged"):
        harness.model(
            {
                "operation": "revise",
                "updates": {"decision": "Rename the decision but retain the broken plan"},
                "reason": "Unrelated consequential field changed",
                "addresses_event_ids": [comparison_event_id],
            },
            session_id="session-1",
        )

    with pytest.raises(HarnessError, match="substantive"):
        harness.model(
            {
                "operation": "revise",
                "updates": {
                    "mechanisms": [
                        {
                            **_open_args()["mechanisms"][0],
                            "audit_note": "reviewed without changing the claim",
                        }
                    ],
                    "current_plan": [
                        {
                            **_open_args()["current_plan"][0],
                            "audit_note": "reviewed without changing the action",
                        }
                    ],
                },
                "reason": "Metadata-only acknowledgement",
                "addresses_event_ids": [comparison_event_id],
            },
            session_id="session-1",
        )

    revised = harness.model(
        {
            "operation": "revise",
            "updates": {
                "mechanisms": [
                    {
                        "id": "M2",
                        "text": "Task structure, not M1, predicts the observed process",
                        "authority_domain": "external_source",
                        "evidence_event_ids": [comparison_event_id],
                    }
                ],
                "current_plan": [
                    {
                        "id": "P2",
                        "action": "Test the task-structure account",
                        "depends_on": ["M2"],
                    }
                ],
            },
            "reason": "Replace the contradicted mechanism and dependent plan",
            "addresses_event_ids": [comparison_event_id],
        },
        session_id="session-1",
    )
    assert revised["case"]["stale"] is False
    assert revised["case"]["stale_claim_ids"] == []
    assert revised["replay"]["usable"] is True


def test_pause_resume_and_close_retire_gateway_without_losing_case(tmp_path):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="session-1")

    paused = harness.model(
        {"operation": "pause", "reason": "Waiting for a collaborator decision"},
        session_id="session-1",
    )
    assert paused["case"]["status"] == "paused"
    assert harness.store.get_active_case("session-1") is None

    ordinary_calls = []
    result = harness.tool_execution_middleware(
        tool_name="read_file",
        args={"path": "unrelated.txt"},
        session_id="session-1",
        next_call=lambda args: ordinary_calls.append(args) or "ordinary result",
    )
    assert result == "ordinary result"
    assert ordinary_calls == [{"path": "unrelated.txt"}]

    resumed = harness.model(
        {"operation": "resume", "case_id": "research-case"},
        session_id="session-1",
    )
    assert resumed["case"]["status"] == "active"
    assert resumed["replay"]["usable"] is True

    closed = harness.model(
        {
            "operation": "close",
            "outcome": "unresolved",
            "summary": "The available source cannot distinguish the accounts",
            "transfer": {
                "decision": "candidate",
                "lesson": "Do not turn absence of a distinction in one paper into a field-level gap",
                "scope": "research-gap judgment",
                "evidence_case_ids": ["research-case"],
            },
        },
        session_id="session-1",
    )
    assert closed["case"]["status"] == "closed"
    assert closed["case"]["closure"]["outcome"] == "unresolved"
    assert closed["case"]["closure"]["transfer"]["decision"] == "candidate"
    assert harness.store.get_active_case("session-1") is None

    after_close_calls = []
    after_close = harness.tool_execution_middleware(
        tool_name="web_search",
        args={"query": "ordinary query"},
        session_id="session-1",
        next_call=lambda args: after_close_calls.append(args) or "ordinary search",
    )
    assert after_close == "ordinary search"
    assert after_close_calls == [{"query": "ordinary query"}]


def test_session_switch_requires_explicit_case_retirement(tmp_path):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="old-session")

    with pytest.raises(HarnessError, match="explicitly pause or close"):
        harness.require_settled_before_session_switch(
            old_session_id="old-session",
            new_session_id="target-session",
            reason="resume",
        )

    harness.model(
        {
            "operation": "pause",
            "reason": "switching to a different transcript",
            "expected_case_revision": harness.store.get_case("research-case")[
                "runtime_revision"
            ],
        },
        session_id="old-session",
    )
    harness.require_settled_before_session_switch(
        old_session_id="old-session",
        new_session_id="target-session",
        reason="resume",
    )


def test_session_reset_pauses_old_case_and_new_session_starts_dormant(tmp_path):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="old-session")

    reset = harness.session_reset(
        old_session_id="old-session",
        new_session_id="new-session",
    )

    assert reset is not None
    assert reset["case"]["status"] == "paused"
    assert harness.store.get_active_case("old-session") is None
    calls = []
    assert harness.tool_execution_middleware(
        tool_name="read_file",
        args={"path": "unrelated.txt"},
        session_id="new-session",
        next_call=lambda args: calls.append(args) or "dormant",
    ) == "dormant"
    assert calls == [{"path": "unrelated.txt"}]

    resumed = harness.model(
        {"operation": "resume", "case_id": "research-case"},
        session_id="new-session",
    )
    assert resumed["case"]["status"] == "active"
    assert resumed["case"]["session_id"] == "new-session"
    assert [event["type"] for event in harness.store.list_events("research-case")][-2:] == [
        "case_paused",
        "case_resumed",
    ]


def test_subagent_stop_pauses_unfinished_case_and_preserves_observed_probe(tmp_path):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="child-session")
    probe_args = {
        "operation": "commit",
        "purpose": "learn",
        "unknown_or_goal": "Whether the source supports the working account",
        "tool_name": "read_file",
        "tool_args": {"path": "evidence.txt"},
        "predicted_outcomes": [
            {"outcome": "support", "meaning": "The source supports the account"}
        ],
        "why_this_action": "A direct read distinguishes the accounts",
        "would_change_belief": "Evidence incompatible with the predicted outcome",
        "authority_domain": "external_source",
    }
    harness.probe(probe_args, session_id="child-session")
    harness.tool_execution_middleware(
        tool_name="read_file",
        args={"path": "evidence.txt"},
        session_id="child-session",
        next_call=lambda _args: "support",
    )

    paused = harness.pause_active_subagent_case(
        child_session_id="child-session",
        child_status="completed",
    )

    assert paused is not None
    assert paused["case"]["status"] == "paused"
    assert paused["case"]["pending_probe"]["status"] == "observed"
    assert harness.store.get_active_case("child-session") is None
    assert paused["event"]["type"] == "case_paused"
    assert {
        key: paused["event"]["payload"][key]
        for key in (
            "reason",
            "lifecycle",
            "child_status",
            "pending_probe_status",
        )
    } == {
        "reason": "delegated epistemic worker stopped before closing its case",
        "lifecycle": "subagent_stop",
        "child_status": "completed",
        "pending_probe_status": "observed",
    }

    resumed = harness.model(
        {"operation": "resume", "case_id": "research-case"},
        session_id="recovery-session",
    )
    assert resumed["case"]["pending_probe"]["status"] == "observed"
    compared = harness.probe(
        {
            "operation": "compare",
            "disposition": "match",
            "material": False,
            "rationale": "The preserved result matches the prediction",
            "affected_claim_ids": [],
            "evidence": _external_evidence(),
        },
        session_id="recovery-session",
    )
    assert compared["replay"]["usable"] is True


def test_subagent_stop_is_noop_without_child_identity_or_active_case(tmp_path):
    harness = EpistemicHarness(tmp_path)
    assert harness.pause_active_subagent_case(
        child_session_id=None,
        child_status="failed",
    ) is None
    assert harness.pause_active_subagent_case(
        child_session_id="missing-child",
        child_status="failed",
    ) is None


def test_compression_rebinds_active_case_and_pending_probe_to_child_session(tmp_path):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="old-session")
    harness.probe(
        {
            "operation": "commit",
            "purpose": "learn",
            "unknown_or_goal": "Whether the continuation preserves the probe",
            "tool_name": "read_file",
            "tool_args": {"path": "continuation.txt"},
            "predicted_outcomes": [
                {"outcome": "preserved", "meaning": "Continue the same case"}
            ],
            "why_this_action": "The source is still the designated evidence",
            "would_change_belief": "Evidence incompatible with the predicted outcome",
            "authority_domain": "external_source",
        },
        session_id="old-session",
    )

    migrated = harness.session_compress(
        old_session_id="old-session",
        new_session_id="new-session",
    )

    assert migrated["status"] == "migrated"
    assert harness.store.get_active_case("old-session") is None
    continued = harness.store.get_active_case("new-session")
    assert continued is not None
    assert continued["pending_probe"]["tool_name"] == "read_file"
    assert harness.store.list_events("research-case")[-1]["type"] == "session_migrated"


def test_prepared_compression_recovers_binding_after_process_restart(tmp_path):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="old-session")
    prepared = harness.prepare_compression(
        old_session_id="old-session",
        new_session_id="new-session",
    )
    assert prepared["status"] == "prepared"

    restarted = EpistemicHarness(tmp_path)
    calls: list[dict] = []
    ordinary = restarted.tool_execution_middleware(
        tool_name="terminal",
        args={"command": "must-not-run"},
        session_id="new-session",
        next_call=lambda args: calls.append(args) or "ordinary",
    )

    assert ordinary == "ordinary"
    assert calls == [{"command": "must-not-run"}]
    assert restarted.store.get_active_case("old-session") is None
    assert restarted.store.get_active_case("new-session")["case_id"] == "research-case"
    assert restarted.store.list_events("research-case")[-1]["type"] == "session_migrated"


def test_prepared_compression_cannot_migrate_during_probe_execution(tmp_path):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="old-session")
    harness.probe(
        {
            "operation": "commit",
            "purpose": "learn",
            "unknown_or_goal": "Whether the result survives compression",
            "tool_name": "read_file",
            "tool_args": {"path": "evidence.txt"},
            "predicted_outcomes": [
                {"outcome": "found", "meaning": "Record the evidence before migration"}
            ],
            "why_this_action": "The observation must not be lost at a session boundary",
            "would_change_belief": "Evidence incompatible with the predicted outcome",
            "authority_domain": "external_source",
        },
        session_id="old-session",
    )
    action_started = threading.Event()
    release_action = threading.Event()
    worker_outcome: dict[str, object] = {}

    def tool(_args):
        action_started.set()
        assert release_action.wait(timeout=5)
        return "durable observation"

    def execute_probe() -> None:
        try:
            worker_outcome["result"] = harness.tool_execution_middleware(
                tool_name="read_file",
                args={"path": "evidence.txt"},
                session_id="old-session",
                next_call=tool,
            )
        except BaseException as exc:
            worker_outcome["error"] = exc

    worker = threading.Thread(target=execute_probe)
    worker.start()
    assert action_started.wait(timeout=5)
    harness.prepare_compression(
        old_session_id="old-session",
        new_session_id="new-session",
    )

    with pytest.raises(store_module.StoreError, match="already executing"):
        harness.store.recover_prepared_compression("new-session")
    assert harness.store.get_active_case("old-session")["pending_probe"]["status"] == "executing"
    assert harness.store.get_active_case("new-session") is None

    release_action.set()
    worker.join(timeout=5)
    assert not worker.is_alive()
    assert worker_outcome == {"result": "durable observation"}
    observed = harness.store.get_active_case("old-session")
    assert observed["pending_probe"]["status"] == "observed"

    migrated = harness.store.recover_prepared_compression("new-session")
    assert migrated["case"]["session_id"] == "new-session"
    continued = harness.store.get_active_case("new-session")
    assert continued["pending_probe"]["status"] == "observed"
    assert [event["type"] for event in harness.store.list_events("research-case")][-3:] == [
        "probe_started",
        "probe_observed",
        "session_migrated",
    ]


def test_recovered_child_is_bound_before_it_can_prepare_another_compression(tmp_path):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="parent-session")
    harness.prepare_compression(
        old_session_id="parent-session",
        new_session_id="child-session",
    )

    prepared = harness.prepare_compression(
        old_session_id="child-session",
        new_session_id="grandchild-session",
    )

    assert prepared["status"] == "prepared"
    assert harness.store.get_active_case("child-session")["case_id"] == "research-case"
    assert harness.store.get_active_case("parent-session") is None


def test_prepublication_compression_crash_replaces_intent_only_when_target_absent(tmp_path):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="parent-session")
    harness.prepare_compression(
        old_session_id="parent-session",
        new_session_id="unpublished-child",
    )

    replacement = harness.prepare_compression(
        old_session_id="parent-session",
        new_session_id="fresh-child",
        session_exists=lambda session_id: session_id == "parent-session",
    )
    assert replacement["status"] == "prepared"
    assert harness.store.get_prepared_compression_from("parent-session")[
        "new_session_id"
    ] == "fresh-child"


def test_stale_compression_intent_fails_closed_without_core_session_proof(tmp_path):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="parent-session")
    harness.prepare_compression(
        old_session_id="parent-session",
        new_session_id="possibly-published-child",
    )

    with pytest.raises(HarnessError, match="proof"):
        harness.prepare_compression(
            old_session_id="parent-session",
            new_session_id="fresh-child",
        )


def test_compression_abort_restores_prepared_binding_to_parent(tmp_path):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="old-session")
    harness.prepare_compression(
        old_session_id="old-session",
        new_session_id="new-session",
    )

    aborted = harness.abort_compression(
        old_session_id="old-session",
        new_session_id="new-session",
    )

    assert aborted["status"] == "aborted"
    assert harness.store.get_active_case("old-session")["case_id"] == "research-case"
    assert harness.store.get_active_case("new-session") is None


def test_prepared_compression_becomes_dormant_if_case_is_paused_before_publish(tmp_path):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="old-session")
    harness.prepare_compression(
        old_session_id="old-session",
        new_session_id="new-session",
    )
    paused = harness.model(
        {"operation": "pause", "reason": "concurrent session reset"},
        session_id="old-session",
    )
    assert paused["case"]["status"] == "paused"

    calls: list[dict] = []
    result = harness.tool_execution_middleware(
        tool_name="terminal",
        args={"command": "safe-dormant"},
        session_id="new-session",
        next_call=lambda args: calls.append(args) or "forwarded",
    )

    assert result == "forwarded"
    assert calls == [{"command": "safe-dormant"}]
    assert list((tmp_path / "compression-boundaries").glob("*.json")) == []


def test_concurrent_resumes_cannot_bind_two_cases_to_one_session(tmp_path, monkeypatch):
    harness = EpistemicHarness(tmp_path)
    for case_id, session_id in (("case-a", "session-a"), ("case-b", "session-b")):
        open_args = _open_args()
        open_args["case_id"] = case_id
        harness.model(open_args, session_id=session_id)
        harness.model(
            {"operation": "pause", "reason": "prepare concurrent resume"},
            session_id=session_id,
        )

    original_get_active = harness.store.get_active_case
    barrier = threading.Barrier(2)

    def synchronized_get_active(session_id):
        result = original_get_active(session_id)
        if session_id == "target-session":
            barrier.wait(timeout=3)
        return result

    monkeypatch.setattr(harness.store, "get_active_case", synchronized_get_active)

    def resume(case_id):
        try:
            resumed = harness.model(
                {"operation": "resume", "case_id": case_id},
                session_id="target-session",
            )
            return ("ok", resumed["case"]["case_id"])
        except HarnessError as exc:
            return ("error", str(exc))

    with ThreadPoolExecutor(max_workers=2) as pool:
        outcomes = list(pool.map(resume, ("case-a", "case-b")))

    assert sum(status == "ok" for status, _ in outcomes) == 1
    active = [
        case_id
        for case_id in ("case-a", "case-b")
        if (
            harness.store.get_case(case_id).get("status") == "active"
            and harness.store.get_case(case_id).get("session_id") == "target-session"
        )
    ]
    assert len(active) == 1


def test_dormant_session_cannot_show_another_sessions_active_case(tmp_path):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="session-1")

    with pytest.raises(HarnessError, match="bound to another session"):
        harness.model(
            {"operation": "show", "case_id": "research-case"},
            session_id="session-2",
        )


def test_active_case_can_read_closed_other_case_without_rebinding(tmp_path):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="session-1")
    harness.model(
        {
            "operation": "close",
            "outcome": "unresolved",
            "summary": "First case closed",
            "transfer": {"decision": "none"},
        },
        session_id="session-1",
    )
    second = _open_args()
    second["case_id"] = "second-case"
    harness.model(second, session_id="session-1")

    shown_closed = harness.model(
        {"operation": "show", "case_id": "research-case"},
        session_id="session-1",
    )
    assert shown_closed["case"]["status"] == "closed"
    assert shown_closed["replay"]["usable"] is True

    shown = harness.model(
        {"operation": "show", "case_id": "second-case"},
        session_id="session-1",
    )
    assert shown["case"]["case_id"] == "second-case"


def test_new_case_surfaces_prior_candidate_lesson_without_auto_promoting_it(tmp_path):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="session-1")
    harness.model(
        {
            "operation": "close",
            "outcome": "unresolved",
            "summary": "The paper bounded the construct but did not establish a literature gap",
            "transfer": {
                "decision": "candidate",
                "lesson": "A construct distinction is not itself a checked research gap",
                "scope": "research-gap judgment",
                "evidence_case_ids": ["research-case"],
            },
        },
        session_id="session-1",
    )

    next_args = _open_args()
    next_args["case_id"] = "next-research-case"
    next_args["prior_case_ids"] = ["research-case"]
    next_args["applied_prior_lessons"] = []
    opened = harness.model(next_args, session_id="session-2")

    assert opened["case"]["model"]["prior_case_ids"] == ["research-case"]
    assert opened["case"]["model"]["applied_prior_lessons"] == []
    assert opened["case"]["model"]["retrieved_prior_lessons"] == opened[
        "prior_case_lessons"
    ]
    assert opened["prior_case_lessons"] == [
        {
            "case_id": "research-case",
            "decision": "candidate",
            "lesson": "A construct distinction is not itself a checked research gap",
            "scope": "research-gap judgment",
            "evidence_case_ids": ["research-case"],
            "calibration": "hypothesis",
        }
    ]


def test_new_case_automatically_retrieves_only_relevant_closed_lessons(tmp_path):
    harness = EpistemicHarness(tmp_path)
    source = _open_args()
    source["case_id"] = "collaboration-source"
    harness.model(source, session_id="source-session")
    harness.model(
        {
            "operation": "close",
            "outcome": "unresolved",
            "summary": "The mechanism remains bounded but reusable.",
            "transfer": {
                "decision": "candidate",
                "lesson": (
                    "Collaboration surfaces context while social influence can "
                    "cause convergence"
                ),
                "scope": "collaboration and delegation design",
                "evidence_case_ids": ["collaboration-source"],
            },
        },
        session_id="source-session",
    )

    relevant = _open_args()
    relevant["case_id"] = "collaboration-followup"
    relevant["decision"] = "Choose a collaboration or delegation design"
    relevant["top_unknown"] = (
        "Whether collaboration surfaces context while causing convergence"
    )
    opened = harness.model(relevant, session_id="relevant-session")

    assert opened["case"]["model"]["prior_case_ids"] == ["collaboration-source"]
    assert len(opened["prior_case_lessons"]) == 1
    assert opened["prior_case_lessons"][0]["relevance"] >= 0.12
    assert opened["case"]["model"]["applied_prior_lessons"] == []

    unrelated = _open_args()
    unrelated["case_id"] = "database-followup"
    unrelated["decision"] = "Choose a database backup schedule"
    unrelated["top_unknown"] = "Which backup schedule minimizes database downtime"
    opened_unrelated = harness.model(unrelated, session_id="unrelated-session")
    assert opened_unrelated["case"]["model"]["prior_case_ids"] == []
    assert opened_unrelated["prior_case_lessons"] == []


def test_transfer_candidate_must_be_bound_to_case_being_closed(tmp_path):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="session-1")

    with pytest.raises(HarnessError, match="case being closed"):
        harness.model(
            _close_args(
                {
                    "decision": "candidate",
                    "lesson": "A plausible lesson",
                    "scope": "bounded scope",
                    "evidence_case_ids": ["invented-case"],
                }
            ),
            session_id="session-1",
        )


def test_repeated_pattern_requires_two_closed_matching_source_cases(tmp_path):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="session-1")

    with pytest.raises(HarnessError, match="two distinct closed"):
        harness.model(
            _close_args(
                {
                    "decision": "repeated_pattern",
                    "lesson": "A repeated-looking lesson",
                    "scope": "bounded scope",
                    "evidence_case_ids": ["research-case"],
                }
            ),
            session_id="session-1",
        )


def test_repeated_pattern_accepts_two_closed_replay_valid_matching_cases(tmp_path):
    harness = EpistemicHarness(tmp_path)
    lesson = "Use the bounded process check before field-level extrapolation"
    scope = "research-gap judgment"
    for index, case_id in enumerate(("source-case-one", "source-case-two"), start=1):
        opened = _open_args()
        opened["case_id"] = case_id
        session_id = f"source-session-{index}"
        harness.model(opened, session_id=session_id)
        harness.model(
            _close_args(
                {
                    "decision": "candidate",
                    "lesson": lesson,
                    "scope": scope,
                    "evidence_case_ids": [case_id],
                }
            ),
            session_id=session_id,
        )

    current = _open_args()
    current["case_id"] = "current-case"
    harness.model(current, session_id="current-session")
    closed = harness.model(
        _close_args(
            {
                "decision": "repeated_pattern",
                "lesson": lesson,
                "scope": scope,
                "evidence_case_ids": ["source-case-one", "source-case-two"],
            }
        ),
        session_id="current-session",
    )

    assert closed["case"]["closure"]["transfer"]["decision"] == "repeated_pattern"


def test_explicit_user_correction_requires_typed_user_event(tmp_path):
    harness = EpistemicHarness(tmp_path)
    opened = _open_args()
    opened["initial_evidence"] = [
        {
            "id": "I1",
            "summary": "An external page states a correction",
            "source_ref": "external-page",
            "authority_domain": "external_source",
            "source_role": "secondary",
            "access_scope": "full",
        }
    ]
    harness.model(opened, session_id="session-1")

    with pytest.raises(HarnessError, match="user-authoritative"):
        harness.model(
            _close_args(
                {
                    "decision": "explicit_user_correction",
                    "lesson": "The user corrected the operating assumption",
                    "scope": "user workflow",
                    "evidence_case_ids": ["research-case"],
                    "evidence_event_ids": ["E000001"],
                }
            ),
            session_id="session-1",
        )


def test_explicit_user_correction_accepts_exact_typed_user_event(tmp_path):
    harness = EpistemicHarness(tmp_path)
    opened = _open_args()
    opened["initial_evidence"] = [
        {
            "id": "I1",
            "summary": "the user explicitly corrected the operating assumption",
            "source_ref": "current-user-message",
            "authority_domain": "user_testimony",
        }
    ]
    harness.model(opened, session_id="session-1")
    closed = harness.model(
        _close_args(
            {
                "decision": "explicit_user_correction",
                "lesson": "The user corrected the operating assumption",
                "scope": "user workflow",
                "evidence_case_ids": ["research-case"],
                "evidence_event_ids": ["E000001"],
            }
        ),
        session_id="session-1",
    )

    assert closed["case"]["closure"]["transfer"]["evidence_event_ids"] == [
        "E000001"
    ]


def test_timeline_tampering_fails_replay_and_denies_tool_execution(tmp_path):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="session-1")
    timeline = tmp_path / "cases" / "research-case" / "events.jsonl"
    original = timeline.read_text()
    timeline.write_text(original.replace("case_opened", "case_altered", 1))

    replayed = harness.replay("research-case")
    assert replayed["usable"] is False
    assert replayed["timeline"]["valid"] is False

    calls = []
    with pytest.raises(HarnessError, match="Timeline integrity failure"):
        harness.tool_execution_middleware(
            tool_name="read_file",
            args={"path": "must-not-run.txt"},
            session_id="session-1",
            next_call=lambda args: calls.append(args) or "unsafe",
        )
    assert calls == []


def test_current_case_tampering_fails_replay_even_with_valid_timeline(tmp_path):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="session-1")
    case_path = tmp_path / "cases" / "research-case" / "case.md"
    text = case_path.read_text()
    start_marker = "<!-- EPISTEMIC-HARNESS-DATA\n"
    end_marker = "\nEND-EPISTEMIC-HARNESS-DATA -->"
    start = text.index(start_marker) + len(start_marker)
    end = text.index(end_marker, start)
    data = json.loads(text[start:end])
    data["model"]["decision"] = "Tampered assertion never produced by a model event"
    case_path.write_text(text[:start] + json.dumps(data, sort_keys=True) + text[end:])

    replay = harness.replay("research-case")
    assert replay["usable"] is False
    assert "case snapshot integrity" in replay["reason"].lower()


def test_formal_clarify_probe_preserves_typed_authority_and_allows_unresolved(tmp_path):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="session-1")

    with pytest.raises(HarnessError, match="clarify authority"):
        harness.probe(
            {
                "operation": "commit",
                "purpose": "learn",
                "unknown_or_goal": "Whether an external scientific proposition is true",
                "tool_name": "clarify",
                "tool_args": {"question": "Is proposition X scientifically true?"},
                "predicted_outcomes": [
                    {"outcome": "yes", "meaning": "Treat X as universal empirical truth"}
                ],
                "why_this_action": "Ask the user",
                "would_change_belief": "Evidence incompatible with the predicted outcome",
                "authority_domain": "external_source",
            },
            session_id="session-1",
        )

    harness.probe(
        {
            "operation": "commit",
            "purpose": "learn",
            "unknown_or_goal": "What the user intended by the phrase",
            "tool_name": "clarify",
            "tool_args": {"question": "What did you mean by different reasoning style?"},
            "predicted_outcomes": [
                {"outcome": "task decomposition", "meaning": "Model decomposition fit"},
                {"outcome": "I do not know", "meaning": "Leave the construct unresolved"},
            ],
            "why_this_action": "Only the user can resolve the user's intended meaning",
            "would_change_belief": "Evidence incompatible with the predicted outcome",
            "authority_domain": "user_intent",
        },
        session_id="session-1",
    )
    harness.tool_execution_middleware(
        tool_name="clarify",
        args={"question": "What did you mean by different reasoning style?"},
        session_id="session-1",
        next_call=lambda _args: "I do not know yet",
    )
    compared = harness.probe(
        {
            "operation": "compare",
            "disposition": "unresolved",
            "material": False,
            "rationale": "Non-knowledge does not discriminate between the candidate meanings",
            "affected_claim_ids": [],
        },
        session_id="session-1",
    )
    assert compared["case"]["pending_probe"] is None
    assert compared["case"]["stale"] is False
    assert compared["replay"]["usable"] is True


def test_user_testimony_cannot_ground_an_external_empirical_claim(tmp_path):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="session-1")
    question_args = {"question": "Is the Moon made of cheese?"}
    harness.probe(
        {
            "operation": "commit",
            "purpose": "learn",
            "unknown_or_goal": "What the user reports believing",
            "tool_name": "clarify",
            "tool_args": question_args,
            "predicted_outcomes": [
                {"outcome": "yes", "meaning": "Record the answer as user testimony"}
            ],
            "why_this_action": "The user is authoritative about their own report",
            "would_change_belief": "Evidence incompatible with the predicted outcome",
            "authority_domain": "user_testimony",
        },
        session_id="session-1",
    )
    harness.tool_execution_middleware(
        tool_name="clarify",
        args=question_args,
        session_id="session-1",
        next_call=lambda _args: "Yes",
    )
    observation_event_id = harness.store.get_active_case("session-1")[
        "pending_probe"
    ]["observation_event_id"]
    harness.probe(
        {
            "operation": "compare",
            "disposition": "match",
            "material": False,
            "rationale": "The answer matched one predicted report",
            "affected_claim_ids": [],
        },
        session_id="session-1",
    )

    with pytest.raises(HarnessError, match="cannot ground.*external"):
        harness.model(
            {
                "operation": "revise",
                "reason": "Treat the user's report as external empirical evidence",
                "updates": {
                    "state_grounding": [
                        {
                            "id": "S1",
                            "text": "The Moon is made of cheese",
                            "authority_domain": "external_source",
                            "evidence_event_ids": [observation_event_id],
                            "epistemic_status": "grounded",
                        }
                    ]
                },
            },
            session_id="session-1",
        )


def test_probe_started_event_cannot_ground_a_claim(tmp_path):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="session-1")
    tool_args = {"path": "evidence.txt"}
    harness.probe(
        {
            "operation": "commit",
            "purpose": "learn",
            "unknown_or_goal": "What the source says",
            "tool_name": "read_file",
            "tool_args": tool_args,
            "predicted_outcomes": [
                {"outcome": "evidence", "meaning": "Revise the model from the observation"}
            ],
            "why_this_action": "The source is relevant",
            "would_change_belief": "Evidence incompatible with the predicted outcome",
            "authority_domain": "external_source",
        },
        session_id="session-1",
    )
    harness.tool_execution_middleware(
        tool_name="read_file",
        args=tool_args,
        session_id="session-1",
        next_call=lambda _args: "observed evidence",
    )
    started_event_id = next(
        event["event_id"]
        for event in harness.store.list_events("research-case")
        if event["type"] == "probe_started"
    )
    harness.probe(
        {
            "operation": "compare",
            "disposition": "match",
            "material": False,
            "rationale": "The observation matched",
            "affected_claim_ids": [],
            "evidence": _external_evidence(),
        },
        session_id="session-1",
    )

    with pytest.raises(HarnessError, match="without evidence authority"):
        harness.model(
            {
                "operation": "revise",
                "reason": "Incorrectly cite dispatch as evidence",
                "updates": {
                    "mechanisms": [
                        {
                            "id": "M2",
                            "text": "The source establishes a new mechanism",
                            "authority_domain": "external_source",
                            "evidence_event_ids": [started_event_id],
                        }
                    ],
                    "current_plan": [
                        {
                            "id": "P2",
                            "action": "Test the claimed mechanism",
                            "depends_on": ["M2"],
                        }
                    ],
                },
            },
            session_id="session-1",
        )


def test_large_observation_is_secret_redacted_and_spilled_to_archive(tmp_path):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="session-1")
    harness.probe(
        {
            "operation": "commit",
            "purpose": "learn",
            "unknown_or_goal": "What the large source contains",
            "tool_name": "read_file",
            "tool_args": {"path": "large.txt"},
            "predicted_outcomes": [
                {"outcome": "relevant evidence", "meaning": "Update the current model"}
            ],
            "why_this_action": "The source is directly relevant",
            "would_change_belief": "Evidence incompatible with the predicted outcome",
            "authority_domain": "external_source",
        },
        session_id="session-1",
    )
    fake_secret = "ghp_" + "X" * 36
    raw_result = "evidence\n" * 3000 + fake_secret
    returned = harness.tool_execution_middleware(
        tool_name="read_file",
        args={"path": "large.txt"},
        session_id="session-1",
        next_call=lambda _args: raw_result,
    )
    assert returned == raw_result

    events = harness.store.list_events("research-case")
    observed = next(event for event in events if event["type"] == "probe_observed")
    payload = observed["payload"]
    assert payload["storage"] == "artifact"
    assert "result" not in payload
    assert fake_secret not in json.dumps(payload)
    artifact = tmp_path / "cases" / "research-case" / payload["artifact_path"]
    assert artifact.exists()
    artifact_text = artifact.read_text()
    assert fake_secret not in artifact_text
    assert "[REDACTED]" in artifact_text


def test_replay_fails_when_observation_artifact_is_missing(tmp_path):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="session-1")
    harness.probe(
        {
            "operation": "commit",
            "purpose": "learn",
            "unknown_or_goal": "What the large source contains",
            "tool_name": "read_file",
            "tool_args": {"path": "large.txt"},
            "predicted_outcomes": [
                {"outcome": "relevant evidence", "meaning": "Update the current model"}
            ],
            "why_this_action": "The source is directly relevant",
            "would_change_belief": "Evidence incompatible with the predicted outcome",
            "authority_domain": "external_source",
        },
        session_id="session-1",
    )
    harness.tool_execution_middleware(
        tool_name="read_file",
        args={"path": "large.txt"},
        session_id="session-1",
        next_call=lambda _args: "evidence\n" * 3000,
    )
    observed = next(
        event
        for event in harness.store.list_events("research-case")
        if event["type"] == "probe_observed"
    )
    harness.probe(
        {
            "operation": "compare",
            "disposition": "match",
            "material": False,
            "rationale": "The source matched the committed expectation",
            "affected_claim_ids": [],
            "evidence": _external_evidence(),
        },
        session_id="session-1",
    )
    assert harness.replay("research-case")["usable"] is True

    second_args = {"command": "inspect follow-up"}
    harness.probe(
        {
            "operation": "commit",
            "purpose": "learn",
            "unknown_or_goal": "Whether the follow-up changes the interpretation",
            "tool_name": "terminal",
            "tool_args": second_args,
            "predicted_outcomes": [
                {"outcome": "new evidence", "meaning": "Revise the mechanism"}
            ],
            "why_this_action": "The follow-up directly tests the remaining unknown",
            "would_change_belief": "Evidence incompatible with the predicted outcome",
            "authority_domain": "external_source",
        },
        session_id="session-1",
    )

    artifact = (
        tmp_path
        / "cases"
        / "research-case"
        / observed["payload"]["artifact_path"]
    )
    artifact.unlink()

    replay = harness.replay("research-case")
    assert replay["usable"] is False
    assert "artifact" in replay["reason"].lower()

    calls: list[dict] = []
    blocked = harness.tool_execution_middleware(
        tool_name="terminal",
        args=second_args,
        session_id="session-1",
        next_call=lambda effective: calls.append(effective) or "must not run",
    )
    assert calls == []
    assert "artifact integrity" in json.loads(blocked)["error"]


def test_tool_failure_is_captured_as_observation_and_case_can_recover(tmp_path):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="session-1")
    harness.probe(
        {
            "operation": "commit",
            "purpose": "advance",
            "unknown_or_goal": "Whether the diagnostic script succeeds",
            "tool_name": "terminal",
            "tool_args": {"command": "run-diagnostic"},
            "predicted_outcomes": [
                {"outcome": "success", "meaning": "Use the diagnostic result"},
                {"outcome": "failure", "meaning": "Treat the result as unresolved"},
            ],
            "why_this_action": "The script is the designated diagnostic",
            "would_change_belief": "Evidence incompatible with the predicted outcome",
            "authority_domain": "empirical_test",
        },
        session_id="session-1",
    )

    fake_secret = "ghp_" + "Y" * 36
    with pytest.raises(RuntimeError, match="diagnostic failed"):
        harness.tool_execution_middleware(
            tool_name="terminal",
            args={"command": "run-diagnostic"},
            session_id="session-1",
            next_call=lambda _args: (_ for _ in ()).throw(
                RuntimeError(f"diagnostic failed with {fake_secret}")
            ),
        )

    active = harness.store.get_active_case("session-1")
    assert active["pending_probe"]["status"] == "observed"
    observed = [
        event for event in harness.store.list_events("research-case")
        if event["type"] == "probe_observed"
    ][0]
    assert observed["payload"]["result_status"] == "error"
    assert fake_secret not in json.dumps(observed)

    compared = harness.probe(
        {
            "operation": "compare",
            "disposition": "unresolved",
            "material": False,
            "rationale": "Execution failure produced no discriminating evidence",
            "affected_claim_ids": [],
        },
        session_id="session-1",
    )
    assert compared["replay"]["usable"] is True


def test_returned_tool_error_is_captured_as_error_observation(tmp_path):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="session-1")
    harness.probe(
        {
            "operation": "commit",
            "purpose": "learn",
            "unknown_or_goal": "Whether the file exists",
            "tool_name": "read_file",
            "tool_args": {"path": "missing.txt"},
            "predicted_outcomes": [
                {"outcome": "error", "meaning": "The source is unavailable"}
            ],
            "why_this_action": "Availability changes the next step",
            "would_change_belief": "Evidence incompatible with the predicted outcome",
            "authority_domain": "external_source",
        },
        session_id="session-1",
    )

    returned = harness.tool_execution_middleware(
        tool_name="read_file",
        args={"path": "missing.txt"},
        session_id="session-1",
        next_call=lambda _args: json.dumps({"error": "file not found"}),
    )

    assert json.loads(returned)["error"] == "file not found"
    case = harness.store.get_active_case("session-1")
    assert case["pending_probe"]["result_status"] == "tool_error"
    observation = next(
        event
        for event in harness.store.list_events("research-case")
        if event["type"] == "probe_observed"
    )
    assert observation["payload"]["result_status"] == "tool_error"


def test_committed_tool_arguments_are_redacted_but_still_match_exactly(tmp_path):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="session-1")
    secret = "opaque correct horse battery staple 7319"
    opaque_api_key = "ordinary-unprefixed-key-483920"
    tool_args = {
        "command": "authenticate to example.invalid",
        "password": secret,
        "nested": {"api_key": opaque_api_key},
    }

    harness.probe(
        {
            "operation": "commit",
            "purpose": "advance",
            "unknown_or_goal": "Whether the authenticated endpoint responds",
            "tool_name": "terminal",
            "tool_args": tool_args,
            "predicted_outcomes": [
                {"outcome": "response", "meaning": "Continue from the endpoint state"}
            ],
            "why_this_action": f"The endpoint password {secret} determines access",
            "would_change_belief": "Evidence incompatible with the predicted outcome",
            "authority_domain": "system_state",
        },
        session_id="session-1",
    )

    case_text = (tmp_path / "cases" / "research-case" / "case.md").read_text()
    timeline_text = (tmp_path / "cases" / "research-case" / "events.jsonl").read_text()
    assert secret not in case_text
    assert secret not in timeline_text
    assert opaque_api_key not in case_text
    assert opaque_api_key not in timeline_text
    for persisted in tmp_path.rglob("*"):
        if persisted.is_file():
            raw = persisted.read_bytes()
            assert secret.encode() not in raw, persisted
            assert opaque_api_key.encode() not in raw, persisted
    case = harness.store.get_active_case("session-1")
    assert case["pending_probe"]["tool_args_sha256"]

    result = harness.tool_execution_middleware(
        tool_name="terminal",
        args=tool_args,
        session_id="session-1",
        next_call=lambda _args: "authenticated response",
    )
    assert result == "authenticated response"


def test_free_text_transition_fields_are_secret_redacted_before_persistence(tmp_path):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="session-1")
    secret = "sk-examplecredential123456789"
    harness.probe(
        {
            "operation": "commit",
            "purpose": "learn",
            "unknown_or_goal": "Whether the bounded source is readable",
            "tool_name": "read_file",
            "tool_args": {"path": "evidence.txt"},
            "predicted_outcomes": [{"outcome": "read", "meaning": "Continue"}],
            "why_this_action": f"Do not persist {secret}",
            "would_change_belief": "Evidence incompatible with the predicted outcome",
            "authority_domain": "system_state",
        },
        session_id="session-1",
    )
    harness.tool_execution_middleware(
        tool_name="read_file",
        args={"path": "evidence.txt"},
        session_id="session-1",
        next_call=lambda _args: "bounded result",
    )
    harness.probe(
        {
            "operation": "compare",
            "disposition": "match",
            "material": False,
            "rationale": f"The result matched without retaining {secret}",
            "affected_claim_ids": [],
        },
        session_id="session-1",
    )
    harness.model(
        {
            "operation": "revise",
            "reason": f"Update after checking {secret}",
            "updates": {"top_unknown": "No remaining bounded unknown"},
            "addresses_event_ids": [],
        },
        session_id="session-1",
    )
    harness.model(
        {
            "operation": "close",
            "outcome": "unresolved",
            "summary": f"Closed without retaining {secret}",
            "transfer": {
                "decision": "candidate",
                "lesson": f"Never persist {secret}",
                "scope": "credential-safe testing",
                "evidence_case_ids": ["research-case"],
            },
        },
        session_id="session-1",
    )

    for persisted in tmp_path.rglob("*"):
        if persisted.is_file():
            assert secret.encode() not in persisted.read_bytes(), persisted


def test_transition_recovers_after_crash_between_timeline_and_case_write(tmp_path, monkeypatch):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="session-1")
    original_write_case = harness.store._write_case
    failed_once = False

    def fail_after_event_append(case):
        nonlocal failed_once
        if not failed_once and case.get("pending_probe") is not None:
            failed_once = True
            raise OSError("injected crash after Timeline commit")
        return original_write_case(case)

    monkeypatch.setattr(harness.store, "_write_case", fail_after_event_append)
    with pytest.raises(OSError, match="injected crash"):
        harness.probe(
            {
                "operation": "commit",
                "purpose": "learn",
                "unknown_or_goal": "Whether crash recovery preserves the committed probe",
                "tool_name": "read_file",
                "tool_args": {"path": "recovery.txt"},
                "predicted_outcomes": [
                    {"outcome": "content", "meaning": "Continue from the captured source"}
                ],
                "why_this_action": "The exact source is the designated evidence",
                "would_change_belief": "Evidence incompatible with the predicted outcome",
                "authority_domain": "external_source",
            },
            session_id="session-1",
        )

    monkeypatch.setattr(harness.store, "_write_case", original_write_case)
    recovered = EpistemicHarness(tmp_path)
    case = recovered.store.get_active_case("session-1")
    assert case is not None
    assert case["pending_probe"]["status"] == "committed"
    assert case["pending_probe"]["tool_name"] == "read_file"
    assert [
        event["type"] for event in recovered.store.list_events("research-case")
    ].count("probe_committed") == 1

    observed = recovered.tool_execution_middleware(
        tool_name="read_file",
        args={"path": "recovery.txt"},
        session_id="session-1",
        next_call=lambda _args: "recovered content",
    )
    assert observed == "recovered content"
    assert recovered.store.get_active_case("session-1")["pending_probe"]["status"] == "observed"


def test_recover_rejects_while_committed_action_is_still_running(tmp_path):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="session-1")
    tool_args = {"command": "slow-external-action"}
    harness.probe(
        {
            "operation": "commit",
            "purpose": "advance",
            "unknown_or_goal": "Whether the external action completes",
            "tool_name": "terminal",
            "tool_args": tool_args,
            "predicted_outcomes": [
                {"outcome": "completed", "meaning": "Continue from the result"},
                {"outcome": "unknown", "meaning": "Recover after a real interruption"},
            ],
            "why_this_action": "The action is the designated next step",
            "would_change_belief": "Evidence incompatible with the predicted outcome",
            "authority_domain": "system_state",
        },
        session_id="session-1",
    )
    entered = threading.Event()
    release = threading.Event()
    outcome = {}

    def slow_action(_args):
        entered.set()
        assert release.wait(timeout=5)
        return "completed"

    def invoke_action():
        try:
            outcome["result"] = harness.tool_execution_middleware(
                tool_name="terminal",
                args=tool_args,
                session_id="session-1",
                next_call=slow_action,
            )
        except Exception as exc:  # pragma: no cover - asserted below
            outcome["error"] = exc

    worker = threading.Thread(target=invoke_action)
    worker.start()
    assert entered.wait(timeout=3)

    with pytest.raises(HarnessError, match="still running"):
        harness.probe(
            {"operation": "recover", "reason": "mistaken crash recovery"},
            session_id="session-1",
        )

    release.set()
    worker.join(timeout=5)
    assert not worker.is_alive()
    assert outcome == {"result": "completed"}
    assert harness.store.get_active_case("session-1")["pending_probe"]["status"] == "observed"


def test_interrupted_execution_requires_explicit_recovery_and_never_retries(tmp_path):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="session-1")
    harness.probe(
        {
            "operation": "commit",
            "purpose": "advance",
            "unknown_or_goal": "Whether the external action completed",
            "tool_name": "terminal",
            "tool_args": {"command": "external-action"},
            "predicted_outcomes": [
                {"outcome": "completed", "meaning": "Continue from the result"},
                {"outcome": "unknown", "meaning": "Inspect state before any retry"},
            ],
            "why_this_action": "The action advances the case",
            "would_change_belief": "Evidence incompatible with the predicted outcome",
            "authority_domain": "system_state",
        },
        session_id="session-1",
    )
    case = harness.store.get_active_case("session-1")
    case["pending_probe"]["status"] = "executing"
    harness.store.transition_case(
        case,
        event_type="probe_started",
        event_payload={"probe_id": "PR000001", "tool_name": "terminal"},
        event_id_target=("pending_probe", "started_event_id"),
    )

    fresh = EpistemicHarness(tmp_path)
    assert fresh.replay("research-case")["reason"] == "probe execution is incomplete"
    retried = []
    blocked = fresh.tool_execution_middleware(
        tool_name="terminal",
        args={"command": "external-action"},
        session_id="session-1",
        next_call=lambda args: retried.append(args) or "unsafe retry",
    )
    assert "pending probe" in json.loads(blocked)["error"]
    assert retried == []

    recovered = fresh.probe(
        {
            "operation": "recover",
            "reason": "Agent process ended after dispatch began and before result capture",
        },
        session_id="session-1",
    )
    assert recovered["probe"]["status"] == "observed"
    observation = recovered["event"]
    assert observation["payload"]["result_status"] == "interrupted"
    assert "may or may not have completed" in observation["payload"]["result"]


def test_replay_accounts_for_every_observation_and_model_event_reference(tmp_path):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="session-1")
    orphan = harness.store.append_event(
        "research-case",
        event_type="probe_observed",
        session_id="session-1",
        payload={
            "probe_id": "PR-ORPHAN",
            "tool_name": "read_file",
            "result_status": "ok",
            "storage": "inline",
            "result": "orphan observation",
        },
    )

    replayed = harness.replay("research-case")
    assert replayed["usable"] is False
    assert replayed["accounting"]["uncompared_observation_event_ids"] == [
        orphan["event_id"]
    ]

    case = harness.store.get_active_case("session-1")
    case["model"]["state_grounding"][0]["evidence_event_ids"] = ["E999999"]
    harness.store.transition_case(
        case,
        event_type="case_metadata_test",
        event_payload={"purpose": "inject invalid reference for replay test"},
    )
    replayed_again = harness.replay("research-case")
    assert "E999999" in replayed_again["accounting"]["invalid_model_event_ids"]

    with pytest.raises(HarnessError, match="replay is not usable"):
        harness.probe(
            {
                "operation": "commit",
                "purpose": "learn",
                "unknown_or_goal": "This must not run against an inconsistent model",
                "tool_name": "read_file",
                "tool_args": {"path": "blocked.txt"},
                "predicted_outcomes": [
                    {"outcome": "anything", "meaning": "No action is permitted yet"}
                ],
                "why_this_action": "Attempted only to test the replay gate",
                "would_change_belief": "Evidence incompatible with the predicted outcome",
                "authority_domain": "external_source",
            },
            session_id="session-1",
        )


def test_concurrent_identical_calls_execute_committed_probe_exactly_once(tmp_path, monkeypatch):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="session-1")
    harness.probe(
        {
            "operation": "commit",
            "purpose": "advance",
            "unknown_or_goal": "Execute one designated action",
            "tool_name": "terminal",
            "tool_args": {"command": "one-action"},
            "predicted_outcomes": [
                {"outcome": "done", "meaning": "Compare the single result"}
            ],
            "why_this_action": "It is the single committed action",
            "would_change_belief": "Evidence incompatible with the predicted outcome",
            "authority_domain": "system_state",
        },
        session_id="session-1",
    )

    original_get_active = harness.store.get_active_case
    first_reads = threading.Barrier(2)
    read_count = 0
    count_lock = threading.Lock()

    def synchronized_get_active(session_id):
        nonlocal read_count
        case = original_get_active(session_id)
        with count_lock:
            read_count += 1
            should_wait = read_count <= 2
        if should_wait:
            first_reads.wait(timeout=2)
        return case

    monkeypatch.setattr(harness.store, "get_active_case", synchronized_get_active)
    dispatched = []
    dispatch_lock = threading.Lock()

    def external_action(args):
        with dispatch_lock:
            dispatched.append(args)
        return "done"

    def invoke():
        try:
            return harness.tool_execution_middleware(
                tool_name="terminal",
                args={"command": "one-action"},
                session_id="session-1",
                next_call=external_action,
            )
        except HarnessError as exc:
            return f"blocked:{exc}"

    with ThreadPoolExecutor(max_workers=2) as pool:
        results = list(pool.map(lambda _index: invoke(), range(2)))

    assert dispatched == [{"command": "one-action"}]
    assert results.count("done") == 1
    blocked = [json.loads(result) for result in results if result != "done"]
    assert len(blocked) == 1
    blocked_error = blocked[0]["error"]
    assert blocked_error.startswith("EPISTEMIC GATEWAY BLOCKED:")
    assert any(
        marker in blocked_error
        for marker in (
            "another invocation",
            "already claimed or changed",
            "probe execution is incomplete",
        )
    )
    event_types = [event["type"] for event in harness.store.list_events("research-case")]
    assert event_types.count("probe_started") == 1
    assert event_types.count("probe_observed") == 1


def test_record_evidence_derives_mixed_claim_calibration_from_access_scope(tmp_path):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="session-1")
    recorded = harness.model(
        {
            "operation": "record_evidence",
            "evidence": [
                {
                    "summary": "The abstract reports the proposed association",
                    "source_ref": "doi:10.0000/abstract-only",
                    "source_role": "primary",
                    "access_scope": "abstract",
                    "limitation": "The methods and results tables were unavailable",
                    "authority_domain": "external_source",
                },
                {
                    "summary": "The full paper reports the mechanism and boundary condition",
                    "source_ref": "doi:10.0000/full-text",
                    "source_role": "primary",
                    "access_scope": "full",
                    "locator": "Results, pp. 8-10",
                    "authority_domain": "external_source",
                },
            ],
        },
        session_id="session-1",
    )
    abstract_id, full_id = recorded["evidence_event_ids"]
    assert [event["type"] for event in recorded["events"]] == [
        "evidence_recorded",
        "evidence_recorded",
    ]

    harness.model(
        {
            "operation": "revise",
            "reason": "Link each claim to the source scope actually inspected",
            "updates": {
                "state_grounding": [
                    {
                        "id": "S1",
                        "text": "The association is reported",
                        "evidence_event_ids": [abstract_id],
                        "authority_domain": "external_source",
                    },
                    {
                        "id": "S2",
                        "text": "The mechanism has the stated boundary condition",
                        "evidence_event_ids": [full_id],
                        "authority_domain": "external_source",
                    },
                ]
            },
        },
        session_id="session-1",
    )
    closed = harness.model(
        {
            "operation": "close",
            "outcome": "resolved",
            "summary": "One claim is provisional and one is supported.",
            "transfer": {"decision": "none"},
        },
        session_id="session-1",
    )
    calibration = closed["case"]["closure"]["calibration"]
    assert calibration["label"] == "mixed"
    assert {
        item["claim_id"]: item["label"]
        for item in calibration["claims"]
        if item["claim_id"] in {"S1", "S2"}
    } == {"S1": "provisional", "S2": "supported"}
    abstract = next(item for item in calibration["claims"] if item["claim_id"] == "S1")
    assert abstract["access_scopes"] == ["abstract"]
    assert abstract["limitations"] == ["The methods and results tables were unavailable"]


def test_record_evidence_recovers_stringified_json_array_from_provider(tmp_path):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="session-1")
    recorded = harness.model(
        {
            "operation": "record_evidence",
            "evidence": json.dumps(
                [
                    {
                        "summary": "The abstract reports the proposed association",
                        "source_ref": "doi:10.0000/stringified",
                        "source_role": "primary",
                        "access_scope": "abstract",
                        "limitation": "Only the abstract was accessible",
                        "authority_domain": "external_source",
                    }
                ]
            ),
        },
        session_id="session-1",
    )

    assert recorded["evidence_event_ids"] == ["E000002"]
    assert recorded["events"][0]["payload"]["access_scope"] == "abstract"


def test_record_evidence_rejects_plain_string_instead_of_treating_it_as_evidence(tmp_path):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="session-1")

    with pytest.raises(HarnessError, match="not plain text"):
        harness.model(
            {"operation": "record_evidence", "evidence": "an abstract says so"},
            session_id="session-1",
        )


def test_limited_evidence_requires_limitation_and_prior_lessons_keep_calibration(tmp_path):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="session-1")
    with pytest.raises(HarnessError, match="requires an explicit limitation"):
        harness.model(
            {
                "operation": "record_evidence",
                "evidence": {
                    "summary": "A search result mentions the claim",
                    "source_ref": "https://example.test/result",
                    "source_role": "unknown",
                    "access_scope": "snippet",
                    "authority_domain": "external_source",
                },
            },
            session_id="session-1",
        )

    harness.model(
        {
            "operation": "close",
            "outcome": "unresolved",
            "summary": "No assessed evidence was available.",
            "transfer": {
                "decision": "candidate",
                "lesson": "A search lead is not a checked result",
                "scope": "literature review",
                "evidence_case_ids": ["research-case"],
            },
        },
        session_id="session-1",
    )
    next_case = _open_args()
    next_case["case_id"] = "next-case"
    next_case["prior_case_ids"] = ["research-case"]
    opened = harness.model(next_case, session_id="session-2")
    assert opened["prior_case_lessons"][0]["calibration"] == "hypothesis"


def test_repeated_pattern_support_cannot_exceed_weakest_source():
    from epistemic_harness.harness import _transfer_support_ceiling

    assert _transfer_support_ceiling(["supported", "provisional"]) == "provisional"
    assert _transfer_support_ceiling(["supported", "mixed"]) == "provisional"
    assert _transfer_support_ceiling(["supported", "supported"]) == "supported"


def test_probe_compare_rejects_user_authority_evidence_for_system_state_probe(tmp_path):
    # F6 regression: a user-testimony evidence object cannot be bound to a
    # system_state probe — that would launder user testimony into a
    # non-user grounded claim.
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="session-1")
    harness.probe(
        {
            "operation": "commit",
            "purpose": "advance",
            "unknown_or_goal": "What the local state reveals",
            "tool_name": "read_file",
            "tool_args": {"path": "state-observation.txt"},
            "predicted_outcomes": ["The file reports the expected state"],
            "would_change_belief": "A value incompatible with the predicted state",
            "why_this_action": "The file is the direct system-state observation",
            "authority_domain": "system_state",
        },
        session_id="session-1",
    )
    harness.tool_execution_middleware(
        tool_name="read_file",
        args={"path": "state-observation.txt"},
        session_id="session-1",
        next_call=lambda _args: "some text",
    )
    with pytest.raises(HarnessError, match="incompatible with probe authority"):
        harness.probe(
            {
                "operation": "compare",
                "disposition": "match",
                "material": False,
                "rationale": "The observation matched",
                "affected_claim_ids": ["M1"],
                "evidence": {
                    "summary": "the user said so",
                    "source_ref": "current-user-message",
                    "authority_domain": "user_testimony",
                },
            },
            session_id="session-1",
        )


def test_initial_evidence_rejects_untyped_external_source(tmp_path):
    # F7 regression: opening with untyped external_source initial evidence
    # (no source_role/access_scope) must be rejected, not promoted to
    # "supported" calibration.
    harness = EpistemicHarness(tmp_path)
    args = _open_args()
    args["initial_evidence"] = [
        {
            "id": "I1",
            "summary": "An external observation",
            "source_ref": "external-page",
            "authority_domain": "external_source",
        }
    ]
    with pytest.raises(HarnessError, match="source_role"):
        harness.model(args, session_id="session-1")


def test_event_authority_map_reattributes_laundered_probe_compared_to_user(tmp_path):
    # F6 part 2: a probe_compared event whose nested evidence is user_testimony
    # must be attributed as user authority (not the probe's declared domain), so a
    # non-user claim citing it is rejected.
    from epistemic_harness.harness import _event_authority_map

    events = [
        {
            "type": "probe_committed",
            "event_id": "E000001",
            "payload": {
                "probe": {
                    "probe_id": "P1",
                    "authority_domain": "system_state",
                }
            },
        },
        {
            "type": "probe_compared",
            "event_id": "E000002",
            "payload": {
                "probe_id": "P1",
                "authority_domain": "system_state",
                "evidence": {
                    "summary": "the user said so",
                    "source_ref": "current-user-message",
                    "authority_domain": "user_testimony",
                },
            },
        },
    ]
    authorities = _event_authority_map(events)
    # The compared event must be attributed user_testimony, not system_state.
    assert authorities["E000002"] == "user_testimony"




def test_close_accepts_json_string_transfer(tmp_path):
    # Regression (2026-09-01 audit, F1): provider function-calling layers can
    # collapse nested objects into JSON strings. A close whose transfer arrives
    # as a string must parse it, not reject the close.
    harness = EpistemicHarness(tmp_path)
    harness.model(
        {"operation": "open", "case_id": "json-transfer", "top_unknown": "u"},
        session_id="s1",
    )
    harness.model(
        {
            "operation": "close",
            "case_id": "json-transfer",
            "outcome": "stopped",
            "summary": "planning complete",
            "transfer": '{"decision": "none"}',
        },
        session_id="s1",
    )
    case = harness.store.get_case("json-transfer")
    assert case["status"] == "closed"
    assert case["closure"]["transfer"]["decision"] == "none"


def test_close_rejects_invalid_json_string_transfer(tmp_path):
    # A NON-JSON string must still fail — the fallback parses, it does not
    # fabricate.
    harness = EpistemicHarness(tmp_path)
    harness.model(
        {"operation": "open", "case_id": "bad-json-transfer", "top_unknown": "u"},
        session_id="s1",
    )
    import pytest as _pytest

    from epistemic_harness.harness import HarnessError

    with _pytest.raises(HarnessError):
        harness.model(
            {
                "operation": "close",
                "case_id": "bad-json-transfer",
                "outcome": "stopped",
                "summary": "planning complete",
                "transfer": "not-json",
            },
            session_id="s1",
        )


def test_lesson_retrieval_survives_word_family_gap(tmp_path):
    # Regression (2026-09-01 audit, F2): the questioning-seed synthesis opened
    # with a query about "the value of questioning and perceptions of
    # question-asking machines" and retrieved ZERO lessons, while a directly
    # relevant lesson (human-AI delegation: separate model-side capability
    # from human-side valuation) sat closed the previous day. Exact-lexical
    # overlap scored 0.057 (floor 0.12). The stem-class channel must retrieve
    # it.
    harness = EpistemicHarness(tmp_path)
    # Close a case carrying the 0718-relevant lesson, phrased as the 1450
    # synthesizer actually phrased it.
    harness.model(
        {
            "operation": "open",
            "case_id": "prior-delegation-synthesis",
            "decision": "Frontier synthesis of human-AI delegation completed",
            "top_unknown": "Whether users value an agent that asks",
        },
        session_id="prior",
    )
    harness.model(
        {
            "operation": "close",
            "case_id": "prior-delegation-synthesis",
            "outcome": "stopped",
            "summary": "done",
            "transfer": {
                "decision": "candidate",
                "lesson": (
                    "In frontier syntheses of human-AI delegation, separate "
                    "model-side capability evidence (benchmark asking rates, "
                    "scaffold gains) from human-side valuation evidence "
                    "(costly revealed selection, willingness-to-pay, reward) "
                    "and state which side each claim rests on."
                ),
                "scope": (
                    "Frontier/literature scans for experimental-economics "
                    "human-AI interaction work where benchmark capability "
                    "results must not be read as valuation."
                ),
                "evidence_case_ids": ["prior-delegation-synthesis"],
            },
        },
        session_id="prior",
    )
    # Open with the 0718 question, verbatim shape.
    opened = harness.model(
        {
            "operation": "open",
            "case_id": "questioning-seed-synthesis",
            "decision": "candidate",
            "top_unknown": (
                "What do the seed literatures establish about the value of "
                "questioning and perceptions of question-asking machines, and "
                "does that evidence survive into agentic AI?"
            ),
        },
        session_id="new",
    )
    retrieved = opened["prior_case_lessons"]
    case_ids = [item["case_id"] for item in retrieved]
    assert "prior-delegation-synthesis" in case_ids, (
        f"word-family retrieval must surface the directly relevant lesson; "
        f"got: {case_ids}"
    )
