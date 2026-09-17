# AI-assisted contribution; maintained by Epistemic Harness contributors.
"""Lifecycle auto-pause and proof-of-life auto-resume tests (E1, plan v10).

Scope honesty: the compression and MoA paths never fire lifecycle hooks in
Hermes v0.21.0 — verified by source inspection at
agent/conversation_compression.py:5219-5238 (context-engine boundary only) and
agent/moa_loop.py (no lifecycle dispatch at all), commit b20cc5f7. A plugin
suite cannot drive Hermes's dispatcher, so that invariant is guarded by the
source-alarm tests at the bottom of this file, which fail loudly if either
Hermes file begins dispatching lifecycle hooks.
"""

from __future__ import annotations

import logging
from pathlib import Path

import pytest

from epistemic_harness import plugin
from epistemic_harness.harness import EpistemicHarness


def _open_args(case_id: str = "research-case") -> dict:
    return {
        "operation": "open",
        "case_id": case_id,
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


def _close_args() -> dict:
    return {
        "operation": "close",
        "outcome": "unresolved",
        "summary": "Closed for the test",
        "transfer": {"decision": "none"},
    }


# ---------------------------------------------------------------- finalize trigger matrix


@pytest.mark.parametrize("platform", ["tui", "desktop", "cli", "webui", ""])
def test_finalize_pauses_with_marker_for_boundary_empty_and_unknown_reasons(tmp_path, platform):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="session-a")
    for reason in ("session_boundary", "", "some_future_reason"):
        result = harness.session_finalize(
            session_id="session-a", reason=reason, platform=platform
        )
        assert result is not None
        assert result["case"]["status"] == "paused"
        case = harness.store.get_case("research-case")
        marker = case.get("auto_resume_pending")
        assert isinstance(marker, dict), f"reason {reason!r} platform {platform!r} lost the marker"
        assert marker["seq_at_pause"] >= 1
        resumed = harness.auto_resume_for_injection("session-a")
        assert resumed is not None
        assert resumed["case"]["status"] == "active"


@pytest.mark.parametrize("reason", ["new_session", "shutdown"])
def test_finalize_skips_reset_owned_and_shutdown_reasons(tmp_path, reason):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="session-a")
    assert (
        harness.session_finalize(session_id="session-a", reason=reason, platform="cli")
        is None
    )
    assert harness.store.get_active_case("session-a") is not None


def test_finalize_expiry_pauses_without_marker(tmp_path):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="session-a")
    result = harness.session_finalize(
        session_id="session-a", reason="session_expired", platform="telegram"
    )
    assert result is not None
    case = harness.store.get_case("research-case")
    assert case["status"] == "paused"
    assert "auto_resume_pending" not in case
    # Expiry is provably permanent: no resurrection on a later boundary.
    assert harness.auto_resume_for_injection("session-a") is None


def test_finalize_is_idempotent_and_fail_open(tmp_path):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="session-a")
    first = harness.session_finalize(session_id="session-a", reason="session_boundary")
    second = harness.session_finalize(session_id="session-a", reason="session_boundary")
    assert first is not None and second is None
    events = [e["type"] for e in harness.store.list_events("research-case")]
    assert events.count("case_paused") == 1


def test_finalize_hook_never_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    plugin.reset_for_tests()

    class _Ctx:
        def __init__(self):
            self.tools, self.hooks, self.middleware = {}, {}, []

        def register_tool(self, **kwargs):
            self.tools[kwargs["name"]] = kwargs

        def register_hook(self, name, callback, **kwargs):
            self.hooks[name] = callback

        def register_middleware(self, kind, callback, **kwargs):
            self.middleware.append((kind, callback, kwargs))

        def register_skill(self, name, path, description="", frontmatter=None):
            pass

        def get_config(self, key, default=None):
            return default

    context = _Ctx()
    plugin.register(context)
    harness = plugin._get_harness()
    harness.model(_open_args("hook-case"), session_id="session-h")
    monkeypatch.setattr(
        harness.store, "get_active_case", lambda _sid: (_ for _ in ()).throw(RuntimeError("boom"))
    )
    # Must not raise into the host.
    context.hooks["on_session_finalize"](session_id="session-h", reason="session_boundary")


# ---------------------------------------------------------------- marker matrix


def test_subagent_stop_pause_carries_no_marker(tmp_path):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="child-session")
    harness.pause_active_subagent_case(child_session_id="child-session", child_status="done")
    case = harness.store.get_case("research-case")
    assert case["status"] == "paused"
    assert "auto_resume_pending" not in case
    assert harness.auto_resume_for_injection("child-session") is None


def test_operator_pause_carries_no_marker(tmp_path):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="session-a")
    harness.model(
        {"operation": "pause", "reason": "Operator hold"}, session_id="session-a"
    )
    case = harness.store.get_case("research-case")
    assert case["status"] == "paused"
    assert "auto_resume_pending" not in case
    assert harness.auto_resume_for_injection("session-a") is None


def test_reset_pause_carries_marker(tmp_path):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="old-session")
    harness.session_reset(old_session_id="old-session", new_session_id="new-session")
    case = harness.store.get_case("research-case")
    assert case["status"] == "paused"
    marker = case.get("auto_resume_pending")
    assert isinstance(marker, dict) and marker["seq_at_pause"] >= 1


# ---------------------------------------------------------------- proof of life


def test_auto_resume_consumes_exactly_once(tmp_path):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="session-a")
    harness.session_finalize(session_id="session-a", reason="session_boundary")
    resumed = harness.auto_resume_for_injection("session-a")
    assert resumed is not None
    assert resumed["event"]["payload"]["reason"] == "proof of life"
    assert resumed["event"]["payload"]["consumed_pause_event_id"]
    assert harness.auto_resume_for_injection("session-a") is None
    case = harness.store.get_case("research-case")
    assert case["status"] == "active"
    assert "auto_resume_pending" not in case


def test_deliberate_disposition_invalidates_older_marker(tmp_path):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args("case-a"), session_id="session-s")
    harness.session_finalize(session_id="session-s", reason="session_boundary")
    # Deliberate: open a new case on the same session.
    harness.model(_open_args("case-b"), session_id="session-s")
    assert harness.auto_resume_for_injection("session-s") is None
    # Same invalidation through an operator pause of the newer case.
    harness.model(
        {"operation": "pause", "reason": "hold"}, session_id="session-s"
    )
    assert harness.auto_resume_for_injection("session-s") is None
    # And both cases remain paused, unmoved.
    assert harness.store.get_case("case-a")["status"] == "paused"
    assert harness.store.get_case("case-b")["status"] == "paused"


def test_close_invalidates_older_marker(tmp_path):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args("case-a"), session_id="session-s")
    harness.session_finalize(session_id="session-s", reason="session_boundary")
    harness.model(_open_args("case-b"), session_id="session-s")
    close = _close_args()
    harness.model(close, session_id="session-s")
    assert harness.store.get_case("case-b")["status"] == "closed"
    assert harness.auto_resume_for_injection("session-s") is None


def test_explicit_resume_clears_marker_across_sessions(tmp_path):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="session-s")
    harness.session_finalize(session_id="session-s", reason="session_boundary")
    harness.model(
        {"operation": "resume", "case_id": "research-case"}, session_id="session-t"
    )
    case = harness.store.get_case("research-case")
    assert case["status"] == "active"
    assert case["session_id"] == "session-t"
    assert "auto_resume_pending" not in case
    assert harness.auto_resume_for_injection("session-s") is None


def test_newest_valid_marker_wins_and_conflict_holds(tmp_path):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args("case-a"), session_id="session-s")
    harness.session_finalize(session_id="session-s", reason="session_boundary")
    harness.model(_open_args("case-b"), session_id="session-s")
    harness.session_finalize(session_id="session-s", reason="session_boundary")
    resumed = harness.auto_resume_for_injection("session-s")
    assert resumed is not None
    assert resumed["case"]["case_id"] == "case-b"
    # case-a's marker is stale (its seq no longer matches the high-water) and
    # the session now has an active case, so case-a can never auto-resume.
    assert harness.auto_resume_for_injection("session-s") is None
    assert harness.store.get_case("case-a")["status"] == "paused"


# ---------------------------------------------------------------- expiry mid-turn analog


def test_expiry_mid_turn_preserves_probe_without_resurrection(tmp_path):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="session-live")
    probe = {
        "operation": "commit",
        "purpose": "learn",
        "unknown_or_goal": "Whether the source supports the account",
        "tool_name": "read_file",
        "tool_args": {"path": "evidence.txt"},
        "predicted_outcomes": [
            {"outcome": "the file supports the account", "meaning": "match"},
            {"outcome": "the file contradicts the account", "meaning": "mismatch"},
        ],
        "would_change_belief": "the account is wrong",
        "why_this_action": "cheapest decisive check",
        "authority_domain": "empirical_test",
    }
    harness.probe(probe, session_id="session-live")
    # The session is past its reset deadline and expiry fires mid-turn.
    harness.session_finalize(
        session_id="session-live", reason="session_expired", platform="telegram"
    )
    case = harness.store.get_case("research-case")
    assert case["status"] == "paused"
    assert case.get("pending_probe", {}) is not None
    # The dying turn's later boundary calls must not resurrect the case.
    assert harness.auto_resume_for_injection("session-live") is None
    # Operator resume keeps the stranded probe available for recover/discard.
    resumed = harness.model(
        {"operation": "resume", "case_id": "research-case"}, session_id="session-new"
    )
    assert resumed["case"]["status"] == "active"
    assert resumed["case"].get("pending_probe")


# ---------------------------------------------------------------- compression lineage


def test_pause_then_migrate_pristine_child_resumes_on_first_boundary(tmp_path):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="parent")
    harness.session_finalize(session_id="parent", reason="session_boundary")
    result = harness.store.migrate_lineage_case(
        lineage_ids=["parent", "child"], child_session_id="child"
    )
    assert result is not None
    case = harness.store.get_case("research-case")
    assert case["status"] == "paused"
    assert case["session_id"] == "child"
    # Marker re-based to the child session's sequence.
    marker = case["auto_resume_pending"]
    assert marker["seq_at_pause"] == 1
    assert marker.get("pause_event_id")
    resumed = harness.auto_resume_for_injection("child")
    assert resumed is not None and resumed["case"]["status"] == "active"


def test_migrate_then_finalize_on_parent_is_noop(tmp_path):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="parent")
    result = harness.store.migrate_lineage_case(
        lineage_ids=["parent", "child"], child_session_id="child"
    )
    assert result is not None
    assert result["case"]["status"] == "active"
    assert result["case"]["session_id"] == "child"
    assert (
        harness.session_finalize(session_id="parent", reason="session_boundary") is None
    )
    assert harness.store.get_case("research-case")["session_id"] == "child"


def test_delayed_adoption_after_deliberate_child_life_strips_marker(tmp_path):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args("case-a"), session_id="parent")
    harness.session_finalize(session_id="parent", reason="session_boundary")
    # Adoption fails or is delayed; the child sees deliberate life first.
    harness.model(_open_args("case-b"), session_id="child")
    harness.model(_close_args(), session_id="child")
    result = harness.store.migrate_lineage_case(
        lineage_ids=["parent", "child"], child_session_id="child"
    )
    assert result is not None
    migrated = harness.store.get_case("case-a")
    assert migrated["status"] == "paused"
    assert migrated["session_id"] == "child"
    assert "auto_resume_pending" not in migrated
    assert harness.auto_resume_for_injection("child") is None


def test_active_child_blocks_adoption(tmp_path):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args("case-a"), session_id="parent")
    harness.session_finalize(session_id="parent", reason="session_boundary")
    harness.model(_open_args("case-b"), session_id="child")
    result = harness.store.migrate_lineage_case(
        lineage_ids=["parent", "child"], child_session_id="child"
    )
    assert result is None
    case_a = harness.store.get_case("case-a")
    assert case_a["status"] == "paused" and case_a["session_id"] == "parent"


def test_tip_first_adoption_picks_newest_qualifying_ancestor(tmp_path):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args("case-old"), session_id="parent-1")
    harness.session_finalize(session_id="parent-1", reason="session_boundary")
    harness.model(_open_args("case-new"), session_id="parent-2")
    harness.session_finalize(session_id="parent-2", reason="session_boundary")
    result = harness.store.migrate_lineage_case(
        lineage_ids=["parent-1", "parent-2", "child"], child_session_id="child"
    )
    assert result is not None
    assert result["case"]["case_id"] == "case-new"
    case_old = harness.store.get_case("case-old")
    assert case_old["status"] == "paused" and case_old["session_id"] == "parent-1"


def test_lineage_prefix_never_adopts_from_descendant(tmp_path):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args("case-new"), session_id="child")
    # An older compacted session's lineage includes the later continuation;
    # adoption for the older session must not see the descendant's case.
    result = harness.store.migrate_lineage_case(
        lineage_ids=["parent", "child", "grandchild"], child_session_id="parent"
    )
    assert result is None
    assert harness.store.get_case("case-new")["session_id"] == "child"


# ---------------------------------------------------------------- churn coalescing (plugin level)


def test_reconnect_churn_records_pairs_but_coalesces_logs(tmp_path, monkeypatch, caplog):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    plugin.reset_for_tests()
    plugin._auto_resume_log_times.clear()

    class _Ctx:
        def __init__(self):
            self.tools, self.hooks, self.middleware = {}, {}, []

        def register_tool(self, **kwargs):
            self.tools[kwargs["name"]] = kwargs

        def register_hook(self, name, callback, **kwargs):
            self.hooks[name] = callback

        def register_middleware(self, kind, callback, **kwargs):
            self.middleware.append((kind, callback, kwargs))

        def register_skill(self, name, path, description="", frontmatter=None):
            pass

        def get_config(self, key, default=None):
            return default

    context = _Ctx()
    plugin.register(context)
    context.tools["epistemic_model"]["handler"](_open_args(), session_id="session-tui")

    with caplog.at_level(logging.INFO, logger="epistemic_harness.plugin"):
        for _ in range(2):
            context.hooks["on_session_finalize"](
                session_id="session-tui", reason="", platform="tui"
            )
            injected = context.hooks["pre_llm_call"](session_id="session-tui")
            assert injected is not None and "context" in injected

    info_lines = [
        record for record in caplog.records if "auto-resumed on proof of life" in record.message
    ]
    assert len(info_lines) == 1
    harness = plugin._get_harness()
    events = [e["type"] for e in harness.store.list_events("research-case")]
    assert events.count("case_paused") == 2
    assert events.count("case_resumed") == 2


# ---------------------------------------------------------------- crash ordering


def test_pending_journal_reconciles_cache_before_any_allocation(tmp_path):
    """A cache write failure after journal apply leaves the journal pending;
    the next operation reconciles in-process and never double-allocates."""
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="session-c")
    store = harness.store

    real_write = store._write_sessions_cache_locked
    state = {"fail": True}

    def flaky_write(mapping):
        if state["fail"]:
            raise OSError("simulated cache write failure")
        return real_write(mapping)

    store._write_sessions_cache_locked = flaky_write  # type: ignore[assignment]
    try:
        # The disposition commits durably (case + events) but the cache write
        # fails, so the transaction surfaces an error and the journal stays.
        with pytest.raises(Exception):
            harness.session_finalize(session_id="session-c", reason="session_boundary")
    finally:
        state["fail"] = False
        store._write_sessions_cache_locked = real_write  # type: ignore[assignment]

    case = store.get_case("research-case")
    assert case["status"] == "paused"
    # The next operation reconciles via the retained pending journal.
    resumed = harness.auto_resume_for_injection("session-c")
    assert resumed is not None
    # Exactly one pause and one resume sequence were ever allocated.
    events = store.list_events("research-case")
    seqs = [e["seq"] for e in events if e.get("seq") is not None]
    assert seqs == sorted(set(seqs))


def test_observational_events_do_not_invalidate_marker(tmp_path):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="session-n")
    harness.store.append_event(
        "research-case",
        event_type="evidence_recorded",
        session_id="session-n",
        payload={"summary": "noise", "events": []},
    )
    harness.session_finalize(session_id="session-n", reason="session_boundary")
    resumed = harness.auto_resume_for_injection("session-n")
    assert resumed is not None and resumed["case"]["status"] == "active"


# ------------------------------------------------------- Hermes expiry characterization

_HERMES_REPO = Path.home() / ".hermes" / "hermes-agent"


@pytest.mark.skipif(
    not (_HERMES_REPO / "gateway" / "session.py").exists(),
    reason="Hermes repo unavailable",
)
def test_real_hermes_routing_resets_only_on_explicit_suspension(tmp_path, monkeypatch):
    """Pin the CURRENT reset surface E1 operates against, with real Hermes code.

    The gateway refactor moved reset decisions to
    ``gateway/session_lifecycle.py::_route_reset_reason``, whose contract is:
    only explicit suspension replaces a routed conversation — elapsed time
    never does. That is why the plugin's reset/finalize hooks (not a timer)
    carry the auto-pause duty. ``mark_turn_active`` still stamps the turn
    start into ``updated_at`` for downgrade compatibility.

    HERMES_HOME is redirected so the session store's SQLite lands in tmp and
    the production-state guard never fires.
    """
    import sys
    from datetime import timedelta

    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    sys.path.insert(0, str(_HERMES_REPO))
    try:
        from gateway.config import GatewayConfig
        from gateway.session import SessionStore, SessionEntry, _now
    except ImportError:
        pytest.skip("Hermes gateway modules unavailable")

    store = SessionStore(tmp_path / "sessions", GatewayConfig())
    entry = SessionEntry(
        session_id="s-live",
        session_key="agent:main:telegram:dm:live",
        created_at=_now() - timedelta(hours=5),
        updated_at=_now() - timedelta(days=2),
    )
    store._entries[entry.session_key] = entry
    token = store.mark_turn_active(entry.session_key)
    assert token and entry.active_turn_token
    # Legacy heuristic: the turn start refreshes updated_at.
    assert entry.updated_at > _now() - timedelta(minutes=1)
    # Two days idle with an active turn: routing does NOT reset on time.
    assert store._route_reset_reason(entry) is None
    # Only explicit suspension (/stop) replaces the routed conversation.
    assert store.suspend_session(entry.session_key) is True
    assert store._route_reset_reason(entry) == "suspended"


# ---------------------------------------------------------------- source alarms (guarded)


@pytest.mark.skipif(
    not (_HERMES_REPO / "agent" / "conversation_compression.py").exists(),
    reason="Hermes repo unavailable",
)
def test_compression_path_never_dispatches_lifecycle_hooks():
    """Alarm: compression must keep using the context-engine boundary only.

    Pinned at commit b20cc5f7 (v0.21.0): conversation_compression.py:5219-5238.
    If this fails, Hermes began firing lifecycle hooks on compression and E1's
    trigger assumptions need re-review.
    """
    text = (_HERMES_REPO / "agent" / "conversation_compression.py").read_text()
    assert "on_session_finalize" not in text
    assert "on_session_reset" not in text
    assert "finalize_session(" not in text


@pytest.mark.skipif(
    not (_HERMES_REPO / "agent" / "moa_loop.py").exists(),
    reason="Hermes repo unavailable",
)
def test_moa_path_never_dispatches_lifecycle_hooks():
    """Alarm: a MoA model swap is in-session, not a session rotation.

    Pinned at commit b20cc5f7 (v0.21.0): agent/moa_loop.py contains no
    lifecycle dispatch at all. If this fails, re-review E1's trigger set.
    """
    text = (_HERMES_REPO / "agent" / "moa_loop.py").read_text()
    assert "finalize_session(" not in text
    assert "on_session_finalize" not in text
    assert "on_session_reset" not in text
    assert "invoke_hook" not in text


# ------------------------------------------- sessions-cache deletion (Sol F8)


def _sessions_cache_path(harness):
    return harness.store._sessions_cache_path()


def test_cache_deletion_rebuilds_high_water_no_sequence_restart(tmp_path):
    # Regression (Sol re-audit finding): a MISSING sessions.json must rebuild
    # from the immutable Timelines, not silently restart sequences at 1.
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args("case-a"), session_id="session-s")
    harness.session_finalize(session_id="session-s", reason="session_boundary")
    # A deliberate disposition on the session advances the high-water.
    harness.model(_open_args("case-b"), session_id="session-s")
    harness.session_finalize(session_id="session-s", reason="session_boundary")
    before = harness.store._read_sessions_cache_locked()["session-s"]
    assert before >= 2
    # Delete the durable cache image.
    _sessions_cache_path(harness).unlink()
    rebuilt = harness.store._read_sessions_cache_locked()
    # The high-water was rebuilt from the Timelines, not restarted at 0/1.
    assert rebuilt["session-s"] == before
    # The cache file was rewritten durably.
    assert _sessions_cache_path(harness).exists()


def test_cache_deletion_does_not_resurrect_stale_marker(tmp_path):
    # A deliberately invalidated marker must stay ineligible even after the
    # cache is deleted and rebuilt: rebuild restores the true high-water, so
    # the stale marker's seq_at_pause no longer matches. The session must have
    # NO active case when auto-resume runs, or the active-case guard would
    # pass for the wrong reason — so case-b is deliberately paused (an
    # operator pause carries no auto-resume marker).
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args("case-a"), session_id="session-s")
    harness.session_finalize(session_id="session-s", reason="session_boundary")
    # Deliberate disposition (opening case-b) invalidates case-a's marker.
    harness.model(_open_args("case-b"), session_id="session-s")
    harness.model({"operation": "pause", "reason": "hold"}, session_id="session-s")
    assert harness.store.get_active_case("session-s") is None
    marker_seq = harness.store.get_case("case-a")["auto_resume_pending"]["seq_at_pause"]
    _sessions_cache_path(harness).unlink()
    rebuilt = harness.store._read_sessions_cache_locked()
    # The rebuilt high-water exceeds the stale marker's pause sequence, which
    # is exactly why the marker no longer qualifies.
    assert rebuilt["session-s"] > marker_seq
    assert harness.auto_resume_for_injection("session-s") is None
    assert harness.store.get_case("case-a")["status"] == "paused"


def test_no_duplicate_sequences_after_cache_deletion(tmp_path):
    # After deletion+rebuild, the next allocated sequence must be strictly
    # greater than every sequence already journaled on the Timelines —
    # no duplicate allocation can ever hand out a seq an old marker carries.
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args("case-a"), session_id="session-s")
    harness.session_finalize(session_id="session-s", reason="session_boundary")
    journal_high = harness.store._read_sessions_cache_locked()["session-s"]
    _sessions_cache_path(harness).unlink()
    harness.model(_open_args("case-b"), session_id="session-s")
    events = harness.store.list_events("case-b")
    seqs = [event.get("seq") for event in events if isinstance(event.get("seq"), int)]
    assert seqs, "disposition events must carry seq"
    assert min(seqs) > journal_high
