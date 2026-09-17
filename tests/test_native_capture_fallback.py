"""Fail-open and correlation tests for native observation capture."""

from __future__ import annotations

import contextvars
import json
import threading

from epistemic_harness import plugin
from epistemic_harness.harness import current_native_capture_state


def _start_probe(tmp_path, monkeypatch, case_id: str):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    plugin.reset_for_tests()
    harness = plugin._get_harness()
    session_id = f"{case_id}-session"
    harness.model(
        {"operation": "open", "case_id": case_id, "top_unknown": "Whether the result is usable"},
        session_id=session_id,
    )
    harness.probe(
        {
            "operation": "commit",
            "purpose": "learn",
            "unknown_or_goal": "Whether the result is usable",
            "tool_name": "native-fixture",
            "tool_args": {"query": "fixture"},
            "predicted_outcomes": [{"outcome": "usable result"}],
            "why_this_action": "The fixture is the designated action",
            "would_change_belief": "The result is incompatible",
            "authority_domain": "system_state",
        },
        session_id=session_id,
    )
    return harness, session_id


def _events(harness, case_id: str):
    return harness.store.list_events(case_id)


def _start_lock_holder(lock):
    held = threading.Event()
    release = threading.Event()

    def hold():
        lock.acquire()
        held.set()
        release.wait(timeout=5)
        lock.release()

    thread = threading.Thread(target=hold, daemon=True)
    thread.start()
    assert held.wait(timeout=2)
    return thread, release


def test_missing_post_tool_call_boundary_records_unknown_and_does_not_reexecute(
    tmp_path, monkeypatch
):
    harness, session_id = _start_probe(tmp_path, monkeypatch, "missing-post")
    executions = []

    def next_call(args):
        executions.append(args)
        # Simulate the transform boundary without a post_tool_call observer.
        plugin.transform_tool_result(
            tool_name="native-fixture",
            args=args,
            result=json.dumps({"value": "handler-result"}),
            session_id=session_id,
            task_id="missing-post-task",
            tool_call_id="missing-post-call",
        )
        return "DISPLAY-RESULT"

    returned = harness.tool_execution_middleware(
        tool_name="native-fixture",
        args={"query": "fixture"},
        next_call=next_call,
        session_id=session_id,
        native_middleware=True,
        task_id="missing-post-task",
        tool_call_id="missing-post-call",
    )
    observed = next(event for event in _events(harness, "missing-post") if event["type"] == "probe_observed")
    assert returned == "DISPLAY-RESULT"
    assert executions == [{"query": "fixture"}]
    assert observed["payload"]["result_status"] == "unknown"
    assert observed["payload"]["raw_result_provenance"] == "unknown"
    assert observed["payload"]["observation_boundary"] == "transform_tool_result_input"


def test_failed_or_unrelated_post_observer_cannot_replace_unknown_capture(
    tmp_path, monkeypatch
):
    harness, session_id = _start_probe(tmp_path, monkeypatch, "failed-post")
    original = harness.record_native_tool_result

    def fail_observer(**_kwargs):
        raise RuntimeError("observer fixture failure")

    monkeypatch.setattr(harness, "record_native_tool_result", fail_observer)

    def next_call(args):
        # The unrelated callback is rejected before it can reach the store.
        plugin.post_tool_call(
            tool_name="native-fixture",
            args=args,
            result=json.dumps({"value": "handler-result"}),
            session_id=session_id,
            task_id="failed-post-task",
            tool_call_id="unrelated-call",
        )
        # The host isolates this observer failure; the transform boundary still
        # has to preserve an unknown provenance result.
        try:
            plugin.post_tool_call(
                tool_name="native-fixture",
                args=args,
                result=json.dumps({"value": "handler-result"}),
                session_id=session_id,
                task_id="failed-post-task",
                tool_call_id="failed-post-call",
            )
        except RuntimeError:
            pass
        plugin.transform_tool_result(
            tool_name="native-fixture",
            args=args,
            result=json.dumps({"value": "handler-result"}),
            session_id=session_id,
            task_id="failed-post-task",
            tool_call_id="failed-post-call",
        )
        return "DISPLAY-RESULT"

    returned = harness.tool_execution_middleware(
        tool_name="native-fixture",
        args={"query": "fixture"},
        next_call=next_call,
        session_id=session_id,
        native_middleware=True,
        task_id="failed-post-task",
        tool_call_id="failed-post-call",
    )
    monkeypatch.setattr(harness, "record_native_tool_result", original)
    late = plugin.post_tool_call(
        tool_name="native-fixture",
        args={"query": "fixture"},
        result=json.dumps({"value": "late-result"}),
        session_id=session_id,
        task_id="failed-post-task",
        tool_call_id="failed-post-call",
    )
    events = _events(harness, "failed-post")
    observations = [event for event in events if event["type"] == "probe_observed"]
    assert returned == "DISPLAY-RESULT"
    assert late is None
    assert len(observations) == 1
    assert observations[0]["payload"]["raw_result_provenance"] == "unknown"


def test_native_capture_scope_records_unknown_when_no_observer_fires(tmp_path, monkeypatch):
    harness, session_id = _start_probe(tmp_path, monkeypatch, "no-native-observer")
    executions = []

    returned = harness.tool_execution_middleware(
        tool_name="native-fixture",
        args={"query": "fixture"},
        next_call=lambda args: executions.append(args) or "DIRECT-RESULT",
        session_id=session_id,
        native_middleware=True,
        task_id="no-native-observer-task",
        tool_call_id="no-native-observer-call",
    )

    observations = [
        event for event in _events(harness, "no-native-observer")
        if event["type"] == "probe_observed"
    ]
    assert returned == "DIRECT-RESULT"
    assert executions == [{"query": "fixture"}]
    assert current_native_capture_state() is None
    assert len(observations) == 1
    assert observations[0]["payload"]["observation_boundary"] == "tool_execution_middleware_return"
    assert observations[0]["payload"]["raw_result_provenance"] == "unknown"
    assert observations[0]["payload"]["result_status"] == "unknown"
    assert observations[0]["payload"]["result"] == "DIRECT-RESULT"

    # A post event arriving after the middleware scope has ended cannot upgrade
    # the bounded unknown observation.
    assert plugin.post_tool_call(
        tool_name="native-fixture",
        args={"query": "fixture"},
        result="LATE-RAW-RESULT",
        session_id=session_id,
        task_id="no-native-observer-task",
        tool_call_id="no-native-observer-call",
    ) is None
    assert len([
        event for event in _events(harness, "no-native-observer")
        if event["type"] == "probe_observed"
    ]) == 1


def test_failed_native_post_observer_still_finalizes_and_uses_unknown_fallback(
    tmp_path, monkeypatch
):
    harness, session_id = _start_probe(tmp_path, monkeypatch, "failed-native-post")

    def fail_observer(**_kwargs):
        raise RuntimeError("observer fixture failure")

    monkeypatch.setattr(harness, "record_native_tool_result", fail_observer)

    def next_call(args):
        try:
            plugin.post_tool_call(
                tool_name="native-fixture",
                args=args,
                result=json.dumps({"value": "handler-result"}),
                session_id=session_id,
                task_id="failed-native-post-task",
                tool_call_id="failed-native-post-call",
            )
        except RuntimeError:
            pass
        return "DISPLAY-RESULT"

    returned = harness.tool_execution_middleware(
        tool_name="native-fixture",
        args={"query": "fixture"},
        next_call=next_call,
        session_id=session_id,
        native_middleware=True,
        task_id="failed-native-post-task",
        tool_call_id="failed-native-post-call",
    )

    observations = [
        event for event in _events(harness, "failed-native-post")
        if event["type"] == "probe_observed"
    ]
    assert returned == "DISPLAY-RESULT"
    assert current_native_capture_state() is None
    assert len(observations) == 1
    assert observations[0]["payload"]["observation_boundary"] == "tool_execution_middleware_return"
    assert observations[0]["payload"]["raw_result_provenance"] == "unknown"


def test_native_fallback_does_not_wait_on_python_state_lock(tmp_path, monkeypatch):
    harness, session_id = _start_probe(tmp_path, monkeypatch, "state-lock-fallback")
    holder = {}

    def next_call(_args):
        state = current_native_capture_state()
        assert state is not None
        thread, release = _start_lock_holder(state.lock)
        holder.update(thread=thread, release=release, state=state)
        return "STATE-LOCKED-RESULT"

    returned = harness.tool_execution_middleware(
        tool_name="native-fixture",
        args={"query": "fixture"},
        next_call=next_call,
        session_id=session_id,
        native_middleware=True,
        task_id="state-lock-fallback-task",
        tool_call_id="state-lock-fallback-call",
    )
    assert returned == "STATE-LOCKED-RESULT"
    assert current_native_capture_state() is None
    assert holder["state"].finalized is True
    assert not [event for event in _events(harness, "state-lock-fallback") if event["type"] == "probe_observed"]
    holder["release"].set()
    holder["thread"].join(timeout=2)
    assert not holder["thread"].is_alive()


def test_native_fallback_does_not_wait_on_python_store_lock(tmp_path, monkeypatch):
    harness, session_id = _start_probe(tmp_path, monkeypatch, "store-lock-fallback")
    holder = {}

    def next_call(_args):
        thread, release = _start_lock_holder(harness.store._lock)
        holder.update(thread=thread, release=release)
        return "STORE-LOCKED-RESULT"

    returned = harness.tool_execution_middleware(
        tool_name="native-fixture",
        args={"query": "fixture"},
        next_call=next_call,
        session_id=session_id,
        native_middleware=True,
        task_id="store-lock-fallback-task",
        tool_call_id="store-lock-fallback-call",
    )
    assert returned == "STORE-LOCKED-RESULT"
    assert current_native_capture_state() is None
    assert not [event for event in _events(harness, "store-lock-fallback") if event["type"] == "probe_observed"]
    holder["release"].set()
    holder["thread"].join(timeout=2)
    assert not holder["thread"].is_alive()


def test_native_post_observer_fails_open_on_python_harness_cache_lock(tmp_path, monkeypatch):
    harness, session_id = _start_probe(tmp_path, monkeypatch, "cache-lock-fallback")
    with_lock = _start_lock_holder(plugin._HARNESS_LOCK)
    try:
        returned = harness.tool_execution_middleware(
            tool_name="native-fixture",
            args={"query": "fixture"},
            next_call=lambda args: plugin.post_tool_call(
                tool_name="native-fixture",
                args=args,
                result=json.dumps({"value": "handler-result"}),
                session_id=session_id,
                task_id="cache-lock-fallback-task",
                tool_call_id="cache-lock-fallback-call",
            ) or "CACHE-LOCKED-RESULT",
            session_id=session_id,
            native_middleware=True,
            task_id="cache-lock-fallback-task",
            tool_call_id="cache-lock-fallback-call",
        )
    finally:
        with_lock[1].set()
        with_lock[0].join(timeout=2)
    assert returned == "CACHE-LOCKED-RESULT"
    assert current_native_capture_state() is None
    observations = [
        event for event in _events(harness, "cache-lock-fallback")
        if event["type"] == "probe_observed"
    ]
    assert len(observations) == 1
    assert observations[0]["payload"]["raw_result_provenance"] == "unknown"


def test_native_exposure_fails_open_on_python_store_lock(tmp_path, monkeypatch):
    harness, session_id = _start_probe(tmp_path, monkeypatch, "exposure-lock-fallback")
    holder = {}

    def next_call(args):
        plugin.post_tool_call(
            tool_name="native-fixture",
            args=args,
            result=json.dumps({"value": "raw-result"}),
            session_id=session_id,
            task_id="exposure-lock-fallback-task",
            tool_call_id="exposure-lock-fallback-call",
        )
        thread, release = _start_lock_holder(harness.store._lock)
        holder.update(thread=thread, release=release)
        return "DISPLAY-RESULT"

    returned = harness.tool_execution_middleware(
        tool_name="native-fixture",
        args={"query": "fixture"},
        next_call=next_call,
        session_id=session_id,
        native_middleware=True,
        task_id="exposure-lock-fallback-task",
        tool_call_id="exposure-lock-fallback-call",
    )
    assert returned == "DISPLAY-RESULT"
    assert current_native_capture_state() is None
    events = _events(harness, "exposure-lock-fallback")
    assert len([event for event in events if event["type"] == "probe_observed"]) == 1
    assert not [event for event in events if event["type"] == "probe_exposed"]
    holder["release"].set()
    holder["thread"].join(timeout=2)
    assert not holder["thread"].is_alive()


def test_finalized_native_scope_rejects_late_callback_with_reused_empty_ids(tmp_path, monkeypatch):
    harness, session_id = _start_probe(tmp_path, monkeypatch, "reused-empty-ids")
    saved_context = {}

    def next_call(_args):
        saved_context["context"] = contextvars.copy_context()
        return "FIRST-RESULT"

    returned = harness.tool_execution_middleware(
        tool_name="native-fixture",
        args={"query": "fixture"},
        next_call=next_call,
        session_id=session_id,
        native_middleware=True,
    )
    assert returned == "FIRST-RESULT"
    assert current_native_capture_state() is None

    late = saved_context["context"].run(
        plugin.post_tool_call,
        tool_name="native-fixture",
        args={"query": "fixture"},
        result="REUSED-ID-LATE-RESULT",
        session_id=session_id,
    )
    observations = [
        event for event in _events(harness, "reused-empty-ids")
        if event["type"] == "probe_observed"
    ]
    assert late is None
    assert len(observations) == 1
    assert observations[0]["payload"]["raw_result_provenance"] == "unknown"


def test_native_capture_scope_finalizes_when_action_raises(tmp_path, monkeypatch):
    harness, session_id = _start_probe(tmp_path, monkeypatch, "native-action-error")

    def next_call(_args):
        raise RuntimeError("action fixture failure")

    try:
        harness.tool_execution_middleware(
            tool_name="native-fixture",
            args={"query": "fixture"},
            next_call=next_call,
            session_id=session_id,
            native_middleware=True,
            task_id="native-action-error-task",
            tool_call_id="native-action-error-call",
        )
    except RuntimeError as exc:
        assert str(exc) == "action fixture failure"
    else:
        raise AssertionError("action exception was swallowed")
    assert current_native_capture_state() is None
    observations = [
        event for event in _events(harness, "native-action-error")
        if event["type"] == "probe_observed"
    ]
    assert len(observations) == 1
    assert observations[0]["payload"]["observation_boundary"] == "tool_execution_middleware_return"
    assert observations[0]["payload"]["raw_result_provenance"] == "unknown"
