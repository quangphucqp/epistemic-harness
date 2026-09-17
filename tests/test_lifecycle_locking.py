# AI-assisted contribution; maintained by Epistemic Harness contributors.
"""Bounded lifecycle lock-acquisition regressions."""

from __future__ import annotations

import logging
import subprocess
import sys
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator

import pytest

from epistemic_harness import plugin
from epistemic_harness.harness import EpistemicHarness
from epistemic_harness.locking import LockTimeoutError, acquire_lock, lock_deadline


_REAL_FLOCK_HOLDER = (
    "import fcntl, sys\n"
    "handle = open(sys.argv[1], 'a+')\n"
    "fcntl.flock(handle.fileno(), fcntl.LOCK_EX)\n"
    "print('locked', flush=True)\n"
    "sys.stdin.readline()\n"
    "fcntl.flock(handle.fileno(), fcntl.LOCK_UN)\n"
    "handle.close()\n"
)


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


@contextmanager
def _held_process_lock(path: Path) -> Iterator[subprocess.Popen[str]]:
    process = subprocess.Popen(
        [sys.executable, "-c", _REAL_FLOCK_HOLDER, str(path)],
        stdin=subprocess.PIPE,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
    )
    try:
        assert process.stdout is not None
        assert process.stdout.readline().strip() == "locked"
        yield process
    finally:
        if process.poll() is None:
            assert process.stdin is not None
            process.stdin.write("\n")
            process.stdin.flush()
        process.wait(timeout=5)


def _snapshot(harness: EpistemicHarness, case_id: str) -> tuple[bytes, bytes]:
    case_dir = harness.store._case_dir(case_id)
    return (
        (case_dir / "case.md").read_bytes(),
        (case_dir / "events.jsonl").read_bytes(),
    )


def _seed_plugin_case(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
    *,
    case_id: str = "research-case",
    session_id: str = "session-a",
) -> EpistemicHarness:
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    plugin.reset_for_tests()
    harness = plugin._get_harness()
    harness.model(_open_args(case_id), session_id=session_id)
    return harness


def _run_async(callback: Callable[[], object]) -> tuple[threading.Thread, threading.Event, list[BaseException]]:
    done = threading.Event()
    errors: list[BaseException] = []

    def run() -> None:
        try:
            callback()
        except BaseException as exc:  # pragma: no cover - assertion below reports it
            errors.append(exc)
        finally:
            done.set()

    thread = threading.Thread(target=run, daemon=True)
    thread.start()
    return thread, done, errors


def _invoke_lifecycle(name: str, session_id: str) -> None:
    if name == "session_finalize":
        plugin.on_session_finalize(session_id=session_id, reason="session_boundary")
    elif name == "session_reset":
        plugin.on_session_reset(old_session_id=session_id, new_session_id="new-session")
    elif name == "subagent_stop":
        plugin.on_subagent_stop(child_session_id=session_id, child_status="failed")
    else:  # pragma: no cover - parametrization guard
        raise AssertionError(name)


def test_lock_deadline_resets_and_failed_acquisition_does_not_release_owner():
    lock = threading.Lock()
    lock.acquire()
    try:
        with lock_deadline(0.01):
            with pytest.raises(LockTimeoutError, match="deadline exceeded"):
                with acquire_lock(lock, label="test lock"):
                    pytest.fail("a held lock was acquired")
        # The failed acquisition did not release the lock held by this caller.
        lock.release()
        with acquire_lock(lock, label="test lock"):
            pass
    finally:
        # The successful context above released it; this is only defensive.
        if lock.acquire(blocking=False):
            lock.release()


@pytest.mark.parametrize("callback_name", ["session_reset", "session_finalize", "subagent_stop"])
def test_each_lifecycle_callback_fails_open_on_real_held_process_lock(
    tmp_path, monkeypatch, caplog, callback_name
):
    harness = _seed_plugin_case(monkeypatch, tmp_path, session_id="lifecycle-session")
    before = _snapshot(harness, "research-case")
    lock_path = harness.store.root / ".store.lock"

    with _held_process_lock(lock_path) as holder:
        with caplog.at_level(logging.WARNING, logger="epistemic_harness.plugin"):
            thread, done, errors = _run_async(
                lambda: _invoke_lifecycle(callback_name, "lifecycle-session")
            )
            assert done.wait(2.0), f"{callback_name} remained blocked on a held flock"
        assert holder.poll() is None
        assert errors == []
        assert _snapshot(harness, "research-case") == before

    thread.join(timeout=2.0)
    assert not thread.is_alive()
    time.sleep(0.05)
    assert _snapshot(harness, "research-case") == before
    assert any(
        callback_name in record.message
        and "lock acquisition exceeded" in record.message
        and "pause was not persisted" in record.message
        for record in caplog.records
    )

    # No retry was queued. A later uncontended callback performs the intended
    # synchronous pause.
    _invoke_lifecycle(callback_name, "lifecycle-session")
    assert harness.store.get_case("research-case")["status"] == "paused"


def test_cold_harness_construction_is_bounded_by_real_held_process_lock(
    tmp_path, monkeypatch, caplog
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    seed = EpistemicHarness(tmp_path / "epistemic-harness")
    seed.model(_open_args(), session_id="cold-session")
    before = _snapshot(seed, "research-case")
    plugin.reset_for_tests()

    with _held_process_lock(seed.store.root / ".store.lock") as holder:
        thread, done, errors = _run_async(
            lambda: plugin.on_session_finalize(
                session_id="cold-session", reason="session_boundary"
            )
        )
        assert done.wait(2.0), "cold lifecycle callback remained blocked on a held flock"
        assert holder.poll() is None
        assert errors == []
        assert plugin._HARNESSES == {}
        assert _snapshot(seed, "research-case") == before

    thread.join(timeout=2.0)
    assert not thread.is_alive()
    time.sleep(0.05)
    assert _snapshot(seed, "research-case") == before

    # The failed cold construction did not poison the cache; the next ordinary
    # lifecycle call constructs a warm harness and persists the pause.
    plugin.on_session_finalize(session_id="cold-session", reason="session_boundary")
    assert plugin._get_harness().store.get_case("research-case")["status"] == "paused"
    assert any(
        "session_finalize" in record.message
        and "pause was not persisted" in record.message
        for record in caplog.records
    )


def test_global_harness_cache_lock_is_bounded_and_has_no_late_write(tmp_path, monkeypatch, caplog):
    harness = _seed_plugin_case(monkeypatch, tmp_path, session_id="global-session")
    before = _snapshot(harness, "research-case")
    held = threading.Event()
    release = threading.Event()

    def hold_cache_lock() -> None:
        plugin._HARNESS_LOCK.acquire()
        held.set()
        release.wait(timeout=5)
        plugin._HARNESS_LOCK.release()

    holder = threading.Thread(target=hold_cache_lock, daemon=True)
    holder.start()
    assert held.wait(2.0)
    with caplog.at_level(logging.WARNING, logger="epistemic_harness.plugin"):
        thread, done, errors = _run_async(
            lambda: plugin.on_session_finalize(
                session_id="global-session", reason="session_boundary"
            )
        )
        assert done.wait(2.0), "lifecycle callback remained blocked on the harness cache lock"
    assert errors == []
    assert _snapshot(harness, "research-case") == before
    release.set()
    holder.join(timeout=2.0)
    thread.join(timeout=2.0)
    assert not holder.is_alive() and not thread.is_alive()
    time.sleep(0.05)
    assert _snapshot(harness, "research-case") == before
    assert any(
        "session_finalize" in record.message
        and "lock acquisition exceeded" in record.message
        for record in caplog.records
    )

    plugin.on_session_finalize(session_id="global-session", reason="session_boundary")
    assert harness.store.get_case("research-case")["status"] == "paused"


def test_store_rlock_timeout_preserves_owner_and_same_thread_reentrancy(tmp_path, monkeypatch, caplog):
    harness = _seed_plugin_case(monkeypatch, tmp_path, session_id="rlock-session")
    before = _snapshot(harness, "research-case")
    held = threading.Event()
    release = threading.Event()

    def hold_store_lock() -> None:
        harness.store._lock.acquire()
        held.set()
        release.wait(timeout=5)
        harness.store._lock.release()

    holder = threading.Thread(target=hold_store_lock, daemon=True)
    holder.start()
    assert held.wait(2.0)
    with caplog.at_level(logging.WARNING, logger="epistemic_harness.plugin"):
        thread, done, errors = _run_async(
            lambda: plugin.on_session_finalize(
                session_id="rlock-session", reason="session_boundary"
            )
        )
        assert done.wait(2.0), "lifecycle callback remained blocked on the store RLock"
    assert errors == []
    assert _snapshot(harness, "research-case") == before
    release.set()
    holder.join(timeout=2.0)
    thread.join(timeout=2.0)
    assert not holder.is_alive() and not thread.is_alive()
    assert _snapshot(harness, "research-case") == before
    assert any("lock acquisition exceeded" in record.message for record in caplog.records)

    with harness.store._lock:
        plugin.on_session_finalize(session_id="rlock-session", reason="session_boundary")
        assert harness.store.get_case("research-case")["status"] == "paused"
        acquired_by_other_thread: list[bool] = []
        probe_done = threading.Event()

        def probe_outer_lock() -> None:
            acquired = harness.store._lock.acquire(timeout=0.1)
            acquired_by_other_thread.append(acquired)
            if acquired:
                harness.store._lock.release()
            probe_done.set()

        probe = threading.Thread(target=probe_outer_lock)
        probe.start()
        assert probe_done.wait(1.0)
        probe.join(timeout=1.0)
        assert acquired_by_other_thread == [False]


def test_unscoped_normal_model_operation_retains_blocking_lock_semantics(tmp_path):
    harness = EpistemicHarness(tmp_path / "epistemic-harness")
    lock_path = harness.store.root / ".store.lock"
    with _held_process_lock(lock_path):
        thread, done, errors = _run_async(
            lambda: harness.model(_open_args("normal-case"), session_id="normal-session")
        )
        assert not done.wait(0.15)
    assert done.wait(2.0)
    thread.join(timeout=2.0)
    assert not thread.is_alive()
    assert errors == []
    assert harness.store.get_case("normal-case")["status"] == "active"


def test_native_manager_dispatch_uses_bounded_callback_on_real_held_flock(
    tmp_path, monkeypatch, caplog
):
    try:
        from hermes_cli.plugins import PluginContext, PluginManager, PluginManifest
    except ImportError:
        pytest.skip("frozen Hermes core is unavailable")

    harness = _seed_plugin_case(monkeypatch, tmp_path, session_id="native-session")
    manager = PluginManager(scope_key=str(tmp_path))
    context = PluginContext(
        PluginManifest(
            name="epistemic-harness-native-test",
            version="1.0.0",
            description="isolated native callback test",
            source="user",
        ),
        manager,
    )
    context.register_hook("on_session_finalize", plugin.on_session_finalize)
    before = _snapshot(harness, "research-case")

    with _held_process_lock(harness.store.root / ".store.lock") as holder:
        with caplog.at_level(logging.WARNING, logger="epistemic_harness.plugin"):
            thread, done, errors = _run_async(
                lambda: manager.invoke_hook(
                    "on_session_finalize",
                    session_id="native-session",
                    reason="session_boundary",
                )
            )
            assert done.wait(2.0), "native manager callback remained blocked on a held flock"
        assert holder.poll() is None
        assert errors == []
        assert _snapshot(harness, "research-case") == before

    thread.join(timeout=2.0)
    assert not thread.is_alive()
    assert _snapshot(harness, "research-case") == before
    assert any(
        "session_finalize" in record.message
        and "pause was not persisted" in record.message
        for record in caplog.records
    )

    manager.invoke_hook(
        "on_session_finalize",
        session_id="native-session",
        reason="session_boundary",
    )
    assert harness.store.get_case("research-case")["status"] == "paused"
