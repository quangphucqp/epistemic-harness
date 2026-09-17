# AI-assisted contribution; maintained by Epistemic Harness contributors.
"""Bounded case lifecycle and model revision logic."""

from __future__ import annotations

from contextvars import ContextVar
from dataclasses import dataclass, field
import hashlib
import json
import logging
import math
import re
import threading
from datetime import datetime, timezone
from functools import wraps
from pathlib import Path
from typing import Any, Callable

from .locking import LockTimeoutError, acquire_lock, lock_deadline
from .store import CaseStore, ProbeExecutionBusyError, StoreError

logger = logging.getLogger(__name__)


def _bounded_native_observer(method: Callable[..., Any]) -> Callable[..., Any]:
    """Run best-effort native bookkeeping without waiting on store locks."""
    @wraps(method)
    def bounded(self: Any, *args: Any, **kwargs: Any) -> Any:
        try:
            with lock_deadline(0):
                return method(self, *args, **kwargs)
        except LockTimeoutError:
            logger.warning(
                "Epistemic native observer skipped: lock unavailable; observation was not persisted"
            )
            return None

    return bounded

_REQUIRED_MODEL_FIELDS = {
    "decision",
    "stopping_condition",
    "state_grounding",
    "mechanisms",
    "alternatives",
    "top_unknown",
    "current_plan",
}
_OPTIONAL_MODEL_FIELDS = {
    "prior_case_ids",
    "retrieved_prior_lessons",
    "applied_prior_lessons",
}
_MODEL_FIELDS = _REQUIRED_MODEL_FIELDS | _OPTIONAL_MODEL_FIELDS
_REVISION_SERVER_MANAGED_FIELDS = {"retrieved_prior_lessons", "prior_case_ids"}
_REVISION_MODEL_FIELDS = _MODEL_FIELDS - _REVISION_SERVER_MANAGED_FIELDS
_USER_AUTHORITY_DOMAINS = {
    "user_preference",
    "user_intent",
    "private_context",
    "user_decision",
    "user_testimony",
}
_EXTERNAL_AUTHORITY_DOMAINS = {"external_source", "system_state", "empirical_test"}
_AUTHORITY_DOMAINS = _USER_AUTHORITY_DOMAINS | _EXTERNAL_AUTHORITY_DOMAINS
_SOURCE_ROLES = {"primary", "synthesis", "secondary", "unknown"}
_ACCESS_SCOPES = {"snippet", "abstract", "partial", "full", "data"}
_LIMITED_ACCESS_SCOPES = {"snippet", "abstract", "partial"}
_FAILED_OBSERVATION_STATUSES = {
    "error",
    "tool_error",
    "unavailable",
    "interrupted",
    "failed",
    "failure",
    "cancelled",
    "canceled",
    "timeout",
}
_NATIVE_RESULT_PROVENANCE = {"unknown", "native_handler_result"}
_INLINE_OBSERVATION_LIMIT = 12_000
_SECRET_KEY_NAMES = {
    "password", "passwd", "secret", "token", "apikey", "accesstoken",
    "refreshtoken", "authtoken", "authorization", "cookie", "credential",
    "privatekey", "sessionkey",
}
_LESSON_STOPWORDS = {
    "about", "after", "again", "against", "also", "because", "before",
    "being", "between", "could", "does", "from", "have", "into", "itself",
    "might", "more", "most", "other", "should", "than", "that", "their",
    "there", "these", "this", "those", "through", "under", "using", "what",
    "when", "where", "which", "while", "with", "would",
}


class HarnessError(RuntimeError):
    """Raised when an epistemic transition violates the case contract."""


class _NativeCaptureStale(RuntimeError):
    """Raised when a native callback loses its middleware identity before a write."""


@dataclass
class _NativeCaptureState:
    """Correlation state shared by native post/transform observers."""

    session_id: str
    task_id: str
    tool_call_id: str
    turn_id: str
    api_request_id: str
    case_id: str
    probe_id: str
    tool_name: str
    raw_captured: bool = False
    unknown_captured: bool = False
    transform_seen: bool = False
    finalized: bool = False
    lock: threading.Lock = field(default_factory=threading.Lock, repr=False)

    def matches(
        self,
        *,
        session_id: str,
        task_id: str,
        tool_call_id: str,
        turn_id: str,
        api_request_id: str,
        tool_name: str,
    ) -> bool:
        return (
            not self.finalized
            and self.session_id == str(session_id or "")
            and self.task_id == str(task_id or "")
            and self.tool_call_id == str(tool_call_id or "")
            and self.turn_id == str(turn_id or "")
            and self.api_request_id == str(api_request_id or "")
            and self.tool_name == str(tool_name or "")
        )


_NATIVE_CAPTURE_STATE: ContextVar[_NativeCaptureState | None] = ContextVar(
    "epistemic_native_capture_state",
    default=None,
)


def current_native_capture_state() -> _NativeCaptureState | None:
    """Return the current native execution correlation, if one is active."""
    return _NATIVE_CAPTURE_STATE.get()


class EpistemicHarness:
    """Service implementing the minimal model–probe–compare–revise loop."""

    def __init__(self, root: str | Path):
        self.store = CaseStore(root)

    def model(self, args: dict[str, Any], *, session_id: str) -> dict[str, Any]:
        operation = str(args.get("operation") or "").strip().lower()
        if operation not in {
            "open",
            "list",
            "show",
            "record_evidence",
            "revise",
            "pause",
            "resume",
            "close",
        }:
            raise HarnessError(
                "operation must be open, list, show, record_evidence, revise, pause, resume, or close"
            )
        try:
            self.store.recover_prepared_compression(session_id)
            if operation == "open":
                return self._open(args, session_id=session_id)
            if operation == "list":
                return self._list(args)
            if operation == "show":
                case_id = str(args.get("case_id") or "").strip()
                active = self.store.get_active_case(session_id)
                if not case_id:
                    case = active or self._require_active(session_id)
                elif active is not None and case_id == active["case_id"]:
                    case = active
                else:
                    # A caller may inspect a settled child while retaining a
                    # different active case. An active target still requires
                    # its owning session, so this never creates a cross-session
                    # authorization path.
                    case = self.store.get_case(case_id)
                    if case.get("status") == "active":
                        raise HarnessError(
                            "active case is bound to another session; pause it before cross-session inspection"
                        )
                    if case.get("status") not in {"paused", "closed"}:
                        raise HarnessError(
                            "only paused or closed cases can be inspected across an active case"
                        )
                return {"case": case, "replay": self.replay(case["case_id"])}
            if operation == "record_evidence":
                return self._record_evidence(args, session_id=session_id)
            if operation == "revise":
                return self._revise(args, session_id=session_id)
            if operation == "pause":
                return self._pause(args, session_id=session_id)
            if operation == "resume":
                return self._resume(args, session_id=session_id)
            if operation == "close":
                return self._close(args, session_id=session_id)
        except StoreError as exc:
            raise HarnessError(str(exc)) from exc
        raise HarnessError(
            "operation must be open, list, show, record_evidence, revise, pause, resume, or close"
        )

    def _list(self, args: dict[str, Any]) -> dict[str, Any]:
        """Expose a bounded, read-only view of the cross-session case portfolio."""
        status = str(args.get("status_filter") or "").strip().lower() or None
        if status is not None and status not in {"active", "paused", "closed"}:
            raise HarnessError("status_filter must be active, paused, or closed")
        session_filter = str(args.get("session_id_filter") or "").strip() or None
        raw_limit = args.get("limit", 50)
        if isinstance(raw_limit, bool) or not isinstance(raw_limit, int):
            raise HarnessError("list limit must be an integer from 1 to 200")
        if raw_limit < 1 or raw_limit > 200:
            raise HarnessError("list limit must be an integer from 1 to 200")
        try:
            matches = self.store.list_case_summaries(
                status=status,
                session_id=session_filter,
            )
        except StoreError as exc:
            raise HarnessError(str(exc)) from exc
        cases = matches[:raw_limit]
        return {
            "cases": cases,
            "returned": len(cases),
            "total_matching": len(matches),
            "case_errors": self.store.case_errors(),
            "filters": {
                "status": status,
                "session_id": session_filter,
                "limit": raw_limit,
            },
        }

    def session_reset(
        self,
        *,
        old_session_id: str | None,
        new_session_id: str | None,
    ) -> dict[str, Any] | None:
        """Pause an old session's active case without importing it into the new one."""
        if not old_session_id:
            return None
        try:
            return self.store.lifecycle_pause(
                old_session_id,
                event_payload={
                    "reason": "Hermes session reset",
                    "old_session_id": old_session_id,
                    "new_session_id": new_session_id,
                    "auto_resumable": True,
                },
                marker={
                    "reason": "Hermes session reset",
                    "seq_at_pause": None,
                },
            )
        except StoreError as exc:
            raise HarnessError(str(exc)) from exc

    def session_finalize(
        self,
        *,
        session_id: str | None,
        reason: str | None,
        platform: str | None = None,
    ) -> dict[str, Any] | None:
        """Pause a session's active case on a lifecycle finalization.

        Trigger discipline (verified against the Hermes v0.21.0 runtimes):
        - ``new_session`` is skipped: the paired gateway reset carries both
          ids and owns that pause through ``session_reset``.
        - ``shutdown`` is skipped: the process is stopping and the session
          resumes in place on restart; pausing would add a pause/resume event
          pair per restart for no continuity gain.
        - Every other reason — ``session_boundary`` (CLI /new), the empty
          reason (TUI/desktop end-of-session, where terminality is unknowable
          at hook time), and any unknown future reason — auto-pauses with a
          proof-of-life marker, so a session that turns out alive self-heals
          on its next LLM call and a dead session leaves the case paused.
        - ``session_expired`` pauses WITHOUT the marker: expiry is the one
          provably permanent transition (the gateway persists
          expiry_finalized), and it can fire mid-turn, so a dying turn must
          not resurrect the case. A pending probe is preserved by the pause
          for explicit recover/discard after an operator resume.
        """
        session_id = str(session_id or "").strip()
        if not session_id:
            return None
        reason_text = str(reason or "")
        if reason_text in ("new_session", "shutdown"):
            return None
        try:
            auto_resumable = reason_text != "session_expired"
            return self.store.lifecycle_pause(
                session_id,
                event_payload={
                    "reason": f"Hermes session finalize ({reason_text or 'no-reason'})",
                    "session_id": session_id,
                    "platform": str(platform or ""),
                    "auto_resumable": auto_resumable,
                },
                marker=(
                    {
                        "reason": f"Hermes session finalize ({reason_text or 'no-reason'})",
                        "platform": str(platform or ""),
                        "seq_at_pause": None,
                    }
                    if auto_resumable
                    else None
                ),
            )
        except StoreError as exc:
            raise HarnessError(str(exc)) from exc

    def auto_resume_for_injection(self, session_id: str) -> dict[str, Any] | None:
        """Proof-of-life resume for a paused, marker-eligible case.

        Called from the pre-LLM boundary when the session has no active case.
        The replay gate runs INSIDE the store's locked transaction
        (``try_auto_resume(replay_ok=...)``): a case whose Timeline no longer
        validates is left paused with its marker intact, never resurrected,
        and no other process ever sees an active-but-invalid case.
        """
        session_id = str(session_id or "").strip()
        if not session_id:
            return None

        def _replay_ok(case: dict[str, Any]) -> bool:
            # Lock-free: this callable runs INSIDE try_auto_resume's locked
            # transaction (store _process_lock + _lock held), so it must NOT
            # re-enter harness.replay() — verify_event_chain re-takes the
            # process lock and self-deadlocks. validate_case_chain_and_snapshot
            # reads the events/snapshot files without taking either lock.
            return self.store.validate_case_chain_and_snapshot(case)

        return self.store.try_auto_resume(session_id, replay_ok=_replay_ok)

    def pause_active_subagent_case(
        self,
        *,
        child_session_id: str | None,
        child_status: str | None,
    ) -> dict[str, Any] | None:
        """Retire an unfinished child-owned case when its subagent stops."""
        session_id = str(child_session_id or "").strip()
        if not session_id:
            return None
        try:
            case = self.store.get_active_case(session_id)
            if case is None:
                return None
            pending = case.get("pending_probe")
            case["status"] = "paused"
            return self.store.transition_case(
                case,
                event_type="case_paused",
                event_payload={
                    "reason": "delegated epistemic worker stopped before closing its case",
                    "lifecycle": "subagent_stop",
                    "child_status": _redact(str(child_status or "unknown").strip())[:80],
                    "pending_probe_status": (
                        pending.get("status") if isinstance(pending, dict) else None
                    ),
                },
            )
        except StoreError as exc:
            raise HarnessError(str(exc)) from exc

    def require_settled_before_session_switch(
        self,
        *,
        old_session_id: str | None,
        new_session_id: str | None,
        reason: str,
    ) -> None:
        """Reject transcript-preserving/session-resume switches while a case is active."""
        if not old_session_id or old_session_id == new_session_id:
            return
        try:
            case = self.store.get_active_case(old_session_id)
        except StoreError as exc:
            raise HarnessError(str(exc)) from exc
        if case is not None:
            raise HarnessError(
                f"cannot {reason or 'switch sessions'} while case {case['case_id']!r} is active; "
                "explicitly pause or close the case first"
            )

    def session_compress(
        self,
        *,
        old_session_id: str | None,
        new_session_id: str | None,
    ) -> dict[str, Any]:
        """Move an active case to the compression child session."""
        if not old_session_id or not new_session_id:
            raise HarnessError("compression requires old and new session IDs")
        if old_session_id == new_session_id:
            case = self.store.get_active_case(old_session_id)
            return {"status": "unchanged", "case": case}
        try:
            migrated = self.store.recover_prepared_compression(new_session_id)
            if migrated is None:
                migrated = self.store.migrate_active_case(
                    old_session_id=old_session_id,
                    new_session_id=new_session_id,
                )
        except StoreError as exc:
            raise HarnessError(str(exc)) from exc
        if migrated is None:
            return {"status": "dormant", "case": None}
        migrated["status"] = "migrated"
        return migrated

    def prepare_compression(
        self,
        *,
        old_session_id: str | None,
        new_session_id: str | None,
        session_exists: Callable[[str], bool] | None = None,
    ) -> dict[str, Any]:
        """Persist a recoverable migration intent before core session rotation."""
        if not old_session_id or not new_session_id:
            raise HarnessError("compression preparation requires old and new session IDs")
        try:
            # A published child may reach another compression boundary before
            # its ordinary pre-LLM/tool recovery hook runs. Recover it here so
            # the next prepared boundary cannot skip the case binding.
            self.store.recover_prepared_compression(old_session_id)
            existing = self.store.get_prepared_compression_from(old_session_id)
            if existing and existing.get("new_session_id") != new_session_id:
                stale_target = str(existing.get("new_session_id") or "")
                if session_exists is None:
                    raise HarnessError(
                        "cannot replace a prepared compression without core proof that its target is absent"
                    )
                try:
                    target_exists = bool(session_exists(stale_target))
                except Exception as exc:
                    raise HarnessError(
                        f"could not prove prepared target absence: {type(exc).__name__}: {exc}"
                    ) from exc
                if target_exists:
                    raise HarnessError(
                        "prepared compression target exists; recover that child instead of replacing it"
                    )
                self.store.abort_prepared_compression(
                    old_session_id=old_session_id,
                    new_session_id=stale_target,
                )
            prepared = self.store.prepare_compression(
                old_session_id=old_session_id,
                new_session_id=new_session_id,
            )
        except StoreError as exc:
            raise HarnessError(str(exc)) from exc
        if prepared is None:
            return {"status": "dormant", "case": None}
        return {"status": "prepared", "case_id": prepared["case_id"]}

    def abort_compression(
        self,
        *,
        old_session_id: str | None,
        new_session_id: str | None,
    ) -> dict[str, Any]:
        """Restore a prepared case binding before core reopens the parent."""
        if not old_session_id or not new_session_id:
            raise HarnessError("compression abort requires old and new session IDs")
        try:
            aborted = self.store.abort_prepared_compression(
                old_session_id=old_session_id,
                new_session_id=new_session_id,
            )
        except StoreError as exc:
            raise HarnessError(str(exc)) from exc
        if aborted is None:
            return {"status": "dormant", "case": None}
        return {"status": "aborted", "case": aborted["case"]}

    def probe(self, args: dict[str, Any], *, session_id: str) -> dict[str, Any]:
        """Commit or compare one epistemic probe."""
        operation = str(args.get("operation") or "").strip().lower()
        if operation not in {"commit", "compare", "recover", "discard"}:
            raise HarnessError("operation must be commit, compare, recover, or discard")
        try:
            self.store.recover_prepared_compression(session_id)
            if operation == "commit":
                return self._commit_probe(args, session_id=session_id)
            if operation == "compare":
                return self._compare_probe(args, session_id=session_id)
            if operation == "recover":
                return self._recover_probe(args, session_id=session_id)
            if operation == "discard":
                return self._discard_probe(args, session_id=session_id)
        except StoreError as exc:
            raise HarnessError(str(exc)) from exc
        raise HarnessError("operation must be commit, compare, recover, or discard")

    def tool_execution_middleware(
        self,
        *,
        tool_name: str,
        args: dict[str, Any],
        next_call: Any,
        session_id: str = "",
        native_middleware: bool = False,
        **_metadata: Any,
    ) -> Any:
        """Capture an explicitly committed probe without taxing ordinary tool use."""
        if tool_name in {"epistemic_model", "epistemic_probe"}:
            return next_call(args)

        if not session_id:
            return next_call(args)

        native_state: _NativeCaptureState | None = None
        native_token = None
        try:
            self.store.recover_prepared_compression(session_id)
            case = self.store.get_active_case(session_id)
            if case is None:
                return next_call(args)
            replay = self.replay(case["case_id"])
            if not replay["timeline"].get("valid"):
                raise HarnessError(
                    f"Timeline integrity failure: {replay['timeline'].get('error', 'unknown error')}"
                )
            probe = case.get("pending_probe")
            if (
                isinstance(probe, dict)
                and probe.get("status") == "executing"
                and probe.get("tool_name") == tool_name
            ):
                return _blocked(
                    "pending probe execution is incomplete; recover or discard it "
                    "before retrying the same action"
                )
            if (
                not isinstance(probe, dict)
                or probe.get("status") != "committed"
                or probe.get("tool_name") != tool_name
            ):
                return next_call(args)
            if not replay["usable"]:
                return _blocked(
                    f"committed probe cannot execute until recovered or discarded: "
                    f"{replay['reason']}"
                )
            expected_args_sha256 = str(probe.get("tool_args_sha256") or "")
            if not expected_args_sha256:
                expected_args_sha256 = _sha256(_canonical(probe.get("tool_args")))
            supplied_args_digest = (
                self.store.commitment_digest(_canonical(args))
                if probe.get("tool_args_commitment_scheme") == "hmac-sha256-v1"
                else _sha256(_canonical(args))
            )
            probe["execution_tool_args_sha256"] = supplied_args_digest
            probe["execution_args_match_commitment"] = (
                supplied_args_digest == expected_args_sha256
            )

            try:
                lease = self.store.probe_execution_lease(case["case_id"])
                lease.__enter__()
            except ProbeExecutionBusyError:
                return _blocked(
                    "another invocation is already executing the committed probe; "
                    "this action was not run"
                )
            try:
                # Parallel tool calls can all read the same committed snapshot
                # before one of them claims the probe lease. Re-read after the
                # claim so a sibling never transitions stale state or turns an
                # ordinary concurrency collision into a critical policy error.
                current = self.store.get_active_case(session_id)
                current_probe = (
                    current.get("pending_probe")
                    if isinstance(current, dict)
                    else None
                )
                if (
                    not isinstance(current, dict)
                    or current.get("case_id") != case.get("case_id")
                    or not isinstance(current_probe, dict)
                    or current_probe.get("status") != "committed"
                    or current_probe.get("probe_id") != probe.get("probe_id")
                    or current_probe.get("tool_name") != tool_name
                ):
                    return _blocked(
                        "another invocation already claimed or changed the committed "
                        "probe; this action was not run"
                    )
                case = current
                probe = current_probe
                expected_args_sha256 = str(probe.get("tool_args_sha256") or "")
                if not expected_args_sha256:
                    expected_args_sha256 = _sha256(_canonical(probe.get("tool_args")))
                supplied_args_digest = (
                    self.store.commitment_digest(_canonical(args))
                    if probe.get("tool_args_commitment_scheme") == "hmac-sha256-v1"
                    else _sha256(_canonical(args))
                )
                probe["execution_tool_args_sha256"] = supplied_args_digest
                probe["execution_args_match_commitment"] = (
                    supplied_args_digest == expected_args_sha256
                )
                probe["status"] = "executing"
                execution_tool_call_id = str(_metadata.get("tool_call_id") or "")
                execution_task_id = str(_metadata.get("task_id") or "")
                execution_turn_id = str(_metadata.get("turn_id") or "")
                execution_api_request_id = str(_metadata.get("api_request_id") or "")
                if execution_tool_call_id:
                    probe["execution_tool_call_id"] = execution_tool_call_id
                if execution_task_id:
                    probe["execution_task_id"] = execution_task_id
                if execution_turn_id:
                    probe["execution_turn_id"] = execution_turn_id
                if execution_api_request_id:
                    probe["execution_api_request_id"] = execution_api_request_id
                started = self.store.transition_case(
                    case,
                    event_type="probe_started",
                    event_payload={
                        "probe_id": probe["probe_id"],
                        "tool_name": tool_name,
                        "args_sha256": supplied_args_digest,
                        "committed_args_sha256": expected_args_sha256,
                        "args_match_commitment": probe[
                            "execution_args_match_commitment"
                        ],
                        **(
                            {"tool_call_id": execution_tool_call_id}
                            if execution_tool_call_id
                            else {}
                        ),
                        **(
                            {"task_id": execution_task_id}
                            if execution_task_id
                            else {}
                        ),
                        **(
                            {"turn_id": execution_turn_id}
                            if execution_turn_id
                            else {}
                        ),
                        **(
                            {"api_request_id": execution_api_request_id}
                            if execution_api_request_id
                            else {}
                        ),
                    },
                    event_id_target=("pending_probe", "started_event_id"),
                )
                case = started["case"]
                if native_middleware:
                    native_state = _NativeCaptureState(
                        session_id=str(session_id or ""),
                        task_id=str(_metadata.get("task_id") or ""),
                        tool_call_id=str(_metadata.get("tool_call_id") or ""),
                        turn_id=str(_metadata.get("turn_id") or ""),
                        api_request_id=str(_metadata.get("api_request_id") or ""),
                        case_id=str(case.get("case_id") or ""),
                        probe_id=str(probe.get("probe_id") or ""),
                        tool_name=tool_name,
                    )
                    native_token = _NATIVE_CAPTURE_STATE.set(native_state)
                try:
                    result = next_call(args)
                except Exception as exc:
                    if native_state is None:
                        self._capture_observation(
                            case,
                            tool_name=tool_name,
                            result_text=f"{type(exc).__name__}: {exc}",
                            result_status="error",
                        )
                    else:
                        try:
                            self.record_unknown_native_tool_result(
                                tool_name=tool_name,
                                args=args,
                                result=f"{type(exc).__name__}: {exc}",
                                session_id=session_id,
                                task_id=str(_metadata.get("task_id") or ""),
                                tool_call_id=str(_metadata.get("tool_call_id") or ""),
                                turn_id=str(_metadata.get("turn_id") or ""),
                                api_request_id=str(_metadata.get("api_request_id") or ""),
                                observation_boundary="tool_execution_middleware_return",
                            )
                        except Exception:
                            logging.getLogger(__name__).warning(
                                "native epistemic observation fallback failed", exc_info=True
                            )
                    raise
                if native_state is not None:
                    if native_state.raw_captured:
                        try:
                            self.record_tool_result_exposure(
                                tool_name=tool_name,
                                args=args,
                                result=result,
                                session_id=session_id,
                                tool_call_id=str(_metadata.get("tool_call_id") or ""),
                                task_id=str(_metadata.get("task_id") or ""),
                                turn_id=str(_metadata.get("turn_id") or ""),
                                api_request_id=str(_metadata.get("api_request_id") or ""),
                                exposure_boundary="tool_execution_middleware_return",
                            )
                        except Exception:
                            # The raw observation is already durable. Exposure
                            # bookkeeping is best-effort and cannot block the
                            # host's already-completed action.
                            logging.getLogger(__name__).warning(
                                "native epistemic exposure observation failed", exc_info=True
                            )
                    if not native_state.raw_captured and not native_state.unknown_captured:
                        try:
                            self.record_unknown_native_tool_result(
                                tool_name=tool_name,
                                args=args,
                                result=result,
                                session_id=session_id,
                                task_id=str(_metadata.get("task_id") or ""),
                                tool_call_id=str(_metadata.get("tool_call_id") or ""),
                                turn_id=str(_metadata.get("turn_id") or ""),
                                api_request_id=str(_metadata.get("api_request_id") or ""),
                                observation_boundary="tool_execution_middleware_return",
                            )
                        except Exception:
                            logging.getLogger(__name__).warning(
                                "native epistemic unknown-boundary fallback failed", exc_info=True
                            )
                    return result
                result_status = _classify_tool_result(result)
                self._capture_observation(
                    case,
                    tool_name=tool_name,
                    result_text=str(result),
                    result_status=result_status,
                )
                return result
            finally:
                if native_state is not None:
                    # Native callbacks may run in abandoned host-hook workers.
                    # Invalidate the shared object without taking its lock: a
                    # callback holding that lock must never hold finalization up.
                    native_state.finalized = True
                    if native_token is not None:
                        _NATIVE_CAPTURE_STATE.reset(native_token)
                lease.__exit__(None, None, None)
        except StoreError as exc:
            raise HarnessError(str(exc)) from exc

    def record_native_tool_result(
        self,
        *,
        tool_name: str,
        args: dict[str, Any] | None,
        result: Any,
        session_id: str,
        task_id: str = "",
        tool_call_id: str = "",
        turn_id: str = "",
        api_request_id: str = "",
    ) -> dict[str, Any] | None:
        """Capture the handler result reported by native ``post_tool_call``."""
        state = current_native_capture_state()
        if state is None or not state.matches(
            session_id=session_id,
            task_id=task_id,
            tool_call_id=tool_call_id,
            turn_id=turn_id,
            api_request_id=api_request_id,
            tool_name=tool_name,
        ):
            return None
        try:
            # Observer persistence is best-effort. A timed-out host callback
            # must not leave a worker waiting on the correlation lock or the
            # store locks after the middleware has returned.
            with lock_deadline(0):
                with acquire_lock(state.lock, label="native post_tool_call state lock"):
                    # Recheck after acquiring the state lock: an abandoned
                    # callback may have inherited this object before scope
                    # finalization invalidated it.
                    if not state.matches(
                        session_id=session_id,
                        task_id=task_id,
                        tool_call_id=tool_call_id,
                        turn_id=turn_id,
                        api_request_id=api_request_id,
                        tool_name=tool_name,
                    ):
                        return None
                    if state.raw_captured or state.unknown_captured:
                        return None
                    if not isinstance(args, dict):
                        return None
                    case = self.store.get_active_case(session_id)
                    probe = case.get("pending_probe") if isinstance(case, dict) else None
                    if (
                        not isinstance(case, dict)
                        or case.get("case_id") != state.case_id
                        or not isinstance(probe, dict)
                        or probe.get("probe_id") != state.probe_id
                        or probe.get("status") != "executing"
                        or probe.get("tool_name") != tool_name
                        or not _probe_args_match(self.store, probe, args)
                    ):
                        return None
                    if not state.matches(
                        session_id=session_id,
                        task_id=task_id,
                        tool_call_id=tool_call_id,
                        turn_id=turn_id,
                        api_request_id=api_request_id,
                        tool_name=tool_name,
                    ):
                        return None
                    observed = self._capture_observation(
                        case,
                        tool_name=tool_name,
                        result_text=_result_text(result),
                        result_status=_classify_tool_result(result),
                        observation_boundary="post_tool_call",
                        raw_result_provenance="native_handler_result",
                        final_exposure_status="unknown",
                        capture_guard=lambda: state.matches(
                            session_id=session_id,
                            task_id=task_id,
                            tool_call_id=tool_call_id,
                            turn_id=turn_id,
                            api_request_id=api_request_id,
                            tool_name=tool_name,
                        ),
                    )
                    state.raw_captured = True
                    return observed
        except LockTimeoutError:
            logger.warning(
                "Epistemic native post_tool_call observation skipped: lock unavailable; "
                "observation was not persisted"
            )
            return None
        except _NativeCaptureStale:
            return None

    def record_unknown_native_tool_result(
        self,
        *,
        tool_name: str,
        args: dict[str, Any] | None,
        result: Any,
        session_id: str,
        task_id: str = "",
        tool_call_id: str = "",
        turn_id: str = "",
        api_request_id: str = "",
        observation_boundary: str = "unknown",
    ) -> dict[str, Any] | None:
        """Record a native result whose raw handler provenance is unavailable."""
        state = current_native_capture_state()
        if state is None or not state.matches(
            session_id=session_id,
            task_id=task_id,
            tool_call_id=tool_call_id,
            turn_id=turn_id,
            api_request_id=api_request_id,
            tool_name=tool_name,
        ):
            return None
        try:
            with lock_deadline(0):
                with acquire_lock(state.lock, label="native unknown-result state lock"):
                    if not state.matches(
                        session_id=session_id,
                        task_id=task_id,
                        tool_call_id=tool_call_id,
                        turn_id=turn_id,
                        api_request_id=api_request_id,
                        tool_name=tool_name,
                    ):
                        return None
                    if state.raw_captured or state.unknown_captured:
                        return None
                    if not isinstance(args, dict):
                        return None
                    case = self.store.get_active_case(session_id)
                    probe = case.get("pending_probe") if isinstance(case, dict) else None
                    if (
                        not isinstance(case, dict)
                        or case.get("case_id") != state.case_id
                        or not isinstance(probe, dict)
                        or probe.get("probe_id") != state.probe_id
                        or probe.get("status") != "executing"
                        or probe.get("tool_name") != tool_name
                        or not _probe_args_match(self.store, probe, args)
                    ):
                        return None
                    if not state.matches(
                        session_id=session_id,
                        task_id=task_id,
                        tool_call_id=tool_call_id,
                        turn_id=turn_id,
                        api_request_id=api_request_id,
                        tool_name=tool_name,
                    ):
                        return None
                    observed = self._capture_observation(
                        case,
                        tool_name=tool_name,
                        result_text=_result_text(result),
                        result_status="unknown",
                        observation_boundary=observation_boundary,
                        raw_result_provenance="unknown",
                        final_exposure_status="unknown",
                        capture_guard=lambda: state.matches(
                            session_id=session_id,
                            task_id=task_id,
                            tool_call_id=tool_call_id,
                            turn_id=turn_id,
                            api_request_id=api_request_id,
                            tool_name=tool_name,
                        ),
                    )
                    state.unknown_captured = True
                    return observed
        except LockTimeoutError:
            logger.warning(
                "Epistemic native unknown-result observation skipped: lock unavailable; "
                "observation was not persisted"
            )
            return None
        except _NativeCaptureStale:
            return None

    def _capture_observation(
        self,
        case: dict[str, Any],
        *,
        tool_name: str,
        result_text: str,
        result_status: str,
        observation_boundary: str | None = None,
        raw_result_provenance: str | None = None,
        final_exposure_status: str | None = None,
        capture_guard: Callable[[], bool] | None = None,
    ) -> dict[str, Any]:
        if capture_guard is not None and not capture_guard():
            raise _NativeCaptureStale("native capture scope finalized before persistence")
        safe_result = _redact(result_text)
        probe = case.get("pending_probe")
        if not isinstance(probe, dict):
            raise HarnessError("pending probe disappeared during execution")
        probe["status"] = "observed"
        probe["result_sha256"] = _sha256(safe_result)
        probe["result_status"] = result_status
        probe["execution_result_status"] = result_status
        probe["display_result_status"] = result_status
        if observation_boundary is not None:
            probe["observation_boundary"] = observation_boundary
        if raw_result_provenance is not None:
            probe["raw_result_provenance"] = raw_result_provenance
        if final_exposure_status is not None:
            probe["final_exposure_status"] = final_exposure_status
        observation_payload = {
            "probe_id": probe["probe_id"],
            "tool_name": tool_name,
            "authority_domain": probe.get("authority_domain"),
            "result_status": result_status,
            "execution_result_status": result_status,
            "display_result_status": result_status,
            "result_sha256": probe["result_sha256"],
        }
        if observation_boundary is not None:
            observation_payload["observation_boundary"] = observation_boundary
        if raw_result_provenance is not None:
            observation_payload["raw_result_provenance"] = raw_result_provenance
        if final_exposure_status is not None:
            observation_payload["final_exposure_status"] = final_exposure_status
        if capture_guard is not None and not capture_guard():
            raise _NativeCaptureStale("native capture scope finalized before persistence")
        if len(safe_result.encode("utf-8")) > _INLINE_OBSERVATION_LIMIT:
            observation_payload.update(
                self.store.store_artifact(case["case_id"], safe_result)
            )
        else:
            observation_payload.update({"storage": "inline", "result": safe_result})
        if capture_guard is not None and not capture_guard():
            raise _NativeCaptureStale("native capture scope finalized before persistence")
        observed = self.store.transition_case(
            case,
            event_type="probe_observed",
            event_payload=observation_payload,
            event_id_target=("pending_probe", "observation_event_id"),
        )
        if observed["case"]["pending_probe"]["status"] != "observed":
            raise HarnessError("failed to persist probe observation")
        return observed

    def _native_exposure_scope_matches(
        self,
        state: _NativeCaptureState,
        *,
        case: dict[str, Any],
        probe: dict[str, Any],
        tool_name: str,
        args: dict[str, Any] | None,
        session_id: str,
        task_id: str,
        tool_call_id: str,
        turn_id: str,
        api_request_id: str,
    ) -> bool:
        """Admit native exposure only from its live capture scope."""
        if not isinstance(args, dict):
            return False
        try:
            with acquire_lock(state.lock, label="native exposure state lock"):
                return (
                    state.matches(
                        session_id=session_id,
                        task_id=task_id,
                        tool_call_id=tool_call_id,
                        turn_id=turn_id,
                        api_request_id=api_request_id,
                        tool_name=tool_name,
                    )
                    and (state.raw_captured or state.unknown_captured)
                ) and (
                    state.case_id == str(case.get("case_id") or "")
                    and state.probe_id == str(probe.get("probe_id") or "")
                    and probe.get("status") == "observed"
                    and probe.get("tool_name") == tool_name
                    and _probe_args_match(self.store, probe, args)
                )
        except LockTimeoutError:
            return False

    @_bounded_native_observer
    def record_tool_result_exposure(
        self,
        *,
        tool_name: str,
        args: dict[str, Any] | None,
        result: Any,
        session_id: str,
        tool_call_id: str = "",
        task_id: str = "",
        turn_id: str = "",
        api_request_id: str = "",
        exposure_boundary: str = "transform_tool_result_input",
    ) -> dict[str, Any] | None:
        """Bind the probe to a result representation observed at a named boundary.

        The transform hook receives the original input, not necessarily the final
        model-visible bytes; callers should identify the boundary explicitly.
        """
        native_state = current_native_capture_state()
        native_scope_guard: Callable[[], bool] | None = None
        try:
            self.store.recover_prepared_compression(session_id)
            case = self.store.get_active_case(session_id)
            if case is None:
                return None
            probe = case.get("pending_probe")
            if not isinstance(probe, dict) or probe.get("status") != "observed":
                return None
            if probe.get("raw_result_provenance") in _NATIVE_RESULT_PROVENANCE:
                if native_state is None:
                    return None
                native_scope_guard = lambda: self._native_exposure_scope_matches(
                    native_state,
                    case=case,
                    probe=probe,
                    tool_name=tool_name,
                    args=args,
                    session_id=session_id,
                    task_id=task_id,
                    tool_call_id=tool_call_id,
                    turn_id=turn_id,
                    api_request_id=api_request_id,
                )
                if not native_scope_guard():
                    return None
            expected_tool_call_id = str(probe.get("execution_tool_call_id") or "")
            exposed_tool_call_id = str(tool_call_id or "")
            if expected_tool_call_id:
                if not exposed_tool_call_id:
                    raise HarnessError(
                        "final tool result is missing its execution correlation"
                    )
                if exposed_tool_call_id != expected_tool_call_id:
                    return None
            elif exposed_tool_call_id:
                # A legacy observation has no execution correlation. A later
                # denied or otherwise unrelated tool result must not be
                # mistaken for that observation merely because its tool name
                # happens to match.
                return None
            if probe.get("tool_name") != tool_name:
                raise HarnessError("final tool result does not match the pending probe tool")
            for field_name, supplied_identity in (
                ("execution_task_id", task_id),
                ("execution_turn_id", turn_id),
                ("execution_api_request_id", api_request_id),
            ):
                expected_identity = str(probe.get(field_name) or "")
                if expected_identity and str(supplied_identity or "") != expected_identity:
                    return None
            if args is not None:
                expected_digest = str(
                    probe.get("execution_tool_args_sha256")
                    or probe.get("tool_args_sha256")
                    or ""
                )
                supplied_digest = (
                    self.store.commitment_digest(_canonical(args))
                    if probe.get("tool_args_commitment_scheme") == "hmac-sha256-v1"
                    else _sha256(_canonical(args))
                )
                if not expected_digest:
                    expected_digest = _sha256(_canonical(probe.get("tool_args")))
                if supplied_digest != expected_digest:
                    raise HarnessError("final tool result arguments do not match the pending probe")

            result_text = _result_text(result)
            safe_result = _redact(result_text)
            if native_scope_guard is not None and not native_scope_guard():
                return None
            if _sha256(safe_result) == probe.get("result_sha256"):
                return {"status": "unchanged", "case": case, "event": None}
            display_result_status = _classify_tool_result(safe_result)
            execution_result_status = _probe_execution_result_status(probe)
            probe["execution_result_status"] = execution_result_status
            probe["display_result_status"] = display_result_status
            probe["result_sha256"] = _sha256(safe_result)
            # Keep the historical field as the latest model-visible result
            # classification. Calibration follows the immutable execution
            # provenance and the supersedes chain instead.
            probe["result_status"] = display_result_status
            payload = {
                "probe_id": probe["probe_id"],
                "tool_name": tool_name,
                "authority_domain": probe.get("authority_domain"),
                "result_status": display_result_status,
                "execution_result_status": execution_result_status,
                "display_result_status": display_result_status,
                "result_sha256": probe["result_sha256"],
                "supersedes_event_id": probe.get("observation_event_id"),
                "observation_boundary": exposure_boundary,
                "final_exposure_status": "unknown",
            }
            if len(safe_result.encode("utf-8")) > _INLINE_OBSERVATION_LIMIT:
                payload.update(self.store.store_artifact(case["case_id"], safe_result))
            else:
                payload.update({"storage": "inline", "result": safe_result})
            if native_scope_guard is not None and not native_scope_guard():
                return None
            return self.store.transition_case(
                case,
                event_type="probe_exposed",
                event_payload=payload,
                event_id_target=("pending_probe", "observation_event_id"),
            )
        except StoreError as exc:
            raise HarnessError(str(exc)) from exc

    def replay(self, case_id: str) -> dict[str, Any]:
        """Backtest structural consistency against the complete recorded Timeline."""
        chain = self.store.verify_event_chain(case_id)
        snapshot = self.store.verify_case_snapshot(case_id)
        if (
            not snapshot.get("valid")
            and "identity does not match storage slot"
            in str(snapshot.get("error") or "")
        ):
            return {
                "usable": False,
                "timeline": chain,
                "case_snapshot": snapshot,
                "accounting": {
                    "uncompared_observation_event_ids": [],
                    "invalid_comparison_observation_event_ids": [],
                    "unaddressed_material_mismatch_event_ids": [],
                    "invalid_model_event_ids": [],
                    "invalid_claim_authority_bindings": [],
                    "invalid_plan_dependency_ids": [],
                    "duplicate_model_ids": [],
                    "invalid_observation_artifacts": [],
                },
                "stale": False,
                "pending_probe": None,
                "reason": f"case snapshot integrity check failed: {snapshot['error']}",
            }
        case = self.store.get_case(case_id)
        events = self.store.list_events(case_id)
        event_ids = {str(event.get("event_id")) for event in events}

        latest_observation_by_probe: dict[str, str] = {}
        for event in events:
            if event.get("type") not in {"probe_observed", "probe_exposed"}:
                continue
            probe_id = str((event.get("payload") or {}).get("probe_id") or "")
            if probe_id:
                latest_observation_by_probe[probe_id] = str(event["event_id"])
        observed_ids = set(latest_observation_by_probe.values())
        compared_observation_ids = {
            str((event.get("payload") or {}).get("observation_event_id"))
            for event in events
            if event.get("type") == "probe_compared"
            and (event.get("payload") or {}).get("observation_event_id")
        }
        discarded_observation_ids = {
            str((event.get("payload") or {}).get("observation_event_id"))
            for event in events
            if event.get("type") == "probe_discarded"
            and (event.get("payload") or {}).get("observation_event_id")
        }
        uncompared = sorted(
            observed_ids - compared_observation_ids - discarded_observation_ids
        )
        invalid_comparison_refs = sorted(compared_observation_ids - observed_ids)
        invalid_artifacts: list[str] = []
        for event in events:
            payload = event.get("payload") or {}
            if event.get("type") not in {"probe_observed", "probe_exposed"} or payload.get("storage") != "artifact":
                continue
            verification = self.store.verify_artifact(case_id, payload)
            if not verification.get("valid"):
                invalid_artifacts.append(
                    f"{event.get('event_id')}: {verification.get('error', 'unknown error')}"
                )

        material_mismatches = {
            str(event["event_id"])
            for event in events
            if event.get("type") == "probe_compared"
            and (event.get("payload") or {}).get("disposition") == "mismatch"
            and (event.get("payload") or {}).get("material") is True
        }
        repaired_mismatches = {
            str((event.get("payload") or {}).get("repaired_mismatch_event_id"))
            for event in events
            if event.get("type") == "model_revised"
            and (event.get("payload") or {}).get("repaired_mismatch_event_id")
        }
        unaddressed_mismatches = sorted(material_mismatches - repaired_mismatches)

        model = case.get("model") or {}
        invalid_model_event_ids: set[str] = set()
        invalid_claim_authority_bindings: list[str] = []
        event_authorities = _event_authority_map(events)
        model_item_ids: list[str] = []
        for field in ("state_grounding", "mechanisms", "alternatives"):
            items = model.get(field) or []
            if not isinstance(items, list):
                continue
            for item in items:
                if not isinstance(item, dict):
                    continue
                if item.get("id"):
                    model_item_ids.append(str(item["id"]))
                for reference in item.get("evidence_event_ids") or []:
                    if str(reference) not in event_ids:
                        invalid_model_event_ids.add(str(reference))
                try:
                    _validate_claim_authority(
                        str(item.get("id") or "unknown"),
                        _copy(item),
                        [str(ref) for ref in (item.get("evidence_event_ids") or [])],
                        event_authorities=event_authorities,
                    )
                except HarnessError as exc:
                    invalid_claim_authority_bindings.append(str(exc))
        plan = model.get("current_plan") or []
        plan_ids = [
            str(item.get("id")) for item in plan
            if isinstance(item, dict) and item.get("id")
        ]
        known_dependency_ids = set(model_item_ids) | set(plan_ids)
        invalid_plan_dependencies = sorted(
            {
                str(dependency)
                for item in plan
                if isinstance(item, dict)
                for dependency in (item.get("depends_on") or [])
                if str(dependency) not in known_dependency_ids
            }
        )
        duplicate_model_ids = sorted(
            {item_id for item_id in model_item_ids + plan_ids if (model_item_ids + plan_ids).count(item_id) > 1}
        )

        accounting = {
            "uncompared_observation_event_ids": uncompared,
            "invalid_comparison_observation_event_ids": invalid_comparison_refs,
            "unaddressed_material_mismatch_event_ids": unaddressed_mismatches,
            "invalid_model_event_ids": sorted(invalid_model_event_ids),
            "invalid_claim_authority_bindings": sorted(
                invalid_claim_authority_bindings
            ),
            "invalid_plan_dependency_ids": invalid_plan_dependencies,
            "duplicate_model_ids": duplicate_model_ids,
            "invalid_observation_artifacts": invalid_artifacts,
        }
        accounting_clean = not any(accounting.values())
        pending = case.get("pending_probe")
        usable = (
            bool(chain.get("valid"))
            and bool(snapshot.get("valid"))
            and accounting_clean
            and not bool(case.get("stale"))
        )
        reason = "model and complete Timeline are structurally consistent"
        if isinstance(pending, dict) and pending.get("status") == "observed":
            usable = False
            reason = "probe result awaits comparison"
        elif isinstance(pending, dict) and pending.get("status") == "executing":
            usable = False
            reason = "probe execution is incomplete"
        elif case.get("stale"):
            reason = "material mismatch requires consequential revision"
        elif not chain.get("valid"):
            reason = "Timeline integrity check failed"
        elif not snapshot.get("valid"):
            reason = f"case snapshot integrity check failed: {snapshot.get('error', 'unknown error')}"
        elif invalid_artifacts:
            reason = "observation artifact integrity check failed"
        elif invalid_claim_authority_bindings:
            reason = "claim authority accounting failed"
        elif not accounting_clean:
            reason = "complete-history accounting failed"
        return {
            "usable": usable,
            "timeline": chain,
            "case_snapshot": snapshot,
            "accounting": accounting,
            "stale": bool(case.get("stale")),
            "pending_probe": pending,
            "reason": reason,
        }

    def _open(self, args: dict[str, Any], *, session_id: str) -> dict[str, Any]:
        if "retrieved_prior_lessons" in args:
            raise HarnessError(
                "retrieved_prior_lessons is server-managed; provide prior_case_ids "
                "when opening a new case"
            )
        top_unknown = str(args.get("top_unknown") or "").strip()
        if not top_unknown:
            raise HarnessError("top_unknown is required to open a case")
        defaults = {
            "decision": "none",
            "stopping_condition": (
                "Answer the top unknown with explicit scope, strongest evidence, "
                "remaining alternatives, and what would change the answer."
            ),
            "state_grounding": [],
            "mechanisms": [],
            "alternatives": [],
            "top_unknown": top_unknown,
            "current_plan": [],
        }
        model = {
            field: _copy(args[field] if field in args else defaults[field])
            for field in _REQUIRED_MODEL_FIELDS
        }
        prior_case_ids = args.get("prior_case_ids") or []
        applied_prior_lessons = args.get("applied_prior_lessons") or []
        if not isinstance(prior_case_ids, list) or not all(
            isinstance(item, str) and item for item in prior_case_ids
        ):
            raise HarnessError("prior_case_ids must be an array of case IDs")
        if len(set(prior_case_ids)) != len(prior_case_ids):
            raise HarnessError("prior_case_ids must contain distinct case IDs")
        if not isinstance(applied_prior_lessons, list) or not all(
            isinstance(item, str) and item for item in applied_prior_lessons
        ):
            raise HarnessError("applied_prior_lessons must be an array of lesson statements")
        model["applied_prior_lessons"] = list(applied_prior_lessons)
        model, initial_events = _prepare_initial_evidence(
            model,
            args.get("initial_evidence") or [],
        )
        if prior_case_ids:
            prior_case_lessons = self._prior_case_lessons(prior_case_ids)
        else:
            prior_case_lessons = self._select_prior_case_lessons(model)
            prior_case_ids = [item["case_id"] for item in prior_case_lessons]
        model["prior_case_ids"] = list(prior_case_ids)
        model["retrieved_prior_lessons"] = _copy(prior_case_lessons)
        case = self.store.open_case(
            session_id=session_id,
            case_id=str(args.get("case_id") or ""),
            model=model,
            initial_events=initial_events,
        )
        return {
            "case": case,
            "replay": self.replay(case["case_id"]),
            "prior_case_lessons": prior_case_lessons,
        }

    def _prior_case_lessons(self, prior_case_ids: list[str]) -> list[dict[str, Any]]:
        lessons: list[dict[str, Any]] = []
        for prior_case_id in prior_case_ids:
            try:
                prior = self.store.get_case(prior_case_id)
            except StoreError as exc:
                raise HarnessError(
                    f"explicit prior case {prior_case_id!r} is unavailable: {exc}"
                ) from exc
            if prior.get("status") != "closed":
                raise HarnessError(
                    f"explicit prior case {prior_case_id!r} must be closed; "
                    "open a new bounded case after closing the requested source"
                )
            try:
                replay = self.replay(prior_case_id)
            except (StoreError, HarnessError) as exc:
                raise HarnessError(
                    f"explicit prior case {prior_case_id!r} failed replay validation: {exc}"
                ) from exc
            if not replay.get("usable"):
                raise HarnessError(
                    f"explicit prior case {prior_case_id!r} is not replay-valid: "
                    f"{replay.get('reason', 'integrity check failed')}"
                )
            lesson = _prior_case_lesson(prior, prior_case_id)
            if lesson is None:
                raise HarnessError(
                    f"explicit prior case {prior_case_id!r} has no reusable transfer lesson"
                )
            lessons.append(lesson)
        return lessons

    def _select_prior_case_lessons(
        self,
        model: dict[str, Any],
        *,
        limit: int = 3,
    ) -> list[dict[str, Any]]:
        """Retrieve a few lexically relevant lessons without a new index or service."""
        query_tokens = _lesson_tokens(_model_relevance_text(model))
        if not query_tokens:
            return []
        ranked: list[tuple[float, str, dict[str, Any]]] = []
        for case_id in self.store.list_case_ids():
            try:
                prior = self.store.get_case(case_id)
                if prior.get("status") != "closed":
                    continue
                lesson = _prior_case_lesson(prior, case_id)
                if lesson is None:
                    continue
                if not self.replay(case_id)["usable"]:
                    continue
            except (StoreError, HarnessError):
                continue
            transfer = ((prior.get("closure") or {}).get("transfer") or {})
            lesson_text = str(transfer.get("lesson") or "").strip()
            scope = str(transfer.get("scope") or "").strip()
            candidate_tokens = _lesson_tokens(f"{lesson_text} {scope}")
            overlap = query_tokens & candidate_tokens
            # Second channel (2026-09-01 audit fix): shared word-FAMILY
            # classes count like shared tokens, weighted below exact overlap.
            # A directly relevant lesson phrased in family words ("asks",
            # "delegation") must not be invisible to a query phrased in its
            # siblings ("questioning", "inquiry").
            class_overlap = _lesson_stem_classes(query_tokens) & _lesson_stem_classes(candidate_tokens)
            effective_overlap = len(overlap) + 0.5 * len(class_overlap)
            if len(overlap) < 2 and not class_overlap:
                continue
            score = (2.0 * effective_overlap) / (
                len(query_tokens) + len(candidate_tokens)
            )
            if score < 0.12 and len(class_overlap) < 3:
                # Below the exact-lexical floor, a lesson still qualifies
                # when the word-family channel is strong (≥3 shared classes)
                # — the observed live failure scored 0.057 exactly while
                # sharing question+value+evidence+humanai+delegation+synthesis.
                continue
            ranked.append(
                (
                    score,
                    case_id,
                    {
                        "case_id": case_id,
                        "decision": transfer.get("decision"),
                        "lesson": lesson_text,
                        "scope": scope,
                        "evidence_case_ids": list(
                            transfer.get("evidence_case_ids") or []
                        ),
                        "calibration": transfer.get("calibration")
                        or ((prior.get("closure") or {}).get("calibration") or {}).get("label")
                        or "hypothesis",
                        "relevance": round(score, 3),
                    },
                )
            )
        ranked.sort(key=lambda item: (-item[0], item[1]))
        return [item[2] for item in ranked[:limit]]

    def _record_evidence(
        self,
        args: dict[str, Any],
        *,
        session_id: str,
    ) -> dict[str, Any]:
        """Record assessed external evidence gathered through ordinary tools."""
        case = self._require_active(session_id)
        if case.get("pending_probe") is not None:
            raise HarnessError(
                "compare or discard the pending probe before recording ordinary evidence"
            )
        supplied = args.get("evidence")
        if isinstance(supplied, str):
            try:
                supplied = json.loads(supplied)
            except (json.JSONDecodeError, TypeError) as exc:
                raise HarnessError(
                    "record_evidence requires a JSON evidence object or non-empty array, not plain text"
                ) from exc
        items = supplied if isinstance(supplied, list) else [supplied]
        if not items or any(not isinstance(item, dict) for item in items):
            raise HarnessError("record_evidence requires one evidence object or a non-empty array")

        descriptors = [
            _normalize_evidence_descriptor(
                item,
                require_external_authority=True,
            )
            for item in items
        ]
        events: list[dict[str, Any]] = []
        for descriptor in descriptors:
            transition = self.store.transition_case(
                case,
                event_type="evidence_recorded",
                event_payload=descriptor,
            )
            case = transition["case"]
            events.append(transition["event"])
        return {
            "case": case,
            "events": events,
            "evidence_event_ids": [event["event_id"] for event in events],
            "replay": self.replay(case["case_id"]),
        }

    def _revise(self, args: dict[str, Any], *, session_id: str) -> dict[str, Any]:
        _reject_server_managed_revision_fields(args)
        case = self._require_active(session_id)
        if case.get("pending_probe") is not None:
            raise HarnessError("compare the pending probe before revising the model")
        reason = _redact(str(args.get("reason") or "").strip())
        if not reason:
            raise HarnessError("revision reason is required")
        updates = args.get("updates")
        if updates is None:
            updates = {
                field: _copy(args[field])
                for field in _MODEL_FIELDS
                if field in args
            }
        if not isinstance(updates, dict) or not updates:
            raise HarnessError(
                "provide updates or one or more model fields to revise"
            )
        _reject_server_managed_revision_fields({"updates": updates})
        unknown = sorted(set(updates) - _REVISION_MODEL_FIELDS)
        if unknown:
            raise HarnessError(f"unsupported model fields: {', '.join(unknown)}")

        old_model = case["model"]
        new_model = _copy(old_model)
        for key, value in updates.items():
            new_model[key] = _copy(value)
        events = self.store.list_events(case["case_id"])
        known_event_ids = {event["event_id"] for event in events}
        new_model = _normalize_model_claims(
            new_model,
            known_event_ids=known_event_ids,
            event_authorities=_event_authority_map(events),
        )
        if case.get("stale") and str(case.get("mismatch_event_id") or "") in (
            args.get("addresses_event_ids") or []
        ):
            old_items = _model_items_by_id(old_model)
            new_items = _model_items_by_id(new_model)
            substantively_unchanged = sorted(
                item_id
                for item_id in (case.get("stale_claim_ids") or [])
                if item_id in old_items
                and item_id in new_items
                and not _stale_item_change_is_substantive(
                    old_items[item_id],
                    new_items[item_id],
                )
            )
            if substantively_unchanged:
                raise HarnessError(
                    "revision leaves stale model items unchanged in substantive content: "
                    + ", ".join(substantively_unchanged)
                )
        if _canonical(new_model) == _canonical(old_model):
            raise HarnessError("revision makes no material change")

        addresses = args.get("addresses_event_ids") or []
        if not isinstance(addresses, list) or not all(isinstance(item, str) for item in addresses):
            raise HarnessError("addresses_event_ids must be an array of event IDs")

        if case.get("stale"):
            mismatch_event_id = str(case.get("mismatch_event_id") or "")
            if not mismatch_event_id or mismatch_event_id not in addresses:
                raise HarnessError(
                    "revision must explicitly address the material mismatch event"
                )
            consequential_fields = {
                "decision",
                "stopping_condition",
                "state_grounding",
                "mechanisms",
                "alternatives",
                "current_plan",
            }
            if not consequential_fields.intersection(updates):
                raise HarnessError(
                    "material mismatch requires a consequential model or plan change"
                )
            old_items = _model_items_by_id(old_model)
            new_items = _model_items_by_id(new_model)
            unchanged_stale = sorted(
                item_id
                for item_id in (case.get("stale_claim_ids") or [])
                if item_id in old_items
                and item_id in new_items
                and not _stale_item_change_is_substantive(
                    old_items[item_id],
                    new_items[item_id],
                )
            )
            if unchanged_stale:
                raise HarnessError(
                    "revision leaves stale model items unchanged: "
                    + ", ".join(unchanged_stale)
                )

        case["model"] = new_model
        case["version"] = int(case["version"]) + 1
        repaired_mismatch = str(case.get("mismatch_event_id") or "") or None
        if case.get("stale"):
            case["stale"] = False
            case["stale_claim_ids"] = []
            case["mismatch_event_id"] = None
        result = self.store.update_case(
            case,
            event_type="model_revised",
            event_payload={
                "model_version": case["version"],
                "changed_fields": sorted(updates),
                "reason": reason,
                "addresses_event_ids": addresses,
                "repaired_mismatch_event_id": repaired_mismatch,
            },
        )
        result["replay"] = self.replay(case["case_id"])
        return result

    def _commit_probe(self, args: dict[str, Any], *, session_id: str) -> dict[str, Any]:
        case = self._require_active(session_id)
        if case.get("pending_probe") is not None:
            raise HarnessError("only one pending probe is permitted")
        replay = self.replay(case["case_id"])
        if not replay["usable"]:
            raise HarnessError(f"case replay is not usable: {replay['reason']}")
        if "claim_snapshot" in args or "model_version" in args:
            raise HarnessError(
                "claim_snapshot and model_version are server-managed; provide claim_id only"
            )
        claim_id: str | None = None
        claim_snapshot: dict[str, Any] | None = None
        if "claim_id" in args:
            if not isinstance(args.get("claim_id"), str) or not args["claim_id"].strip():
                raise HarnessError("claim_id must be a non-empty existing claim ID")
            claim_id = args["claim_id"].strip()
            for section in ("state_grounding", "mechanisms", "alternatives"):
                for item in (case.get("model") or {}).get(section) or []:
                    if isinstance(item, dict) and str(item.get("id") or "") == claim_id:
                        claim_snapshot = _copy(item)
                        break
                if claim_snapshot is not None:
                    break
            if claim_snapshot is None:
                raise HarnessError(
                    "claim_id must name an existing claim in state_grounding, mechanisms, or alternatives"
                )
        purpose = str(args.get("purpose") or "").strip()
        if purpose not in {"learn", "advance"}:
            raise HarnessError("purpose must be learn or advance")
        required_text = [
            "unknown_or_goal",
            "tool_name",
            "would_change_belief",
            "why_this_action",
            "authority_domain",
        ]
        missing = [key for key in required_text if not str(args.get(key) or "").strip()]
        if missing:
            raise HarnessError(f"missing probe fields: {', '.join(missing)}")
        tool_name = str(args["tool_name"]).strip()
        if tool_name in {"epistemic_model", "epistemic_probe"}:
            raise HarnessError("a probe must bind one external action or formal question")
        if tool_name == "delegate_task":
            raise HarnessError(
                "delegation is a compound asynchronous action and cannot be bounded by one probe"
            )
        authority_domain = str(args["authority_domain"]).strip()
        if authority_domain not in _AUTHORITY_DOMAINS:
            raise HarnessError(
                f"authority_domain must be one of {', '.join(sorted(_AUTHORITY_DOMAINS))}"
            )
        if tool_name == "clarify" and authority_domain not in _USER_AUTHORITY_DOMAINS:
            raise HarnessError(
                "clarify authority must be user_preference, user_intent, private_context, "
                "user_decision, or user_testimony; it cannot resolve external empirical truth"
            )
        tool_args = _resolve_probe_tool_args(args)
        predictions = args.get("predicted_outcomes")
        if not isinstance(predictions, list) or not predictions:
            raise HarnessError("predicted_outcomes must be a non-empty array")
        normalized_predictions: list[dict[str, str]] = []
        for prediction in predictions:
            if isinstance(prediction, str) and prediction.strip():
                normalized_predictions.append(
                    {
                        "outcome": prediction.strip(),
                        "meaning": "Update the case in light of this observation.",
                    }
                )
                continue
            if not isinstance(prediction, dict):
                raise HarnessError(
                    "each predicted outcome must be text or an object with outcome"
                )
            outcome = str(prediction.get("outcome") or "").strip()
            if not outcome:
                raise HarnessError("each predicted outcome requires outcome")
            normalized_predictions.append(
                {
                    "outcome": outcome,
                    "meaning": str(prediction.get("meaning") or "").strip()
                    or "Update the case in light of this observation.",
                }
            )

        prior_commits = sum(
            event.get("type") == "probe_committed"
            for event in self.store.list_events(case["case_id"])
        )
        canonical_tool_args = _canonical(tool_args)
        known_secrets = _secret_values(tool_args)
        probe = {
            "probe_id": f"PR{prior_commits + 1:06d}",
            "status": "committed",
            "purpose": purpose,
            "unknown_or_goal": _redact_value(
                str(args["unknown_or_goal"]).strip(), known_secrets=known_secrets
            ),
            "tool_name": tool_name,
            "tool_args": _redact_value(tool_args, known_secrets=known_secrets),
            "tool_args_sha256": self.store.commitment_digest(canonical_tool_args),
            "tool_args_commitment_scheme": "hmac-sha256-v1",
            "predicted_outcomes": _redact_value(
                normalized_predictions,
                known_secrets=known_secrets,
            ),
            "would_change_belief": _redact_value(
                str(args["would_change_belief"]).strip(),
                known_secrets=known_secrets,
            ),
            "why_this_action": _redact_value(
                str(args["why_this_action"]).strip(), known_secrets=known_secrets
            ),
            "authority_domain": authority_domain,
        }
        if claim_id is not None and claim_snapshot is not None:
            probe["claim_id"] = claim_id
            probe["claim_snapshot"] = claim_snapshot
            probe["model_version"] = int(case["version"])
        case["pending_probe"] = probe
        transition = self.store.transition_case(
            case,
            event_type="probe_committed",
            event_payload={
                "probe": _copy(probe),
                "args_sha256": probe["tool_args_sha256"],
            },
            event_id_target=("pending_probe", "commit_event_id"),
        )
        return {"probe": transition["case"]["pending_probe"], "event": transition["event"]}

    def _compare_probe(self, args: dict[str, Any], *, session_id: str) -> dict[str, Any]:
        case = self._require_active(session_id)
        probe = case.get("pending_probe")
        if not isinstance(probe, dict) or probe.get("status") != "observed":
            raise HarnessError("a captured probe observation must be compared first")
        disposition = str(args.get("disposition") or "").strip().lower()
        if disposition not in {"match", "mismatch", "unresolved"}:
            raise HarnessError("disposition must be match, mismatch, or unresolved")
        material = args.get("material", False)
        if not isinstance(material, bool):
            raise HarnessError("material must be true or false")
        rationale = _redact(str(args.get("rationale") or "").strip())
        if not rationale:
            raise HarnessError("comparison rationale is required")
        if "claim_snapshot" in args or "model_version" in args:
            raise HarnessError(
                "claim_snapshot and model_version are server-managed; provide claim_id only at commit"
            )
        bound_claim_id = str(probe.get("claim_id") or "").strip() or None
        belief_change: str | None = None
        if bound_claim_id is not None:
            belief_change = str(args.get("belief_change") or "").strip().lower()
            if belief_change not in {
                "strengthened",
                "weakened",
                "narrowed",
                "unchanged",
                "unresolved",
            }:
                raise HarnessError(
                    "belief_change is required for a claim-bound comparison and must be "
                    "strengthened, weakened, narrowed, unchanged, or unresolved"
                )

        if "affected_claim_ids" not in args or args.get("affected_claim_ids") is None:
            affected = [bound_claim_id] if bound_claim_id is not None else []
        else:
            affected = args.get("affected_claim_ids")
        if not isinstance(affected, list) or not all(isinstance(item, str) for item in affected):
            raise HarnessError("affected_claim_ids must be an array of claim IDs")
        known_item_ids = set(_model_items_by_id(case.get("model") or {}))
        unknown_affected = sorted(set(affected) - known_item_ids)
        if unknown_affected:
            raise HarnessError(
                f"affected_claim_ids contains unknown model items: {', '.join(unknown_affected)}"
            )
        if bound_claim_id is not None:
            if not affected or bound_claim_id not in affected:
                raise HarnessError(
                    "a bound comparison must affect its bound claim by default; "
                    "unrelated claim IDs are not permitted"
                )
            related_ids = set(
                _expand_stale_model_ids(case.get("model") or {}, [bound_claim_id])
            )
            unrelated = sorted(set(affected) - related_ids)
            if unrelated:
                raise HarnessError(
                    "a bound comparison cannot redirect evidence to unrelated claim IDs: "
                    + ", ".join(unrelated)
                )
        if disposition == "mismatch" and material and not affected:
            raise HarnessError("a material mismatch must identify affected claims or plans")

        argument_match = probe.get("execution_args_match_commitment")
        if not isinstance(argument_match, bool):
            argument_match = None
        argument_match_status = (
            "matched"
            if argument_match is True
            else "mismatched"
            if argument_match is False
            else "unknown"
        )
        argument_drift_reason = _redact(
            str(args.get("argument_drift_reason") or "").strip()
        )
        if argument_match is False and not argument_drift_reason:
            raise HarnessError(
                "argument drift was recorded; provide a nonempty argument drift reason "
                "or discard the probe"
            )

        evidence_descriptor: dict[str, Any] | None = None
        supplied_evidence = args.get("evidence")
        if supplied_evidence is not None:
            if not isinstance(supplied_evidence, dict):
                raise HarnessError("probe comparison evidence must be one object")
            evidence_descriptor = _normalize_evidence_descriptor(
                supplied_evidence,
                require_external_authority=(
                    probe.get("authority_domain") == "external_source"
                ),
            )
            # F6: the comparison evidence must be authority-compatible with the
            # probe it is bound to. A user-authority evidence object cannot be
            # laundered into a non-user (system_state/empirical_test) probe, and
            # vice versa.
            probe_authority = str(probe.get("authority_domain") or "")
            evidence_authority = str(evidence_descriptor.get("authority_domain") or "")
            if (
                probe_authority in _AUTHORITY_DOMAINS
                and evidence_authority in _AUTHORITY_DOMAINS
                and (probe_authority in _USER_AUTHORITY_DOMAINS)
                != (evidence_authority in _USER_AUTHORITY_DOMAINS)
            ):
                raise HarnessError(
                    f"comparison evidence authority {evidence_authority!r} is "
                    f"incompatible with probe authority {probe_authority!r}"
                )
        if (
            probe.get("authority_domain") == "external_source"
            and probe.get("result_status") == "ok"
            and evidence_descriptor is None
        ):
            raise HarnessError(
                "successful external-source comparison requires an evidence descriptor"
            )

        event_payload = {
            "probe_id": probe["probe_id"],
            "commit_event_id": probe.get("commit_event_id"),
            "observation_event_id": probe.get("observation_event_id"),
            "authority_domain": probe.get("authority_domain"),
            "disposition": disposition,
            "material": material,
            "rationale": rationale,
            "affected_claim_ids": affected,
            "execution_args_match_commitment": argument_match,
            "argument_match_status": argument_match_status,
            **(
                {"argument_drift_reason": argument_drift_reason}
                if argument_drift_reason
                else {}
            ),
            **(
                {
                    "claim_id": bound_claim_id,
                    "claim_snapshot": _copy(probe["claim_snapshot"]),
                    "model_version": probe.get("model_version"),
                    "belief_change": belief_change,
                }
                if bound_claim_id is not None
                else {}
            ),
            **(
                {"evidence": evidence_descriptor}
                if evidence_descriptor is not None
                else {}
            ),
        }
        case["pending_probe"] = None
        event_target: str | None = None
        if disposition == "mismatch" and material:
            case["stale"] = True
            case["stale_claim_ids"] = _expand_stale_model_ids(
                case.get("model") or {},
                affected,
            )
            event_target = "mismatch_event_id"
        transition = self.store.transition_case(
            case,
            event_type="probe_compared",
            event_payload=event_payload,
            event_id_target=event_target,
        )
        transition["replay"] = self.replay(case["case_id"])
        return transition

    def _recover_probe(self, args: dict[str, Any], *, session_id: str) -> dict[str, Any]:
        case = self._require_active(session_id)
        probe = case.get("pending_probe")
        if not isinstance(probe, dict) or probe.get("status") != "executing":
            raise HarnessError("only an interrupted executing probe can be recovered")
        if self.store.probe_execution_is_live(case["case_id"]):
            raise HarnessError("cannot recover a probe while its external action is still running")
        reason = _redact(str(args.get("reason") or "").strip())
        if not reason:
            raise HarnessError("recovery reason is required")
        observed = self._capture_observation(
            case,
            tool_name=str(probe.get("tool_name") or "unknown"),
            result_text=(
                "Execution was interrupted after dispatch began and before a result was "
                f"durably captured; the action may or may not have completed. Recovery reason: {reason}"
            ),
            result_status="interrupted",
        )
        return {
            "probe": observed["case"]["pending_probe"],
            "event": observed["event"],
            "replay": self.replay(case["case_id"]),
        }

    def _discard_probe(self, args: dict[str, Any], *, session_id: str) -> dict[str, Any]:
        case = self._require_active(session_id)
        probe = case.get("pending_probe")
        if not isinstance(probe, dict):
            raise HarnessError("there is no pending probe to discard")
        if (
            probe.get("status") == "executing"
            and self.store.probe_execution_is_live(case["case_id"])
        ):
            raise HarnessError("cannot discard a probe while its action is still running")
        reason = _redact(str(args.get("reason") or "").strip())
        if not reason:
            raise HarnessError("discard reason is required")
        payload = {
            "probe_id": probe.get("probe_id"),
            "prior_status": probe.get("status"),
            "commit_event_id": probe.get("commit_event_id"),
            "observation_event_id": probe.get("observation_event_id"),
            "reason": reason,
        }
        case["pending_probe"] = None
        transition = self.store.transition_case(
            case,
            event_type="probe_discarded",
            event_payload=payload,
        )
        transition["replay"] = self.replay(case["case_id"])
        return transition

    def _pause(self, args: dict[str, Any], *, session_id: str) -> dict[str, Any]:
        case = self._require_active(session_id)
        self._require_settled(case, transition="pause")
        if not self.replay(case["case_id"])["usable"]:
            raise HarnessError("case replay must pass before pause")
        reason = _redact(str(args.get("reason") or "").strip())
        if not reason:
            raise HarnessError("pause reason is required")
        case["status"] = "paused"
        transition = self.store.transition_case(
            case,
            event_type="case_paused",
            event_payload={"reason": reason},
        )
        transition["replay"] = self.replay(case["case_id"])
        return transition

    def _resume(self, args: dict[str, Any], *, session_id: str) -> dict[str, Any]:
        case_id = str(args.get("case_id") or "").strip()
        if not case_id:
            raise HarnessError("case_id is required to resume")
        case = self.store.get_case(case_id)
        if case.get("status") != "paused":
            raise HarnessError("only a paused case can be resumed")
        replay = self.replay(case_id)
        if not replay["timeline"].get("valid"):
            raise HarnessError(f"cannot resume case: {replay['reason']}")
        transition = self.store.resume_paused_case(
            case_id=case_id,
            session_id=session_id,
        )
        transition["replay"] = self.replay(case_id)
        return transition

    def _close(self, args: dict[str, Any], *, session_id: str) -> dict[str, Any]:
        case = self._require_active(session_id)
        self._require_settled(case, transition="close")
        if not self.replay(case["case_id"])["usable"]:
            raise HarnessError("case replay must pass before close")
        outcome = str(args.get("outcome") or "").strip().lower()
        if outcome not in {"resolved", "unresolved", "stopped", "replaced"}:
            raise HarnessError("outcome must be resolved, unresolved, stopped, or replaced")
        summary = _redact(str(args.get("summary") or "").strip())
        if not summary:
            raise HarnessError("closure summary is required")
        calibration = _closure_calibration(
            case,
            events=self.store.list_events(case["case_id"]),
        )
        if outcome == "resolved" and calibration["label"] == "hypothesis":
            raise HarnessError(
                "resolved closure requires at least one evidence-linked grounded claim; "
                "link the evidence and revise the model, or use unresolved"
            )
        if "transfer" not in args:
            raise HarnessError(
                "closure requires an explicit transfer review; record one candidate "
                "lesson when reusable, otherwise choose none"
            )
        transfer = args.get("transfer")
        if isinstance(transfer, str):
            # Some provider function-calling layers collapse nested objects with
            # sparse schemas into a JSON string. Accept that form rather than
            # rejecting a semantically valid close (observed live: the 0718
            # report executor could not close its planning case).
            try:
                transfer = json.loads(transfer)
            except ValueError:
                transfer = None
        if not isinstance(transfer, dict):
            raise HarnessError("transfer review is required")
        decision = str(transfer.get("decision") or "").strip().lower()
        allowed_transfer = {"none", "candidate", "explicit_user_correction", "repeated_pattern"}
        if decision not in allowed_transfer:
            raise HarnessError(
                "transfer decision must be none, candidate, explicit_user_correction, or repeated_pattern"
            )
        normalized_transfer = {"decision": decision}
        if decision != "none":
            lesson = _redact(str(transfer.get("lesson") or "").strip())
            scope = _redact(str(transfer.get("scope") or "").strip())
            evidence_case_ids = transfer.get("evidence_case_ids") or []
            if not lesson or not scope:
                raise HarnessError("a transferable lesson requires lesson and scope")
            if (
                not isinstance(evidence_case_ids, list)
                or not evidence_case_ids
                or not all(isinstance(item, str) and item for item in evidence_case_ids)
            ):
                raise HarnessError("evidence_case_ids must be a non-empty array of case IDs")
            normalized_transfer.update(
                {
                    "lesson": lesson,
                    "scope": scope,
                    "evidence_case_ids": list(evidence_case_ids),
                }
            )
            normalized_transfer.update(
                self._validate_transfer_evidence(
                    case=case,
                    current_calibration=calibration,
                    decision=decision,
                    lesson=lesson,
                    scope=scope,
                    evidence_case_ids=list(evidence_case_ids),
                    transfer=transfer,
                )
            )
        closure = {
            "outcome": outcome,
            "summary": summary,
            "closed_at": _utc_now(),
            "calibration": calibration,
            "transfer": normalized_transfer,
        }
        case["status"] = "closed"
        case["closure"] = closure
        transition = self.store.transition_case(
            case,
            event_type="case_closed",
            event_payload=_copy(closure),
            event_id_target=("closure", "event_id"),
        )
        transition["replay"] = self.replay(case["case_id"])
        return transition

    def _validate_transfer_evidence(
        self,
        *,
        case: dict[str, Any],
        current_calibration: dict[str, Any],
        decision: str,
        lesson: str,
        scope: str,
        evidence_case_ids: list[str],
        transfer: dict[str, Any],
    ) -> dict[str, Any]:
        current_case_id = str(case["case_id"])
        if len(set(evidence_case_ids)) != len(evidence_case_ids):
            raise HarnessError("transfer evidence_case_ids must be distinct")
        if decision == "candidate":
            if evidence_case_ids != [current_case_id]:
                raise HarnessError(
                    "a candidate lesson must cite exactly the case being closed"
                )
            return {"calibration": str(current_calibration.get("label") or "hypothesis")}
        if decision == "explicit_user_correction":
            if evidence_case_ids != [current_case_id]:
                raise HarnessError(
                    "an explicit user correction must cite exactly the case being closed"
                )
            event_ids = transfer.get("evidence_event_ids") or []
            if (
                not isinstance(event_ids, list)
                or not event_ids
                or not all(isinstance(item, str) and item for item in event_ids)
                or len(set(event_ids)) != len(event_ids)
            ):
                raise HarnessError(
                    "explicit_user_correction requires distinct evidence_event_ids"
                )
            authorities = _event_authority_map(
                self.store.list_events(current_case_id)
            )
            invalid = [
                event_id
                for event_id in event_ids
                if authorities.get(event_id) not in _USER_AUTHORITY_DOMAINS
            ]
            if invalid:
                raise HarnessError(
                    "explicit_user_correction requires user-authoritative evidence events: "
                    + ", ".join(invalid)
                )
            return {
                "evidence_event_ids": list(event_ids),
                "calibration": str(current_calibration.get("label") or "hypothesis"),
            }

        # Promotion requires at least two independently closed, replay-valid
        # cases that recorded the same scoped lesson.
        if len(evidence_case_ids) < 2 or current_case_id in evidence_case_ids:
            raise HarnessError(
                "repeated_pattern requires two distinct closed source cases"
            )
        source_calibrations: list[str] = []
        for source_case_id in evidence_case_ids:
            try:
                source_case = self.store.get_case(source_case_id)
            except StoreError as exc:
                raise HarnessError(
                    f"repeated_pattern source case is unavailable: {source_case_id}"
                ) from exc
            if source_case.get("status") != "closed":
                raise HarnessError(
                    f"repeated_pattern source case is not closed: {source_case_id}"
                )
            source_transfer = ((source_case.get("closure") or {}).get("transfer") or {})
            if (
                str(source_transfer.get("lesson") or "").strip() != lesson
                or str(source_transfer.get("scope") or "").strip() != scope
            ):
                raise HarnessError(
                    "repeated_pattern source does not record the same lesson and scope: "
                    + source_case_id
                )
            if not self.replay(source_case_id)["usable"]:
                raise HarnessError(
                    f"repeated_pattern source replay is unusable: {source_case_id}"
                )
            source_calibrations.append(
                str(source_transfer.get("calibration") or "")
                or str(((source_case.get("closure") or {}).get("calibration") or {}).get("label") or "hypothesis")
            )
        return {
            "calibration": _transfer_support_ceiling(source_calibrations),
            "source_calibrations": source_calibrations,
        }

    def reject_synthetic_result_exposure(
        self,
        *,
        session_id: str,
        event_type: str,
    ) -> None:
        """Permit delayed results; explicit probes remain the only captured evidence."""
        session_id = str(session_id or "").strip()
        if not session_id:
            raise HarnessError("synthetic result exposure requires session_id")
        self.store.recover_prepared_compression(session_id)

    @staticmethod
    def _require_settled(case: dict[str, Any], *, transition: str) -> None:
        if case.get("pending_probe") is not None:
            raise HarnessError(f"compare the pending probe before attempting to {transition}")
        if case.get("stale"):
            raise HarnessError(f"repair the material mismatch before attempting to {transition}")

    def _require_active(self, session_id: str) -> dict[str, Any]:
        case = self.store.get_active_case(session_id)
        if case is None:
            raise HarnessError("no active epistemic case for this session")
        return case


def _reject_server_managed_revision_fields(args: dict[str, Any]) -> None:
    """Reject caller attempts to write derived lesson/import metadata."""
    if "retrieved_prior_lessons" in args:
        raise HarnessError(
            "retrieved_prior_lessons is server-managed and cannot be supplied in a revision"
        )
    updates = args.get("updates")
    if isinstance(updates, dict) and "retrieved_prior_lessons" in updates:
        raise HarnessError(
            "retrieved_prior_lessons is server-managed and cannot be supplied in a revision"
        )
    if "prior_case_ids" in args or (
        isinstance(updates, dict) and "prior_case_ids" in updates
    ):
        raise HarnessError(
            "prior_case_ids is open-time-only metadata; open a new bounded case "
            "if a new import set is needed"
        )


def _prior_case_lesson(
    prior: dict[str, Any],
    case_id: str,
) -> dict[str, Any] | None:
    """Extract one already-validated reusable closure lesson."""
    closure = prior.get("closure")
    if not isinstance(closure, dict):
        return None
    transfer = closure.get("transfer")
    if not isinstance(transfer, dict):
        return None
    decision = str(transfer.get("decision") or "").strip().lower()
    if not decision or decision == "none":
        return None
    lesson = str(transfer.get("lesson") or "").strip()
    scope = str(transfer.get("scope") or "").strip()
    evidence_case_ids = transfer.get("evidence_case_ids") or []
    if (
        not lesson
        or not scope
        or not isinstance(evidence_case_ids, list)
        or not evidence_case_ids
        or not all(isinstance(item, str) and item for item in evidence_case_ids)
    ):
        return None
    closure_calibration = closure.get("calibration")
    if not isinstance(closure_calibration, dict):
        closure_calibration = {}
    return {
        "case_id": case_id,
        "decision": decision,
        "lesson": lesson,
        "scope": scope,
        "evidence_case_ids": list(evidence_case_ids),
        "calibration": transfer.get("calibration")
        or closure_calibration.get("label")
        or "hypothesis",
    }


def _normalize_evidence_descriptor(
    item: dict[str, Any],
    *,
    require_external_authority: bool = False,
) -> dict[str, Any]:
    """Normalize factual source/access descriptors without inventing a score."""
    summary = _redact(str(item.get("summary") or "").strip())
    source_ref = _redact(str(item.get("source_ref") or "").strip())
    authority = str(item.get("authority_domain") or "").strip()
    source_role = str(item.get("source_role") or "").strip().lower()
    access_scope = str(item.get("access_scope") or "").strip().lower()
    locator = _redact(str(item.get("locator") or "").strip())
    limitation = _redact(str(item.get("limitation") or "").strip())

    if not summary or not source_ref:
        raise HarnessError("evidence requires summary and source_ref")
    if authority not in _AUTHORITY_DOMAINS:
        raise HarnessError("evidence has an invalid authority_domain")
    if require_external_authority and authority != "external_source":
        raise HarnessError("record_evidence is for authority_domain=external_source")
    if authority == "external_source":
        if source_role not in _SOURCE_ROLES:
            raise HarnessError(
                "external evidence source_role must be primary, synthesis, secondary, or unknown"
            )
        if access_scope not in _ACCESS_SCOPES:
            raise HarnessError(
                "external evidence access_scope must be snippet, abstract, partial, full, or data"
            )
        if access_scope in _LIMITED_ACCESS_SCOPES and not limitation:
            raise HarnessError(
                f"{access_scope} evidence requires an explicit limitation"
            )

    normalized: dict[str, Any] = {
        "summary": summary,
        "source_ref": source_ref,
        "authority_domain": authority,
    }
    if authority == "external_source":
        normalized.update(
            {
                "source_role": source_role,
                "access_scope": access_scope,
            }
        )
    if locator:
        normalized["locator"] = locator
    if limitation:
        normalized["limitation"] = limitation
    return normalized


def _prepare_initial_evidence(
    model: dict[str, Any],
    initial_evidence: Any,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if not isinstance(initial_evidence, list):
        raise HarnessError("initial_evidence must be an array")
    reference_map: dict[str, str] = {}
    authority_map: dict[str, str] = {}
    events: list[dict[str, Any]] = []
    for index, item in enumerate(initial_evidence, start=1):
        if not isinstance(item, dict):
            raise HarnessError("each initial_evidence item must be an object")
        reference = str(item.get("id") or "").strip()
        # F7: new initial_evidence must carry the same typing as record_evidence.
        # Untyped external-source evidence (no source_role/access_scope) is rejected
        # rather than being promoted to "supported" by the calibration compat branch.
        descriptor = _normalize_evidence_descriptor(
            item,
            require_external_authority=False,
        )
        authority = str(descriptor["authority_domain"])
        if not re.fullmatch(r"I[1-9][0-9]*", reference):
            raise HarnessError("initial evidence IDs must match I1, I2, ...")
        if reference in reference_map:
            raise HarnessError(f"duplicate initial evidence ID: {reference}")
        event_id = f"E{index:06d}"
        reference_map[reference] = event_id
        authority_map[event_id] = authority
        events.append(
            {
                "type": "initial_observation",
                "payload": {
                    "initial_ref": reference,
                    **descriptor,
                },
            }
        )
    normalized = _normalize_model_claims(
        model,
        initial_reference_map=reference_map,
        known_event_ids=set(reference_map.values()),
        event_authorities=authority_map,
    )
    return normalized, events


def _normalize_model_claims(
    model: dict[str, Any],
    *,
    initial_reference_map: dict[str, str] | None = None,
    known_event_ids: set[str] | None = None,
    event_authorities: dict[str, str] | None = None,
) -> dict[str, Any]:
    normalized = _copy(model)
    for field in ("decision", "stopping_condition", "top_unknown"):
        if not isinstance(normalized.get(field), str) or not normalized[field].strip():
            raise HarnessError(f"{field} must be a non-empty string")

    seen_ids: set[str] = set()
    for section in ("state_grounding", "mechanisms", "alternatives"):
        items = normalized.get(section)
        if not isinstance(items, list):
            raise HarnessError(f"{section} must be an array")
        for item in items:
            if not isinstance(item, dict):
                raise HarnessError(f"each {section} item must be an object")
            claim_id = str(item.get("id") or "").strip()
            text = str(item.get("text") or "").strip()
            if not claim_id or not text:
                raise HarnessError(f"each {section} item requires id and text")
            if claim_id in seen_ids:
                raise HarnessError(f"duplicate model item ID: {claim_id}")
            seen_ids.add(claim_id)
            refs = item.get("evidence_event_ids") or []
            if not isinstance(refs, list) or not all(isinstance(ref, str) for ref in refs):
                raise HarnessError(f"{claim_id} evidence_event_ids must be an array")
            mapped_refs: list[str] = []
            for ref in refs:
                if initial_reference_map is not None:
                    if ref not in initial_reference_map:
                        raise HarnessError(
                            f"new case claim {claim_id} cites undeclared initial evidence {ref}"
                        )
                    ref = initial_reference_map[ref]
                if known_event_ids is not None and ref not in known_event_ids:
                    raise HarnessError(f"claim {claim_id} cites unknown Timeline event {ref}")
                mapped_refs.append(ref)
            item["evidence_event_ids"] = mapped_refs
            status = str(item.get("epistemic_status") or "").strip().lower()
            if not status:
                status = "grounded" if mapped_refs else "hypothesis"
            if status not in {"grounded", "hypothesis", "unresolved"}:
                raise HarnessError(
                    f"{claim_id} epistemic_status must be grounded, hypothesis, or unresolved"
                )
            if status == "grounded" and not mapped_refs:
                raise HarnessError(f"grounded claim {claim_id} must cite a Timeline event")
            item["epistemic_status"] = status
            _validate_claim_authority(
                claim_id,
                item,
                mapped_refs,
                event_authorities=event_authorities,
            )

    plan = normalized.get("current_plan")
    if not isinstance(plan, list):
        raise HarnessError("current_plan must be an array")
    plan_ids: set[str] = set()
    for step in plan:
        if not isinstance(step, dict):
            raise HarnessError("each current_plan item must be an object")
        step_id = str(step.get("id") or "").strip()
        action = str(step.get("action") or "").strip()
        dependencies = step.get("depends_on") or []
        if not step_id or not action:
            raise HarnessError("each current_plan item requires id and action")
        if step_id in seen_ids or step_id in plan_ids:
            raise HarnessError(f"duplicate model item ID: {step_id}")
        if not isinstance(dependencies, list) or not all(
            isinstance(dependency, str) for dependency in dependencies
        ):
            raise HarnessError(f"plan step {step_id} depends_on must be an array")
        plan_ids.add(step_id)
    available_ids = seen_ids | plan_ids
    for step in plan:
        unknown = sorted(set(step.get("depends_on") or []) - available_ids)
        if unknown:
            raise HarnessError(
                f"plan step {step['id']} has unknown dependencies: {', '.join(unknown)}"
            )
    return normalized


def _event_authority_map(events: list[dict[str, Any]]) -> dict[str, str]:
    """Resolve evidence authority, including older events that only carry probe IDs."""
    probe_authorities: dict[str, str] = {}
    for event in events:
        if event.get("type") != "probe_committed":
            continue
        probe = (event.get("payload") or {}).get("probe") or {}
        probe_id = str(probe.get("probe_id") or "")
        authority = str(probe.get("authority_domain") or "")
        if probe_id and authority in _AUTHORITY_DOMAINS:
            probe_authorities[probe_id] = authority

    authorities: dict[str, str] = {}
    evidence_event_types = {
        "initial_observation",
        "evidence_recorded",
        "probe_observed",
        "probe_exposed",
        "probe_compared",
    }
    for event in events:
        if event.get("type") not in evidence_event_types:
            continue
        event_id = str(event.get("event_id") or "")
        payload = event.get("payload") or {}
        authority = str(payload.get("authority_domain") or "")
        # F6: for probe_compared events, the nested evidence descriptor carries the
        # observation's true authority. Prefer it over the probe's declared domain so
        # a user-testimony evidence object cannot be attributed to a non-user probe.
        if event.get("type") == "probe_compared":
            evidence = payload.get("evidence")
            if isinstance(evidence, dict):
                nested = str(evidence.get("authority_domain") or "")
                if nested in _AUTHORITY_DOMAINS:
                    authority = nested
        if authority not in _AUTHORITY_DOMAINS:
            authority = probe_authorities.get(str(payload.get("probe_id") or ""), "")
        if event_id and authority in _AUTHORITY_DOMAINS:
            authorities[event_id] = authority
    return authorities


def _validate_claim_authority(
    claim_id: str,
    item: dict[str, Any],
    evidence_event_ids: list[str],
    *,
    event_authorities: dict[str, str] | None,
) -> None:
    claim_authority = str(item.get("authority_domain") or "").strip()
    if claim_authority and claim_authority not in _AUTHORITY_DOMAINS:
        raise HarnessError(f"claim {claim_id} has an invalid authority_domain")
    if not evidence_event_ids:
        item.pop("grounding_authority_domains", None)
        return
    if not claim_authority:
        raise HarnessError(
            f"claim {claim_id} with evidence requires an authority_domain"
        )
    cited_authorities = {
        str((event_authorities or {}).get(event_id) or "")
        for event_id in evidence_event_ids
    }
    if "" in cited_authorities:
        raise HarnessError(
            f"claim {claim_id} cites a Timeline event without evidence authority"
        )
    claim_group = (
        "user" if claim_authority in _USER_AUTHORITY_DOMAINS else "external"
    )
    cited_groups = {
        "user" if authority in _USER_AUTHORITY_DOMAINS else "external"
        for authority in cited_authorities
    }
    incompatible = sorted(cited_groups - {claim_group})
    if incompatible:
        source_group = incompatible[0]
        raise HarnessError(
            f"{source_group} authority cannot ground {claim_group} claim {claim_id}"
        )
    item["authority_domain"] = claim_authority
    item["grounding_authority_domains"] = sorted(cited_authorities)


def _model_items_by_id(model: dict[str, Any]) -> dict[str, dict[str, Any]]:
    items: dict[str, dict[str, Any]] = {}
    for section in ("state_grounding", "mechanisms", "alternatives", "current_plan"):
        for item in model.get(section) or []:
            if isinstance(item, dict) and item.get("id"):
                items[str(item["id"])] = item
    return items


def _substantive_item_signature(item: dict[str, Any]) -> str:
    if "action" in item:
        material = {
            "action": " ".join(str(item.get("action") or "").split()),
            "depends_on": sorted(str(value) for value in (item.get("depends_on") or [])),
        }
    else:
        material = {"text": " ".join(str(item.get("text") or "").split())}
    return _canonical(material)


_UNCERTAINTY_DOWNGRADES = {
    ("grounded", "hypothesis"),
    ("grounded", "unresolved"),
    ("hypothesis", "unresolved"),
}


def _stale_item_change_is_substantive(
    old_item: dict[str, Any],
    new_item: dict[str, Any],
) -> bool:
    """Return whether a stale item received a consequential repair.

    Text/action changes remain the normal repair path. A status-only change is
    substantive only when it moves toward uncertainty; a status upgrade cannot
    clear a material mismatch without a new assertion or evidence-backed edit.
    Metadata and whitespace-only edits therefore remain ineffective.
    """
    if _substantive_item_signature(old_item) != _substantive_item_signature(new_item):
        return True
    if "action" in old_item or "action" in new_item:
        return False
    old_status = str(old_item.get("epistemic_status") or "").strip().lower()
    new_status = str(new_item.get("epistemic_status") or "").strip().lower()
    return (old_status, new_status) in _UNCERTAINTY_DOWNGRADES


def _expand_stale_model_ids(model: dict[str, Any], affected: list[str]) -> list[str]:
    """Include every plan step that transitively depends on an affected item."""
    stale = set(affected)
    plan = [item for item in (model.get("current_plan") or []) if isinstance(item, dict)]
    changed = True
    while changed:
        changed = False
        for step in plan:
            step_id = str(step.get("id") or "")
            dependencies = {str(item) for item in (step.get("depends_on") or [])}
            if step_id and step_id not in stale and dependencies.intersection(stale):
                stale.add(step_id)
                changed = True
    return sorted(stale)


def _model_relevance_text(model: dict[str, Any]) -> str:
    parts = [
        str(model.get("decision") or ""),
        str(model.get("top_unknown") or ""),
    ]
    for section in ("state_grounding", "mechanisms", "alternatives"):
        for item in model.get(section) or []:
            if isinstance(item, dict):
                parts.append(str(item.get("text") or ""))
    return " ".join(parts)


def _lesson_tokens(text: str) -> set[str]:
    return {
        token
        for token in re.findall(r"[a-z0-9]+", text.lower())
        if len(token) >= 4 and token not in _LESSON_STOPWORDS
    }


# Word families the lexical retrieval historically missed (2026-09-01 audit:
# the questioning-seed synthesis retrieved zero lessons despite a directly
# relevant lesson closed the previous day — "questioning/asks/asking" never
# matched "delegation/asks" at the 0.12 floor). A stem-class expansion gives
# the scorer a second channel: overlapping token CLASSES count like shared
# tokens. Deliberately small and hand-audited — this is a patch on lexical
# retrieval, not an attempt at embeddings.
_LESSON_STEM_CLASSES: dict[str, tuple[str, ...]] = {
    "question": ("question", "questions", "questioning", "asks", "asking", "asked", "inquiry", "inquiries", "interrogate", "interrogation"),
    "value": ("value", "values", "valuation", "worth", "reward", "payoff", "utility", "benefit", "demand"),
    "evidence": ("evidence", "empirical", "data", "findings", "results", "literature", "sources"),
    "humanai": ("human", "humans", "user", "users", "people", "agent", "agents", "ai", "machine", "machines", "model", "models"),
    "delegation": ("delegate", "delegates", "delegation", "delegating", "outsourcing", "handoff", "asking", "consultation", "advice", "advising"),
    "synthesis": ("synthesis", "synthesize", "report", "reports", "scan", "frontier", "landscape", "memo"),
    "experiment": ("experiment", "experiments", "experimental", "empirical", "field", "laboratory", "randomized", "treatment"),
}
_STEM_CLASS_BY_TOKEN: dict[str, str] = {
    token: name
    for name, family in _LESSON_STEM_CLASSES.items()
    for token in family
}


def _lesson_stem_classes(tokens: set[str]) -> set[str]:
    """Token classes present in a token set (second retrieval channel)."""
    return {
        name
        for token in tokens
        if (name := _STEM_CLASS_BY_TOKEN.get(token)) is not None
    }


def _transfer_support_ceiling(labels: list[str]) -> str:
    """Repeated observations cannot become stronger than their weakest source."""
    if not labels:
        return "hypothesis"
    levels = {
        "hypothesis": 0,
        "provisional": 1,
        "mixed": 1,
        "supported": 2,
    }
    weakest = min(levels.get(str(label), 0) for label in labels)
    return {0: "hypothesis", 1: "provisional", 2: "supported"}[weakest]


def _normalize_observation_status(value: Any) -> str:
    """Map one recorded status to the conservative calibration vocabulary."""
    if not isinstance(value, str) or not value.strip():
        return "unknown"
    normalized = value.strip().lower()
    if normalized == "ok":
        return "ok"
    if normalized in _FAILED_OBSERVATION_STATUSES:
        return "failed"
    return "unknown"


def _recorded_execution_status(payload: dict[str, Any]) -> str:
    """Resolve raw status fields without trusting an inconsistent pair."""
    result_status = _normalize_observation_status(payload.get("result_status"))
    if "execution_result_status" not in payload:
        return result_status
    execution_status = _normalize_observation_status(
        payload.get("execution_result_status")
    )
    # A failure in either field is safer than promoting a contradictory record;
    # two different non-failure statuses remain unknown.
    if "failed" in {result_status, execution_status}:
        return "failed"
    if (
        result_status != "unknown"
        and execution_status != "unknown"
        and result_status != execution_status
    ):
        return "unknown"
    return execution_status if execution_status != "unknown" else result_status


def _display_observation_status(payload: dict[str, Any]) -> str:
    """Resolve the latest display classification, conservatively."""
    result_status = _normalize_observation_status(payload.get("result_status"))
    if "display_result_status" not in payload:
        return result_status
    display_status = _normalize_observation_status(
        payload.get("display_result_status")
    )
    if "failed" in {result_status, display_status}:
        return "unknown"
    if (
        result_status != "unknown"
        and display_status != "unknown"
        and result_status != display_status
    ):
        return "unknown"
    return display_status if display_status != "unknown" else result_status


def _probe_execution_result_status(probe: dict[str, Any]) -> str:
    """Read the immutable execution status on a current probe.

    A legacy probe has no separate execution field. Its historical Timeline is
    resolved by the calibration chain; this helper must not guess that a
    model-visible ``result_status=ok`` was an execution success.
    """
    if "execution_result_status" not in probe:
        return "unknown"
    value = probe.get("execution_result_status")
    if not isinstance(value, str) or not value.strip():
        return "unknown"
    normalized = value.strip().lower()
    if normalized == "ok" or normalized in _FAILED_OBSERVATION_STATUSES:
        return normalized
    return "unknown"


def _observation_result_status(event: dict[str, Any]) -> str:
    """Classify one raw observation without inventing exposure provenance.

    An exposed representation cannot be classified in isolation: its
    ``supersedes_event_id`` must be resolved against the complete Timeline.
    Callers with the Timeline use ``_resolve_observation_result_status``.
    """
    if event.get("type") == "probe_exposed":
        return "unknown"
    payload = event.get("payload") or {}
    if not isinstance(payload, dict):
        return "unknown"
    return _recorded_execution_status(payload)


def _resolve_observation_result_status(
    event_id: str,
    events_by_id: dict[str, dict[str, Any]],
    *,
    expected_probe_id: str | None = None,
    visiting: set[str] | None = None,
) -> str:
    """Resolve execution outcome through an authoritative exposure chain.

    ``probe_observed`` is the execution boundary. ``probe_exposed`` records a
    later representation and must name the immediately preceding observation
    for the same probe. A failed predecessor remains failed through every
    display transformation. Missing, malformed, cyclic, or cross-probe links
    are unknown, never successful. A display error after an otherwise
    successful execution is also unknown: it describes no positive source
    observation.
    """
    event_id = str(event_id or "")
    if not event_id or event_id in (visiting or set()):
        return "unknown"
    event = events_by_id.get(event_id)
    if not isinstance(event, dict):
        return "unknown"
    event_type = event.get("type")
    if event_type not in {"probe_observed", "probe_exposed"}:
        return "unknown"
    payload = event.get("payload")
    if not isinstance(payload, dict):
        return "unknown"
    probe_id = str(payload.get("probe_id") or "").strip()
    if expected_probe_id is not None and probe_id != expected_probe_id:
        return "unknown"

    if event_type == "probe_observed":
        if payload.get("raw_result_provenance") == "unknown":
            return "unknown"
        return _recorded_execution_status(payload)

    # Exposure provenance is not optional. Requiring the source probe ID here
    # is what makes a valid-looking reference from another probe ineligible.
    if not probe_id:
        return "unknown"
    supersedes = payload.get("supersedes_event_id")
    if not isinstance(supersedes, str) or not supersedes.strip():
        return "unknown"
    predecessor = events_by_id.get(supersedes.strip())
    if not isinstance(predecessor, dict) or predecessor.get("type") not in {
        "probe_observed",
        "probe_exposed",
    }:
        return "unknown"
    predecessor_payload = predecessor.get("payload")
    if not isinstance(predecessor_payload, dict):
        return "unknown"
    predecessor_probe_id = str(predecessor_payload.get("probe_id") or "").strip()
    if not predecessor_probe_id or predecessor_probe_id != probe_id:
        return "unknown"

    next_visiting = set(visiting or set())
    next_visiting.add(event_id)
    predecessor_status = _resolve_observation_result_status(
        supersedes.strip(),
        events_by_id,
        expected_probe_id=probe_id,
        visiting=next_visiting,
    )
    if predecessor_status == "failed":
        return "failed"
    if predecessor_status != "ok":
        return "unknown"

    # ``result_status`` is retained for schema-v1 compatibility. New exposure
    # events carry the explicit display field so execution provenance is never
    # confused with the representation shown to the model.
    display_status = _display_observation_status(payload)
    return "ok" if display_status == "ok" else "unknown"


def _closure_calibration(
    case: dict[str, Any],
    *,
    events: list[dict[str, Any]],
) -> dict[str, Any]:
    """Derive claim and case support from the evidence actually inspected.

    A probe execution failure is evidence that the execution failed, not
    positive support for the external proposition the action was meant to
    inspect. This distinction is applied both when a claim cites the raw
    observation and when it cites the later comparison event.
    """
    events_by_id = {
        str(event.get("event_id") or ""): event
        for event in events
        if isinstance(event, dict) and event.get("event_id")
    }
    claim_assessments: list[dict[str, Any]] = []
    grounded_claim_ids: list[str] = []
    evidence_event_ids: set[str] = set()
    lead_only_ids: set[str] = set()
    direct_ids: set[str] = set()
    failed_observation_ids: set[str] = set()
    legacy_observation_ids: set[str] = set()
    for section in ("state_grounding", "mechanisms", "alternatives"):
        for item in (case.get("model") or {}).get(section) or []:
            if not isinstance(item, dict):
                continue
            references = {
                str(event_id)
                for event_id in (item.get("evidence_event_ids") or [])
                if event_id
            }
            claim_id = str(item.get("id") or "unknown")
            assessment: dict[str, Any] = {
                "claim_id": claim_id,
                "section": section,
                "evidence_event_ids": sorted(references),
            }
            if item.get("epistemic_status") != "grounded" or not references:
                assessment.update({"label": "hypothesis", "source_roles": [], "access_scopes": []})
                claim_assessments.append(assessment)
                continue

            grounded_claim_ids.append(claim_id)
            evidence_event_ids.update(references)
            claim_leads: set[str] = set()
            claim_direct: set[str] = set()
            claim_failed: set[str] = set()
            claim_legacy: set[str] = set()
            source_roles: set[str] = set()
            access_scopes: set[str] = set()
            limitations: list[str] = []
            for event_id in sorted(references):
                event = events_by_id.get(event_id) or {}
                payload = event.get("payload")
                if not isinstance(payload, dict):
                    payload = {}
                descriptor = payload.get("evidence") if event.get("type") == "probe_compared" else payload
                if not isinstance(descriptor, dict):
                    descriptor = {}
                role = str(descriptor.get("source_role") or "").strip().lower()
                scope = str(descriptor.get("access_scope") or "").strip().lower()
                limitation = str(descriptor.get("limitation") or "").strip()
                if role in _SOURCE_ROLES:
                    source_roles.add(role)
                if scope in _ACCESS_SCOPES:
                    access_scopes.add(scope)
                if limitation:
                    limitations.append(limitation)

                # A comparison is only as strong as the actual linked
                # observation. Never let a caller-supplied comparison
                # descriptor replace a failed, unavailable, interrupted, or
                # legacy execution status.
                observation = event
                observation_payload = payload
                event_type = event.get("type")
                observation_status = (
                    "ok"
                    if event_type in {"initial_observation", "evidence_recorded"}
                    else "unknown"
                )
                expected_probe_id = str(payload.get("probe_id") or "").strip() or None
                if event_type in {"probe_observed", "probe_exposed"}:
                    observation_status = _resolve_observation_result_status(
                        event_id,
                        events_by_id,
                        expected_probe_id=expected_probe_id,
                    )
                elif event_type == "probe_compared":
                    observation_id = str(payload.get("observation_event_id") or "")
                    observation = events_by_id.get(observation_id) or {}
                    observation_payload = observation.get("payload")
                    if not isinstance(observation_payload, dict):
                        observation_payload = {}
                    observation_status = _resolve_observation_result_status(
                        observation_id,
                        events_by_id,
                        expected_probe_id=expected_probe_id,
                    )

                if event_type in {
                    "probe_observed",
                    "probe_exposed",
                    "probe_compared",
                }:
                    if observation_status == "failed":
                        claim_failed.add(event_id)
                        failed_observation_ids.add(event_id)
                        continue
                    if observation_status != "ok":
                        claim_legacy.add(event_id)
                        legacy_observation_ids.add(event_id)
                        continue

                tool_name = str(observation_payload.get("tool_name") or "").lower()
                if (
                    observation.get("type") in {"probe_observed", "probe_exposed"}
                    and "search" in tool_name
                ):
                    # Search results remain leads even if a comparison payload
                    # claims a broader access scope. Full-text support requires
                    # a separately inspected source observation.
                    claim_leads.add(event_id)
                    continue
                if scope in {"snippet", "abstract"}:
                    claim_leads.add(event_id)
                    continue
                if scope in {"partial", "full", "data"}:
                    claim_direct.add(event_id)
                    continue

                # Backward compatibility for schema-v1 evidence: searches are
                # leads; other successful typed observations retain their
                # historical direct-evidence treatment until re-assessed.
                if (
                    observation.get("type") in {"probe_observed", "probe_exposed"}
                    and "search" in tool_name
                ):
                    claim_leads.add(event_id)
                else:
                    claim_direct.add(event_id)

            lead_only_ids.update(claim_leads)
            direct_ids.update(claim_direct)
            assessment.update(
                {
                    "label": "supported" if claim_direct else "provisional" if claim_leads else "hypothesis",
                    "source_roles": sorted(source_roles),
                    "access_scopes": sorted(access_scopes),
                    "limitations": sorted(set(limitations)),
                    "lead_only_event_ids": sorted(claim_leads),
                    "direct_event_ids": sorted(claim_direct),
                    "failed_observation_event_ids": sorted(claim_failed),
                    "legacy_observation_event_ids": sorted(claim_legacy),
                }
            )
            claim_assessments.append(assessment)

    grounded_labels = {
        item["label"]
        for item in claim_assessments
        if item["claim_id"] in grounded_claim_ids
    }
    # An ungrounded hypothesis among the ASSERTED beliefs (state_grounding,
    # mechanisms) caps the overall label: the case cannot read "supported"
    # while part of what it holds true is still a hypothesis. Alternatives are
    # the option set — rivals remain hypotheses by design and do not cap.
    has_ungrounded_hypothesis = any(
        item["label"] == "hypothesis"
        and item["claim_id"] not in grounded_claim_ids
        and item["section"] in {"state_grounding", "mechanisms"}
        for item in claim_assessments
    )
    if grounded_labels == {"supported"} and not has_ungrounded_hypothesis:
        label = "supported"
    elif grounded_labels == {"provisional"} and not has_ungrounded_hypothesis:
        label = "provisional"
    elif grounded_labels and grounded_labels != {"hypothesis"}:
        label = "mixed"
    else:
        label = "hypothesis"
    return {
        "label": label,
        "grounded_claim_ids": sorted(grounded_claim_ids),
        "evidence_event_ids": sorted(evidence_event_ids),
        "lead_only_event_ids": sorted(lead_only_ids),
        "direct_event_ids": sorted(direct_ids),
        "failed_observation_event_ids": sorted(failed_observation_ids),
        "legacy_observation_event_ids": sorted(legacy_observation_ids),
        "claims": claim_assessments,
        "language": (
            "supported, not confirmed or proven"
            if label == "supported"
            else "mixed support; distinguish grounded claims from provisional or ungrounded ones"
            if label == "mixed"
            else "provisional; inspect primary evidence before stronger language"
            if label == "provisional"
            else "hypothesis; do not describe as resolved"
        ),
    }


class _DuplicateJSONKeyError(ValueError):
    """Raised when a provider JSON object contains a duplicate key."""


def _ensure_finite_json_values(value: Any) -> None:
    """Reject values that cannot be represented as strict JSON."""
    if isinstance(value, float) and not math.isfinite(value):
        raise HarnessError("tool arguments must contain only finite JSON numbers")
    if isinstance(value, dict):
        for item in value.values():
            _ensure_finite_json_values(item)
    elif isinstance(value, list):
        for item in value:
            _ensure_finite_json_values(item)


def _resolve_probe_tool_args(args: dict[str, Any]) -> dict[str, Any]:
    """Resolve the provider-safe JSON argument route before a probe is committed."""
    if "tool_args_json" not in args:
        tool_args = args.get("tool_args")
        if not isinstance(tool_args, dict):
            raise HarnessError("tool_args must be an object")
        _ensure_finite_json_values(tool_args)
        return tool_args

    raw_json = args.get("tool_args_json")
    if not isinstance(raw_json, str):
        raise HarnessError("tool_args_json must be a JSON object string")

    def _object_pairs(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        result: dict[str, Any] = {}
        for key, value in pairs:
            if key in result:
                raise _DuplicateJSONKeyError(f"duplicate key {key!r}")
            result[key] = value
        return result

    def _reject_constant(value: str) -> Any:
        raise ValueError(f"non-finite number {value}")

    try:
        decoded = json.loads(
            raw_json,
            object_pairs_hook=_object_pairs,
            parse_constant=_reject_constant,
        )
    except _DuplicateJSONKeyError as exc:
        raise HarnessError(f"tool_args_json contains duplicate keys: {exc}") from exc
    except (TypeError, ValueError, json.JSONDecodeError) as exc:
        raise HarnessError(
            "tool_args_json must be valid JSON with unique keys and finite numbers"
        ) from exc
    if not isinstance(decoded, dict):
        raise HarnessError("tool_args_json must decode to a JSON object")
    _ensure_finite_json_values(decoded)

    if "tool_args" in args:
        legacy = args.get("tool_args")
        if legacy is not None and legacy != {}:
            if not isinstance(legacy, dict):
                raise HarnessError(
                    "legacy tool_args must be absent, null, or an object when tool_args_json is supplied"
                )
            _ensure_finite_json_values(legacy)
            if _canonical(legacy) != _canonical(decoded):
                raise HarnessError(
                    "tool_args_json is authoritative; nonempty legacy tool_args must canonically agree"
                )
    return decoded


def _canonical(value: Any) -> str:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _probe_args_match(store: CaseStore, probe: dict[str, Any], args: dict[str, Any]) -> bool:
    """Check native observer arguments against the committed execution digest."""
    expected = str(
        probe.get("execution_tool_args_sha256")
        or probe.get("tool_args_sha256")
        or ""
    )
    if not expected:
        return False
    supplied = (
        store.commitment_digest(_canonical(args))
        if probe.get("tool_args_commitment_scheme") == "hmac-sha256-v1"
        else _sha256(_canonical(args))
    )
    return supplied == expected


def _result_text(result: Any) -> str:
    """Serialize a native result without losing structured JSON values."""
    return (
        result
        if isinstance(result, str)
        else json.dumps(result, ensure_ascii=False, sort_keys=True, default=str)
    )


def _copy(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))


def _sha256(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def _blocked(reason: str) -> str:
    return json.dumps({"error": f"EPISTEMIC GATEWAY BLOCKED: {reason}"}, ensure_ascii=False)


def _classify_tool_result(result: Any) -> str:
    parsed = result
    if isinstance(result, str):
        try:
            parsed = json.loads(result)
        except (TypeError, json.JSONDecodeError):
            return "ok"
    if not isinstance(parsed, dict):
        return "ok"
    error_value = parsed.get("error")
    empty_error = (
        error_value is None
        or error_value is False
        or (isinstance(error_value, str) and error_value == "")
    )
    if "error" in parsed and not empty_error:
        return "tool_error"
    if parsed.get("success") is False:
        return "tool_error"
    return "ok"


def _redact_value(value: Any, *, known_secrets: set[str] | None = None) -> Any:
    if isinstance(value, str):
        text = _redact(value)
        for secret in known_secrets or set():
            text = text.replace(secret, "[REDACTED]")
        return text
    if isinstance(value, list):
        return [_redact_value(item, known_secrets=known_secrets) for item in value]
    if isinstance(value, dict):
        redacted: dict[str, Any] = {}
        for key, item in value.items():
            safe_key = _redact(str(key))
            compact_key = re.sub(r"[^a-z0-9]", "", str(key).lower())
            redacted[safe_key] = (
                "[REDACTED]"
                if isinstance(item, str) and compact_key in _SECRET_KEY_NAMES
                else _redact_value(item, known_secrets=known_secrets)
            )
        return redacted
    return _copy(value)


def _secret_values(value: Any) -> set[str]:
    secrets: set[str] = set()
    if isinstance(value, dict):
        for key, item in value.items():
            compact_key = re.sub(r"[^a-z0-9]", "", str(key).lower())
            if isinstance(item, str) and compact_key in _SECRET_KEY_NAMES and item:
                secrets.add(item)
            else:
                secrets.update(_secret_values(item))
    elif isinstance(value, list):
        for item in value:
            secrets.update(_secret_values(item))
    return secrets


def _redact(text: str) -> str:
    # Durable evidence is a stricter privacy boundary than transient display:
    # replace common credential-shaped tokens outright before the broader
    # Hermes redactor applies its provider-specific masking.
    text = re.sub(r"\bgh[pousr]_[A-Za-z0-9_]{8,}\b", "[REDACTED]", text)
    text = re.sub(r"\bsk-[A-Za-z0-9_-]{12,}\b", "[REDACTED]", text)
    text = re.sub(
        r"(?i)(authorization\s*[:=]\s*(?:bearer\s+)?)[^\s,;]+",
        r"\1[REDACTED]",
        text,
    )
    try:
        from agent.redact import redact_sensitive_text

        return redact_sensitive_text(text, force=True)
    except ImportError:
        return text


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()
