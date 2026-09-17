# AI-assisted contribution; maintained by Epistemic Harness contributors.
"""Profile-scoped, append-only storage for epistemic cases."""

from __future__ import annotations

import errno
import hashlib
import hmac
import json
import os
import re
import threading
import time
import uuid
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from .case_index import CaseIndex
from .locking import LockTimeoutError, acquire_lock, remaining_lock_seconds

try:
    import fcntl
except ImportError:  # pragma: no cover - non-POSIX fallback
    fcntl = None  # type: ignore[assignment]

_SCHEMA_VERSION = 1
_ZERO_HASH = "0" * 64
_CASE_ID_RE = re.compile(r"^[a-z0-9][a-z0-9-]{0,63}$")
_DATA_START = "<!-- EPISTEMIC-HARNESS-DATA\n"
_DATA_END = "\nEND-EPISTEMIC-HARNESS-DATA -->"

# Event types that change a case's lifecycle disposition (open, pause, resume,
# close, migrate). Only these advance the per-session high-water sequence used
# by the auto-resume eligibility marker; observational events (evidence,
# revisions, probe lifecycle) never move it, so noise cannot invalidate a
# valid marker. The set names the types the code has always emitted —
# renaming types would break replay compatibility with existing Timelines.
_DISPOSITION_EVENT_TYPES = frozenset({
    "case_opened",
    "case_paused",
    "case_resumed",
    "case_closed",
    "session_migrated",
    "session_migration_aborted",
})
_SESSIONS_CACHE_NAME = "sessions.json"


class StoreError(RuntimeError):
    """Raised when durable case invariants cannot be satisfied."""


class ProbeExecutionBusyError(StoreError):
    """Raised when another invocation already owns a case's probe lease."""


class CaseStore:
    """Store cases as ``case.md``, ``events.jsonl``, and ``archive/``."""

    def __init__(self, root: str | Path):
        if fcntl is None:
            raise StoreError(
                "epistemic-harness requires POSIX fcntl cross-process locking; "
                "no safe backend is available on this platform"
            )
        self.root = Path(root)
        self.cases_dir = self.root / "cases"
        self.compression_boundaries_dir = self.root / "compression-boundaries"
        self._ensure_dir_durable(self.root)
        self._ensure_dir_durable(self.cases_dir)
        self._ensure_dir_durable(self.compression_boundaries_dir)
        self._lock = threading.RLock()
        self._case_index = CaseIndex()
        self._live_execution_leases: set[str] = set()
        self._commitment_key = self._load_or_create_commitment_key()
        self._recover_pending_transactions()

    def commitment_digest(self, canonical_value: str) -> str:
        """Bind exact arguments without exposing an offline unsalted hash oracle."""
        return hmac.new(
            self._commitment_key,
            canonical_value.encode("utf-8"),
            hashlib.sha256,
        ).hexdigest()

    def _load_or_create_commitment_key(self) -> bytes:
        key_path = self.root / ".commitment.key"
        try:
            descriptor = os.open(key_path, os.O_WRONLY | os.O_CREAT | os.O_EXCL, 0o600)
        except FileExistsError:
            try:
                key = key_path.read_bytes()
            except OSError as exc:
                raise StoreError(f"cannot read commitment key: {exc}") from exc
        else:
            key = os.urandom(32)
            try:
                with os.fdopen(descriptor, "wb") as handle:
                    handle.write(key)
                    handle.flush()
                    os.fsync(handle.fileno())
                self._fsync_dir(key_path.parent)
            except Exception:
                try:
                    key_path.unlink()
                except OSError:
                    pass
                raise
        if len(key) != 32:
            raise StoreError("commitment key is malformed")
        return key

    def open_case(
        self,
        *,
        session_id: str,
        case_id: str,
        model: dict[str, Any],
        initial_events: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        """Create and bind one active case to ``session_id``."""
        self._validate_case_id(case_id)
        if not session_id.strip():
            raise StoreError("session_id is required")
        if not isinstance(model, dict):
            raise StoreError("model must be an object")

        with self._store_lock(), self._process_lock():
            self._recover_pending_transactions_locked()
            if self._get_active_case_no_recovery(session_id) is not None:
                raise StoreError(f"session {session_id!r} already has an active case")
            case_dir = self._case_dir(case_id)
            self._discard_empty_incomplete_case_dir(case_dir)
            if case_dir.exists():
                raise StoreError(f"case {case_id!r} already exists")
            self._ensure_dir_durable(case_dir)
            self._ensure_dir_durable(case_dir / "archive")
            self._fsync_dir(case_dir)
            self._fsync_dir(self.cases_dir)

            now = _utc_now()
            case = {
                "schema_version": _SCHEMA_VERSION,
                "case_id": case_id,
                "status": "active",
                "session_id": session_id,
                "version": 1,
                "created_at": now,
                "updated_at": now,
                "stale": False,
                "pending_probe": None,
                "model": model,
            }
            event_specs = [
                {
                    "type": str(item.get("type") or "initial_observation"),
                    "session_id": session_id,
                    "payload": dict(item.get("payload") or {}),
                }
                for item in (initial_events or [])
            ]
            event_specs.append(
                {
                    "type": "case_opened",
                    "session_id": session_id,
                    "payload": {"model_version": 1},
                }
            )
            self._commit_transaction(case, event_specs, write_snapshot=True)
            return _copy(case)

    def get_case(self, case_id: str) -> dict[str, Any]:
        self._validate_case_id(case_id)
        with self._store_lock(), self._process_lock():
            self._recover_pending_transactions_locked()
            path = self._case_dir(case_id) / "case.md"
            if not path.exists():
                raise StoreError(f"case {case_id!r} does not exist")
            return self._read_case(path, expected_case_id=case_id)

    def get_active_case(self, session_id: str) -> dict[str, Any] | None:
        """Return the active case bound to ``session_id``, if any."""
        with self._store_lock(), self._process_lock():
            self._recover_pending_transactions_locked()
            return self._get_active_case_no_recovery(session_id)

    def has_any_active_case(self) -> bool:
        """Return whether any session currently owns an active case."""
        with self._store_lock(), self._process_lock():
            self._recover_pending_transactions_locked()
            for case in self._case_index.select(self):
                if case.get("status") == "active":
                    return True
            return False

    def list_case_ids(self) -> list[str]:
        """Return the stable IDs of complete cases without reading their contents."""
        with self._store_lock(), self._process_lock():
            self._recover_pending_transactions_locked()
            return [
                path.parent.name
                for path in sorted(self.cases_dir.glob("*/case.md"))
                if _CASE_ID_RE.fullmatch(path.parent.name)
            ]

    def case_errors(self) -> list[str]:
        """IDs excluded from the healthy portfolio due to unreadable records."""
        with self._store_lock(), self._process_lock():
            self._case_index.refresh(self)
            return sorted(self._case_index.errors)

    def list_case_summaries(
        self,
        *,
        status: str | None = None,
        session_id: str | None = None,
    ) -> list[dict[str, Any]]:
        """Return compact portfolio metadata without mutating case state."""
        if status is not None and status not in {"active", "paused", "closed"}:
            raise StoreError("case status filter must be active, paused, or closed")
        if session_id is not None and not session_id.strip():
            raise StoreError("case session filter cannot be empty")

        summaries: list[dict[str, Any]] = []
        with self._store_lock(), self._process_lock():
            self._recover_pending_transactions_locked()
            for case in self._case_index.select(self):
                case_path = self._case_dir(case["case_id"]) / "case.md"
                if status is not None and case.get("status") != status:
                    continue
                events = self._read_events(case_path.parent / "events.jsonl")
                session_ids: list[str] = []
                seen_sessions: set[str] = set()
                for event in events:
                    event_session = str(event.get("session_id") or "").strip()
                    if event_session and event_session not in seen_sessions:
                        session_ids.append(event_session)
                        seen_sessions.add(event_session)
                current_session = str(case.get("session_id") or "").strip()
                if current_session and current_session not in seen_sessions:
                    session_ids.append(current_session)
                if session_id is not None and session_id not in session_ids:
                    continue
                model = case.get("model") if isinstance(case.get("model"), dict) else {}
                summary = str(model.get("decision") or model.get("top_unknown") or "").strip()
                summary = " ".join(summary.split())[:240]
                pending = case.get("pending_probe")
                summaries.append({
                    "case_id": str(case.get("case_id") or ""),
                    "status": str(case.get("status") or ""),
                    "created_at": str(case.get("created_at") or ""),
                    "updated_at": str(case.get("updated_at") or ""),
                    "summary": summary,
                    "session_id": current_session,
                    "session_ids": session_ids,
                    "event_count": len(events),
                    "pending_probe_status": (
                        str(pending.get("status") or "")
                        if isinstance(pending, dict)
                        else None
                    ),
                    "epistemically_stale": bool(case.get("stale")),
                })
        summaries.sort(key=lambda item: item["case_id"])
        summaries.sort(key=lambda item: item["updated_at"], reverse=True)
        return summaries

    def migrate_active_case(
        self,
        *,
        old_session_id: str,
        new_session_id: str,
    ) -> dict[str, Any] | None:
        """Atomically rebind one active case to its compressed continuation."""
        if not old_session_id or not new_session_id:
            raise StoreError("both old and new session IDs are required for migration")
        with self._store_lock(), self._process_lock():
            self._recover_pending_transactions_locked()
            result = self._migrate_active_case_locked(
                old_session_id=old_session_id,
                new_session_id=new_session_id,
                event_type="session_migrated",
                reason="Hermes conversation compression",
            )
            return result

    def prepare_compression(
        self,
        *,
        old_session_id: str,
        new_session_id: str,
    ) -> dict[str, Any] | None:
        """Durably record a case migration before Hermes publishes the child."""
        if not old_session_id or not new_session_id or old_session_id == new_session_id:
            raise StoreError("distinct old and new session IDs are required")
        with self._store_lock(), self._process_lock():
            self._recover_pending_transactions_locked()
            case = self._get_active_case_no_recovery(old_session_id)
            if case is None:
                return None
            conflict = self._get_active_case_no_recovery(new_session_id)
            if conflict is not None and conflict.get("case_id") != case.get("case_id"):
                raise StoreError(f"new session {new_session_id!r} already has an active case")
            for _, intent in self._compression_intents_locked():
                if intent["old_session_id"] == old_session_id and intent["new_session_id"] != new_session_id:
                    raise StoreError("old session already has a different prepared compression")
            intent = {
                "schema_version": _SCHEMA_VERSION,
                "case_id": case["case_id"],
                "old_session_id": old_session_id,
                "new_session_id": new_session_id,
                "created_at": _utc_now(),
            }
            self._atomic_write(
                self._compression_intent_path(old_session_id, new_session_id),
                json.dumps(intent, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            )
            return _copy(intent)

    def recover_prepared_compression(self, new_session_id: str) -> dict[str, Any] | None:
        """Complete a prepared migration before the child can execute anything."""
        if not new_session_id:
            return None
        with self._store_lock(), self._process_lock():
            self._recover_pending_transactions_locked()
            matches = [
                (path, intent)
                for path, intent in self._compression_intents_locked()
                if intent["new_session_id"] == new_session_id
            ]
            if not matches:
                return None
            if len(matches) != 1:
                raise StoreError("multiple prepared compressions target the same session")
            _, intent = matches[0]
            old_session_id = intent["old_session_id"]
            case_id = intent["case_id"]
            active_new = self._get_active_case_no_recovery(new_session_id)
            active_old = self._get_active_case_no_recovery(old_session_id)
            if active_new is not None:
                if active_new.get("case_id") != case_id:
                    raise StoreError("prepared compression targets a conflicting active case")
                result = {"case": _copy(active_new), "event": None}
            elif active_old is not None:
                if active_old.get("case_id") != case_id:
                    raise StoreError("prepared compression source case changed")
                result = self._migrate_active_case_locked(
                    old_session_id=old_session_id,
                    new_session_id=new_session_id,
                    event_type="session_migrated",
                    reason="recovered prepared Hermes compression",
                )
            else:
                case_path = self._case_dir(case_id) / "case.md"
                case = (
                    self._read_case(case_path, expected_case_id=case_id)
                    if case_path.exists()
                    else None
                )
                if case is None or case.get("status") == "active":
                    raise StoreError("prepared compression lost its active case binding")
                result = None
            self._clear_compression_intent_locked(old_session_id, new_session_id)
            return result

    def get_prepared_compression_from(
        self,
        old_session_id: str,
    ) -> dict[str, Any] | None:
        """Return the unique durable intent originating at ``old_session_id``."""
        with self._store_lock(), self._process_lock():
            self._recover_pending_transactions_locked()
            matches = [
                intent
                for _, intent in self._compression_intents_locked()
                if intent["old_session_id"] == old_session_id
            ]
            if len(matches) > 1:
                raise StoreError(
                    f"multiple prepared compressions originate at {old_session_id!r}"
                )
            return _copy(matches[0]) if matches else None

    def abort_prepared_compression(
        self,
        *,
        old_session_id: str,
        new_session_id: str,
    ) -> dict[str, Any] | None:
        """Compensate a failed core rotation before Hermes reopens the parent."""
        with self._store_lock(), self._process_lock():
            self._recover_pending_transactions_locked()
            path = self._compression_intent_path(old_session_id, new_session_id)
            if not path.exists():
                return None
            intent = self._read_compression_intent(path)
            active_old = self._get_active_case_no_recovery(old_session_id)
            active_new = self._get_active_case_no_recovery(new_session_id)
            if active_old is not None:
                if active_old.get("case_id") != intent["case_id"]:
                    raise StoreError("prepared compression source case changed")
                result = {"case": _copy(active_old), "event": None}
            elif active_new is not None:
                if active_new.get("case_id") != intent["case_id"]:
                    raise StoreError("prepared compression target case changed")
                result = self._migrate_active_case_locked(
                    old_session_id=new_session_id,
                    new_session_id=old_session_id,
                    event_type="session_migration_aborted",
                    reason="Hermes compression rollback",
                )
            else:
                raise StoreError("prepared compression lost its active case binding")
            self._clear_compression_intent_locked(old_session_id, new_session_id)
            return result

    def _migrate_active_case_locked(
        self,
        *,
        old_session_id: str,
        new_session_id: str,
        event_type: str,
        reason: str,
    ) -> dict[str, Any] | None:
        case = self._get_active_case_no_recovery(old_session_id)
        if case is None:
            return None
        conflict = self._get_active_case_no_recovery(new_session_id)
        if conflict is not None and conflict.get("case_id") != case.get("case_id"):
            raise StoreError(f"new session {new_session_id!r} already has an active case")
        case_id = str(case.get("case_id") or "")
        # A session binding is part of the epistemic transaction.  Fence every
        # migration with the same lease held from probe_started through
        # probe_observed so compression cannot strand a committed observation
        # in the executing state.  The lease is non-blocking: recovery fails
        # closed and leaves its durable intent for a later retry.
        with self.probe_execution_lease(case_id):
            case["session_id"] = new_session_id
            case["updated_at"] = _utc_now()
            events = self._commit_transaction(
                case,
                [{
                    "type": event_type,
                    "session_id": new_session_id,
                    "payload": {
                        "old_session_id": old_session_id,
                        "new_session_id": new_session_id,
                        "reason": reason,
                    },
                }],
                write_snapshot=False,
            )
        return {"case": _copy(case), "event": events[-1]}

    def _compression_intent_path(self, old_session_id: str, new_session_id: str) -> Path:
        digest = hashlib.sha256(
            f"{old_session_id}\0{new_session_id}".encode("utf-8")
        ).hexdigest()
        return self.compression_boundaries_dir / f"{digest}.json"

    def _compression_intents_locked(self) -> list[tuple[Path, dict[str, Any]]]:
        return [
            (path, self._read_compression_intent(path))
            for path in sorted(self.compression_boundaries_dir.glob("*.json"))
        ]

    def _read_compression_intent(self, path: Path) -> dict[str, Any]:
        try:
            intent = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise StoreError(f"invalid prepared compression intent: {path.name}") from exc
        if not isinstance(intent, dict):
            raise StoreError("prepared compression intent must be an object")
        case_id = str(intent.get("case_id") or "")
        self._validate_case_id(case_id)
        old_session_id = str(intent.get("old_session_id") or "")
        new_session_id = str(intent.get("new_session_id") or "")
        if not old_session_id or not new_session_id or old_session_id == new_session_id:
            raise StoreError("prepared compression intent has invalid session IDs")
        if path != self._compression_intent_path(old_session_id, new_session_id):
            raise StoreError("prepared compression intent filename does not match its contents")
        return intent

    def _clear_compression_intent_locked(
        self,
        old_session_id: str,
        new_session_id: str,
    ) -> None:
        path = self._compression_intent_path(old_session_id, new_session_id)
        try:
            path.unlink()
        except FileNotFoundError:
            return
        self._fsync_dir(self.compression_boundaries_dir)

    def lifecycle_pause(
        self,
        session_id: str,
        *,
        event_payload: dict[str, Any],
        marker: dict[str, Any] | None,
    ) -> dict[str, Any] | None:
        """Pause the session's active case atomically for a lifecycle event.

        Reads the active case and applies the transition under one store lock
        so a concurrent probe/evidence transition cannot cause an optimistic
        conflict and drop the pause (the finding that motivated this). The
        case is re-read inside the lock; the marker, when provided, is stamped
        and its ``seq_at_pause`` filled from the pause event's disposition
        sequence. Returns the transition result, or None when the session has
        no active case.
        """
        if not session_id:
            return None
        with self._store_lock(), self._process_lock():
            self._recover_pending_transactions_locked()
            case = self._get_active_case_no_recovery(session_id)
            if case is None:
                return None
            pending = case.get("pending_probe")
            case["status"] = "paused"
            if marker is not None:
                case["auto_resume_pending"] = dict(marker)
            payload = dict(event_payload)
            payload.setdefault("pending_probe_status", (
                pending.get("status") if isinstance(pending, dict) else None
            ))
            events = self._commit_transaction(
                case,
                [{"type": "case_paused", "session_id": session_id, "payload": payload}],
                write_snapshot=False,
            )
            return {"case": _copy(case), "event": events[-1]}

    def resume_paused_case(self, *, case_id: str, session_id: str) -> dict[str, Any]:
        """Atomically claim a free session and reactivate exactly one paused case."""
        self._validate_case_id(case_id)
        if not session_id.strip():
            raise StoreError("session_id is required")
        with self._store_lock(), self._process_lock():
            self._recover_pending_transactions_locked()
            if self._get_active_case_no_recovery(session_id) is not None:
                raise StoreError("this session already has an active epistemic case")
            case_path = self._case_dir(case_id) / "case.md"
            if not case_path.exists():
                raise StoreError(f"case {case_id!r} does not exist")
            case = self._read_case(case_path, expected_case_id=case_id)
            if case.get("status") != "paused":
                raise StoreError("only a paused case can be resumed")
            if not self.validate_case_chain_and_snapshot(
                case,
                expected_case_id=case_id,
            ):
                raise StoreError(
                    "cannot resume case: authoritative case snapshot integrity check failed"
                )
            # An explicit resume consumes the case's auto-resume marker,
            # regardless of which session the case was bound to when paused.
            case.pop("auto_resume_pending", None)
            prior_session_id = str(case.get("session_id") or "")
            case["session_id"] = session_id
            case["status"] = "active"
            case["updated_at"] = _utc_now()
            events = self._commit_transaction(
                case,
                [{
                    "type": "case_resumed",
                    "session_id": session_id,
                    "payload": {"prior_session_id": prior_session_id},
                }],
                write_snapshot=False,
            )
            return {"case": _copy(case), "event": events[-1]}

    def _get_active_case_no_recovery(self, session_id: str) -> dict[str, Any] | None:
        if not session_id:
            return None
        active = [case for case in self._case_index.select(self, session_id) if case.get("status") == "active"]
        if len(active) > 1:
            raise StoreError("multiple active cases bound to this session")
        return active[0] if active else None

    # -- per-session disposition high-water (auto-resume eligibility) --------
    #
    # Every journaled DISPOSITION event carries an immutable top-level "seq":
    # the next sequence number of the session that disposition acted on. The
    # sessions.json cache beside the cases directory is the durable
    # write-through image of each session's high-water mark. It is reconciled
    # from pending journals before they are unlinked, so the pending journal
    # is the durable cursor proving which sequences the cache incorporates.

    def _sessions_cache_path(self) -> Path:
        return self.root / _SESSIONS_CACHE_NAME

    def _read_sessions_cache_locked(self) -> dict[str, int]:
        """Read the high-water cache, rebuilding from Timelines on corruption.

        Legacy events carry no "seq"; their sessions simply start at 0. Any
        non-integral or negative value is corruption and triggers a rebuild
        from the immutable event logs, never a silent restart for that
        session (which would let a stale marker re-validate).
        """
        path = self._sessions_cache_path()
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
            sessions = data.get("sessions") if isinstance(data, dict) else None
            if not isinstance(sessions, dict):
                raise ValueError("missing sessions mapping")
            result: dict[str, int] = {}
            for key, value in sessions.items():
                if not isinstance(key, str) or not isinstance(value, int) or value < 0:
                    raise ValueError(f"invalid session high-water entry {key!r}/{value!r}")
                result[key] = value
            return result
        except (OSError, ValueError, TypeError):
            # Missing OR malformed: in both cases the cache cannot be trusted
            # as the durable high-water image. Rebuild from the immutable
            # Timelines — never silently restart sequences (a deleted cache
            # would let disposition sequences restart at 1 and eventually
            # re-qualify an invalidated stale marker for auto-resume).
            # Journal-held sequences are folded in first: every public entry
            # point runs _recover_pending_transactions_locked() before any
            # allocation, so the rebuild happens after journal recovery.
            rebuilt = self._rebuild_sessions_cache_locked()
            self._write_sessions_cache_locked(rebuilt)
            return rebuilt

    def _rebuild_sessions_cache_locked(self) -> dict[str, int]:
        """Derive every session's high-water from the immutable event logs."""
        high_water: dict[str, int] = {}
        for events_path in sorted(self.cases_dir.glob("*/events.jsonl")):
            for event in self._read_events(events_path):
                seq = event.get("seq")
                session_id = str(event.get("session_id") or "")
                if session_id and isinstance(seq, int) and seq >= 1:
                    if seq > high_water.get(session_id, 0):
                        high_water[session_id] = seq
        return high_water

    def _write_sessions_cache_locked(self, mapping: dict[str, int]) -> None:
        payload = {
            "schema_version": _SCHEMA_VERSION,
            "sessions": dict(sorted(mapping.items())),
        }
        self._atomic_write(
            self._sessions_cache_path(),
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
        )

    def _next_session_seq_locked(self, allocated: dict[str, int], session_id: str) -> int:
        """Allocate one disposition sequence number inside the current lock.

        ``allocated`` carries numbers already handed out by this transaction so
        multi-event transactions stay strictly increasing per session. The
        durable write happens when the journal is applied; allocation alone
        mutates nothing.
        """
        if session_id not in allocated:
            allocated[session_id] = self._read_sessions_cache_locked().get(session_id, 0)
        allocated[session_id] += 1
        return allocated[session_id]

    def _reconcile_sessions_cache_locked(self, events: list[dict[str, Any]]) -> None:
        """Fold one journal's disposition sequences into the durable cache.

        Called while the journal is still pending: if this write fails, the
        journal stays discoverable and the next recovery pass reconciles
        before any eligibility check or sequence allocation.
        """
        deltas: dict[str, int] = {}
        for event in events:
            seq = event.get("seq")
            session_id = str(event.get("session_id") or "")
            if session_id and isinstance(seq, int) and seq >= 1:
                if seq > deltas.get(session_id, 0):
                    deltas[session_id] = seq
        if not deltas:
            return
        cache = self._read_sessions_cache_locked()
        changed = False
        for session_id, seq in deltas.items():
            if seq > cache.get(session_id, 0):
                cache[session_id] = seq
                changed = True
        if changed:
            self._write_sessions_cache_locked(cache)

    def try_auto_resume(self, session_id: str, *, replay_ok: Any = None) -> dict[str, Any] | None:
        """Atomically resume the session's one marker-eligible paused case.

        Eligibility: the case is paused, bound to ``session_id``, carries an
        ``auto_resume_pending`` marker, and the marker's ``seq_at_pause``
        equals the session's current high-water — i.e. no deliberate
        disposition has happened on the session since the lifecycle pause.
        At most one marker can validate per session because sequences are
        unique per session. The resume consumes the marker and advances the
        high-water, so a consumed marker can never re-validate.

        ``replay_ok`` is an optional callable ``replay_ok(case) -> bool`` run
        INSIDE the same locked transaction before the case is published
        active. When it returns False or raises, the case is left paused (no
        marker consumed) and None is returned, so no other process ever sees
        an active case whose Timeline is invalid.
        """
        if not session_id:
            return None
        with self._store_lock(), self._process_lock():
            self._recover_pending_transactions_locked()
            if self._get_active_case_no_recovery(session_id) is not None:
                return None
            return self._auto_resume_locked(session_id, replay_ok=replay_ok)

    def prepare_injection(self, session_id: str, *, replay_ok: Any = None) -> dict[str, Any] | None:
        """The case to inject, or None — one locked pass over the portfolio.

        Combines the active-case lookup and the proof-of-life auto-resume so a
        dormant pre-LLM turn scans the portfolio exactly once instead of up to
        four times. Returns the active case as a case dict; if no active case
        exists it resumes the newest marker-eligible paused case (validated by
        ``replay_ok``) and returns that case dict with a transient
        ``auto_resume_is_new`` flag set on the returned copy; otherwise None.
        The flag is transient — it is set only on the returned copy, never
        persisted.

        ``replay_ok`` MUST be lock-free: it is called with the store's
        ``_process_lock`` (file flock) already held, so it cannot re-enter a
        store call that takes that lock (e.g. ``harness.replay``). The harness
        passes a ``verify_case_snapshot``-style validator instead.
        """
        if not session_id:
            return None
        with self._store_lock(), self._process_lock():
            self._recover_pending_transactions_locked()
            active = self._get_active_case_no_recovery(session_id)
            if active is not None:
                return _copy(active)
            resumed = self._auto_resume_locked(session_id, replay_ok=replay_ok)
            if resumed is None:
                return None
            case = _copy(resumed["case"])
            case["auto_resume_is_new"] = True
            return case

    def _auto_resume_locked(self, session_id: str, *, replay_ok: Any = None) -> dict[str, Any] | None:
        """Auto-resume body; caller must hold the store lock + process lock."""
        if not session_id:
            return None
        if self._get_active_case_no_recovery(session_id) is not None:
            return None
        high_water = self._read_sessions_cache_locked().get(session_id, 0)
        candidates: list[dict[str, Any]] = []
        for case in self._case_index.select(self, session_id):
            if case.get("status") != "paused":
                continue
            if str(case.get("session_id") or "") != session_id:
                continue
            marker = case.get("auto_resume_pending")
            if not isinstance(marker, dict):
                continue
            if marker.get("seq_at_pause") != high_water or high_water < 1:
                continue
            candidates.append(case)
        if not candidates:
            return None
        candidates.sort(key=lambda item: str(item.get("updated_at") or ""))
        candidate = candidates[-1]
        try:
            if replay_ok is not None and not replay_ok(candidate):
                return None
        except Exception:
            return None
        marker = candidate.pop("auto_resume_pending")
        candidate["status"] = "active"
        candidate["updated_at"] = _utc_now()
        events = self._commit_transaction(
            candidate,
            [{
                "type": "case_resumed",
                "session_id": session_id,
                "payload": {
                    "reason": "proof of life",
                    "auto": True,
                    "consumed_pause_event_id": marker.get("pause_event_id"),
                    "consumed_seq_at_pause": marker.get("seq_at_pause"),
                    "pause_reason": marker.get("reason"),
                },
            }],
            write_snapshot=False,
        )
        return {"case": _copy(candidate), "event": events[-1]}

    def validate_case_chain_and_snapshot(
        self,
        case: dict[str, Any],
        *,
        expected_case_id: str | None = None,
    ) -> bool:
        """Lock-free replay gate: is this candidate's Timeline chain intact and
        its snapshot bound to the tip?

        Runs on an already-unlocked read inside the caller's critical section,
        so it must NOT take the ``_process_lock`` or ``_lock``. It verifies the
        event hash chain, the snapshot integrity metadata, and the bind across
        the same files ``verify_event_chain``/``verify_case_snapshot`` check.
        """
        case_id = str(case.get("case_id") or "")
        if not case_id or (
            expected_case_id is not None and case_id != expected_case_id
        ):
            return False
        events_path = self._case_dir(case_id) / "events.jsonl"
        try:
            events = self._read_events(events_path)
        except StoreError:
            return False
        if not events:
            return False
        expected_prev = _ZERO_HASH
        for index, event in enumerate(events, start=1):
            if event.get("event_id") != f"E{index:06d}":
                return False
            if event.get("prev_hash") != expected_prev:
                return False
            if event.get("hash") != _event_hash(event):
                return False
            expected_prev = event["hash"]
        return self._case_snapshot_binds_to_events(case, events)

    @staticmethod
    def _case_snapshot_binds_to_events(
        case: dict[str, Any],
        events: list[dict[str, Any]],
    ) -> bool:
        """Check one snapshot against a supplied, already-read Timeline.

        The recovery path may see a Timeline that already contains a pending
        journal while ``case.md`` still describes the predecessor. It uses this
        lock-free predicate against the predecessor prefix before completing the
        journal; normal writes use the complete Timeline.
        """
        if not events:
            return False
        integrity = case.get("integrity")
        if not isinstance(integrity, dict):
            return False
        last_event_id = integrity.get("last_event_id")
        if last_event_id != events[-1].get("event_id"):
            return False
        state_hash = integrity.get("state_sha256")
        if state_hash != _case_state_hash(case):
            return False
        event_state_hash = (events[-1].get("payload") or {}).get("case_state_sha256")
        return bool(event_state_hash) and event_state_hash == state_hash

    def migrate_lineage_case(
        self,
        *,
        lineage_ids: list[str],
        child_session_id: str,
    ) -> dict[str, Any] | None:
        """Adopt the newest eligible ancestor case in one locked primitive.

        Hermes's compression lineage deliberately continues through later
        compression tips when called with an older session, so candidates are
        restricted to the lineage PREFIX ending at ``child_session_id`` — a
        descendant is never treated as an ancestor. Within the prefix, the
        tip-first newest ancestor owning an active case or a validly marked
        paused case is adopted; older qualifying ancestors never migrate. An
        operator-paused or expiry-paused ancestor (no valid marker) is never
        adopted. A paused adoption keeps the case paused: the marker survives
        only when the child is pristine (no case record names it, current or
        historical), re-based to the child-session sequence; on a non-pristine
        child the marker is stripped and the case simply stays paused.
        """
        if not child_session_id:
            return None
        lineage = [str(item) for item in (lineage_ids or [])]
        if child_session_id not in lineage:
            return None
        prefix = lineage[: lineage.index(child_session_id)]
        if not prefix:
            return None
        with self._store_lock(), self._process_lock():
            self._recover_pending_transactions_locked()
            if self._get_active_case_no_recovery(child_session_id) is not None:
                return None
            child_pristine = self._session_is_pristine_locked(child_session_id)
            for ancestor in reversed(prefix):
                if ancestor == child_session_id:
                    continue
                active = self._get_active_case_no_recovery(ancestor)
                if active is not None:
                    return self._migrate_active_case_locked(
                        old_session_id=ancestor,
                        new_session_id=child_session_id,
                        event_type="session_migrated",
                        reason="Hermes conversation compression",
                    )
                paused = self._get_marked_paused_case_locked(ancestor)
                if paused is None:
                    continue
                return self._migrate_paused_case_locked(
                    paused,
                    child_session_id=child_session_id,
                    keep_marker=child_pristine,
                )
            return None

    def _session_is_pristine_locked(self, session_id: str) -> bool:
        """A session is pristine while no case record names it — currently or
        historically (any Timeline event with that session id)."""
        self._case_index.select(self, session_id)
        for case in self._case_index.select(self):
            case_path = self._case_dir(case["case_id"]) / "case.md"
            if str(case.get("session_id") or "") == session_id:
                return False
            events = self._read_events(case_path.parent / "events.jsonl")
            for event in events:
                if str(event.get("session_id") or "") == session_id:
                    return False
        return True

    def _get_marked_paused_case_locked(self, session_id: str) -> dict[str, Any] | None:
        """The session's paused case whose marker still validates, if any."""
        high_water = self._read_sessions_cache_locked().get(session_id, 0)
        found: dict[str, Any] | None = None
        for case in self._case_index.select(self, session_id):
            if case.get("status") != "paused":
                continue
            if str(case.get("session_id") or "") != session_id:
                continue
            marker = case.get("auto_resume_pending")
            if not isinstance(marker, dict):
                continue
            if high_water < 1 or marker.get("seq_at_pause") != high_water:
                continue
            if found is not None:
                # Multiple valid markers on one session should be impossible
                # (sequences are unique per session); take the newest.
                if str(case.get("updated_at") or "") > str(found.get("updated_at") or ""):
                    found = case
            else:
                found = case
        return found

    def _migrate_paused_case_locked(
        self,
        case: dict[str, Any],
        *,
        child_session_id: str,
        keep_marker: bool,
    ) -> dict[str, Any] | None:
        """Migrate a marker-eligible paused case to its compressed child.

        The case stays paused. When the child is pristine the marker is kept
        and re-based: the migration event allocates the child's next sequence
        and the marker's ``seq_at_pause`` is rewritten to it in the same
        transaction (the commit fills ``None`` markers from the transaction's
        disposition event), while the original pause event stays on the
        Timeline for provenance. Fenced with the case's probe lease like an
        active migration; a live lease fails closed for a later retry.
        """
        case_id = str(case.get("case_id") or "")
        old_session_id = str(case.get("session_id") or "")
        with self.probe_execution_lease(case_id):
            case["session_id"] = child_session_id
            case["updated_at"] = _utc_now()
            marker = case.get("auto_resume_pending")
            origin_pause_event_id = None
            if isinstance(marker, dict):
                origin_pause_event_id = marker.get("pause_event_id")
            if keep_marker and isinstance(marker, dict):
                # Re-based by _commit_transaction from this transaction's
                # disposition event for the child session.
                marker["seq_at_pause"] = None
                marker["rebased_to_session"] = child_session_id
            else:
                case.pop("auto_resume_pending", None)
            events = self._commit_transaction(
                case,
                [{
                    "type": "session_migrated",
                    "session_id": child_session_id,
                    "payload": {
                        "old_session_id": old_session_id,
                        "new_session_id": child_session_id,
                        "reason": "Hermes conversation compression",
                        "migrated_paused": True,
                        "marker_kept": bool(keep_marker),
                        "origin_pause_event_id": origin_pause_event_id,
                    },
                }],
                write_snapshot=False,
            )
        return {"case": _copy(case), "event": events[-1]}


    def append_event(
        self,
        case_id: str,
        *,
        event_type: str,
        session_id: str,
        payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Append one hash-linked event and fsync it before returning."""
        if not event_type.strip():
            raise StoreError("event_type is required")
        self._validate_case_id(case_id)
        events_path = self._case_dir(case_id) / "events.jsonl"

        with self._store_lock(), self._process_lock():
            self._recover_pending_transactions_locked()
            if not events_path.parent.exists():
                raise StoreError(f"case {case_id!r} does not exist")
            case = self._read_case(
                self._case_dir(case_id) / "case.md",
                expected_case_id=case_id,
            )
            events = self._commit_transaction(
                case,
                [{"type": event_type, "session_id": session_id, "payload": payload}],
                write_snapshot=False,
            )
            return _copy(events[-1])

    def verify_event_chain(self, case_id: str) -> dict[str, Any]:
        self._validate_case_id(case_id)
        with self._store_lock(), self._process_lock():
            self._recover_pending_transactions_locked()
            events = self._read_events(self._case_dir(case_id) / "events.jsonl")
        expected_prev = _ZERO_HASH
        for index, event in enumerate(events, start=1):
            if event.get("event_id") != f"E{index:06d}":
                return {"valid": False, "events": len(events), "error": "event_id sequence mismatch"}
            if event.get("prev_hash") != expected_prev:
                return {"valid": False, "events": len(events), "error": "prev_hash mismatch"}
            if event.get("hash") != _event_hash(event):
                return {"valid": False, "events": len(events), "error": "event hash mismatch"}
            expected_prev = event["hash"]
        return {"valid": True, "events": len(events)}

    def verify_case_snapshot(self, case_id: str) -> dict[str, Any]:
        """Verify that the mutable case snapshot is bound to the Timeline tip."""
        self._validate_case_id(case_id)
        with self._store_lock(), self._process_lock():
            self._recover_pending_transactions_locked()
            try:
                case = self._read_case(
                    self._case_dir(case_id) / "case.md",
                    expected_case_id=case_id,
                )
            except StoreError as exc:
                # A diagnostic read should report a cross-slot substitution as
                # an invalid snapshot, while authoritative readers and all
                # writers still reject the object outright.
                if "identity does not match storage slot" not in str(exc):
                    raise
                return {"valid": False, "error": str(exc)}
            events = self._read_events(self._case_dir(case_id) / "events.jsonl")
        if not events:
            return {"valid": False, "error": "Timeline is empty"}
        integrity = case.get("integrity")
        if not isinstance(integrity, dict):
            return {"valid": False, "error": "case snapshot integrity metadata is missing"}
        state_hash = str(integrity.get("state_sha256") or "")
        if integrity.get("last_event_id") != events[-1].get("event_id"):
            return {"valid": False, "error": "case snapshot integrity event mismatch"}
        if state_hash != _case_state_hash(case):
            return {"valid": False, "error": "case snapshot integrity hash mismatch"}
        event_state_hash = str((events[-1].get("payload") or {}).get("case_state_sha256") or "")
        if not event_state_hash or event_state_hash != state_hash:
            return {"valid": False, "error": "Timeline does not bind the current case snapshot"}
        return {"valid": True, "last_event_id": events[-1]["event_id"]}

    def list_events(self, case_id: str) -> list[dict[str, Any]]:
        """Return a defensive copy of a case Timeline."""
        self._validate_case_id(case_id)
        with self._store_lock(), self._process_lock():
            self._recover_pending_transactions_locked()
            return _copy(self._read_events(self._case_dir(case_id) / "events.jsonl"))

    def update_case(
        self,
        case: dict[str, Any],
        *,
        event_type: str,
        event_payload: dict[str, Any],
    ) -> dict[str, Any]:
        """Persist one new case version and its append-only transition event."""
        case_id = str(case.get("case_id") or "")
        self._validate_case_id(case_id)
        with self._store_lock(), self._process_lock():
            self._recover_pending_transactions_locked()
            current = self._read_case(
                self._case_dir(case_id) / "case.md",
                expected_case_id=case_id,
            )
            if int(case.get("version", 0)) != int(current.get("version", 0)) + 1:
                raise StoreError("case version must increase by exactly one")
            case["updated_at"] = _utc_now()
            events = self._commit_transaction(
                case,
                [{
                    "type": event_type,
                    "session_id": str(case.get("session_id") or ""),
                    "payload": event_payload,
                }],
                write_snapshot=True,
            )
            return {"case": _copy(case), "event": events[-1]}

    def transition_case(
        self,
        case: dict[str, Any],
        *,
        event_type: str,
        event_payload: dict[str, Any],
        event_id_target: tuple[str, str] | str | None = None,
    ) -> dict[str, Any]:
        """Persist runtime state without incrementing the model version."""
        case_id = str(case.get("case_id") or "")
        self._validate_case_id(case_id)
        with self._store_lock(), self._process_lock():
            self._recover_pending_transactions_locked()
            current = self._read_case(
                self._case_dir(case_id) / "case.md",
                expected_case_id=case_id,
            )
            if int(case.get("version", 0)) != int(current.get("version", 0)):
                raise StoreError("runtime transition cannot change the model version")
            if case.get("updated_at") != current.get("updated_at"):
                raise StoreError("concurrent case transition detected; reload before retrying")
            case["updated_at"] = _utc_now()
            events = self._commit_transaction(
                case,
                [{
                    "type": event_type,
                    "session_id": str(case.get("session_id") or ""),
                    "payload": event_payload,
                }],
                write_snapshot=False,
                event_id_target=event_id_target,
            )
            return {"case": _copy(case), "event": events[-1]}

    def store_artifact(self, case_id: str, text: str) -> dict[str, Any]:
        """Store redacted evidence content-addressed inside ``archive/``."""
        self._validate_case_id(case_id)
        digest = hashlib.sha256(text.encode("utf-8")).hexdigest()
        relative = Path("archive") / "artifacts" / f"{digest}.txt"
        destination = self._case_dir(case_id) / relative
        with self._store_lock(), self._process_lock():
            self._recover_pending_transactions_locked()
            if not destination.exists():
                self._atomic_write(destination, text)
        return {
            "storage": "artifact",
            "artifact_path": relative.as_posix(),
            "artifact_sha256": digest,
            "artifact_bytes": len(text.encode("utf-8")),
        }

    def verify_artifact(self, case_id: str, payload: dict[str, Any]) -> dict[str, Any]:
        """Verify one Timeline artifact reference without trusting its path."""
        self._validate_case_id(case_id)
        relative = Path(str(payload.get("artifact_path") or ""))
        if (
            relative.is_absolute()
            or ".." in relative.parts
            or relative.parts[:2] != ("archive", "artifacts")
        ):
            return {"valid": False, "error": "invalid artifact path"}
        case_root = self._case_dir(case_id).resolve()
        artifact = (case_root / relative).resolve()
        try:
            artifact.relative_to(case_root)
        except ValueError:
            return {"valid": False, "error": "artifact path escapes the case directory"}
        try:
            content = artifact.read_bytes()
        except OSError as exc:
            return {"valid": False, "error": f"artifact is unavailable: {exc}"}
        expected_bytes = payload.get("artifact_bytes")
        if not isinstance(expected_bytes, int) or len(content) != expected_bytes:
            return {"valid": False, "error": "artifact byte count mismatch"}
        expected_sha256 = str(payload.get("artifact_sha256") or "")
        if hashlib.sha256(content).hexdigest() != expected_sha256:
            return {"valid": False, "error": "artifact digest mismatch"}
        return {"valid": True}

    @contextmanager
    def probe_execution_lease(self, case_id: str):
        """Hold an OS-released lease for the duration of one external action."""
        self._validate_case_id(case_id)
        lease_path = self._case_dir(case_id) / "archive" / ".probe-execution.lock"
        lease_path.parent.mkdir(parents=True, exist_ok=True)
        handle = lease_path.open("a+", encoding="utf-8")
        fallback_acquired = False
        try:
            if fcntl is not None:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                except BlockingIOError as exc:
                    raise ProbeExecutionBusyError(
                        "the committed probe is already executing"
                    ) from exc
            else:  # pragma: no cover - non-POSIX fallback
                with self._store_lock():
                    if case_id in self._live_execution_leases:
                        raise ProbeExecutionBusyError(
                            "the committed probe is already executing"
                        )
                    self._live_execution_leases.add(case_id)
                    fallback_acquired = True
            yield
        finally:
            if fcntl is not None:
                try:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)
                except OSError:
                    pass
            elif fallback_acquired:  # pragma: no cover - non-POSIX fallback
                with self._store_lock():
                    self._live_execution_leases.discard(case_id)
            handle.close()

    def probe_execution_is_live(self, case_id: str) -> bool:
        """Return whether another thread/process still holds the action lease."""
        self._validate_case_id(case_id)
        if fcntl is None:  # pragma: no cover - non-POSIX fallback
            with self._store_lock():
                return case_id in self._live_execution_leases
        lease_path = self._case_dir(case_id) / "archive" / ".probe-execution.lock"
        lease_path.parent.mkdir(parents=True, exist_ok=True)
        with lease_path.open("a+", encoding="utf-8") as handle:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            except BlockingIOError:
                return True
            try:
                return False
            finally:
                fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _case_dir(self, case_id: str) -> Path:
        return self.cases_dir / case_id

    def _commit_transaction(
        self,
        case: dict[str, Any],
        event_specs: list[dict[str, Any]],
        *,
        write_snapshot: bool,
        event_id_target: tuple[str, str] | str | None = None,
    ) -> list[dict[str, Any]]:
        """Journal one case+Timeline transition so recovery is deterministic."""
        case_id = str(case.get("case_id") or "")
        case_path = self._case_dir(case_id) / "case.md"
        if case_path.exists():
            current = self._read_case(
                case_path,
                expected_case_id=case_id,
            )
            # Validate the authoritative on-disk predecessor, not the caller's
            # already-mutated copy. Every existing-case write reaches this
            # boundary, so a sibling operation cannot repair a tampered
            # snapshot by hashing the corruption into a new valid tip.
            if not self.validate_case_chain_and_snapshot(current):
                raise StoreError(
                    "cannot persist transition: authoritative case snapshot integrity check failed"
                )
            expected_revision = int(case.get("runtime_revision", 0) or 0)
            current_revision = int(current.get("runtime_revision", 0) or 0)
            if expected_revision != current_revision:
                raise StoreError("concurrent case transition detected; reload before retrying")
            case["runtime_revision"] = current_revision + 1
        else:
            case["runtime_revision"] = 1
        events_path = self._case_dir(case_id) / "events.jsonl"
        previous = self._read_events(events_path)
        event_specs = _copy(event_specs)
        if not event_specs:
            raise StoreError("a case transaction requires at least one event")

        # Allocate per-session disposition sequences BEFORE any hashing so the
        # case snapshot hash binds the filled marker and every event hash
        # binds its sequence. Allocation mutates only the transaction-local
        # ``allocated`` map; the durable cache write happens when the journal
        # is applied (reconcile-before-unlink).
        allocated: dict[str, int] = {}
        for spec in event_specs:
            spec_type = str(spec.get("type") or "")
            spec_session = str(spec.get("session_id") or "")
            if spec_type in _DISPOSITION_EVENT_TYPES and spec_session:
                spec["seq"] = self._next_session_seq_locked(allocated, spec_session)
        marker = case.get("auto_resume_pending")
        if isinstance(marker, dict) and not marker.get("seq_at_pause"):
            # Fill the marker from this transaction's last disposition event
            # for the case's (possibly newly migrated) session. Covers both
            # the auto-pause path (case_paused on the current session) and the
            # paused-migration re-base (session_migrated on the child).
            case_session = str(case.get("session_id") or "")
            fill_seq = None
            fill_event_id = None
            for offset, spec in reversed(list(enumerate(event_specs, start=1))):
                if spec.get("seq") and str(spec.get("session_id") or "") == case_session:
                    fill_seq = spec["seq"]
                    fill_event_id = f"E{len(previous) + offset:06d}"
                    break
            if fill_seq is None:
                raise StoreError(
                    "an auto-resume marker requires a disposition event in the same transaction"
                )
            marker["seq_at_pause"] = fill_seq
            if not marker.get("pause_event_id"):
                marker["pause_event_id"] = fill_event_id

        final_event_id = f"E{len(previous) + len(event_specs):06d}"
        if event_id_target is not None:
            if isinstance(event_id_target, str):
                case[event_id_target] = final_event_id
            else:
                parent_key, field_key = event_id_target
                parent = case.get(parent_key)
                if not isinstance(parent, dict):
                    raise StoreError(f"event target {parent_key!r} is not an object")
                parent[field_key] = final_event_id

        case["integrity"] = {"last_event_id": final_event_id}
        case_state_sha256 = _case_state_hash(case)
        case["integrity"]["state_sha256"] = case_state_sha256
        final_payload = dict(event_specs[-1].get("payload") or {})
        final_payload["case_state_sha256"] = case_state_sha256
        event_specs[-1]["payload"] = final_payload
        events = self._prepare_events(previous, event_specs)

        journal = self._case_dir(case_id) / "archive" / ".pending-transition.json"
        if journal.exists():
            raise StoreError(f"case {case_id!r} already has a pending durable transition")
        transaction = {
            "schema_version": _SCHEMA_VERSION,
            "case_id": case_id,
            "case": _copy(case),
            "events": _copy(events),
            "write_snapshot": bool(write_snapshot),
        }
        self._atomic_write(
            journal,
            json.dumps(transaction, ensure_ascii=False, sort_keys=True, indent=2),
        )
        self._apply_transaction_journal(journal)
        return events

    @staticmethod
    def _prepare_events(
        previous: list[dict[str, Any]],
        event_specs: list[dict[str, Any]],
    ) -> list[dict[str, Any]]:
        prev_hash = previous[-1]["hash"] if previous else _ZERO_HASH
        events: list[dict[str, Any]] = []
        for offset, spec in enumerate(event_specs, start=1):
            event = {
                "event_id": f"E{len(previous) + offset:06d}",
                "timestamp": _utc_now(),
                "type": str(spec.get("type") or ""),
                "session_id": str(spec.get("session_id") or ""),
                "payload": dict(spec.get("payload") or {}),
                "prev_hash": prev_hash,
            }
            if not event["type"].strip():
                raise StoreError("event_type is required")
            if spec.get("seq") is not None:
                # Immutable per-session disposition sequence (see
                # _DISPOSITION_EVENT_TYPES); hashed into the event.
                event["seq"] = int(spec["seq"])
            event["hash"] = _event_hash(event)
            events.append(event)
            prev_hash = event["hash"]
        return events

    def _recover_pending_transactions(self) -> None:
        with self._store_lock(), self._process_lock():
            self._recover_pending_transactions_locked()

    def _recover_pending_transactions_locked(self) -> None:
        journals = sorted(
            self.cases_dir.glob("*/archive/.pending-transition.json")
        )
        for journal in journals:
            self._apply_transaction_journal(journal)

    def _apply_transaction_journal(self, journal: Path) -> None:
        try:
            transaction = json.loads(journal.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise StoreError(f"cannot read recovery journal {journal}: {exc}") from exc
        if not isinstance(transaction, dict) or transaction.get("schema_version") != _SCHEMA_VERSION:
            raise StoreError(f"unsupported recovery journal {journal}")
        case = transaction.get("case")
        events = transaction.get("events")
        if not isinstance(case, dict) or not isinstance(events, list) or not events:
            raise StoreError(f"malformed recovery journal {journal}")
        case_id = str(transaction.get("case_id") or "")
        self._validate_case_id(case_id)
        journal_case_id = self._storage_case_id_from_path(journal)
        if journal_case_id != case_id:
            raise StoreError(
                "recovery journal storage slot does not match its case_id"
            )
        if case.get("case_id") != case_id:
            raise StoreError(f"recovery journal case mismatch in {journal}")

        events_path = self._case_dir(case_id) / "events.jsonl"
        existing = self._read_events(events_path)
        expected_prev = _ZERO_HASH
        for index, event in enumerate(existing, start=1):
            if (
                event.get("event_id") != f"E{index:06d}"
                or event.get("prev_hash") != expected_prev
                or event.get("hash") != _event_hash(event)
            ):
                raise StoreError(
                    f"cannot recover {case_id!r}: existing Timeline chain is invalid"
                )
            expected_prev = str(event["hash"])
        try:
            first_index = int(str(events[0].get("event_id") or "E000000")[1:]) - 1
        except ValueError as exc:
            raise StoreError(f"malformed recovery event ID in {journal}") from exc
        if first_index < 0 or len(existing) < first_index:
            raise StoreError(f"recovery journal Timeline prefix is missing for {case_id!r}")
        case_path = self._case_dir(case_id) / "case.md"
        if case_path.exists():
            current_case = self._read_case(
                case_path,
                expected_case_id=case_id,
            )
            predecessor = existing[:first_index]
            # Recovery may observe either the predecessor snapshot with the
            # pending events already appended, or the fully applied snapshot
            # with all journal events present. Accept only one of those two
            # authoritative bindings; never hash a damaged predecessor into
            # the recovered state.
            if not (
                self._case_snapshot_binds_to_events(current_case, predecessor)
                or self._case_snapshot_binds_to_events(current_case, existing)
            ):
                raise StoreError(
                    f"cannot recover {case_id!r}: authoritative case snapshot integrity is invalid"
                )
        journal_prev = existing[first_index - 1]["hash"] if first_index else _ZERO_HASH
        for offset, event in enumerate(events, start=1):
            expected_id = f"E{first_index + offset:06d}"
            if (
                event.get("event_id") != expected_id
                or event.get("prev_hash") != journal_prev
                or event.get("hash") != _event_hash(event)
            ):
                raise StoreError(f"recovery journal event chain is invalid in {journal}")
            journal_prev = str(event["hash"])
        overlap = min(max(len(existing) - first_index, 0), len(events))
        if existing[first_index:first_index + overlap] != events[:overlap]:
            raise StoreError(f"recovery journal conflicts with Timeline for {case_id!r}")
        if len(existing) > first_index + len(events):
            raise StoreError(f"Timeline advanced past pending recovery for {case_id!r}")
        integrity = case.get("integrity")
        if (
            not isinstance(integrity, dict)
            or integrity.get("last_event_id") != events[-1].get("event_id")
            or integrity.get("state_sha256") != _case_state_hash(case)
            or (events[-1].get("payload") or {}).get("case_state_sha256")
            != integrity.get("state_sha256")
        ):
            raise StoreError(f"recovery journal case snapshot binding is invalid in {journal}")
        if overlap < len(events):
            self._write_events(events_path, existing + events[overlap:])

        self._write_case(case)
        if bool(transaction.get("write_snapshot")):
            self._write_snapshot(case)
        # Reconcile the per-session high-water cache BEFORE the journal is
        # unlinked: the still-pending journal is the durable cursor proving
        # which sequences the cache incorporates. If the cache write fails,
        # the journal stays discoverable and the next recovery pass reconciles
        # before any eligibility check or sequence allocation. Re-application
        # is idempotent because reconciliation is a max() fold.
        self._reconcile_sessions_cache_locked(events)
        journal.unlink()
        self._fsync_dir(journal.parent)

    def _write_events(self, path: Path, events: list[dict[str, Any]]) -> None:
        text = "".join(
            json.dumps(
                event,
                # JSON permits U+2028/U+2029 inside strings, but Python's
                # splitlines() and some JSONL consumers treat them as record
                # boundaries. Escape non-ASCII characters so only the literal
                # LF appended below can delimit Timeline records.
                ensure_ascii=True,
                sort_keys=True,
                separators=(",", ":"),
            ) + "\n"
            for event in events
        )
        self._atomic_write(path, text)

    @contextmanager
    def _store_lock(self):
        """Acquire the process-local store RLock under the active deadline."""
        with acquire_lock(self._lock, label="epistemic store RLock"):
            yield

    def _acquire_process_flock(self, handle: Any) -> None:
        """Acquire the profile flock, polling only when a deadline is active."""
        if fcntl is None:  # pragma: no cover - constructor rejects this backend
            return
        remaining = remaining_lock_seconds()
        if remaining is None:
            fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
            return

        while True:
            try:
                fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
                return
            except OSError as exc:
                if exc.errno not in {
                    errno.EACCES,
                    errno.EAGAIN,
                    errno.EWOULDBLOCK,
                }:
                    raise
                remaining = remaining_lock_seconds()
                if remaining is None or remaining <= 0:
                    raise LockTimeoutError(
                        "lock acquisition deadline exceeded while acquiring "
                        "epistemic store process lock"
                    ) from exc
                time.sleep(min(0.01, remaining))

    @contextmanager
    def _process_lock(self):
        """Serialize durable transitions across gateway and worker processes.

        Explicit model/probe calls remain blocking. Lifecycle wrappers install a
        scoped deadline, in which case flock uses LOCK_NB polling and raises
        without entering the critical section when the budget expires.
        """
        lock_path = self.root / ".store.lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+", encoding="utf-8") as handle:
            acquired = False
            try:
                self._acquire_process_flock(handle)
                acquired = True
                yield
            finally:
                if acquired and fcntl is not None:
                    fcntl.flock(handle.fileno(), fcntl.LOCK_UN)

    def _write_case(self, case: dict[str, Any]) -> None:
        path = self._case_dir(case["case_id"]) / "case.md"
        self._atomic_write(path, _render_case(case))

    def _write_snapshot(self, case: dict[str, Any]) -> None:
        path = self._case_dir(case["case_id"]) / "archive" / f"case-v{case['version']:04d}.md"
        self._atomic_write(path, _render_case(case))

    def _read_case(
        self,
        path: Path,
        *,
        expected_case_id: str | None = None,
    ) -> dict[str, Any]:
        try:
            text = path.read_text(encoding="utf-8")
            start = text.index(_DATA_START) + len(_DATA_START)
            end = text.index(_DATA_END, start)
            data = json.loads(text[start:end])
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise StoreError(f"cannot read case file {path}: {exc}") from exc
        if not isinstance(data, dict) or data.get("schema_version") != _SCHEMA_VERSION:
            raise StoreError(f"unsupported or malformed case file {path}")
        expected = (
            expected_case_id
            if expected_case_id is not None
            else self._storage_case_id_from_path(path)
        )
        if expected is not None and data.get("case_id") != expected:
            raise StoreError(
                f"case identity does not match storage slot {expected!r}: {path}"
            )
        return data

    def _storage_case_id_from_path(self, path: Path) -> str | None:
        """Return the case ID encoded by a managed case or archive path.

        Use lexical absolute paths rather than ``resolve()`` so a symlink or
        copied snapshot cannot silently re-home a read into another case slot.
        Archive snapshots remain valid reads because their parent case directory
        is the authoritative slot.
        """
        path_abs = Path(os.path.abspath(path))
        cases_abs = Path(os.path.abspath(self.cases_dir))
        try:
            relative = path_abs.relative_to(cases_abs)
        except ValueError:
            return None
        if len(relative.parts) == 2 and relative.parts[1] == "case.md":
            return relative.parts[0]
        if len(relative.parts) == 3 and relative.parts[1] == "archive":
            return relative.parts[0]
        return None

    def _read_events(self, path: Path) -> list[dict[str, Any]]:
        if not path.exists():
            return []
        events: list[dict[str, Any]] = []
        try:
            # JSONL records are delimited by literal LF bytes. Do not use
            # splitlines(): legacy records may contain valid raw U+2028/U+2029
            # characters inside JSON strings.
            for line_number, line in enumerate(path.read_text(encoding="utf-8").split("\n"), start=1):
                if not line.strip():
                    continue
                event = json.loads(line)
                if not isinstance(event, dict):
                    raise ValueError(f"line {line_number} is not an object")
                events.append(event)
        except (OSError, ValueError, json.JSONDecodeError) as exc:
            raise StoreError(f"cannot read Timeline {path}: {exc}") from exc
        return events

    @staticmethod
    def _discard_empty_incomplete_case_dir(case_dir: Path) -> None:
        """Reclaim only the directory skeleton created before the first journal."""
        if not case_dir.exists():
            return
        entries = list(case_dir.iterdir())
        if not entries:
            case_dir.rmdir()
            CaseStore._fsync_dir(case_dir.parent)
            return
        archive = case_dir / "archive"
        if entries == [archive] and archive.is_dir():
            archive_entries = list(archive.iterdir())
            if archive_entries and all(
                entry.is_file() and entry.name.startswith(".") and entry.name.endswith(".tmp")
                for entry in archive_entries
            ):
                for entry in archive_entries:
                    entry.unlink()
                CaseStore._fsync_dir(archive)
                archive_entries = []
            if not archive_entries:
                archive.rmdir()
                case_dir.rmdir()
                CaseStore._fsync_dir(case_dir.parent)

    @staticmethod
    def _atomic_write(path: Path, text: str) -> None:
        CaseStore._ensure_dir_durable(path.parent)
        tmp = path.with_name(f".{path.name}.{uuid.uuid4().hex}.tmp")
        try:
            with tmp.open("w", encoding="utf-8") as handle:
                handle.write(text)
                handle.flush()
                os.fsync(handle.fileno())
            os.replace(tmp, path)
            CaseStore._fsync_dir(path.parent)
        finally:
            if tmp.exists():
                tmp.unlink()

    @staticmethod
    def _fsync_dir(path: Path) -> None:
        try:
            descriptor = os.open(path, os.O_RDONLY)
        except OSError as exc:
            raise StoreError(f"cannot open directory for durability sync {path}: {exc}") from exc
        try:
            os.fsync(descriptor)
        except OSError as exc:
            raise StoreError(f"cannot durability-sync directory {path}: {exc}") from exc
        finally:
            os.close(descriptor)

    @staticmethod
    def _ensure_dir_durable(path: Path) -> None:
        """Create missing directories and durably publish every new entry."""
        missing: list[Path] = []
        cursor = path
        while not cursor.exists():
            missing.append(cursor)
            if cursor.parent == cursor:
                break
            cursor = cursor.parent
        path.mkdir(parents=True, exist_ok=True)
        for created in reversed(missing):
            CaseStore._fsync_dir(created.parent)

    @staticmethod
    def _validate_case_id(case_id: str) -> None:
        if not isinstance(case_id, str) or not _CASE_ID_RE.fullmatch(case_id):
            raise StoreError("case_id must match [a-z0-9][a-z0-9-]{0,63}")


def _render_case(case: dict[str, Any]) -> str:
    model = case.get("model") or {}
    data = json.dumps(case, ensure_ascii=False, sort_keys=True, indent=2)
    lines = [
        f"> **Provenance:** AI-assisted contribution maintained by Epistemic Harness contributors, {_utc_now()[:10]}.",
        "",
        _DATA_START.rstrip("\n"),
        data,
        _DATA_END.lstrip("\n"),
        "",
        f"# Epistemic Case: {case.get('case_id', '')}",
        "",
        f"**Status:** {case.get('status', '')}  ",
        f"**Version:** {case.get('version', '')}  ",
        f"**Stale:** {case.get('stale', False)}  ",
        f"**Decision:** {model.get('decision', '')}",
        "",
        "## Stopping condition",
        str(model.get("stopping_condition", "")),
        "",
        "## State grounding",
        _render_items(model.get("state_grounding", [])),
        "",
        "## Mechanisms",
        _render_items(model.get("mechanisms", [])),
        "",
        "## Alternatives",
        _render_items(model.get("alternatives", [])),
        "",
        "## Top unknown",
        str(model.get("top_unknown", "")),
        "",
        "## Current plan",
        _render_items(model.get("current_plan", []), label_key="action"),
        "",
        "## Prior cases",
        _render_items(model.get("prior_case_ids", [])),
        "",
        "## Retrieved prior lessons",
        _render_items(model.get("retrieved_prior_lessons", []), label_key="lesson"),
        "",
        "## Applied prior lessons",
        _render_items(model.get("applied_prior_lessons", [])),
        "",
        "## Pending probe",
        _render_object(case.get("pending_probe")),
        "",
        "## Closure",
        _render_object(case.get("closure")),
        "",
    ]
    return "\n".join(lines)


def _render_object(value: Any) -> str:
    if value is None:
        return "- None recorded"
    return "```json\n" + json.dumps(value, ensure_ascii=False, sort_keys=True, indent=2) + "\n```"


def _render_items(items: Any, *, label_key: str = "text") -> str:
    if not isinstance(items, list) or not items:
        return "- None recorded"
    rendered = []
    for item in items:
        if isinstance(item, dict):
            prefix = f"{item.get('id')}: " if item.get("id") else ""
            rendered.append(f"- {prefix}{item.get(label_key, '')}")
        else:
            rendered.append(f"- {item}")
    return "\n".join(rendered)


def _case_state_hash(case: dict[str, Any]) -> str:
    material = _copy(case)
    integrity = material.get("integrity")
    if isinstance(integrity, dict):
        integrity.pop("state_sha256", None)
    canonical = json.dumps(material, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _event_hash(event: dict[str, Any]) -> str:
    material = {key: value for key, value in event.items() if key != "hash"}
    canonical = json.dumps(material, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _copy(value: Any) -> Any:
    return json.loads(json.dumps(value, ensure_ascii=False))
