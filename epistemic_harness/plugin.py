# AI-assisted contribution; maintained by Epistemic Harness contributors.
"""Hermes registration layer for the bounded epistemic harness."""

from __future__ import annotations

import json
import logging
import os
import threading
from pathlib import Path
from typing import Any, Callable

from .harness import EpistemicHarness, HarnessError, current_native_capture_state
from .locking import (
    DEFAULT_LIFECYCLE_LOCK_BUDGET_SECONDS,
    LockTimeoutError,
    acquire_lock,
    lock_deadline,
)
from .presentation import ResponseDetailError, project_response, validate_response_detail
from .schemas import MODEL_OPERATIONS, MODEL_SCHEMA, PROBE_OPERATIONS, PROBE_SCHEMA
from .store import StoreError

_HARNESSES: dict[Path, EpistemicHarness] = {}
_HARNESS_LOCK = threading.Lock()
logger = logging.getLogger(__name__)
_RUNTIME_GENERATION = "20260916-prospective-inquiry"
_SKILL_PATH = Path(__file__).resolve().parents[1] / "skills" / "epistemic-inquiry" / "SKILL.md"
_SKILL_DESCRIPTION = (
    "A bounded, session-scoped inquiry workflow: explicit activation, claim-bound "
    "commit, actual tool read, comparison, revision or non-update, and close/readback."
)


def _state_root() -> Path:
    try:
        from hermes_constants import get_hermes_home

        return get_hermes_home() / "epistemic-harness"
    except ImportError:
        return Path(os.environ.get("HERMES_HOME", str(Path.home() / ".hermes"))) / "epistemic-harness"


def _get_harness() -> EpistemicHarness:
    root = _state_root().expanduser().resolve()
    with acquire_lock(_HARNESS_LOCK, label="epistemic harness cache lock"):
        harness = _HARNESSES.get(root)
        if harness is None:
            harness = EpistemicHarness(root)
            _HARNESSES[root] = harness
        return harness


def _adopt_compression_lineage(harness: EpistemicHarness, session_id: str) -> None:
    """Rebind a case from a compressed ancestor using Hermes's public lineage API.

    Hermes publishes compression children atomically in its session database.
    Looking up that already-committed lineage from the next plugin boundary is
    both safer and substantially less coupled than participating in the core
    compression transaction with private pre/commit/abort hooks.

    The actual migration is one locked store primitive
    (``migrate_lineage_case``): it restricts candidates to the lineage prefix
    ending at this session (Hermes deliberately continues the lineage through
    later tips), adopts the newest ancestor owning an active case or a
    validly marked lifecycle-paused case, and decides marker retention from
    child pristineness — so a lifecycle pause can never strand the case on the
    parent, and a delayed adoption can never resurrect an old case after the
    child has seen a deliberate disposition.
    """
    session_id = str(session_id or "").strip()
    if not session_id or harness.store.get_active_case(session_id) is not None:
        return
    try:
        from hermes_state import SessionDB

        database = SessionDB(read_only=True)
        try:
            lineage = database.get_compression_lineage(session_id)
        finally:
            database.close()
    except Exception:
        # A missing/unavailable session database must never make the optional
        # epistemic plugin a host-wide availability dependency.
        return
    if session_id not in lineage:
        return
    try:
        harness.store.migrate_lineage_case(
            lineage_ids=[str(item) for item in lineage],
            child_session_id=session_id,
        )
    except Exception:
        # Adoption failure (a live probe lease, a store error) fails closed
        # for this boundary and is retried at the next one.
        logger.warning("Epistemic lineage adoption failed; will retry at the next boundary")


def _session_harness(session_id: str) -> EpistemicHarness:
    harness = _get_harness()
    _adopt_compression_lineage(harness, session_id)
    return harness


def reset_for_tests() -> None:
    """Clear the profile-root harness cache; intentionally not registered as a tool."""
    with _HARNESS_LOCK:
        _HARNESSES.clear()


def epistemic_model_handler(args: dict[str, Any], **kwargs: Any) -> str:
    try:
        operation, response_detail = _validate_tool_request(
            args,
            tool_name="epistemic_model",
            operations=MODEL_OPERATIONS,
        )
    except (TypeError, ValueError, ResponseDetailError) as exc:
        return _error(str(exc))
    session_id = str(kwargs.get("session_id") or "")
    if not session_id:
        return _error("epistemic_model requires a Hermes session_id")
    try:
        result = _session_harness(session_id).model(args, session_id=session_id)
        if operation == "list":
            result["runtime"] = {"pid": os.getpid(), "generation": _RUNTIME_GENERATION}
        result = project_response(
            result,
            response_detail=response_detail,
            requested_case_id=args.get("case_id"),
        )
        return json.dumps(result, ensure_ascii=False, sort_keys=True)
    except (HarnessError, StoreError, ValueError, TypeError) as exc:
        return _error(str(exc))


def epistemic_probe_handler(args: dict[str, Any], **kwargs: Any) -> str:
    try:
        _operation, response_detail = _validate_tool_request(
            args,
            tool_name="epistemic_probe",
            operations=PROBE_OPERATIONS,
        )
    except (TypeError, ValueError, ResponseDetailError) as exc:
        return _error(str(exc))
    session_id = str(kwargs.get("session_id") or "")
    if not session_id:
        return _error("epistemic_probe requires a Hermes session_id")
    try:
        result = _session_harness(session_id).probe(args, session_id=session_id)
        result = project_response(
            result,
            response_detail=response_detail,
            requested_case_id=args.get("case_id"),
        )
        return json.dumps(result, ensure_ascii=False, sort_keys=True)
    except (HarnessError, StoreError, ValueError, TypeError) as exc:
        return _error(str(exc))


def execution_gateway(**kwargs: Any) -> Any:
    """Apply case discipline without making plugin health a host-wide gate."""
    session_id = str(kwargs.get("session_id") or "")
    return _session_harness(session_id).tool_execution_middleware(
        native_middleware=True,
        **kwargs,
    )


def _best_effort_native_observer(name: str, callback: Callable[[], Any]) -> Any:
    """Keep observer bookkeeping from extending the host action lifetime."""
    try:
        with lock_deadline(0):
            return callback()
    except LockTimeoutError:
        logger.warning(
            "Epistemic native %s skipped: lock unavailable; observation was not persisted",
            name,
        )
        return None


def post_tool_call(**kwargs: Any) -> Any:
    """Capture the native handler result before result transforms are applied."""
    state = current_native_capture_state()
    if state is None:
        return None
    tool_name = str(kwargs.get("tool_name") or "")
    session_id = str(kwargs.get("session_id") or "")
    task_id = str(kwargs.get("task_id") or "")
    tool_call_id = str(kwargs.get("tool_call_id") or "")
    turn_id = str(kwargs.get("turn_id") or "")
    api_request_id = str(kwargs.get("api_request_id") or "")
    if not state.matches(
        session_id=session_id,
        task_id=task_id,
        tool_call_id=tool_call_id,
        turn_id=turn_id,
        api_request_id=api_request_id,
        tool_name=tool_name,
    ):
        return None
    return _best_effort_native_observer(
        "post_tool_call observation",
        lambda: _session_harness(session_id).record_native_tool_result(
            tool_name=tool_name,
            args=kwargs.get("args") if isinstance(kwargs.get("args"), dict) else None,
            result=kwargs.get("result"),
            session_id=session_id,
            task_id=task_id,
            tool_call_id=tool_call_id,
            turn_id=turn_id,
            api_request_id=api_request_id,
        ),
    )


def transform_tool_result(**kwargs: Any) -> Any:
    """Observe the native result-transform boundary without changing it."""
    result = kwargs.get("result")
    session_id = str(kwargs.get("session_id") or "")
    state = current_native_capture_state()
    if state is not None:
        tool_name = str(kwargs.get("tool_name") or "")
        task_id = str(kwargs.get("task_id") or "")
        tool_call_id = str(kwargs.get("tool_call_id") or "")
        turn_id = str(kwargs.get("turn_id") or "")
        api_request_id = str(kwargs.get("api_request_id") or "")
        if not state.matches(
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
                with acquire_lock(state.lock, label="native transform state lock"):
                    if not state.matches(
                        session_id=session_id,
                        task_id=task_id,
                        tool_call_id=tool_call_id,
                        turn_id=turn_id,
                        api_request_id=api_request_id,
                        tool_name=tool_name,
                    ):
                        return None
                    state.transform_seen = True
                    needs_unknown = not state.raw_captured and not state.unknown_captured
        except LockTimeoutError:
            logger.warning(
                "Epistemic native transform observer skipped: lock unavailable; "
                "observation was not persisted"
            )
            return None
        if needs_unknown:
            try:
                _best_effort_native_observer(
                    "transform-boundary observation",
                    lambda: _session_harness(session_id).record_unknown_native_tool_result(
                        tool_name=tool_name,
                        args=kwargs.get("args") if isinstance(kwargs.get("args"), dict) else None,
                        result=result,
                        session_id=session_id,
                        task_id=task_id,
                        tool_call_id=tool_call_id,
                        turn_id=turn_id,
                        api_request_id=api_request_id,
                        observation_boundary="transform_tool_result_input",
                    ),
                )
            except Exception:
                # Hook failures are isolated by Hermes. Leave the state marked
                # as seen so the enclosing middleware can make one final
                # unknown-boundary attempt without redispatching the action.
                logger.warning(
                    "Epistemic native raw-boundary observer failed; provenance is unknown",
                    exc_info=True,
                )
        return None
    _best_effort_native_observer(
        "exposure observation",
        lambda: _session_harness(session_id).record_tool_result_exposure(
            tool_name=str(kwargs.get("tool_name") or ""),
            args=kwargs.get("args") if isinstance(kwargs.get("args"), dict) else None,
            result=result,
            session_id=session_id,
            tool_call_id=str(kwargs.get("tool_call_id") or ""),
            task_id=str(kwargs.get("task_id") or ""),
            turn_id=str(kwargs.get("turn_id") or ""),
            api_request_id=str(kwargs.get("api_request_id") or ""),
        ),
    )
    # None is the native observer result: later transforms remain available
    # and the underlying tool is never redispatched.
    return None


def pre_llm_context(**kwargs: Any) -> dict[str, str] | None:
    """Inject active case state only into its bound session.

    When the session has no active case, proof-of-life auto-resume runs first:
    a lifecycle-paused case whose marker still validates comes back active and
    injects normally; a genuinely dead session never reaches this hook, so its
    case stays paused.
    """
    session_id = str(kwargs.get("session_id") or "")
    if not session_id:
        return None
    try:
        harness = _session_harness(session_id)

        def _replay_ok(case: dict[str, Any]) -> bool:
            # Lock-free: runs inside the store's _process_lock, so it must not
            # re-enter a store call that takes that lock (e.g. harness.replay).
            return harness.store.validate_case_chain_and_snapshot(case)

        case = harness.store.prepare_injection(session_id, replay_ok=_replay_ok)
        if case is None:
            return None
        if case.get("status") == "active" and case.get("auto_resume_is_new", False):
            _log_auto_resume_coalesced({"case": case})
        replay = harness.replay(case["case_id"])
        rendered = _render_context(case, replay)
        if rendered is None:
            return None
        return {"context": rendered}
    except Exception:
        logger.warning("Epistemic case context is unavailable; continuing without injection")
        return None


_AUTO_RESUME_LOG_WINDOW_S = 600.0
_auto_resume_log_times: dict[str, float] = {}


def _log_auto_resume_coalesced(resumed: dict[str, Any]) -> None:
    """One log line per case per 10-minute window during reconnect churn.

    The Timeline always records every auto-pause/auto-resume pair faithfully;
    only the log stream coalesces.
    """
    import time as _time

    case = resumed.get("case") if isinstance(resumed, dict) else None
    case_id = str((case or {}).get("case_id") or "")
    now = _time.monotonic()
    last = _auto_resume_log_times.get(case_id)
    if last is not None and now - last < _AUTO_RESUME_LOG_WINDOW_S:
        return
    _auto_resume_log_times[case_id] = now
    logger.info("Epistemic case %s auto-resumed on proof of life", case_id)


def _run_lifecycle_callback(name: str, callback: Callable[[], Any]) -> Any:
    """Run one optional lifecycle callback with a single lock budget."""
    try:
        with lock_deadline(DEFAULT_LIFECYCLE_LOCK_BUDGET_SECONDS):
            return callback()
    except LockTimeoutError:
        logger.warning(
            "Epistemic lifecycle %s skipped: lock acquisition exceeded %.2fs; "
            "pause was not persisted",
            name,
            DEFAULT_LIFECYCLE_LOCK_BUDGET_SECONDS,
        )
    except Exception:
        logger.warning("Epistemic lifecycle %s observer failed; continuing", name)
    return None


def on_subagent_stop(**kwargs: Any) -> None:
    """Pause an unfinished case owned by a child whose run has ended."""
    _run_lifecycle_callback(
        "subagent_stop",
        lambda: _get_harness().pause_active_subagent_case(
            child_session_id=str(kwargs.get("child_session_id") or ""),
            child_status=str(kwargs.get("child_status") or "unknown"),
        ),
    )


def on_session_reset(
    *,
    old_session_id: str | None = None,
    new_session_id: str | None = None,
    **_kwargs: Any,
) -> None:
    _run_lifecycle_callback(
        "session_reset",
        lambda: _get_harness().session_reset(
            old_session_id=old_session_id,
            new_session_id=new_session_id,
        ),
    )


def on_session_finalize(**kwargs: Any) -> None:
    """Observer: lifecycle auto-pause on finalization. Deliberately
    non-critical: a failure never reaches the host."""
    _run_lifecycle_callback(
        "session_finalize",
        lambda: _get_harness().session_finalize(
            session_id=str(kwargs.get("session_id") or ""),
            reason=kwargs.get("reason"),
            platform=kwargs.get("platform"),
        ),
    )


# --- injection budget (spill guard) ------------------------------------------
#
# Hermes's hook-output spill (tools/hook_output_spill.py, default-on since
# 2026-08-13) replaces any pre_llm_call context piece over the configured
# max_chars (default 10,000) with a preview plus a filesystem pointer. An
# oversized case injection would silently strip the case state that controls
# case-relevant work. The ladder below bounds every emission to an effective
# ceiling resolved per compose; small cases render byte-identical to before.

_DEFAULT_INJECTION_MAX_CHARS = 9000
_SPILL_DEFAULT_MAX_CHARS = 10_000
_SPILL_MARGIN_CHARS = 1000
_INJECTION_BUDGET: dict[str, Any] = {}


def _resolve_injection_ceiling() -> int:
    """Effective ceiling for one compose: the plugin budget, tightened under
    the operator's live spill threshold when Hermes would spill earlier.

    Per-compose cadence matches Hermes's own per-turn resolution. The
    guarantee is relative to this compose-time snapshot: Hermes resolves its
    own spill snapshot after hook return, and an operator reduction landing in
    that microsecond window degrades to Hermes's spill-with-pointer — the
    pre-guard baseline, never worse. Fail-open to the defaults on any error.
    An explicitly configured zero budget is preserved (it is a real operator
    intent to suppress injection).
    """
    try:
        raw = (_INJECTION_BUDGET or {}).get("injection_max_chars")
        budget = _DEFAULT_INJECTION_MAX_CHARS if raw is None else int(raw)
    except (TypeError, ValueError):
        budget = _DEFAULT_INJECTION_MAX_CHARS
    if budget < 0:
        budget = 0
    try:
        from tools.hook_output_spill import get_spill_config

        spill = get_spill_config() or {}
        if spill.get("enabled", True):
            spill_max = int(spill.get("max_chars") or _SPILL_DEFAULT_MAX_CHARS)
            return max(0, min(budget, spill_max - _SPILL_MARGIN_CHARS))
    except Exception:
        pass
    return budget


def _compact_case(case: dict[str, Any], replay: dict[str, Any]) -> dict[str, Any]:
    model = case.get("model") or {}
    return {
        "case_id": case.get("case_id"),
        "status": case.get("status"),
        "model_version": case.get("version"),
        "stale": case.get("stale"),
        "pending_probe": case.get("pending_probe"),
        "decision": model.get("decision"),
        "stopping_condition": model.get("stopping_condition"),
        "state_grounding": model.get("state_grounding"),
        "mechanisms": model.get("mechanisms"),
        "alternatives": model.get("alternatives"),
        "top_unknown": model.get("top_unknown"),
        "current_plan": model.get("current_plan"),
        "prior_case_ids": model.get("prior_case_ids"),
        "retrieved_prior_lessons": model.get("retrieved_prior_lessons"),
        "applied_prior_lessons": model.get("applied_prior_lessons"),
        "replay": replay,
    }


_RULES_TEXT = (
    "Rules:\n"
    "- Ordinary search, recall, reads, code, and delegation remain available without probe wrapping.\n"
    "- For a high-value uncertainty, optionally COMMIT one prediction, execute, then COMPARE.\n"
    "- Run a committed probe's intended tool call alone; parallel same-name siblings are not probe evidence.\n"
    "- Argument drift is recorded rather than blocked; DISCARD a mistaken or stranded probe.\n"
    "- A material mismatch makes the case stale; revise consequentially before closure.\n"
    "- User answers are authoritative only for the typed user domain, not universal empirical truth.\n"
    "- Every evidence-citing claim must declare an authority_domain compatible with its Timeline sources.\n"
    "- After ordinary external research, RECORD_EVIDENCE as a JSON array with source role, access scope, and any access limitation; do not put recalled, user, or system-state observations there.\n"
    "- Use returned Timeline evidence_event_ids verbatim in claims; never guess, invent, or shorten an event ID.\n"
    "- REVISE always requires a reason and one or more changed model fields.\n"
    "- A probe commitment names what would change the belief; its prediction must be falsifiable enough to guide judgment.\n"
    "- Optionally bind a commit to one existing state-grounding, mechanism, or alternative claim with claim_id; the server stores its immutable snapshot.\n"
    "- A claim-bound compare must state belief_change (strengthened, weakened, narrowed, unchanged, or unresolved); this is separate from match/mismatch/unresolved disposition.\n"
    "- Failed, unavailable, interrupted, or legacy observations do not support the unobserved external proposition, even when a comparison descriptor says full access.\n"
    "- If execution arguments drifted, explain the drift at compare or discard the probe; legacy records have unknown argument-match status.\n"
    "- Evidence calibration is derived from what was inspected; do not supply or inflate a confidence score.\n"
    "- Never call a supported or provisional conclusion confirmed, proven, or certain.\n"
    "- Retrieved lessons include their source calibration, remain candidates rather than facts, and never ground a current claim without current evidence.\n"
    "- At closure, record at most one reusable mechanism or caution, or explicitly choose no transfer.\n"
    "- Unresolved is a legitimate outcome. Do not manufacture a conclusion for closure.\n"
    "- Ordinary curiosity need not become a formal question unless it can change the model or decision.\n"
    "- A delegated inquiry may use a case in the worker session when the scope is bounded and ownership is explicit.\n"
    "- Announce the inquiry boundary before opening it; do not turn every mechanical retrieval into a case.\n"
)


def _render_header_and_json(case_id: Any, compact: dict[str, Any], *, indent: int | None) -> str:
    body = json.dumps(compact, ensure_ascii=False, sort_keys=True, indent=indent)
    return (
        f"[ACTIVE EPISTEMIC CASE {case_id}]\n"
        "This bounded case is active. The durable model below controls case-relevant work.\n"
        + _RULES_TEXT
        + "Current case:\n"
        + body
    )


_RETRIEVAL_HINT = "call epistemic_model(show) for the complete value"


def _shed_count_marker(value: Any) -> dict[str, Any]:
    count = len(value) if isinstance(value, list) else (1 if value else 0)
    return {"shed_for_budget": True, "count": count, "retrieve": _RETRIEVAL_HINT}


def _digest_pending_probe(pending: Any) -> Any:
    """Keep probe identity and control fields; digest the committed arguments.

    The full committed arguments live in the durable case store — the marker
    names that retrieval path instead of shipping bulk into the prompt.
    """
    if not isinstance(pending, dict):
        return pending
    digest = {
        key: pending.get(key)
        for key in ("status", "tool_name", "purpose", "unknown_or_goal", "committed_at")
        if key in pending
    }
    args = pending.get("tool_args")
    if args is not None:
        try:
            args_bytes = len(json.dumps(args, ensure_ascii=False))
        except (TypeError, ValueError):
            args_bytes = -1
        digest["tool_args_bytes"] = args_bytes
        digest["tool_args_location"] = "durable case store (epistemic_model show)"
        digest["retrieve"] = _RETRIEVAL_HINT
    return digest


def _cap_field_value(value: Any) -> Any:
    """Deterministic per-field cap preserving valid JSON at every size."""
    marker = "...[truncated-for-budget]"
    if isinstance(value, str):
        return value if len(value) <= 500 else value[:500] + marker
    if isinstance(value, list):
        kept = [_cap_field_value(item) for item in value[:2]]
        if len(value) > 2:
            kept.append({"truncated_for_budget": True, "omitted": len(value) - 2})
        return kept
    if isinstance(value, dict):
        return {key: _cap_field_value(item) for key, item in value.items()}
    return value


def _add_retrieval_marker(field_key: str, capped_value: Any) -> Any:
    """Attach the retrieval hint to a capped field's container, when the cap
    actually truncated something."""
    if isinstance(capped_value, str) and capped_value.endswith("...[truncated-for-budget]"):
        return {"truncated_for_budget": True, "field": field_key, "retrieve": _RETRIEVAL_HINT, "value": capped_value}
    if isinstance(capped_value, list) and any(
        isinstance(item, dict) and item.get("truncated_for_budget")
        for item in capped_value
    ):
        return {"truncated_for_budget": True, "field": field_key, "retrieve": _RETRIEVAL_HINT, "items": capped_value}
    return capped_value


def _render_context(case: dict[str, Any], replay: dict[str, Any]) -> str | None:
    """Render the case injection under the effective spill ceiling.

    Degradation ladder, re-measuring after each substep: compact separators;
    shed retrieved_prior_lessons, then collapse replay, then shed
    applied_prior_lessons; per-field caps over every remaining field
    (pending_probe keeps identity plus a byte count and the retrieval path;
    case_id/status/stale are never truncated); pointer form; and finally no
    injection at all when even the pointer exceeds the ceiling (an
    unrepresentable configuration — the case remains accessible through the
    portfolio tools). Returns None for that last tier.
    """
    ceiling = _resolve_injection_ceiling()
    case_id = case.get("case_id")
    compact = _compact_case(case, replay)

    rendered = _render_header_and_json(case_id, compact, indent=2)
    if len(rendered) <= ceiling:
        return rendered
    tier = 0

    rendered = _render_header_and_json(case_id, compact, indent=None)
    if len(rendered) <= ceiling:
        logger.warning("Epistemic case %s injection budgeted (tier 0)", case_id)
        return rendered
    tier = 1

    tiered = dict(compact)
    for field in ("retrieved_prior_lessons", "replay", "applied_prior_lessons"):
        if tiered.get(field) is None:
            continue
        tiered[field] = (
            {"usable": replay.get("usable"), "shed_for_budget": True, "retrieve": _RETRIEVAL_HINT}
            if field == "replay" and isinstance(replay, dict)
            else _shed_count_marker(tiered[field])
        )
        rendered = _render_header_and_json(case_id, tiered, indent=None)
        if len(rendered) <= ceiling:
            logger.warning(
                "Epistemic case %s injection budgeted (tier 1, through %s)", case_id, field
            )
            return rendered
    tier = 2

    capped = dict(tiered)
    capped["pending_probe"] = _digest_pending_probe(capped.get("pending_probe"))
    for field in (
        "prior_case_ids",
        "current_plan",
        "alternatives",
        "mechanisms",
        "state_grounding",
        "stopping_condition",
        "decision",
        "top_unknown",
    ):
        original = capped.get(field)
        capped[field] = _add_retrieval_marker(field, _cap_field_value(original))
    rendered = _render_header_and_json(case_id, capped, indent=None)
    if len(rendered) <= ceiling:
        logger.warning("Epistemic case %s injection budgeted (tier 2, field caps)", case_id)
        return rendered
    tier = 3

    pointer = (
        f"[ACTIVE EPISTEMIC CASE {case_id}]\n"
        "This bounded case is active. Its state exceeds the prompt budget; "
        "call epistemic_model(show) for the complete case before case-relevant work.\n"
        + json.dumps(
            {
                "case_id": case_id,
                "status": case.get("status"),
                "stale": case.get("stale"),
                "pending_probe": bool(case.get("pending_probe")),
                "budget_exceeded": True,
                "full_state": "call epistemic_model(show) for the complete case",
            },
            ensure_ascii=False,
            sort_keys=True,
        )
    )
    if len(pointer) <= ceiling:
        logger.warning("Epistemic case %s injection budgeted (tier 3, pointer form)", case_id)
        return pointer

    logger.warning(
        "Epistemic case %s injection suppressed: ceiling %d cannot hold the pointer form "
        "(%d chars); the case remains available via epistemic_model list/show",
        case_id,
        ceiling,
        len(pointer),
    )
    return None


def register(ctx: Any) -> None:
    """Register an optional epistemic plugin that cannot disable Hermes."""
    try:
        value = ctx.get_config("injection_max_chars", None)
        if value is not None:
            _INJECTION_BUDGET["injection_max_chars"] = value
    except Exception:
        logger.warning("epistemic-harness config read failed; using injection defaults")
    ctx.register_tool(
        name="epistemic_model",
        toolset="epistemic-harness",
        schema=MODEL_SCHEMA,
        handler=epistemic_model_handler,
        description=(
            "Manage and list lightweight epistemic case models, assessed evidence, and lifecycle. "
            "Always supply an explicit operation; opening needs only case_id and top_unknown, "
            "and richer fields are optional. Use after explicit activation or an announced bounded "
            "inquiry. Record ordinary external research with record_evidence. A named paused or "
            "closed case can be inspected read-only while another case remains active."
        ),
        emoji="🧭",
    )
    ctx.register_tool(
        name="epistemic_probe",
        toolset="epistemic-harness",
        schema=PROBE_SCHEMA,
        handler=epistemic_probe_handler,
        description=(
            "Commit, compare, recover, or discard one case-bound probe. Optionally bind a "
            "commit to an existing claim; bound comparisons state belief_change separately "
            "from observation disposition. Always supply the explicit operation."
        ),
        emoji="🔬",
    )
    ctx.register_skill(
        "epistemic-inquiry",
        _SKILL_PATH,
        description=_SKILL_DESCRIPTION,
        frontmatter={
            "name": "epistemic-inquiry",
            "description": _SKILL_DESCRIPTION,
        },
    )
    # Every hook and middleware is deliberately non-critical. A malformed or
    # temporarily unavailable epistemic store may disable case assistance, but
    # the plugin manager must continue the underlying Hermes turn exactly once.
    ctx.register_hook("pre_llm_call", pre_llm_context)
    ctx.register_hook("post_tool_call", post_tool_call)
    ctx.register_hook("transform_tool_result", transform_tool_result)
    ctx.register_hook("subagent_stop", on_subagent_stop)
    ctx.register_hook("on_session_reset", on_session_reset)
    ctx.register_hook("on_session_finalize", on_session_finalize)
    ctx.register_middleware("tool_execution", execution_gateway)
    logger.info("epistemic-harness registered generation=%s pid=%s", _RUNTIME_GENERATION, os.getpid())


def _error(message: str) -> str:
    return json.dumps({"error": message}, ensure_ascii=False)


def _validate_tool_request(
    args: Any,
    *,
    tool_name: str,
    operations: list[str],
) -> tuple[str, str]:
    """Validate request controls before touching the harness or durable store."""
    if not isinstance(args, dict):
        raise ValueError(
            f"{tool_name} requires an object with the 'operation' field on every call; "
            f"valid operations: {', '.join(operations)}"
        )
    raw_operation = args.get("operation")
    operation = raw_operation.strip().lower() if isinstance(raw_operation, str) else ""
    if not operation:
        raise ValueError(
            f"{tool_name} requires the 'operation' field on every call; "
            f"valid operations: {', '.join(operations)}. "
            f"Example: {{\"operation\":\"{operations[0]}\"}}"
        )
    if operation not in operations:
        raise ValueError(
            f"{tool_name} operation {raw_operation!r} is invalid; "
            f"valid operations: {', '.join(operations)}. "
            f"Example: {{\"operation\":\"{operations[0]}\"}}"
        )
    if "response_detail" in args and args["response_detail"] is None:
        raise ResponseDetailError(
            "response_detail must be full or compact; "
            "example: {\"response_detail\":\"full\"}"
        )
    response_detail = validate_response_detail(args.get("response_detail", "full"))
    return operation, response_detail
