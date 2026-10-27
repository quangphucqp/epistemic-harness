# AI-assisted contribution; maintained by Epistemic Harness contributors.
"""Pure helpers for the optional method-advice Timeline events.

These functions read only the case Timeline. They never touch the network, the
case model, or the store. The harness writes the events; this module decides
what the written history means.
"""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

OFFER_STARTED = "method_offer_started"
OFFERED = "method_offered"
OFFER_FAILED = "method_offer_failed"
CHOSEN = "method_chosen"
EXCUSED = "method_consultation_excused"
METHOD_EVENT_TYPES = {OFFER_STARTED, OFFERED, OFFER_FAILED, CHOSEN, EXCUSED}
DISPOSITIONS = {"continue_same", "continue_different", "stop"}
DECISION_POINTS = {"probe", "close"}

# A failed consultation may be waived only when the failure is not the
# caller's fault. Caller-input errors such as bad_candidates and
# bad_disposition must not become a way to skip the gate.
EXCUSABLE_FAILURES = {
    "transport_error",
    "timeout",
    "http_error",
    "busy",
    "malformed_response",
    "jev_error",
    "state_write_failed",
}



class MethodError(ValueError):
    """Raised when a method-advice request violates its contract."""

    def __init__(self, message: str, *, reason: str = "bad_request"):
        super().__init__(message)
        self.reason = reason


def payload(event: dict[str, Any]) -> dict[str, Any]:
    value = event.get("payload")
    return value if isinstance(value, dict) else {}


def events_of(events: list[dict[str, Any]], event_type: str) -> list[dict[str, Any]]:
    return [
        event
        for event in events
        if isinstance(event, dict) and event.get("type") == event_type
    ]


def current_offer(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Return the newest unanswered offer, including its Timeline event ID."""
    answered = {
        str(payload(event).get("offer_id") or "")
        for event in events_of(events, CHOSEN)
    }
    for event in reversed(events_of(events, OFFERED)):
        data = payload(event)
        offer_id = str(data.get("offer_id") or "")
        if not offer_id or offer_id in answered:
            continue
        return {**data, "event_id": str(event.get("event_id") or "")}
    return None


def offer_by_id(events: list[dict[str, Any]], offer_id: str) -> dict[str, Any] | None:
    for event in reversed(events_of(events, OFFERED)):
        data = payload(event)
        if str(data.get("offer_id") or "") == offer_id:
            return {**data, "event_id": str(event.get("event_id") or "")}
    return None


def last_chosen(events: list[dict[str, Any]]) -> dict[str, Any] | None:
    chosen = events_of(events, CHOSEN)
    return payload(chosen[-1]) if chosen else None


def previous_method_id(events: list[dict[str, Any]]) -> str | None:
    for event in reversed(events_of(events, CHOSEN)):
        method_id = str(payload(event).get("method_id") or "").strip()
        if method_id:
            return method_id
    return None


def decision_point_of(event: dict[str, Any] | None) -> str:
    if not isinstance(event, dict):
        return ""
    return str(payload(event).get("decision_point") or "").strip().lower()


def consultation_at_tip(
    events: list[dict[str, Any]], *, decision_point: str
) -> dict[str, Any] | None:
    """Return the tip consultation for one decision point, if any.

    A consultation is current only while it is the last Timeline event. This
    forces a fresh offer/choose (or a recorded waiver) immediately before the
    decision it informs, rather than allowing one offer to cover several later
    commitments.
    """
    if not events:
        return None
    tip = events[-1]
    if not isinstance(tip, dict):
        return None
    event_type = tip.get("type")
    if decision_point_of(tip) != decision_point:
        return None
    if event_type == CHOSEN:
        return dict(payload(tip))
    if event_type == EXCUSED:
        return dict(payload(tip))
    if event_type == OFFER_FAILED:
        reason = str(payload(tip).get("reason") or "").strip()
        if reason in EXCUSABLE_FAILURES:
            return dict(payload(tip))
    return None



def round_number(events: list[dict[str, Any]]) -> int:
    """Count attempts, so a failed attempt still advances the round."""
    return len(events_of(events, OFFER_STARTED)) + 1


def valid_current_offer(
    events: list[dict[str, Any]], *, now: float
) -> dict[str, Any] | None:
    """Return an unanswered offer that is still the Timeline tip and unexpired."""
    offer = current_offer(events)
    if offer is None:
        return None
    if not events or str(events[-1].get("event_id") or "") != str(
        offer.get("event_id") or ""
    ):
        return None
    expires = parse_time(offer.get("expires_at"))
    if expires is None or now > expires:
        return None
    return offer


def budget_used(events: list[dict[str, Any]]) -> dict[str, int]:
    attempts = max(0, len(events_of(events, OFFER_STARTED)))
    tokens = 0
    for event in events_of(events, OFFERED):
        usage = payload(event).get("usage")
        if isinstance(usage, dict):
            value = usage.get("input_tokens")
            if isinstance(value, int) and not isinstance(value, bool):
                tokens += max(0, value)
    return {"attempts": attempts, "input_tokens": max(0, tokens)}


def parse_time(value: Any) -> float | None:
    text = str(value or "").strip()
    if not text:
        return None
    try:
        if text.endswith("Z"):
            text = text[:-1] + "+00:00"
        parsed = datetime.fromisoformat(text)
    except ValueError:
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=timezone.utc)
    return parsed.timestamp()


def failure_state(events: list[dict[str, Any]]) -> tuple[int, float | None]:
    """Return trailing consecutive failures and the last failure timestamp."""
    count = 0
    last_time: float | None = None
    for event in reversed(events):
        if not isinstance(event, dict):
            continue
        event_type = event.get("type")
        if event_type == OFFERED:
            break
        if event_type == OFFER_FAILED:
            count += 1
            if last_time is None:
                last_time = parse_time(event.get("timestamp"))
    return count, last_time


def cooldown_active(
    events: list[dict[str, Any]],
    *,
    now: float,
    max_consecutive_failures: int,
    cooldown_s: int,
) -> bool:
    if cooldown_s <= 0:
        return False
    count, last_time = failure_state(events)
    if count < max_consecutive_failures or last_time is None:
        return False
    return now - last_time < cooldown_s


def validate_disposition(
    args: dict[str, Any],
    *,
    offer: dict[str, Any],
    previous_method: str | None,
    decision_point: str,
) -> dict[str, Any]:
    disposition = str(args.get("disposition") or "").strip().lower()
    if disposition not in DISPOSITIONS:
        raise MethodError(
            "disposition must be continue_same, continue_different, or stop",
            reason="bad_disposition",
        )
    if decision_point not in DECISION_POINTS:
        raise MethodError(
            "the offer has no valid decision point", reason="bad_decision_point"
        )
    offered = {
        str(item.get("method_id") or "")
        for item in (offer.get("candidates") or [])
        if isinstance(item, dict)
    }
    raw_method = str(args.get("method_id") or "").strip().lower()
    note = str(args.get("note") or "").strip()[:500]
    if decision_point == "close":
        if disposition != "stop":
            raise MethodError(
                "closing the case requires disposition stop; use stop to record "
                "the closure decision, or continue the investigation",
                reason="bad_disposition",
            )
        if raw_method:
            raise MethodError(
                "stop does not take a method_id", reason="bad_disposition"
            )
        return {"disposition": disposition, "method_id": None, "note": note}
    if disposition == "stop":
        raise MethodError(
            "stop does not authorize a probe; choose continue_same or "
            "continue_different to record the method you will run",
            reason="bad_disposition",
        )
    if disposition == "continue_same":
        if raw_method:
            raise MethodError(
                "continue_same uses the previously chosen method; omit method_id",
                reason="bad_disposition",
            )
        if not previous_method:
            raise MethodError(
                "continue_same needs a previously chosen method; use "
                "continue_different with a method_id, or stop",
                reason="no_previous_method",
            )
        if previous_method not in offered:
            raise MethodError(
                "the previous method is not in this offer; use continue_different",
                reason="previous_method_not_offered",
            )
        return {
            "disposition": disposition,
            "method_id": previous_method,
            "note": note,
        }
    if not raw_method:
        raise MethodError(
            "continue_different requires method_id", reason="missing_method"
        )
    if raw_method not in offered:
        raise MethodError(
            "method_id must be one of the offered candidate methods",
            reason="unknown_method",
        )
    if raw_method == previous_method:
        raise MethodError(
            "that is the previous method; use continue_same", reason="same_method"
        )
    return {"disposition": disposition, "method_id": raw_method, "note": note}
