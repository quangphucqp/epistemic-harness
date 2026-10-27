# AI-assisted contribution; maintained by Epistemic Harness contributors.
"""Optional TypeSafe/Jev method advice.

This module is deliberately separate from ``harness.py``: the core case model
stays local, and only this module can make a network call. It sends a bounded,
redacted projection of the case and a caller-supplied list of candidate
investigation methods. Jev ranks those candidates and estimates whether another
round would change the decision. It does not generate methods, execute them,
ground a claim, or close a case.
"""

from __future__ import annotations

import hashlib
import json
import math
import os
import re
import threading
import time
import urllib.error
import urllib.request
from dataclasses import dataclass
from typing import Any, Callable
from urllib.parse import urlparse

_DEFAULT_ENDPOINT = "https://api.typesafe.ai/v1/systemone"
_DEFAULT_MODEL = "jev-latest"
_CONFIG_KEYS = {
    "enabled",
    "model",
    "endpoint",
    "timeout_s",
    "max_calls_per_case",
    "max_input_tokens_per_case",
    "max_state_chars",
    "max_candidates",
    "max_candidate_chars",
    "offer_ttl_s",
    "max_consecutive_failures",
    "cooldown_s",
    "allow_user_domain_state",
}
_MAX_RESPONSE_BYTES = 262_144
_MAX_INFLIGHT_WORKERS = 4
_WORKER_SLOTS = threading.BoundedSemaphore(_MAX_INFLIGHT_WORKERS)


def reset_for_tests() -> None:
    """Replace the worker semaphore so one test cannot exhaust the next."""
    global _WORKER_SLOTS
    _WORKER_SLOTS = threading.BoundedSemaphore(_MAX_INFLIGHT_WORKERS)


_USER_AUTHORITY_DOMAINS = {
    "user_preference",
    "user_intent",
    "private_context",
    "user_decision",
    "user_testimony",
}
_CANDIDATE_ID = re.compile(r"^[a-z0-9][a-z0-9_-]{0,31}$")


class JevError(RuntimeError):
    """Base class for an optional-advice failure."""

    reason = "jev_error"


class JevConfigError(JevError):
    reason = "bad_config"


class JevCandidateError(JevConfigError):
    reason = "bad_candidates"


class JevTransportError(JevError):
    reason = "transport_error"

    def __init__(self, message: str, *, reason: str | None = None) -> None:
        super().__init__(message)
        if reason is not None:
            self.reason = reason


class JevMalformed(JevError):
    reason = "malformed_response"


@dataclass(frozen=True)
class JevConfig:
    """Configuration for the optional hosted judge."""

    enabled: bool = False
    misconfigured: bool = False
    model: str = _DEFAULT_MODEL
    endpoint: str = _DEFAULT_ENDPOINT
    timeout_s: float = 6.0
    max_calls_per_case: int = 4
    max_input_tokens_per_case: int = 30_000
    max_state_chars: int = 6_000
    max_candidates: int = 6
    max_candidate_chars: int = 240
    offer_ttl_s: int = 1_800
    max_consecutive_failures: int = 3
    cooldown_s: int = 600
    allow_user_domain_state: bool = False


def _clamp_int(value: Any, default: int, low: int, high: int) -> int:
    try:
        number = int(value)
    except (TypeError, ValueError):
        return default
    return max(low, min(high, number))


def _clamp_float(value: Any, default: float, low: float, high: float) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if not math.isfinite(number):
        return default
    return max(low, min(high, number))


def load_config(raw: Any) -> JevConfig:
    """Normalize plugin settings without raising into the host turn."""
    if raw is None:
        raw = {}
    if isinstance(raw, bool):
        if not raw:
            return JevConfig(enabled=False)
        raise JevConfigError("jev settings must be a mapping, not a boolean")
    if not isinstance(raw, dict):
        raise JevConfigError("jev settings must be a mapping")
    enabled = raw.get("enabled", False)
    if not isinstance(enabled, bool):
        text = str(enabled).strip().lower()
        if text in {"1", "true", "yes", "on"}:
            enabled = True
        elif text in {"0", "false", "no", "off", ""}:
            enabled = False
        else:
            raise JevConfigError(
                "jev enabled must be a boolean or one of true/false/yes/no/on/off"
            )
    unknown = sorted(set(raw) - _CONFIG_KEYS)
    if unknown:
        raise JevConfigError(
            "unknown jev setting(s): " + ", ".join(unknown)
        )
    if not enabled:
        return JevConfig(enabled=False)
    endpoint = str(raw.get("endpoint") or _DEFAULT_ENDPOINT).strip()
    parsed = urlparse(endpoint)
    if parsed.scheme != "https" or not parsed.netloc:
        raise JevConfigError("jev endpoint must be an https URL")
    model = str(raw.get("model") or _DEFAULT_MODEL).strip() or _DEFAULT_MODEL
    allow_user = raw.get("allow_user_domain_state", False)
    if not isinstance(allow_user, bool):
        allow_user = str(allow_user).strip().lower() in {"1", "true", "yes", "on"}
    return JevConfig(
        enabled=enabled,
        model=model,
        endpoint=endpoint,
        timeout_s=_clamp_float(raw.get("timeout_s"), 6.0, 1.0, 20.0),
        max_calls_per_case=_clamp_int(raw.get("max_calls_per_case"), 4, 1, 12),
        max_input_tokens_per_case=_clamp_int(
            raw.get("max_input_tokens_per_case"), 30_000, 1_000, 200_000
        ),
        max_state_chars=_clamp_int(raw.get("max_state_chars"), 6_000, 1_000, 12_000),
        max_candidates=_clamp_int(raw.get("max_candidates"), 6, 2, 8),
        max_candidate_chars=_clamp_int(
            raw.get("max_candidate_chars"), 240, 40, 500
        ),
        offer_ttl_s=_clamp_int(raw.get("offer_ttl_s"), 1_800, 60, 86_400),
        max_consecutive_failures=_clamp_int(
            raw.get("max_consecutive_failures"), 3, 1, 10
        ),
        cooldown_s=_clamp_int(raw.get("cooldown_s"), 600, 0, 86_400),
        allow_user_domain_state=allow_user,
    )


def api_key() -> str:
    """Resolve the vendor key through Hermes's profile secret scope.

    Standalone use falls back to the process environment. The plugin never reads
    ``.env`` or a credential store directly, and a multiplexed host with no
    active profile scope fails closed instead of reading another profile's
    environment.
    """
    try:
        from agent.secret_scope import UnscopedSecretError, get_secret
    except ImportError:
        return str(os.environ.get("TYPESAFE_API_KEY") or "").strip()
    try:
        value = get_secret("TYPESAFE_API_KEY", "")
    except UnscopedSecretError:
        return ""
    except Exception:
        # An optional advice path must not break the host turn on a secret
        # resolution error. Report no key; the tool stays hidden or unavailable.
        return ""
    return str(value or "").strip()


def _redact(text: Any) -> str:
    value = str(text or "").strip()
    value = re.sub(r"\bgh[pousr]_[A-Za-z0-9_]{8,}\b", "[REDACTED]", value)
    value = re.sub(r"\bsk-[A-Za-z0-9_-]{12,}\b", "[REDACTED]", value)
    value = re.sub(
        r"(?i)(authorization\s*[:=]\s*(?:bearer\s+)?)[^\s,;]+",
        r"\1[REDACTED]",
        value,
    )
    try:
        from agent.redact import redact_sensitive_text

        return redact_sensitive_text(value, force=True)
    except ImportError:
        return value


_LOCATOR_PATTERNS = (
    re.compile(r"(?i)\b(?:https?|ftp|s3|gs|file)://[^\s,;\"']+"),
    re.compile(r"(?i)\bdoi:\s*\S+"),
    re.compile(r"\b10\.\d{4,9}/[^\s,;\"']+"),
    re.compile(r"(?i)\bE\d{6,}\b"),
    re.compile(r"(?i)\b(?:archive/)?artifacts/[^\s,;\"']+"),
    re.compile(r"\\\\[^\s,;\"']+"),
    re.compile(r"(?i)\b[A-Za-z]:[\\/][^\s,;\"']+"),
    re.compile(r"""(?:^|(?<=[\s("'`=:,{\[]))~/[^\s,;)"']*"""),
    re.compile(
        r"""(?:^|(?<=[\s("'`=:,{\[]))/(?:[^\s/]+\.[A-Za-z0-9]{1,8})\b"""
    ),
    re.compile(r"""//[^\s,;)"']+"""),
    re.compile(
        r"""(?<![\w/])(?:[\w.-]+/)+[\w.-]*\.[A-Za-z0-9]{1,8}\b"""
    ),
    re.compile(
        r"""(?:^|(?<=[\s("'`=:,{\[]))(?:\.\./|\./|/)(?:[^\s/]+/)+[^\s,;)"']*"""
    ),
    re.compile(
        r"""(?:^|(?<=[\s("'`=:,{\[]))/(?:etc|usr|opt|var|tmp|root|home|Users|Volumes|private|Library|mnt|media)\b"""
    ),
)


def _scrub_locators(text: str) -> str:
    """Remove common path, URL, DOI, and event-ID shapes from free text.

    This is a best-effort content scrubber, not a guarantee: prose can describe
    a location without matching a pattern, and a path containing a space can
    leave its tail. The structured-field allowlist is the primary boundary; this
    reduces accidental locator disclosure in free text.
    """
    for pattern in _LOCATOR_PATTERNS:
        text = pattern.sub("[REDACTED-LOCATOR]", text)
    return text


class _Budget:
    """Deterministic character budget for the outbound projection."""

    def __init__(self, limit: int):
        self.limit = limit
        self.used = 0
        self.truncated = False
        self.user_domain_omitted = 0
        self.user_domain_included = 0

    def take(self, value: Any, cap: int) -> str | None:
        text = _scrub_locators(_redact(value))
        if len(text) > cap:
            self.truncated = True
            text = text[:cap]
        if not text:
            return None
        if self.used + len(text) > self.limit:
            self.truncated = True
            return None
        self.used += len(text)
        return text

    def take_list(self, values: Any, cap: int, count: int) -> list[str]:
        result: list[str] = []
        if not isinstance(values, list):
            return result
        for value in values[:count]:
            text = self.take(value, cap)
            if text is not None:
                result.append(text)
        return result


def _claim_texts(
    items: Any,
    *,
    budget: _Budget,
    cap: int,
    count: int,
    allow_user_domain_state: bool,
) -> list[str]:
    texts: list[str] = []
    if not isinstance(items, list):
        return texts
    if len(items) > count:
        budget.truncated = True
    for item in items[:count]:
        if not isinstance(item, dict):
            continue
        authority = str(item.get("authority_domain") or "").strip().lower()
        if authority in _USER_AUTHORITY_DOMAINS and not allow_user_domain_state:
            budget.user_domain_omitted += 1
            continue
        if authority in _USER_AUTHORITY_DOMAINS:
            budget.user_domain_included += 1
        text = item.get("text") or item.get("action") or item.get("summary")
        rendered = budget.take(text, cap)
        if rendered is not None:
            texts.append(rendered)
    return texts


def build_state(
    case: dict[str, Any],
    events: list[dict[str, Any]],
    config: JevConfig,
    *,
    round_number: int,
    previous_method: str | None,
) -> tuple[dict[str, Any], dict[str, Any]]:
    """Build the smallest useful, redacted projection of one case.

    The allowlist is explicit. Case IDs, session IDs, evidence locators, probe
    arguments and results, prior lessons, closure text, and artifact paths are
    never included.
    """
    raw_model = case.get("model")
    model: dict[str, Any] = raw_model if isinstance(raw_model, dict) else {}
    budget = _Budget(config.max_state_chars)
    state: dict[str, Any] = {
        "case": {
            "top_unknown": budget.take(model.get("top_unknown"), 600) or "",
            "decision": budget.take(model.get("decision"), 600) or "",
            "stopping_condition": budget.take(model.get("stopping_condition"), 600)
            or "",
            "round": round_number,
        },
        "claims": {
            "alternatives": _claim_texts(
                model.get("alternatives"),
                budget=budget,
                cap=240,
                count=5,
                allow_user_domain_state=config.allow_user_domain_state,
            ),
            "state_grounding": _claim_texts(
                model.get("state_grounding"),
                budget=budget,
                cap=240,
                count=6,
                allow_user_domain_state=config.allow_user_domain_state,
            ),
            "mechanisms": _claim_texts(
                model.get("mechanisms"),
                budget=budget,
                cap=240,
                count=4,
                allow_user_domain_state=config.allow_user_domain_state,
            ),
        },
        "plan": _claim_texts(
            model.get("current_plan"),
            budget=budget,
            cap=200,
            count=5,
            allow_user_domain_state=config.allow_user_domain_state,
        ),
        "recent_events": [],
        "previous_method": budget.take(previous_method, 240) or "",
    }
    recent: list[str] = []
    candidates = [
        event
        for event in events
        if isinstance(event, dict)
        and str(event.get("type") or "").strip()
        and not str(event.get("type") or "").startswith("method_")
    ]
    if len(candidates) > 3:
        budget.truncated = True
    for event in candidates[-3:]:
        event_type = str(event.get("type") or "").strip()
        rendered = budget.take(event_type, 80)
        if rendered is not None:
            recent.append(rendered)
    state["recent_events"] = recent
    privacy = {
        "sent_chars": budget.used + len(str(state["case"].get("round") or "")),
        "truncated": budget.truncated,
        "user_domain_omitted": budget.user_domain_omitted,
        "user_domain_included": budget.user_domain_included > 0,
        "fields": [
            "top_unknown",
            "decision",
            "stopping_condition",
            "round",
            "alternatives",
            "state_grounding",
            "mechanisms",
            "plan",
            "recent_events",
            "previous_method",
        ],
    }
    return state, privacy


def build_request(
    state: dict[str, Any],
    candidates: list[dict[str, str]],
    config: JevConfig,
) -> dict[str, Any]:
    """Build one System One request with a joint choice and a stop question."""
    criteria = {item["method_id"]: item["text"] for item in candidates}
    return {
        "model": config.model,
        "state": state,
        "questions": {
            "method": {
                "type": "choice",
                "instructions": (
                    "Which single investigation method, run next, is most likely to "
                    "change this case's decision or retire its top unknown? Prefer a "
                    "method that discriminates between the listed alternatives over one "
                    "that repeats evidence already recorded. Judge information value "
                    "only; ignore fluency, familiarity, or cost appeal."
                ),
                "criteria": criteria,
            },
            "more_investigation": {
                "type": "noul",
                "instructions": (
                    "Given what has already been recorded, would another investigation "
                    "round be likely to change the decision rather than repeat existing "
                    "evidence?"
                ),
                "criteria": {
                    "true": "Another round could plausibly change the decision",
                    "false": "Another round would probably repeat existing evidence",
                },
            },
        },
    }


class _NoRedirect(urllib.request.HTTPRedirectHandler):
    """Refuse redirects so the projected state cannot be forwarded elsewhere."""

    def redirect_request(self, *args: Any, **kwargs: Any) -> None:  # noqa: D401
        return None


def call(
    request: dict[str, Any],
    *,
    config: JevConfig,
    opener: Callable[..., Any] | None = None,
) -> dict[str, Any]:
    """POST one request and return the decoded JSON object."""
    if urlparse(config.endpoint).scheme != "https":
        raise JevConfigError("jev endpoint must be an https URL")
    key = api_key()
    if not key:
        raise JevConfigError("TYPESAFE_API_KEY is not set")
    body = json.dumps(request, ensure_ascii=False).encode("utf-8")
    http_request = urllib.request.Request(
        config.endpoint,
        data=body,
        method="POST",
        headers={
            "Authorization": f"Bearer {key}",
            "Content-Type": "application/json",
            "User-Agent": "epistemic-harness/1.4.3",
        },
    )
    open_call = opener or urllib.request.build_opener(_NoRedirect()).open
    cancel = threading.Event()

    def fetch() -> dict[str, Any]:
        deadline = time.monotonic() + config.timeout_s
        chunks: list[bytes] = []
        total = 0
        if cancel.is_set():
            raise JevTransportError("timeout", reason="timeout")
        try:
            if cancel.is_set():
                raise JevTransportError("timeout", reason="timeout")
            with open_call(http_request, timeout=config.timeout_s) as response:
                while True:
                    if cancel.is_set() or time.monotonic() >= deadline:
                        raise JevTransportError("timeout", reason="timeout")
                    remaining = _MAX_RESPONSE_BYTES + 1 - total
                    if remaining <= 0:
                        raise JevMalformed("response exceeded the size limit")
                    reader = getattr(response, "read1", None) or response.read
                    chunk = reader(min(8192, remaining))
                    if not chunk:
                        break
                    chunks.append(chunk)
                    total += len(chunk)
        except urllib.error.HTTPError as exc:
            reason = "auth_error" if exc.code in {401, 403} else "http_error"
            raise JevTransportError(
                f"http_error:{exc.code}", reason=reason
            ) from exc
        except JevError:
            raise
        except (urllib.error.URLError, TimeoutError, OSError) as exc:
            raise JevTransportError(
                f"transport_error:{type(exc).__name__}"
            ) from exc
        raw = b"".join(chunks)
        if len(raw) > _MAX_RESPONSE_BYTES:
            raise JevMalformed("response exceeded the size limit")
        try:
            decoded = json.loads(raw.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as exc:
            raise JevMalformed("response was not valid JSON") from exc
        if not isinstance(decoded, dict):
            raise JevMalformed("response was not a JSON object")
        return decoded

    outcome: dict[str, Any] = {}

    slots = _WORKER_SLOTS
    if not slots.acquire(blocking=False):
        raise JevTransportError(
            "too many in-flight transport workers", reason="busy"
        )

    def worker() -> None:
        try:
            outcome["value"] = fetch()
        except BaseException as exc:  # noqa: BLE001 - re-raised in the caller
            outcome["error"] = exc
        finally:
            slots.release()

    thread = threading.Thread(target=worker, name="epistemic-jev", daemon=True)
    try:
        thread.start()
    except BaseException:
        slots.release()
        raise
    thread.join(config.timeout_s)
    if thread.is_alive():
        cancel.set()
        raise JevTransportError("timeout", reason="timeout")
    if "error" in outcome:
        raise outcome["error"]
    return outcome["value"]


def parse_offer(
    response: dict[str, Any],
    *,
    offered_ids: list[str],
    config: JevConfig,
) -> dict[str, Any]:
    """Validate the vendor reply and return an internal offer record."""
    answers = response.get("answers")
    if not isinstance(answers, dict):
        raise JevMalformed("response has no answers object")
    method = answers.get("method")
    if not isinstance(method, dict):
        raise JevMalformed("response has no method answer")
    choice = str(method.get("choice") or "").strip()
    if choice not in offered_ids:
        raise JevMalformed("method choice is not one of the supplied candidates")
    probabilities = method.get("probabilities")
    if not isinstance(probabilities, dict):
        raise JevMalformed("method answer has no probabilities")
    clean: dict[str, float] = {}
    for method_id in offered_ids:
        raw_value = probabilities.get(method_id)
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise JevMalformed(f"missing probability for {method_id}")
        value = float(raw_value)
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            raise JevMalformed(f"invalid probability for {method_id}")
        clean[method_id] = value
    confidence = method.get("confidence")
    if confidence is not None:
        if isinstance(confidence, bool) or not isinstance(confidence, (int, float)):
            raise JevMalformed("confidence is not numeric")
        confidence = float(confidence)
        if not math.isfinite(confidence) or confidence < 0.0 or confidence > 1.0:
            raise JevMalformed("confidence is outside 0..1")
    more = answers.get("more_investigation")
    continue_probability = None
    if isinstance(more, dict):
        value = more.get("noul")
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise JevMalformed("more_investigation answer is not numeric")
        continue_probability = float(value)
        if not math.isfinite(continue_probability) or not 0.0 <= continue_probability <= 1.0:
            raise JevMalformed("more_investigation is outside 0..1")
    raw_usage = response.get("usage")
    usage: dict[str, Any] = raw_usage if isinstance(raw_usage, dict) else {}
    input_tokens = usage.get("input_tokens", 0)
    output_tokens = usage.get("output_tokens", 0)
    if isinstance(input_tokens, bool) or not isinstance(input_tokens, int):
        input_tokens = 0
    if isinstance(output_tokens, bool) or not isinstance(output_tokens, int):
        output_tokens = 0
    input_tokens = max(0, input_tokens)
    output_tokens = max(0, output_tokens)
    ranking = sorted(
        (
            {"method_id": method_id, "probability": value}
            for method_id, value in clean.items()
        ),
        key=lambda item: (-item["probability"], item["method_id"]),
    )
    model_label = str(response.get("model") or config.model).strip()
    if not re.fullmatch(r"[A-Za-z0-9._:/+-]{1,80}", model_label):
        model_label = "unknown"
    return {
        "recommended_method_id": choice,
        "ranking": ranking,
        "confidence": confidence,
        "continue_probability": continue_probability,
        "model": model_label,
        "usage": {"input_tokens": input_tokens, "output_tokens": output_tokens},
    }


def projection_digest(state: dict[str, Any]) -> str:
    """Stable digest used only for local offer provenance."""
    encoded = json.dumps(state, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()


def validate_candidates(
    raw: Any,
    config: JevConfig,
    *,
    candidates_json: Any = None,
) -> list[dict[str, str]]:
    """Validate the caller-supplied candidate list.

    Candidate methods are supplied per case by the agent (or any other
    proposer). The plugin does not ship a fixed method taxonomy.
    """
    if candidates_json is not None:
        if not isinstance(candidates_json, str) or not candidates_json.strip():
            raise JevCandidateError("candidates_json must be a non-empty JSON string")
        try:
            parsed = json.loads(candidates_json)
        except json.JSONDecodeError as exc:
            raise JevCandidateError("candidates_json is not valid JSON") from exc
        # Providers can materialize an omitted array argument as []. Treat an
        # empty list as absent, matching the probe tool's tool_args_json route.
        if raw not in (None, []):
            if not isinstance(raw, list):
                raise JevCandidateError(
                    "candidates must be an array of method objects"
                )
            if json.dumps(raw, sort_keys=True, ensure_ascii=False) != json.dumps(
                parsed, sort_keys=True, ensure_ascii=False
            ):
                raise JevCandidateError(
                    "candidates and candidates_json must describe the same list"
                )
        raw = parsed
    if not isinstance(raw, list):
        raise JevCandidateError("candidates must be an array of method objects")
    if len(raw) < 2:
        raise JevCandidateError("provide at least two candidate methods")
    if len(raw) > config.max_candidates:
        raise JevCandidateError(
            f"provide at most {config.max_candidates} candidate methods"
        )
    result: list[dict[str, str]] = []
    seen: set[str] = set()
    for item in raw:
        if not isinstance(item, dict):
            raise JevCandidateError("each candidate must be an object")
        method_id = str(item.get("method_id") or "").strip().lower()
        text = _scrub_locators(_redact(item.get("text"))).strip()
        if not _CANDIDATE_ID.fullmatch(method_id):
            raise JevCandidateError(
                "method_id must be lowercase letters, digits, hyphen, or underscore"
            )
        if method_id in seen:
            raise JevCandidateError(f"duplicate method_id: {method_id}")
        if not text:
            raise JevCandidateError(f"candidate {method_id} has no text")
        if len(text) > config.max_candidate_chars:
            raise JevCandidateError(
                f"candidate {method_id} exceeds {config.max_candidate_chars} characters"
            )
        seen.add(method_id)
        result.append({"method_id": method_id, "text": text})
    return result
