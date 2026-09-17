"""Conservative, presentation-only projections for model-facing tool receipts.

The harness and store return the full durable receipt. This module is intentionally
small and read-only: compact output may omit verbose lesson prose, but it never
changes a case, Timeline, replay result, or audit receipt.
"""

# AI-assisted modification by Luftballon, 2026-09-16.

from __future__ import annotations

from copy import deepcopy
from typing import Any

_RESPONSE_DETAILS = ("full", "compact")


class ResponseDetailError(ValueError):
    """Raised when a caller supplies an unsupported response presentation."""


def validate_response_detail(value: Any) -> str:
    """Normalize the optional presentation selector without guessing invalid values."""
    if value is None:
        return "full"
    if not isinstance(value, str) or value.strip().lower() not in _RESPONSE_DETAILS:
        raise ResponseDetailError(
            "response_detail must be full or compact; "
            "example: {\"response_detail\":\"full\"}"
        )
    return value.strip().lower()


def project_response(
    result: Any,
    *,
    response_detail: Any = "full",
    requested_case_id: Any = None,
) -> Any:
    """Return a compact view while leaving all non-lesson receipt fields intact.

    Only ``case.model.retrieved_prior_lessons`` and top-level
    ``prior_case_lessons`` are projection targets. For a lesson record at one of
    those locations, compact mode removes exactly its verbose ``lesson`` body,
    keeps every other identity/provenance field, and records an exact full
    readback call. The input result is never mutated; durable state is not
    involved. When no lesson body is omitted, no readback requirement is added.
    """
    detail = validate_response_detail(response_detail)
    if detail == "full" or not isinstance(result, dict):
        return result

    projected = deepcopy(result)
    omitted_paths: list[str] = []

    case = projected.get("case")
    if isinstance(case, dict):
        model = case.get("model")
        if isinstance(model, dict):
            lessons = model.get("retrieved_prior_lessons")
            if isinstance(lessons, list):
                model["retrieved_prior_lessons"] = _project_lessons(
                    lessons,
                    "case.model.retrieved_prior_lessons",
                    omitted_paths,
                )

    lessons = projected.get("prior_case_lessons")
    if isinstance(lessons, list):
        projected["prior_case_lessons"] = _project_lessons(
            lessons,
            "prior_case_lessons",
            omitted_paths,
        )

    projected["response_detail"] = "compact"
    projected["omitted_field_paths"] = omitted_paths

    if not omitted_paths:
        return projected

    case = projected.get("case")
    case_id = None
    if isinstance(case, dict):
        case_id = case.get("case_id")
    if not case_id:
        case_id = requested_case_id
    if isinstance(case_id, str) and case_id.strip():
        projected["readback"] = {
            "tool": "epistemic_model",
            "arguments": {
                "operation": "show",
                "case_id": case_id,
                "response_detail": "full",
            },
        }
    return projected


def _project_lessons(
    lessons: list[Any],
    path: str,
    omitted_paths: list[str],
) -> list[Any]:
    projected: list[Any] = []
    for index, lesson in enumerate(lessons):
        item_path = f"{path}[{index}]"
        if not isinstance(lesson, dict):
            projected.append(deepcopy(lesson))
            continue
        if "lesson" not in lesson:
            projected.append(deepcopy(lesson))
            continue
        summary = deepcopy(lesson)
        del summary["lesson"]
        omitted_paths.append(f"{item_path}.lesson")
        summary["omitted_fields"] = ["lesson"]
        projected.append(summary)
    return projected


__all__ = ["ResponseDetailError", "project_response", "validate_response_detail"]
