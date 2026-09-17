# AI-assisted contribution; maintained by Epistemic Harness contributors.
"""Regression tests for the 2026-09-07 audit fixes.

- Claim-array schemas must carry their sub-properties through Hermes's
  runtime sanitizer (sparse schemas were stripped by some providers, so
  revisions arrived as empty arrays: "revision makes no material change").
- Overall calibration must never read "supported" while an ungrounded
  hypothesis remains in the model.
"""

from __future__ import annotations

from epistemic_harness.harness import _closure_calibration
from epistemic_harness.schemas import MODEL_SCHEMA


def test_claim_arrays_keep_sub_properties_through_the_runtime_sanitizer():
    try:
        from tools.schema_sanitizer import sanitize_tool_schemas
    except ImportError:
        import sys
        from pathlib import Path

        repo = Path.home() / ".hermes" / "hermes-agent"
        if not (repo / "tools" / "schema_sanitizer.py").exists():
            import pytest

            pytest.skip("Hermes repo unavailable")
        sys.path.insert(0, str(repo))
        from tools.schema_sanitizer import sanitize_tool_schemas

    cleaned = sanitize_tool_schemas([{"type": "function", "function": MODEL_SCHEMA}])[0][
        "function"
    ]["parameters"]
    for field in ("state_grounding", "mechanisms", "alternatives"):
        props = cleaned["properties"][field]["items"]["properties"]
        assert {"id", "text", "evidence_event_ids", "authority_domain", "epistemic_status"} <= set(props)
        assert cleaned["properties"][field]["items"]["required"] == ["id", "text"]
    # The updates catch-all and the probe evidence descriptor are the same
    # stripping class; both must carry their sub-properties.
    updates = cleaned["properties"]["updates"]["properties"]
    assert updates["mechanisms"]["items"]["properties"]["epistemic_status"]
    assert updates["current_plan"]["items"]["required"] == ["id", "action"]


def test_probe_evidence_descriptor_keeps_sub_properties():
    from epistemic_harness.schemas import PROBE_SCHEMA

    try:
        from tools.schema_sanitizer import sanitize_tool_schemas
    except ImportError:
        import sys
        from pathlib import Path

        repo = Path.home() / ".hermes" / "hermes-agent"
        if not (repo / "tools" / "schema_sanitizer.py").exists():
            import pytest

            pytest.skip("Hermes repo unavailable")
        sys.path.insert(0, str(repo))
        from tools.schema_sanitizer import sanitize_tool_schemas

    cleaned = sanitize_tool_schemas([{"type": "function", "function": PROBE_SCHEMA}])[0][
        "function"
    ]["parameters"]
    props = cleaned["properties"]["evidence"]["properties"]
    assert {"summary", "source_ref", "source_role", "access_scope", "limitation"} <= set(props)


def test_overall_calibration_caps_when_an_ungrounded_hypothesis_remains():
    case = {
        "model": {
            "state_grounding": [
                {"id": "C1", "epistemic_status": "grounded", "evidence_event_ids": ["E1"]},
                {"id": "C2", "epistemic_status": "hypothesis", "evidence_event_ids": []},
            ]
        }
    }
    events = [
        {
            "event_id": "E1",
            "type": "evidence_recorded",
            "payload": {"source_role": "secondary", "access_scope": "full"},
        }
    ]
    calibration = _closure_calibration(case, events=events)
    assert calibration["label"] == "mixed"
    assert "ungrounded" in calibration["language"]


def test_fully_grounded_supported_case_still_reads_supported():
    case = {
        "model": {
            "state_grounding": [
                {"id": "C1", "epistemic_status": "grounded", "evidence_event_ids": ["E1"]}
            ]
        }
    }
    events = [
        {
            "event_id": "E1",
            "type": "evidence_recorded",
            "payload": {"source_role": "primary", "access_scope": "full"},
        }
    ]
    calibration = _closure_calibration(case, events=events)
    assert calibration["label"] == "supported"
