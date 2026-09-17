"""Public-release behavior contracts for operation safety, compact output, and skill packaging."""

# AI-assisted modification by Luftballon, 2026-09-16.

from __future__ import annotations

import json
from pathlib import Path

import yaml

from epistemic_harness import plugin
from epistemic_harness.presentation import project_response
from epistemic_harness.schemas import MODEL_SCHEMA, PROBE_SCHEMA


def _parsed(value: str) -> dict:
    result = json.loads(value)
    assert isinstance(result, dict)
    return result


def _open(case_id: str, *, response_detail: str | None = None) -> dict:
    args = {
        "operation": "open",
        "case_id": case_id,
        "top_unknown": "Which bounded explanation remains useful?",
    }
    if response_detail is not None:
        args["response_detail"] = response_detail
    return args


def _call_model(args: dict, *, session_id: str) -> dict:
    return _parsed(plugin.epistemic_model_handler(args, session_id=session_id))


def test_operation_and_response_detail_are_explicit_in_both_tool_schemas():
    for schema, operations in (
        (MODEL_SCHEMA, {"open", "list", "show", "record_evidence", "revise", "pause", "resume", "close"}),
        (PROBE_SCHEMA, {"commit", "compare", "recover", "discard"}),
    ):
        properties = schema["parameters"]["properties"]
        assert "operation" in schema["parameters"]["required"]
        assert set(properties["operation"]["enum"]) == operations
        assert properties["operation"].get("description")
        assert properties["response_detail"]["enum"] == ["full", "compact"]
        assert "response_detail" in properties["response_detail"].get("description", "")
        assert "always" in schema["description"].lower()


def test_missing_or_invalid_operation_is_rejected_before_case_recovery_or_mutation(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    plugin.reset_for_tests()

    for args in ({}, {"operation": "not-an-operation"}):
        error = _parsed(plugin.epistemic_model_handler(args, session_id="unbound-session"))
        assert "error" in error
        assert "operation" in error["error"]
        assert "open" in error["error"]

    probe_error = _parsed(plugin.epistemic_probe_handler({}, session_id="unbound-session"))
    assert "operation" in probe_error["error"]
    assert "commit" in probe_error["error"]

    invalid_detail = _parsed(
        plugin.epistemic_model_handler(
            _open("must-not-open", response_detail="verbose"),
            session_id="unbound-session",
        )
    )
    assert "response_detail" in invalid_detail["error"]
    assert not (tmp_path / "epistemic-harness").exists()


def test_compact_projection_is_conservative_and_full_readback_restores_lesson_body(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    plugin.reset_for_tests()
    lesson = ("A long bounded lesson that must remain on disk and be available by full readback. " * 80).strip()

    _call_model(_open("source-case"), session_id="source-session")
    _call_model(
        {
            "operation": "close",
            "outcome": "unresolved",
            "summary": "The source case remains unresolved.",
            "transfer": {
                "decision": "candidate",
                "lesson": lesson,
                "scope": "bounded inquiry",
                "evidence_case_ids": ["source-case"],
            },
        },
        session_id="source-session",
    )

    compact = _call_model(
        {
            **_open("compact-case", response_detail="compact"),
            "prior_case_ids": ["source-case"],
        },
        session_id="compact-session",
    )

    assert compact["response_detail"] == "compact"
    assert compact["omitted_field_paths"] == [
        "case.model.retrieved_prior_lessons[0].lesson",
        "prior_case_lessons[0].lesson",
    ]
    assert compact["readback"] == {
        "tool": "epistemic_model",
        "arguments": {
            "operation": "show",
            "case_id": "compact-case",
            "response_detail": "full",
        },
    }
    assert compact["case"]["case_id"] == "compact-case"
    assert compact["case"]["status"] == "active"
    assert compact["case"]["version"] == 1
    assert compact["case"]["stale"] is False
    assert compact["case"]["pending_probe"] is None
    assert compact["case"]["model"]["top_unknown"] == "Which bounded explanation remains useful?"
    assert compact["case"]["model"]["retrieved_prior_lessons"][0]["case_id"] == "source-case"
    assert "lesson" not in compact["case"]["model"]["retrieved_prior_lessons"][0]
    assert len(json.dumps(compact, ensure_ascii=False)) < len(
        json.dumps(
            _call_model(
                {"operation": "show", "case_id": "compact-case", "response_detail": "full"},
                session_id="compact-session",
            ),
            ensure_ascii=False,
        )
    )

    full = _call_model(
        {"operation": "show", "case_id": "compact-case", "response_detail": "full"},
        session_id="compact-session",
    )
    assert "response_detail" not in full
    assert full["case"]["model"]["retrieved_prior_lessons"][0]["lesson"] == lesson


def test_default_response_detail_remains_full():
    assert "response_detail" not in _open("default-detail")
    assert MODEL_SCHEMA["parameters"]["properties"]["response_detail"]["default"] == "full"
    assert PROBE_SCHEMA["parameters"]["properties"]["response_detail"]["default"] == "full"


def test_compact_projection_preserves_non_lesson_receipts_and_does_not_mutate_input():
    result = {
        "case": {
            "case_id": "receipt-case",
            "status": "active",
            "version": 4,
            "stale": True,
            "stale_claim_ids": ["M1", "P1"],
            "mismatch_event_id": "E000004",
            "pending_probe": {
                "status": "observed",
                "argument_drift_reason": "fallback source used",
                "claim_snapshot": {"id": "M1", "text": "claim"},
            },
            "closure": {"calibration": {"label": "hypothesis"}},
            "model": {
                "decision": "Keep the question open",
                "stopping_condition": "Stop at unresolved",
                "top_unknown": "Whether M1 holds",
                "state_grounding": [],
                "mechanisms": [],
                "alternatives": [],
                "current_plan": [],
                "retrieved_prior_lessons": [
                    {
                        "case_id": "prior-case",
                        "decision": "candidate",
                        "lesson": "verbose lesson body",
                        "scope": "bounded scope",
                        "evidence_case_ids": ["prior-case"],
                        "calibration": "hypothesis",
                    }
                ],
            },
        },
        "event": {"event_id": "E000005", "payload": {"evidence_event_ids": ["E000004"]}},
        "probe": {"claim_snapshot": {"id": "M1"}, "argument_drift_reason": "fallback source used"},
        "replay": {"accounting": {"uncompared_observation_event_ids": ["E000003"]}, "stale": True},
    }
    before = json.loads(json.dumps(result))
    compact = project_response(result, response_detail="compact")

    assert compact["event"] == result["event"]
    assert compact["probe"] == result["probe"]
    assert compact["replay"] == result["replay"]
    assert compact["case"]["stale_claim_ids"] == ["M1", "P1"]
    assert compact["case"]["mismatch_event_id"] == "E000004"
    assert compact["case"]["pending_probe"] == result["case"]["pending_probe"]
    assert compact["case"]["closure"] == result["case"]["closure"]
    assert "lesson" not in compact["case"]["model"]["retrieved_prior_lessons"][0]
    assert result == before


def test_compact_projection_only_targets_known_lesson_locations_and_keeps_provenance():
    result = {
        "case": {
            "case_id": "target-case",
            "model": {
                "retrieved_prior_lessons": [
                    {
                        "case_id": "model-prior",
                        "decision": "candidate",
                        "scope": "bounded inquiry",
                        "evidence_case_ids": ["model-prior"],
                        "evidence_event_ids": ["E000001"],
                        "calibration": "hypothesis",
                        "relevance": 0.75,
                        "source_descriptor": {
                            "role": "primary",
                            "locator": "source://model-prior",
                        },
                        "provenance_note": "small-source-receipt",
                        "lesson": "verbose model lesson",
                    }
                ],
                "prior_case_lessons": [
                    {
                        "case_id": "nested-prior",
                        "evidence_event_ids": ["E000002"],
                        "source_descriptor": {"role": "secondary"},
                        "lesson": "not a projection target",
                    }
                ],
                "nested": {
                    "retrieved_prior_lessons": [
                        {
                            "case_id": "deep-prior",
                            "evidence_event_ids": ["E000003"],
                            "lesson": "also not a projection target",
                        }
                    ]
                },
            },
        },
        "prior_case_lessons": [
            {
                "case_id": "top-prior",
                "decision": "candidate",
                "scope": "top-level scope",
                "evidence_case_ids": ["top-prior"],
                "evidence_event_ids": ["E000004"],
                "calibration": "supported",
                "relevance": 0.5,
                "source_descriptor": {"role": "synthesis"},
                "provenance_note": "top-level-receipt",
                "lesson": "verbose top-level lesson",
            }
        ],
        "event": {
            "event_id": "E000005",
            "prior_case_lessons": [
                {
                    "case_id": "event-prior",
                    "evidence_event_ids": ["E000006"],
                    "source_descriptor": {"kind": "event"},
                    "lesson": "event receipt body",
                }
            ],
        },
        "probe": {
            "retrieved_prior_lessons": [
                {
                    "case_id": "probe-prior",
                    "evidence_event_ids": ["E000007"],
                    "source_descriptor": {"kind": "probe"},
                    "lesson": "probe receipt body",
                }
            ]
        },
        "replay": {
            "prior_case_lessons": [
                {
                    "case_id": "replay-prior",
                    "evidence_event_ids": ["E000008"],
                    "source_descriptor": {"kind": "replay"},
                    "lesson": "replay receipt body",
                }
            ]
        },
    }
    before = json.loads(json.dumps(result))

    compact = project_response(result, response_detail="compact")

    expected_model_lesson = {
        key: value
        for key, value in result["case"]["model"]["retrieved_prior_lessons"][0].items()
        if key != "lesson"
    }
    expected_model_lesson["omitted_fields"] = ["lesson"]
    expected_top_lesson = {
        key: value
        for key, value in result["prior_case_lessons"][0].items()
        if key != "lesson"
    }
    expected_top_lesson["omitted_fields"] = ["lesson"]

    assert compact["case"]["model"]["retrieved_prior_lessons"] == [expected_model_lesson]
    assert compact["prior_case_lessons"] == [expected_top_lesson]
    assert compact["case"]["model"]["prior_case_lessons"] == result["case"]["model"]["prior_case_lessons"]
    assert compact["case"]["model"]["nested"] == result["case"]["model"]["nested"]
    assert compact["event"] == result["event"]
    assert compact["probe"] == result["probe"]
    assert compact["replay"] == result["replay"]
    assert compact["omitted_field_paths"] == [
        "case.model.retrieved_prior_lessons[0].lesson",
        "prior_case_lessons[0].lesson",
    ]
    assert compact["readback"]["arguments"] == {
        "operation": "show",
        "case_id": "target-case",
        "response_detail": "full",
    }
    assert result == before


def test_compact_projection_without_a_lesson_body_has_no_readback_requirement():
    result = {
        "case": {
            "case_id": "metadata-only-case",
            "model": {
                "retrieved_prior_lessons": [
                    {
                        "case_id": "prior-case",
                        "decision": "none",
                        "scope": "bounded scope",
                        "evidence_case_ids": [],
                        "calibration": "hypothesis",
                        "relevance": 0.0,
                        "provenance_note": "metadata-only",
                    }
                ]
            },
        },
        "prior_case_lessons": [],
    }
    before = json.loads(json.dumps(result))

    compact = project_response(result, response_detail="compact")

    assert compact["response_detail"] == "compact"
    assert compact["omitted_field_paths"] == []
    assert "readback" not in compact
    assert compact["case"]["model"]["retrieved_prior_lessons"] == result["case"]["model"]["retrieved_prior_lessons"]
    assert compact["prior_case_lessons"] == []
    assert result == before


def test_full_projection_returns_the_complete_result_unchanged():
    result = {"event": {"prior_case_lessons": [{"lesson": "complete"}]}}

    full = project_response(result, response_detail="full")

    assert full is result


def test_registers_bundled_skill_through_public_context_api(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    plugin.reset_for_tests()

    class Context:
        def __init__(self):
            self.tools = {}
            self.hooks = {}
            self.middleware = []
            self.skills = {}

        def get_config(self, key, default=None):
            return default

        def register_tool(self, **kwargs):
            self.tools[kwargs["name"]] = kwargs

        def register_hook(self, name, callback, **kwargs):
            self.hooks[name] = callback

        def register_middleware(self, kind, callback, **kwargs):
            self.middleware.append((kind, callback))

        def register_skill(self, name, path, description="", frontmatter=None):
            self.skills[name] = {
                "path": Path(path),
                "description": description,
                "frontmatter": dict(frontmatter or {}),
            }

    context = Context()
    plugin.register(context)

    assert set(context.skills) == {"epistemic-inquiry"}
    skill = context.skills["epistemic-inquiry"]
    assert skill["path"].name == "SKILL.md"
    assert skill["path"].is_file()
    body = skill["path"].read_text(encoding="utf-8")
    assert "response_detail" in body
    assert "compact" in body
    assert "addresses_event_ids" in body
    assert "unresolved" in body.lower()
    assert str(Path.home() / ".hermes" / "skills") not in str(skill["path"])


def test_response_detail_survives_native_schema_sanitizer_when_available():
    try:
        from tools.schema_sanitizer import sanitize_tool_schemas
    except ImportError:
        return

    cleaned = sanitize_tool_schemas(
        [
            {"type": "function", "function": MODEL_SCHEMA},
            {"type": "function", "function": PROBE_SCHEMA},
        ]
    )
    for tool in cleaned:
        parameters = tool["function"]["parameters"]
        assert "operation" in parameters["required"]
        assert "response_detail" in parameters["properties"]
        assert parameters["properties"]["response_detail"]["enum"] == ["full", "compact"]


def test_manifest_declares_public_release_metadata():
    root = Path(__file__).parents[1]
    manifest = yaml.safe_load((root / "plugin.yaml").read_text(encoding="utf-8"))
    assert manifest["version"] == "1.2.0"
    assert manifest["license"] == "MIT"
    assert manifest["platforms"] == ["linux", "macos"]
    assert "AI-assisted" in manifest["author"]
