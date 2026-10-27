# AI-assisted contribution; maintained by Epistemic Harness contributors.
"""Tests for the optional Jev method advice and its case contract."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from epistemic_harness import jev, methods as method_events, plugin
from epistemic_harness.harness import EpistemicHarness, HarnessError

CANDIDATES = [
    {"method_id": "m1", "text": "Read one more primary source"},
    {"method_id": "m2", "text": "Run a small discriminating experiment"},
    {"method_id": "m3", "text": "Ask a second model for an independent reading"},
]


def _open_args(case_id: str = "methods-case") -> dict:
    return {
        "operation": "open",
        "case_id": case_id,
        "decision": "Choose an investigation method",
        "stopping_condition": "One method produces decisive evidence or the question remains unresolved",
        "state_grounding": [
            {"id": "S1", "text": "The current reading is provisional", "evidence_event_ids": []}
        ],
        "mechanisms": [
            {"id": "M1", "text": "Mechanism one", "evidence_event_ids": []}
        ],
        "alternatives": [
            {"id": "A1", "text": "Alternative one", "evidence_event_ids": []}
        ],
        "top_unknown": "Which method discriminates",
        "current_plan": [{"id": "P1", "action": "Initial read", "depends_on": ["M1"]}],
    }


def _reply(choice: str = "m2", ids: list[str] | None = None) -> dict:
    offered = ids or [item["method_id"] for item in CANDIDATES]
    return {
        "model": "jev-test",
        "answers": {
            "method": {
                "type": "choice",
                "choice": choice,
                "probabilities": {
                    method_id: (1.0 if method_id == choice else 0.0)
                    for method_id in offered
                },
                "confidence": 1.0,
            },
            "more_investigation": {"type": "noul", "noul": 0.7},
        },
        "usage": {"input_tokens": 120, "output_tokens": 12},
    }


def _config(**overrides) -> jev.JevConfig:
    values = {
        "enabled": True,
        "max_calls_per_case": 2,
        "max_input_tokens_per_case": 5_000,
        "max_state_chars": 4_000,
    }
    values.update(overrides)
    return jev.JevConfig(**values)


@pytest.fixture()
def enabled_key(monkeypatch):
    monkeypatch.setattr(jev, "api_key", lambda: "test-key")


def _offer(harness: EpistemicHarness, session: str = "s", config=None, transport=None):
    return harness.methods(
        {"operation": "offer", "candidates": CANDIDATES},
        session_id=session,
        jev_config=config or _config(),
        transport=transport or (lambda request: _reply("m2")),
    )


def test_config_disabled_by_default():
    config = jev.load_config({})
    assert config.enabled is False
    assert config.model == "jev-latest"
    assert config.endpoint.startswith("https://")


def test_config_rejects_non_https_endpoint():
    with pytest.raises(jev.JevConfigError):
        jev.load_config({"enabled": True, "endpoint": "http://example.com/v1"})


def test_recent_events_cap_sets_truncated():
    case = {
        "case_id": "case",
        "session_id": "session",
        "model": {
            "top_unknown": "Which method discriminates",
            "decision": "Choose",
            "stopping_condition": "Decisive evidence",
            "state_grounding": [],
            "mechanisms": [],
            "alternatives": [],
            "current_plan": [],
        },
    }
    events = [
        {"type": f"event_{index}", "payload": {}}
        for index in range(5)
    ]
    state, privacy = jev.build_state(
        case,
        events,
        _config(),
        round_number=1,
        previous_method=None,
    )
    assert len(state["recent_events"]) == 3
    assert privacy["truncated"] is True


def test_projection_excludes_case_id_session_and_evidence_refs():
    case = {
        "case_id": "SECRET-CASE",
        "session_id": "SECRET-SESSION",
        "model": {
            "top_unknown": "Which method discriminates",
            "decision": "Choose",
            "stopping_condition": "Decisive evidence",
            "state_grounding": [
                {"id": "S1", "text": "A claim", "source_ref": "doi:SECRET"}
            ],
            "mechanisms": [],
            "alternatives": [],
            "current_plan": [{"id": "P1", "action": "Read"}],
        },
        "pending_probe": {"tool_args": {"password": "SECRET"}},
    }
    state, privacy = jev.build_state(
        case,
        [{"type": "case_opened", "payload": {"summary": "opened"}}],
        _config(),
        round_number=1,
        previous_method=None,
    )
    blob = json.dumps(state)
    assert "SECRET-CASE" not in blob
    assert "SECRET-SESSION" not in blob
    assert "source_ref" not in blob
    assert "pending_probe" not in blob
    assert privacy["truncated"] is False


def test_projection_omits_user_domain_claims_unless_allowed():
    case = {
        "model": {
            "top_unknown": "Which method",
            "decision": "Choose",
            "stopping_condition": "Evidence",
            "state_grounding": [
                {
                    "id": "S1",
                    "text": "QP said the moon is cheese",
                    "authority_domain": "user_testimony",
                }
            ],
            "mechanisms": [],
            "alternatives": [],
            "current_plan": [],
        }
    }
    closed, closed_privacy = jev.build_state(
        case, [], _config(), round_number=1, previous_method=None
    )
    assert "moon is cheese" not in json.dumps(closed)
    assert closed_privacy["user_domain_omitted"] == 1
    assert closed_privacy["truncated"] is False
    opened, _ = jev.build_state(
        case,
        [],
        _config(allow_user_domain_state=True),
        round_number=1,
        previous_method=None,
    )
    assert "moon is cheese" in json.dumps(opened)


def test_projection_truncates_deterministically():
    case = {
        "model": {
            "top_unknown": "x" * 2_000,
            "decision": "y" * 2_000,
            "stopping_condition": "z" * 2_000,
            "state_grounding": [],
            "mechanisms": [],
            "alternatives": [],
            "current_plan": [],
        }
    }
    first, privacy = jev.build_state(
        case, [], _config(max_state_chars=1_000), round_number=1, previous_method=None
    )
    second, _ = jev.build_state(
        case, [], _config(max_state_chars=1_000), round_number=1, previous_method=None
    )
    assert first == second
    assert privacy["truncated"] is True
    assert len(first["case"]["top_unknown"]) == 600


def test_candidate_validation_rejects_duplicates_and_long_text():
    with pytest.raises(jev.JevConfigError):
        jev.validate_candidates(
            [
                {"method_id": "m1", "text": "a"},
                {"method_id": "m1", "text": "b"},
            ],
            _config(),
        )
    with pytest.raises(jev.JevConfigError):
        jev.validate_candidates(
            [
                {"method_id": "m1", "text": "a"},
                {"method_id": "m2", "text": "b" * 500},
            ],
            _config(),
        )


def test_candidate_routes_must_agree():
    direct = [
        {"method_id": "m1", "text": "a"},
        {"method_id": "m2", "text": "b"},
    ]
    different = [
        {"method_id": "m1", "text": "a"},
        {"method_id": "m2", "text": "different"},
    ]
    with pytest.raises(jev.JevConfigError, match="same list"):
        jev.validate_candidates(
            direct,
            _config(),
            candidates_json=json.dumps(different),
        )
    assert jev.validate_candidates(
        direct, _config(), candidates_json=json.dumps(direct)
    ) == direct


def test_parse_offer_rejects_unknown_choice_and_nonfinite():
    with pytest.raises(jev.JevMalformed):
        jev.parse_offer(_reply("m9"), offered_ids=["m1", "m2"], config=_config())
    bad = _reply("m2")
    bad["answers"]["method"]["probabilities"]["m1"] = float("nan")
    with pytest.raises(jev.JevMalformed):
        jev.parse_offer(bad, offered_ids=["m1", "m2"], config=_config())


def test_offer_requires_active_case(tmp_path, enabled_key):
    harness = EpistemicHarness(tmp_path)
    with pytest.raises(HarnessError, match="no active epistemic case"):
        _offer(harness)


def test_offer_refuses_pending_probe(tmp_path, enabled_key):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="s")
    harness.probe(
        {
            "operation": "commit",
            "purpose": "learn",
            "unknown_or_goal": "Which method",
            "tool_name": "read_file",
            "tool_args": {"path": "x"},
            "predicted_outcomes": [{"outcome": "read"}],
            "why_this_action": "the designated read",
            "would_change_belief": "the file contradicts",
            "authority_domain": "external_source",
        },
        session_id="s",
    )
    result = _offer(harness)
    assert result == {"available": False, "reason": "probe_pending"}


def test_offer_records_started_then_offered(tmp_path, enabled_key):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="s")
    result = _offer(harness)
    assert result["available"] is True
    assert result["recommended_method_id"] == "m2"
    assert result["continue_probability"] == 0.7
    assert result["usage"] == {"input_tokens": 120, "output_tokens": 12}
    events = harness.store.list_events("methods-case")
    assert [event["type"] for event in events][-2:] == [
        "method_offer_started",
        "method_offered",
    ]
    case = harness.store.get_case("methods-case")
    assert case["version"] == 1
    assert harness.replay("methods-case")["usable"] is True


def test_offer_budget_exhausted(tmp_path, enabled_key):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="s")
    config = _config(max_calls_per_case=1)
    assert _offer(harness, config=config)["available"] is True
    harness.methods(
        {
            "operation": "choose",
            "disposition": "continue_different",
            "method_id": "m2",
        },
        session_id="s",
    )
    second = _offer(harness, config=config)
    assert second["available"] is False
    assert second["reason"] == "budget_exhausted"
    assert second["budget"]["attempts"] == 1


def test_offer_transport_exception_is_fail_open(tmp_path, enabled_key):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="s")

    def boom(_request):
        raise RuntimeError("provider exploded")

    result = _offer(harness, transport=boom)
    assert result["available"] is False
    assert result["reason"] == "transport_error"
    types = [event["type"] for event in harness.store.list_events("methods-case")]
    assert "method_offer_failed" in types


def test_case_id_mismatch_rejected(tmp_path, enabled_key):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="s")
    with pytest.raises(HarnessError, match="does not match"):
        harness.methods(
            {
                "operation": "offer",
                "case_id": "another-case",
                "candidates": CANDIDATES,
            },
            session_id="s",
            jev_config=_config(),
            transport=lambda _request: _reply("m2"),
        )


def test_offer_state_changed_during_call(tmp_path, enabled_key):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="s")

    def transport(_request):
        harness.model(
            {
                "operation": "record_evidence",
                "evidence": [
                    {
                        "summary": "A new observation landed during the call",
                        "source_ref": "local:x",
                        "source_role": "primary",
                        "access_scope": "full",
                        "authority_domain": "external_source",
                    }
                ],
            },
            session_id="s",
        )
        return _reply("m2")

    result = _offer(harness, transport=transport)
    assert result["available"] is False
    assert result["reason"] == "state_changed_during_call"
    types = [event["type"] for event in harness.store.list_events("methods-case")]
    assert "method_offer_failed" in types


def test_choose_continue_same_and_close_stop(tmp_path, enabled_key):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="s")
    config = _config(max_calls_per_case=3)
    first = _offer(harness, config=config)
    chosen = harness.methods(
        {
            "operation": "choose",
            "offer_id": first["offer_id"],
            "disposition": "continue_different",
            "method_id": "m1",
        },
        session_id="s",
    )
    assert chosen["recorded"] is True
    assert chosen["method_id"] == "m1"
    assert chosen["followed_recommendation"] is False

    second = _offer(harness, config=config)
    same = harness.methods(
        {"operation": "choose", "disposition": "continue_same"},
        session_id="s",
    )
    assert same["recorded"] is True
    assert same["method_id"] == "m1"

    third = harness.methods(
        {
            "operation": "offer",
            "decision_point": "close",
            "candidates": CANDIDATES,
        },
        session_id="s",
        jev_config=config,
        transport=lambda request: _reply("m2"),
    )
    stopped = harness.methods(
        {"operation": "choose", "disposition": "stop", "note": "enough"},
        session_id="s",
    )
    assert stopped["recorded"] is True
    assert stopped["method_id"] is None
    case = harness.store.get_case("methods-case")
    assert case["status"] == "active"
    assert case["version"] == 1
    assert third["available"] is True


def test_choose_rejects_stale_offer_after_evidence(tmp_path, enabled_key):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="s")
    first = _offer(harness)
    harness.model(
        {
            "operation": "record_evidence",
            "evidence": [
                {
                    "summary": "New evidence",
                    "source_ref": "local:x",
                    "source_role": "primary",
                    "access_scope": "full",
                    "authority_domain": "external_source",
                }
            ],
        },
        session_id="s",
    )
    result = harness.methods(
        {
            "operation": "choose",
            "offer_id": first["offer_id"],
            "disposition": "continue_different",
            "method_id": "m1",
        },
        session_id="s",
    )
    assert result["recorded"] is False
    assert result["reason"] == "stale_offer"


def test_choose_rejects_expired_offer(tmp_path, enabled_key, monkeypatch):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="s")
    _offer(harness)
    monkeypatch.setattr(method_events, "parse_time", lambda _value: 0.0)
    result = harness.methods(
        {
            "operation": "choose",
            "disposition": "continue_different",
            "method_id": "m1",
        },
        session_id="s",
    )
    assert result["recorded"] is False
    assert result["reason"] == "expired_offer"


def test_choose_rejects_unknown_method_and_same_as_previous(tmp_path, enabled_key):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="s")
    _offer(harness)
    unknown = harness.methods(
        {
            "operation": "choose",
            "disposition": "continue_different",
            "method_id": "m9",
        },
        session_id="s",
    )
    assert unknown["recorded"] is False
    assert unknown["reason"] == "unknown_method"


def test_status_never_calls_network(tmp_path, enabled_key, monkeypatch):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="s")

    def explode(_request, **_kwargs):
        raise AssertionError("status must not call the network")

    monkeypatch.setattr(jev, "call", explode)
    result = harness.methods({"operation": "status"}, session_id="s", jev_config=_config())
    assert result["enabled"] is True
    assert result["budget"]["attempts"] == 0


def test_duplicate_wrapper_same_dispatch_captures_real_result(tmp_path):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="s")
    harness.probe(
        {
            "operation": "commit",
            "purpose": "learn",
            "unknown_or_goal": "Whether the action returns",
            "tool_name": "terminal",
            "tool_args": {"command": "one"},
            "predicted_outcomes": [{"outcome": "returns"}],
            "why_this_action": "the designated action",
            "would_change_belief": "the action returns nothing",
            "authority_domain": "system_state",
        },
        session_id="s",
    )

    def inner(_args):
        return harness.tool_execution_middleware(
            tool_name="terminal",
            args={"command": "one"},
            next_call=lambda _args: "REAL-RESULT",
            session_id="s",
            tool_call_id="call-1",
        )

    result = harness.tool_execution_middleware(
        tool_name="terminal",
        args={"command": "one"},
        next_call=inner,
        session_id="s",
        tool_call_id="call-1",
    )
    assert result == "REAL-RESULT"
    events = harness.store.list_events("methods-case")
    observed = [event for event in events if event["type"] == "probe_observed"]
    assert len(observed) == 1
    assert "EPISTEMIC GATEWAY BLOCKED" not in json.dumps(observed)
    assert observed[0]["payload"]["result_status"] == "ok"
    assert harness.store.get_case("methods-case")["pending_probe"]["status"] == "observed"


def test_handler_disabled_returns_structured_result(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    plugin.reset_for_tests()
    plugin._set_jev_config(jev.JevConfig(enabled=False))
    result = json.loads(
        plugin.epistemic_methods_handler({"operation": "status"}, session_id="s")
    )
    assert result == {"available": False, "reason": "disabled"}


def test_jev_available_tracks_enabled_config(monkeypatch):
    plugin.reset_for_tests()
    plugin._set_jev_config(jev.JevConfig(enabled=False))
    assert plugin._jev_available() is False
    plugin._set_jev_config(jev.JevConfig(enabled=True))
    assert plugin._jev_available() is True
    plugin.reset_for_tests()


def test_jev_config_is_profile_scoped(monkeypatch, tmp_path):
    current = {"root": tmp_path / "profile-a"}
    monkeypatch.setattr(plugin, "_state_root", lambda: current["root"])
    plugin.reset_for_tests()
    plugin._set_jev_config(jev.JevConfig(enabled=True))
    current["root"] = tmp_path / "profile-b"
    plugin._set_jev_config(jev.JevConfig(enabled=False))
    config_b = plugin._jev_config()
    assert config_b is not None and config_b.enabled is False
    current["root"] = tmp_path / "profile-a"
    config_a = plugin._jev_config()
    assert config_a is not None and config_a.enabled is True
    plugin.reset_for_tests()


def test_handler_rejects_invalid_operation_before_store_access(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    plugin.reset_for_tests()
    result = json.loads(
        plugin.epistemic_methods_handler({"operation": "bogus"}, session_id="")
    )
    assert "operation" in result["error"]


def test_schema_declares_nested_candidate_properties():
    from epistemic_harness.schemas import METHODS_SCHEMA

    properties = METHODS_SCHEMA["parameters"]["properties"]
    assert properties["operation"]["enum"] == ["offer", "choose", "status"]
    assert set(properties["candidates"]["items"]["properties"]) == {
        "method_id",
        "text",
    }
    assert properties["disposition"]["enum"] == [
        "continue_same",
        "continue_different",
        "stop",
    ]


def test_offer_pending_blocks_second_offer(tmp_path, enabled_key):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="s")
    first = _offer(harness)
    second = _offer(harness)
    assert first["available"] is True
    assert second["available"] is False
    assert second["reason"] == "offer_pending"
    assert second["offer_id"] == first["offer_id"]


def test_plan_user_domain_is_filtered():
    case = {
        "model": {
            "top_unknown": "Which method",
            "decision": "Choose",
            "stopping_condition": "Evidence",
            "state_grounding": [],
            "mechanisms": [],
            "alternatives": [],
            "current_plan": [
                {
                    "id": "P1",
                    "action": "QP privately said the moon is cheese",
                    "authority_domain": "user_testimony",
                }
            ],
        }
    }
    state, privacy = jev.build_state(
        case, [], _config(), round_number=1, previous_method=None
    )
    assert "moon is cheese" not in json.dumps(state)
    assert privacy["user_domain_omitted"] == 1


def test_plan_authority_survives_open_and_is_filtered(tmp_path, enabled_key):
    harness = EpistemicHarness(tmp_path)
    args = _open_args("plan-authority-case")
    args["current_plan"] = [
        {
            "id": "P1",
            "action": "QP privately said the moon is cheese",
            "authority_domain": "user_testimony",
        }
    ]
    harness.model(args, session_id="s")
    captured: dict = {}

    def transport(request):
        captured["request"] = request
        return _reply("m2")

    result = _offer(harness, transport=transport)
    assert result["available"] is True
    assert "moon is cheese" not in json.dumps(captured["request"])
    assert result["privacy"]["user_domain_omitted"] == 1


def test_projection_scrubs_locators_and_event_ids():
    case = {
        "model": {
            "top_unknown": "Check E000012 at /Users/qp/secret.md and https://example.com/x doi:10.1/abc",
            "decision": "Read ~/private/notes.md",
            "stopping_condition": "Inspect archive/artifacts/deadbeef.txt",
            "state_grounding": [],
            "mechanisms": [],
            "alternatives": [],
            "current_plan": [],
        }
    }
    state, _ = jev.build_state(
        case, [], _config(), round_number=1, previous_method=None
    )
    blob = json.dumps(state)
    assert "E000012" not in blob
    assert "/Users" not in blob
    assert "~/private" not in blob
    assert "https://" not in blob
    assert "doi:" not in blob
    assert "archive/artifacts" not in blob


def test_model_label_is_bounded():
    reply = _reply("m2")
    reply["model"] = "M" * 200
    parsed = jev.parse_offer(
        reply, offered_ids=["m1", "m2"], config=_config()
    )
    assert parsed["model"] == "unknown"


def test_negative_usage_is_clamped():
    reply = _reply("m2")
    reply["usage"]["input_tokens"] = -10_000
    parsed = jev.parse_offer(
        reply, offered_ids=["m1", "m2"], config=_config()
    )
    assert parsed["usage"]["input_tokens"] == 0


def test_candidate_error_reason_is_specific():
    with pytest.raises(jev.JevCandidateError) as exc:
        jev.validate_candidates([{"method_id": "m1", "text": "a"}], _config())
    assert exc.value.reason == "bad_candidates"


def test_candidates_json_accepts_materialized_empty_array():
    assert (
        jev.validate_candidates([], _config(), candidates_json=json.dumps(CANDIDATES))
        == CANDIDATES
    )


def test_status_ignores_answered_offers(tmp_path, enabled_key):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="s")
    config = _config(max_calls_per_case=2)
    first = _offer(harness, config=config)
    harness.methods(
        {
            "operation": "choose",
            "offer_id": first["offer_id"],
            "disposition": "continue_different",
            "method_id": "m1",
        },
        session_id="s",
    )
    second = _offer(harness, config=config)
    harness.methods(
        {
            "operation": "choose",
            "offer_id": second["offer_id"],
            "disposition": "continue_different",
            "method_id": "m2",
        },
        session_id="s",
    )
    status = harness.methods({"operation": "status"}, session_id="s", jev_config=config)
    assert status["current_offer"] is None


def test_previous_method_survives_stop(tmp_path, enabled_key):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="s")
    config = _config(max_calls_per_case=3)
    first = _offer(harness, config=config)
    harness.methods(
        {
            "operation": "choose",
            "offer_id": first["offer_id"],
            "disposition": "continue_different",
            "method_id": "m1",
        },
        session_id="s",
    )
    second = harness.methods(
        {
            "operation": "offer",
            "decision_point": "close",
            "candidates": CANDIDATES,
        },
        session_id="s",
        jev_config=config,
        transport=lambda request: _reply("m2"),
    )
    harness.methods(
        {"operation": "choose", "offer_id": second["offer_id"], "disposition": "stop"},
        session_id="s",
    )
    third = _offer(harness, config=config)
    same = harness.methods(
        {"operation": "choose", "offer_id": third["offer_id"], "disposition": "continue_same"},
        session_id="s",
    )
    assert same["recorded"] is True
    assert same["method_id"] == "m1"


def test_probe_cannot_target_methods_tool(tmp_path):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="s")
    with pytest.raises(HarnessError, match="external action"):
        harness.probe(
            {
                "operation": "commit",
                "purpose": "learn",
                "unknown_or_goal": "Whether the advice tool is an action",
                "tool_name": "epistemic_methods",
                "tool_args": {"operation": "status"},
                "predicted_outcomes": [{"outcome": "advice"}],
                "why_this_action": "test the exclusion",
                "would_change_belief": "it would be an action",
                "authority_domain": "system_state",
            },
            session_id="s",
        )


def test_cross_thread_duplicate_wrapper_runs_once(tmp_path):
    import threading

    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="s")
    harness.probe(
        {
            "operation": "commit",
            "purpose": "learn",
            "unknown_or_goal": "Whether the action returns",
            "tool_name": "terminal",
            "tool_args": {"command": "one"},
            "predicted_outcomes": [{"outcome": "returns"}],
            "why_this_action": "the designated action",
            "would_change_belief": "the action returns nothing",
            "authority_domain": "system_state",
        },
        session_id="s",
    )
    box: dict[str, object] = {}

    def inner(_args):
        def run():
            box["value"] = harness.tool_execution_middleware(
                tool_name="terminal",
                args={"command": "one"},
                next_call=lambda _args: "REAL-THREAD",
                session_id="s",
                tool_call_id="call-thread",
            )

        worker = threading.Thread(target=run)
        worker.start()
        worker.join(timeout=5)
        return box.get("value")

    result = harness.tool_execution_middleware(
        tool_name="terminal",
        args={"command": "one"},
        next_call=inner,
        session_id="s",
        tool_call_id="call-thread",
    )
    assert result == "REAL-THREAD"
    observed = [
        event
        for event in harness.store.list_events("methods-case")
        if event["type"] == "probe_observed"
    ]
    assert len(observed) == 1
    assert "EPISTEMIC GATEWAY BLOCKED" not in json.dumps(observed)


def test_nested_reentry_with_different_args_is_blocked(tmp_path):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="s")
    harness.probe(
        {
            "operation": "commit",
            "purpose": "learn",
            "unknown_or_goal": "Whether the action returns",
            "tool_name": "terminal",
            "tool_args": {"command": "one"},
            "predicted_outcomes": [{"outcome": "returns"}],
            "why_this_action": "the designated action",
            "would_change_belief": "the action returns nothing",
            "authority_domain": "system_state",
        },
        session_id="s",
    )
    executed: list[dict] = []

    def inner(args):
        return harness.tool_execution_middleware(
            tool_name="terminal",
            args={"command": "TWO"},
            next_call=lambda actual: executed.append(actual) or "TWO-RAN",
            session_id="s",
            tool_call_id="call-args",
        )

    result = harness.tool_execution_middleware(
        tool_name="terminal",
        args={"command": "one"},
        next_call=inner,
        session_id="s",
        tool_call_id="call-args",
    )
    assert "arguments do not match" in result
    assert executed == []


def test_total_timeout_bounds_drip_response(enabled_key):
    import time

    class Drip:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read1(self, _size):
            time.sleep(0.6)
            return b"x"

    def opener(_request, timeout=None):
        del timeout
        return Drip()

    started = time.monotonic()
    with pytest.raises(jev.JevTransportError):
        jev.call(
            {"model": "jev-latest", "state": {}, "questions": {}},
            config=jev.JevConfig(enabled=True, timeout_s=1.0),
            opener=opener,
        )
    assert time.monotonic() - started < 2.5


def test_timeout_reports_timeout_reason(enabled_key):
    import time

    class Drip:
        def __enter__(self):
            return self

        def __exit__(self, *_args):
            return False

        def read1(self, _size):
            time.sleep(0.6)
            return b"x"

    def opener(_request, timeout=None):
        del timeout
        return Drip()

    with pytest.raises(jev.JevTransportError) as info:
        jev.call(
            {"model": "jev-latest", "state": {}, "questions": {}},
            config=jev.JevConfig(enabled=True, timeout_s=0.5),
            opener=opener,
        )
    assert info.value.reason == "timeout"


def test_worker_cap_returns_busy(enabled_key):
    import threading
    import time

    gate = threading.Event()
    jev.reset_for_tests()

    def opener(_request, timeout=None):
        del timeout
        gate.wait(5)
        raise TimeoutError("released")

    for _ in range(jev._MAX_INFLIGHT_WORKERS):
        with pytest.raises(jev.JevTransportError):
            jev.call(
                {"model": "jev-latest", "state": {}, "questions": {}},
                config=jev.JevConfig(enabled=True, timeout_s=0.2),
                opener=opener,
            )
    with pytest.raises(jev.JevTransportError) as info:
        jev.call(
            {"model": "jev-latest", "state": {}, "questions": {}},
            config=jev.JevConfig(enabled=True, timeout_s=0.2),
            opener=opener,
        )
    assert info.value.reason == "busy"
    gate.set()
    time.sleep(1.0)


def test_projection_scrubs_other_absolute_paths():
    case = {
        "model": {
            "top_unknown": "Check /Volumes/SEAGATE/cases/private.pdf and /etc/passwd",
            "decision": "Read /opt/data/x and \\\\server\\share\\x",
            "stopping_condition": "See HTTPS://EXAMPLE.COM/x and doi: 10.1234/abc",
            "state_grounding": [],
            "mechanisms": [],
            "alternatives": [],
            "current_plan": [],
        }
    }
    state, _ = jev.build_state(
        case, [], _config(), round_number=1, previous_method=None
    )
    blob = json.dumps(state)
    for marker in (
        "/Volumes",
        "/etc/passwd",
        "/opt/data",
        "\\\\server",
        "HTTPS://",
        "doi:",
        "10.1234/abc",
    ):
        assert marker not in blob


def test_total_timeout_bounds_header_block(enabled_key):
    import time

    def opener(_request, timeout=None):
        del timeout
        time.sleep(3)
        raise AssertionError("the abandoned worker should not control the result")

    started = time.monotonic()
    with pytest.raises(jev.JevTransportError):
        jev.call(
            {"model": "jev-latest", "state": {}, "questions": {}},
            config=jev.JevConfig(enabled=True, timeout_s=1.0),
            opener=opener,
        )
    assert time.monotonic() - started < 2.0


def test_projection_scrubs_quoted_and_embedded_paths():
    case = {
        "model": {
            "top_unknown": (
                'source="/Volumes/SEAGATE/cases/secret.pdf" '
                "file:///Volumes/SEAGATE/cases/other.pdf "
                "`C:/Users/luft/private.docx` "
                '{"path": "/etc/passwd"} read "/opt/private/data.csv" '
                "see ~/secret.pdf and /secret.pdf and //server/share and dir/file.csv"
            ),
            "decision": "Choose",
            "stopping_condition": "Evidence",
            "state_grounding": [],
            "mechanisms": [],
            "alternatives": [],
            "current_plan": [],
        }
    }
    state, _ = jev.build_state(
        case, [], _config(), round_number=1, previous_method=None
    )
    blob = json.dumps(state)
    for marker in (
        "/Volumes",
        "file://",
        "C:/Users",
        "/etc/passwd",
        "/opt/private",
        "~/secret.pdf",
        "/secret.pdf",
        "//server/share",
        "dir/file.csv",
    ):
        assert marker not in blob


def test_closed_case_still_records_terminal_failure(tmp_path, enabled_key):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="s")

    def transport(_request):
        harness.model(
            {
                "operation": "close",
                "outcome": "stopped",
                "summary": "closed while advice was in flight",
                "transfer": {"decision": "none"},
            },
            session_id="s",
        )
        return _reply("m2")

    result = _offer(harness, transport=transport)
    assert result["available"] is False
    assert result["reason"] == "state_changed_during_call"
    types = [event["type"] for event in harness.store.list_events("methods-case")]
    assert "method_offer_failed" in types
    assert harness.store.get_case("methods-case")["status"] == "closed"


def test_paused_case_still_records_terminal_failure(tmp_path, enabled_key):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="s")

    def transport(_request):
        harness.model(
            {"operation": "pause", "reason": "mid-call"}, session_id="s"
        )
        return _reply("m2")

    result = _offer(harness, transport=transport)
    assert result["available"] is False
    assert result["reason"] == "state_changed_during_call"
    types = [event["type"] for event in harness.store.list_events("methods-case")]
    assert "method_offer_failed" in types
    assert harness.store.get_case("methods-case")["status"] == "paused"


def test_plan_authority_tag_is_case_insensitive():
    case = {
        "model": {
            "top_unknown": "Which method",
            "decision": "Choose",
            "stopping_condition": "Evidence",
            "state_grounding": [],
            "mechanisms": [],
            "alternatives": [],
            "current_plan": [
                {
                    "id": "P1",
                    "action": "User said the moon is cheese",
                    "authority_domain": "User_Testimony",
                }
            ],
        }
    }
    state, privacy = jev.build_state(
        case, [], _config(), round_number=1, previous_method=None
    )
    assert "moon is cheese" not in json.dumps(state)
    assert privacy["user_domain_omitted"] == 1


def _probe_args() -> dict:
    return {
        "operation": "commit",
        "purpose": "learn",
        "unknown_or_goal": "Whether the synthetic read changes the claim",
        "tool_name": "read_file",
        "tool_args": {"path": "synthetic.txt"},
        "predicted_outcomes": [{"outcome": "the file is present"}],
        "why_this_action": "the designated synthetic read",
        "would_change_belief": "the file contradicts the claim",
        "authority_domain": "external_source",
    }


def _close_args() -> dict:
    return {
        "operation": "close",
        "outcome": "stopped",
        "summary": "Synthetic close",
        "transfer": {"decision": "none"},
    }


def test_probe_commit_requires_consultation_when_enabled(tmp_path, enabled_key):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="s")
    with pytest.raises(HarnessError, match="consultation is required before a probe"):
        harness.probe(_probe_args(), session_id="s", jev_config=_config())
    assert not any(
        event["type"] == "probe_committed"
        for event in harness.store.list_events("methods-case")
    )

    offered = _offer(harness)
    chosen = harness.methods(
        {
            "operation": "choose",
            "offer_id": offered["offer_id"],
            "disposition": "continue_different",
            "method_id": "m2",
        },
        session_id="s",
    )
    assert chosen["recorded"] is True
    committed = harness.probe(_probe_args(), session_id="s", jev_config=_config())
    assert committed["probe"]["status"] == "committed"


def test_close_requires_consultation_when_enabled(tmp_path, enabled_key):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="s")
    with pytest.raises(HarnessError, match="consultation is required before a close"):
        harness.model(_close_args(), session_id="s", jev_config=_config())
    assert harness.store.get_case("methods-case")["status"] == "active"

    offered = harness.methods(
        {
            "operation": "offer",
            "decision_point": "close",
            "candidates": CANDIDATES,
        },
        session_id="s",
        jev_config=_config(),
        transport=lambda _request: _reply("m2"),
    )
    assert offered["available"] is True
    harness.methods(
        {"operation": "choose", "offer_id": offered["offer_id"], "disposition": "stop"},
        session_id="s",
    )
    closed = harness.model(_close_args(), session_id="s", jev_config=_config())
    assert closed["case"]["status"] == "closed"


def test_probe_waives_after_budget_exhausted(tmp_path, enabled_key):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="s")
    committed = harness.probe(
        _probe_args(),
        session_id="s",
        jev_config=_config(max_calls_per_case=0),
    )
    assert committed["probe"]["status"] == "committed"
    waived = [
        event
        for event in harness.store.list_events("methods-case")
        if event["type"] == "method_consultation_excused"
    ]
    assert waived and waived[-1]["payload"]["reason"] == "budget_exhausted"


def test_probe_waives_after_failure_budget_exhausted(tmp_path, enabled_key):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="s")
    config = _config(max_calls_per_case=4, cooldown_s=0)

    def boom(_request):
        raise jev.JevTransportError("timeout", reason="timeout")

    for _ in range(4):
        assert _offer(harness, config=config, transport=boom)["reason"] == "timeout"
    harness.model(
        {
            "operation": "record_evidence",
            "evidence": [
                {
                    "summary": "Move the Timeline past the final failure",
                    "source_ref": "synthetic:budget",
                    "source_role": "primary",
                    "access_scope": "full",
                    "authority_domain": "external_source",
                }
            ],
        },
        session_id="s",
    )
    committed = harness.probe(_probe_args(), session_id="s", jev_config=config)
    assert committed["probe"]["status"] == "committed"
    waived = [
        event
        for event in harness.store.list_events("methods-case")
        if event["type"] == "method_consultation_excused"
    ]
    assert waived and waived[-1]["payload"]["reason"] == "budget_exhausted"


def test_token_limit_blocks_instead_of_waiving(tmp_path, enabled_key):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="s")
    config = _config(max_calls_per_case=4, max_input_tokens_per_case=1)
    assert _offer(harness, config=config)["available"] is True
    harness.model(
        {
            "operation": "record_evidence",
            "evidence": [
                {
                    "summary": "Move the Timeline past the successful offer",
                    "source_ref": "synthetic:token",
                    "source_role": "primary",
                    "access_scope": "full",
                    "authority_domain": "external_source",
                }
            ],
        },
        session_id="s",
    )
    with pytest.raises(HarnessError, match="input-token budget is exhausted"):
        harness.probe(_probe_args(), session_id="s", jev_config=config)


def test_probe_waives_after_transport_failure(tmp_path, enabled_key):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="s")

    def boom(_request):
        raise jev.JevTransportError("timeout", reason="timeout")

    failed = _offer(harness, transport=boom)
    assert failed["reason"] == "timeout"
    committed = harness.probe(_probe_args(), session_id="s", jev_config=_config())
    assert committed["probe"]["status"] == "committed"


def test_bad_candidates_do_not_waive_consultation(tmp_path, enabled_key):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="s")
    bad = harness.methods(
        {
            "operation": "offer",
            "candidates": [{"method_id": "m1", "text": "Only one candidate"}],
        },
        session_id="s",
        jev_config=_config(),
    )
    assert bad["reason"] == "bad_candidates"
    with pytest.raises(HarnessError, match="consultation is required before a probe"):
        harness.probe(_probe_args(), session_id="s", jev_config=_config())


def test_plan_authority_invalid_value_is_rejected(tmp_path):
    harness = EpistemicHarness(tmp_path)
    args = _open_args()
    args["current_plan"][0]["authority_domain"] = "user-testimony"
    with pytest.raises(HarnessError, match="invalid authority_domain"):
        harness.model(args, session_id="s")


def test_revise_preserves_an_existing_authority_tag(tmp_path, enabled_key):
    harness = EpistemicHarness(tmp_path)
    args = _open_args()
    args["current_plan"][0]["authority_domain"] = "user_testimony"
    harness.model(args, session_id="s")
    harness.model(
        {
            "operation": "revise",
            "reason": "clarify the plan wording",
            "updates": {
                "current_plan": [
                    {
                        "id": "P1",
                        "action": "Initial read, clarified",
                        "depends_on": ["M1"],
                    }
                ]
            },
        },
        session_id="s",
    )
    captured: dict = {}

    def transport(request):
        captured["request"] = request
        return _reply("m2")

    result = _offer(harness, transport=transport)
    assert result["privacy"]["user_domain_omitted"] == 1
    assert "Initial read, clarified" not in json.dumps(captured["request"])


def test_followed_recommendation_is_false_for_close_stop(tmp_path, enabled_key):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="s")
    offered = harness.methods(
        {
            "operation": "offer",
            "decision_point": "close",
            "candidates": CANDIDATES,
        },
        session_id="s",
        jev_config=_config(),
        transport=lambda request: _reply("m2"),
    )
    chosen = harness.methods(
        {"operation": "choose", "offer_id": offered["offer_id"], "disposition": "stop"},
        session_id="s",
    )
    assert chosen["followed_recommendation"] is False


def test_choose_rejects_a_mismatched_decision_point(tmp_path, enabled_key):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="s")
    offered = _offer(harness)
    chosen = harness.methods(
        {
            "operation": "choose",
            "offer_id": offered["offer_id"],
            "decision_point": "close",
            "disposition": "stop",
        },
        session_id="s",
    )
    assert chosen["recorded"] is False
    assert chosen["reason"] == "decision_point_mismatch"


def test_probe_choose_stop_is_rejected(tmp_path, enabled_key):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="s")
    offered = _offer(harness)
    chosen = harness.methods(
        {"operation": "choose", "offer_id": offered["offer_id"], "disposition": "stop"},
        session_id="s",
    )
    assert chosen["recorded"] is False
    assert chosen["reason"] == "bad_disposition"
    with pytest.raises(HarnessError, match="a Jev offer is waiting"):
        harness.probe(_probe_args(), session_id="s", jev_config=_config())


def test_close_choose_continue_is_rejected(tmp_path, enabled_key):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="s")
    offered = harness.methods(
        {
            "operation": "offer",
            "decision_point": "close",
            "candidates": CANDIDATES,
        },
        session_id="s",
        jev_config=_config(),
        transport=lambda request: _reply("m2"),
    )
    chosen = harness.methods(
        {
            "operation": "choose",
            "offer_id": offered["offer_id"],
            "disposition": "continue_different",
            "method_id": "m2",
        },
        session_id="s",
    )
    assert chosen["recorded"] is False
    assert chosen["reason"] == "bad_disposition"


def test_config_disabled_ignores_invalid_endpoint():
    config = jev.load_config({"enabled": False, "endpoint": "http://insecure"})
    assert config.enabled is False


def test_config_rejects_non_mapping_and_unknown_keys():
    with pytest.raises(jev.JevConfigError):
        jev.load_config(True)
    with pytest.raises(jev.JevConfigError):
        jev.load_config({"enable": True})


def test_config_rejects_unknown_enabled_value():
    with pytest.raises(jev.JevConfigError):
        jev.load_config({"enabled": "ture"})


def test_misconfigured_gate_blocks_probe(tmp_path):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="s")
    with pytest.raises(HarnessError, match="configuration is invalid"):
        harness.probe(
            _probe_args(),
            session_id="s",
            jev_config=jev.JevConfig(enabled=True, misconfigured=True),
        )


def test_auth_error_does_not_waive_consultation(tmp_path, enabled_key):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="s")

    def unauthorized(_request):
        raise jev.JevTransportError("http_error:401", reason="auth_error")

    failed = _offer(harness, transport=unauthorized)
    assert failed["reason"] == "auth_error"
    with pytest.raises(HarnessError, match="consultation is required before a probe"):
        harness.probe(_probe_args(), session_id="s", jev_config=_config())


def test_auth_errors_do_not_trigger_a_cooldown_waiver(tmp_path, enabled_key):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="s")

    def unauthorized(_request):
        raise jev.JevTransportError("http_error:401", reason="auth_error")

    config = _config(max_calls_per_case=4)
    for _ in range(3):
        failed = _offer(harness, config=config, transport=unauthorized)
        assert failed["reason"] == "auth_error"
    with pytest.raises(HarnessError, match="consultation is required before a probe"):
        harness.probe(_probe_args(), session_id="s", jev_config=config)
    assert not any(
        event["type"] == "method_consultation_excused"
        for event in harness.store.list_events("methods-case")
    )


def test_four_auth_errors_do_not_exhaust_the_budget(tmp_path, enabled_key):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="s")
    config = _config(max_calls_per_case=4)

    def unauthorized(_request):
        raise jev.JevTransportError("http_error:401", reason="auth_error")

    for _ in range(12):
        failed = _offer(harness, config=config, transport=unauthorized)
        assert failed["reason"] == "auth_error"
    status = harness.methods({"operation": "status"}, session_id="s", jev_config=config)
    assert status["budget"]["attempts"] == 0
    with pytest.raises(HarnessError, match="consultation is required before a probe"):
        harness.probe(_probe_args(), session_id="s", jev_config=config)
    with pytest.raises(HarnessError, match="consultation is required before a close"):
        harness.model({"operation": "close", **_close_args()}, session_id="s", jev_config=config)
    assert not any(
        event["type"] == "method_consultation_excused"
        for event in harness.store.list_events("methods-case")
    )


def test_inflight_starts_do_not_exhaust_budget(tmp_path, enabled_key):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="s")
    case = harness.store.get_case("methods-case")
    for index in range(4):
        case = harness.store.transition_case(
            case,
            event_type=method_events.OFFER_STARTED,
            event_payload={
                "offer_id": f"offer-inflight-{index}",
                "round": index + 1,
                "decision_point": "probe",
            },
        )["case"]
    with pytest.raises(HarnessError, match="consultation is required before a probe"):
        harness.probe(_probe_args(), session_id="s", jev_config=_config())
    with pytest.raises(HarnessError, match="consultation is required before a close"):
        harness.model(_close_args(), session_id="s", jev_config=_config())


def test_busy_does_not_waive_consultation(tmp_path, enabled_key):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="s")

    def busy(_request):
        raise jev.JevTransportError("busy", reason="busy")

    for _ in range(3):
        assert _offer(harness, transport=busy)["reason"] == "busy"
    with pytest.raises(HarnessError, match="consultation is required before a probe"):
        harness.probe(_probe_args(), session_id="s", jev_config=_config())
    assert not any(
        event["type"] == "method_consultation_excused"
        for event in harness.store.list_events("methods-case")
    )


def test_unanswered_successful_offers_do_not_waive_budget(tmp_path, enabled_key):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="s")
    config = _config(max_calls_per_case=4)
    for index in range(4):
        assert _offer(harness, config=config)["available"] is True
        harness.model(
            {
                "operation": "record_evidence",
                "evidence": [
                    {
                        "summary": f"Invalidating observation {index}",
                        "source_ref": f"synthetic:{index}",
                        "source_role": "primary",
                        "access_scope": "full",
                        "authority_domain": "external_source",
                    }
                ],
            },
            session_id="s",
        )
    with pytest.raises(HarnessError, match="consultation is required before a probe"):
        harness.probe(_probe_args(), session_id="s", jev_config=config)
    assert not any(
        event["type"] == "method_consultation_excused"
        for event in harness.store.list_events("methods-case")
    )


def test_missing_key_blocks_probe(tmp_path, monkeypatch):
    monkeypatch.setattr(jev, "api_key", lambda: "")
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="s")
    with pytest.raises(HarnessError, match="no key is available"):
        harness.probe(_probe_args(), session_id="s", jev_config=_config())


def test_invalid_probe_does_not_write_a_waiver(tmp_path, enabled_key):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="s")
    bad = _probe_args()
    bad.pop("tool_name")
    with pytest.raises(HarnessError, match="missing probe fields"):
        harness.probe(
            bad,
            session_id="s",
            jev_config=_config(max_calls_per_case=0),
        )
    assert not any(
        event["type"] == "method_consultation_excused"
        for event in harness.store.list_events("methods-case")
    )


def test_invalid_close_does_not_write_a_waiver(tmp_path, enabled_key):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="s")
    bad = _close_args()
    bad["outcome"] = "bogus"
    with pytest.raises(HarnessError, match="outcome must be"):
        harness.model(
            bad,
            session_id="s",
            jev_config=_config(max_calls_per_case=0),
        )
    assert not any(
        event["type"] == "method_consultation_excused"
        for event in harness.store.list_events("methods-case")
    )


def test_consultation_is_invalidated_by_a_later_case_event(tmp_path, enabled_key):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="s")
    offered = _offer(harness)
    harness.methods(
        {
            "operation": "choose",
            "offer_id": offered["offer_id"],
            "disposition": "continue_different",
            "method_id": "m2",
        },
        session_id="s",
    )
    harness.model(
        {
            "operation": "record_evidence",
            "evidence": [
                {
                    "summary": "A synthetic observation",
                    "source_ref": "synthetic:observation",
                    "source_role": "primary",
                    "access_scope": "full",
                    "authority_domain": "external_source",
                }
            ],
        },
        session_id="s",
    )
    with pytest.raises(HarnessError, match="a Jev offer is waiting|consultation is required"):
        harness.probe(_probe_args(), session_id="s", jev_config=_config())


def test_unanswered_offer_blocks_commit_even_at_budget(tmp_path, enabled_key):
    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="s")
    config = _config(max_calls_per_case=1)
    offered = _offer(harness, config=config)
    assert offered["available"] is True
    with pytest.raises(HarnessError, match="offer is waiting"):
        harness.probe(_probe_args(), session_id="s", jev_config=config)
    assert not any(
        event["type"] == "probe_committed"
        for event in harness.store.list_events("methods-case")
    )
    harness.methods(
        {
            "operation": "choose",
            "offer_id": offered["offer_id"],
            "disposition": "continue_different",
            "method_id": "m2",
        },
        session_id="s",
    )
    assert (
        harness.probe(_probe_args(), session_id="s", jev_config=config)["probe"]["status"]
        == "committed"
    )


def test_plugin_handlers_enforce_consultation(tmp_path, monkeypatch, enabled_key):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    plugin.reset_for_tests()
    plugin._set_jev_config(_config())
    harness = plugin._get_harness()
    harness.model(_open_args(), session_id="s")

    blocked = json.loads(
        plugin.epistemic_probe_handler(_probe_args(), session_id="s")
    )
    assert "consultation is required before a probe" in blocked["error"]

    offered = _offer(harness)
    harness.methods(
        {
            "operation": "choose",
            "offer_id": offered["offer_id"],
            "disposition": "continue_different",
            "method_id": "m2",
        },
        session_id="s",
    )
    committed = json.loads(
        plugin.epistemic_probe_handler(_probe_args(), session_id="s")
    )
    assert committed["probe"]["status"] == "committed"


def test_dispatch_registry_clears_on_exception(tmp_path):
    import epistemic_harness.harness as harness_module

    harness = EpistemicHarness(tmp_path)
    harness.model(_open_args(), session_id="s")
    harness.probe(
        {
            "operation": "commit",
            "purpose": "learn",
            "unknown_or_goal": "Whether the action returns",
            "tool_name": "terminal",
            "tool_args": {"command": "one"},
            "predicted_outcomes": [{"outcome": "returns"}],
            "why_this_action": "the designated action",
            "would_change_belief": "the action returns nothing",
            "authority_domain": "system_state",
        },
        session_id="s",
    )

    def boom(_args):
        raise RuntimeError("host failure")

    with pytest.raises(RuntimeError):
        harness.tool_execution_middleware(
            tool_name="terminal",
            args={"command": "one"},
            next_call=boom,
            session_id="s",
            tool_call_id="call-boom",
        )
    assert harness_module._DISPATCH_IN_FLIGHT == {}
