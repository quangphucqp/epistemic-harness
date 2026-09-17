# AI-assisted contribution; maintained by Epistemic Harness contributors.
"""Real Hermes runtime smoke for the standalone epistemic plugin."""

from __future__ import annotations

import json
import sys
from pathlib import Path

from hermes_cli.middleware import run_tool_execution_middleware
from hermes_cli.plugins import get_plugin_manager
from model_tools import get_tool_definitions, handle_function_call as native_dispatch
from tools.registry import registry

def handle_function_call(name, args, **kwargs):
    if name in {"epistemic_model", "epistemic_probe"}:
        return native_dispatch("tool_call", {"calls": [{"name": name, "arguments": args}]}, **kwargs)
    return native_dispatch(name, args, **kwargs)


def outer_dispatch(name, args, **kwargs):
    """Exercise the agent's outer execution boundary, not lower-level dispatch."""
    return run_tool_execution_middleware(
        name,
        args,
        lambda actual: native_dispatch(
            name,
            actual,
            skip_pre_tool_call_hook=True,
            skip_tool_request_middleware=True,
            skip_tool_execution_middleware=True,
            **kwargs,
        ),
        **kwargs,
    )

def _parsed(value: str) -> dict:
    result = json.loads(value)
    if isinstance(result, dict) and result.get("error"):
        raise AssertionError(result["error"])
    return result


def _tool_names() -> set[str]:
    return {
        str((definition.get("function") or {}).get("name") or "")
        for definition in get_tool_definitions(
            enabled_toolsets=["epistemic-harness"],
            quiet_mode=True,
        )
    }


def main() -> None:
    home = Path(sys.argv[1]).resolve()
    manager = get_plugin_manager()
    manager.discover_and_load()

    assert registry.get_entry("epistemic_model") is not None
    assert registry.get_entry("epistemic_probe") is not None
    skill_path = manager.find_plugin_skill("epistemic-harness:epistemic-inquiry")
    assert skill_path is not None and skill_path.is_file()
    assert "claim-bound commit" in skill_path.read_text(encoding="utf-8")
    catalog = json.dumps(get_tool_definitions(enabled_toolsets=["epistemic-harness"], quiet_mode=True))
    assert all(name in catalog for name in ("epistemic_model", "epistemic_probe"))
    described = native_dispatch("tool_describe", {"names": ["epistemic_model", "epistemic_probe"]}, session_id="runtime-smoke-session")
    assert "state_grounding" in described and "predicted_outcomes" in described
    assert manager.has_hook("pre_llm_call")
    assert manager.has_hook("post_tool_call")
    assert manager.has_hook("transform_tool_result")
    assert manager.has_hook("on_session_reset")
    assert manager.has_hook("on_session_finalize")
    assert manager.has_middleware("tool_execution")

    fixture = home / "runtime-fixture.txt"
    fixture.write_text("observed-runtime-value\n", encoding="utf-8")
    ordinary_fixture = home / "runtime-ordinary-fixture.txt"
    ordinary_fixture.write_text("ordinary-runtime-value\n", encoding="utf-8")
    session_id = "runtime-smoke-session"
    case_id = "runtime-smoke"

    malformed = handle_function_call(
        "epistemic_model",
        {},
        session_id=session_id,
        task_id="runtime-smoke",
    )
    malformed_text = malformed if isinstance(malformed, str) else json.dumps(malformed, sort_keys=True)
    assert "operation" in malformed_text and "NOT invoked" in malformed_text
    assert not (home / "epistemic-harness" / "cases").exists()

    opened = _parsed(handle_function_call(
        "epistemic_model",
        {
            "operation": "open",
            "case_id": case_id,
            "decision": "Verify the standalone plugin runtime",
            "stopping_condition": "The native tool path captures and compares one probe",
            "state_grounding": [
                {"id": "S1", "text": "The plugin loaded", "evidence_event_ids": []}
            ],
            "mechanisms": [
                {"id": "M1", "text": "Native middleware captures the probe", "evidence_event_ids": []}
            ],
            "alternatives": [{"id": "A1", "text": "The native path loses the probe"}],
            "top_unknown": "Whether native Hermes contracts are sufficient",
            "current_plan": [{"id": "P1", "action": "Commit and execute one read"}],
            "response_detail": "compact",
        },
        session_id=session_id,
        task_id="runtime-smoke",
    ))
    assert opened["case"]["status"] == "active"
    assert opened["response_detail"] == "compact"
    assert opened["omitted_field_paths"] == []
    assert "readback" not in opened
    full_open = _parsed(handle_function_call(
        "epistemic_model",
        {"operation": "show", "case_id": case_id, "response_detail": "full"},
        session_id=session_id,
        task_id="runtime-smoke",
    ))
    assert "response_detail" not in full_open
    assert manager.invoke_hook("pre_llm_call", session_id=session_id)

    ordinary = handle_function_call(
        "read_file", {"path": str(ordinary_fixture)},
        session_id=session_id, task_id="runtime-smoke",
    )
    assert "ordinary-runtime-value" in ordinary

    probe_args = {"path": str(fixture)}
    committed = _parsed(handle_function_call(
        "epistemic_probe",
        {
            "operation": "commit",
            "claim_id": "M1",
            "purpose": "learn",
            "unknown_or_goal": "Observe the real registry result",
            "tool_name": "read_file",
            "tool_args": probe_args,
            "predicted_outcomes": [
                {"outcome": "fixture content", "meaning": "Native middleware executed"}
            ],
            "why_this_action": "It crosses the actual Hermes registry",
            "would_change_belief": "The content is absent or the call fails",
            "authority_domain": "system_state",
        },
        session_id=session_id,
        task_id="runtime-smoke",
    ))
    assert committed["probe"]["status"] == "committed"

    observed = outer_dispatch(
        "read_file", probe_args,
        session_id=session_id, task_id="runtime-smoke",
    )
    assert "observed-runtime-value" in observed, observed
    native_events_path = home / "epistemic-harness" / "cases" / case_id / "events.jsonl"
    native_events = [json.loads(line) for line in native_events_path.read_text().splitlines()]
    native_observed = next(event for event in native_events if event["type"] == "probe_observed")
    assert native_observed["payload"]["observation_boundary"] == "post_tool_call"
    assert native_observed["payload"]["raw_result_provenance"] == "native_handler_result"

    compared = _parsed(handle_function_call(
        "epistemic_probe",
        {
            "operation": "compare",
            "disposition": "match",
            "material": False,
            "rationale": "The native call returned the committed fixture",
            "belief_change": "unchanged",
        },
        session_id=session_id,
        task_id="runtime-smoke",
    ))
    assert compared["replay"]["usable"] is True
    assert compared["event"]["payload"]["claim_id"] == "M1"
    assert compared["event"]["payload"]["belief_change"] == "unchanged"
    assert compared["event"]["payload"]["affected_claim_ids"] == ["M1"]
    evidence_id = compared["event"]["payload"]["observation_event_id"]

    revised = _parsed(handle_function_call(
        "epistemic_model",
        {
            "operation": "revise",
            "reason": "Bind the native observation to the mechanism",
            "mechanisms": [{
                "id": "M1",
                "text": "Native middleware captures the probe",
                "epistemic_status": "grounded",
                "authority_domain": "system_state",
                "evidence_event_ids": [evidence_id],
            }],
        },
        session_id=session_id,
        task_id="runtime-smoke",
    ))
    assert revised["case"]["model"]["mechanisms"][0]["epistemic_status"] == "grounded"

    closed = _parsed(handle_function_call(
        "epistemic_model",
        {
            "operation": "close",
            "outcome": "resolved",
            "summary": "The standalone plugin completed its native runtime path",
            "transfer": {"decision": "none"},
        },
        session_id=session_id,
        task_id="runtime-smoke",
    ))
    assert closed["case"]["status"] == "closed"
    assert manager.invoke_hook("pre_llm_call", session_id=session_id) == []

    direct_session = "runtime-direct-unknown-session"
    direct_case = "runtime-direct-unknown"
    _parsed(handle_function_call(
        "epistemic_model",
        {
            "operation": "open",
            "case_id": direct_case,
            "top_unknown": "Whether direct dispatch proves M1",
            "mechanisms": [{"id": "M1", "text": "Direct dispatch proves the mechanism"}],
        },
        session_id=direct_session,
        task_id=direct_case,
    ))
    _parsed(handle_function_call(
        "epistemic_probe",
        {
            "operation": "commit",
            "claim_id": "M1",
            "purpose": "learn",
            "unknown_or_goal": "Whether direct dispatch proves M1",
            "tool_name": "read_file",
            "tool_args": probe_args,
            "predicted_outcomes": [{"outcome": "fixture content"}],
            "why_this_action": "The fixture is the designated direct check",
            "would_change_belief": "The fixture is unavailable",
            "authority_domain": "system_state",
        },
        session_id=direct_session,
        task_id=direct_case,
    ))
    direct_returned = handle_function_call(
        "read_file",
        probe_args,
        session_id=direct_session,
        task_id=direct_case,
    )
    assert "observed-runtime-value" in direct_returned
    direct_events_path = home / "epistemic-harness" / "cases" / direct_case / "events.jsonl"
    direct_events = [json.loads(line) for line in direct_events_path.read_text().splitlines()]
    direct_observed = next(event for event in direct_events if event["type"] == "probe_observed")
    assert direct_observed["payload"]["observation_boundary"] == "tool_execution_middleware_return"
    assert direct_observed["payload"]["raw_result_provenance"] == "unknown"
    direct_compared = _parsed(handle_function_call(
        "epistemic_probe",
        {
            "operation": "compare",
            "disposition": "match",
            "material": False,
            "rationale": "Direct dispatch returned the fixture without a native raw boundary",
            "belief_change": "unchanged",
        },
        session_id=direct_session,
        task_id=direct_case,
    ))
    assert direct_compared["replay"]["usable"] is True
    _parsed(handle_function_call(
        "epistemic_model",
        {
            "operation": "revise",
            "reason": "Record the direct result's unknown provenance",
            "mechanisms": [{
                "id": "M1",
                "text": "Direct dispatch proves the mechanism",
                "epistemic_status": "grounded",
                "authority_domain": "system_state",
                "evidence_event_ids": [direct_observed["event_id"]],
            }],
        },
        session_id=direct_session,
        task_id=direct_case,
    ))
    direct_close = json.loads(handle_function_call(
        "epistemic_model",
        {
            "operation": "close",
            "outcome": "resolved",
            "summary": "Attempt closure from direct dispatch",
            "transfer": {"decision": "none"},
        },
        session_id=direct_session,
        task_id=direct_case,
    ))
    assert "resolved closure requires" in direct_close["error"]

    uncertainty_session = "runtime-uncertainty-session"
    uncertainty_case = "runtime-uncertainty"
    _parsed(handle_function_call(
        "epistemic_model",
        {
            "operation": "open",
            "case_id": uncertainty_case,
            "decision": "Keep the mechanism uncertain when the check is contrary",
            "stopping_condition": "The mechanism is revised or remains unresolved",
            "state_grounding": [
                {"id": "S1", "text": "A bounded mechanism is under review"}
            ],
            "mechanisms": [
                {
                    "id": "M1",
                    "text": "The fixture supports the mechanism",
                    "epistemic_status": "hypothesis",
                }
            ],
            "alternatives": [{"id": "A1", "text": "The fixture is unrelated"}],
            "top_unknown": "Whether the fixture supports the mechanism",
            "current_plan": [{"id": "P1", "action": "Check the fixture", "depends_on": ["M1"]}],
        },
        session_id=uncertainty_session,
        task_id=uncertainty_case,
    ))
    uncertainty_probe = _parsed(handle_function_call(
        "epistemic_probe",
        {
            "operation": "commit",
            "claim_id": "M1",
            "purpose": "learn",
            "unknown_or_goal": "Whether the fixture supports M1",
            "tool_name": "read_file",
            "tool_args": probe_args,
            "predicted_outcomes": [{"outcome": "contrary value", "meaning": "Downgrade M1"}],
            "why_this_action": "The fixture is the designated check",
            "would_change_belief": "The fixture does not support M1",
            "authority_domain": "system_state",
        },
        session_id=uncertainty_session,
        task_id=uncertainty_case,
    ))
    assert uncertainty_probe["probe"]["claim_id"] == "M1"
    handle_function_call(
        "read_file",
        probe_args,
        session_id=uncertainty_session,
        task_id=uncertainty_case,
    )
    uncertainty_compared = _parsed(handle_function_call(
        "epistemic_probe",
        {
            "operation": "compare",
            "disposition": "mismatch",
            "material": True,
            "belief_change": "weakened",
            "rationale": "The fixture is contrary to the mechanism prediction",
        },
        session_id=uncertainty_session,
        task_id=uncertainty_case,
    ))
    uncertainty_mismatch_id = uncertainty_compared["event"]["event_id"]
    downgraded = _parsed(handle_function_call(
        "epistemic_model",
        {
            "operation": "revise",
            "reason": "The contrary fixture leaves the mechanism unresolved",
            "addresses_event_ids": [uncertainty_mismatch_id],
            "mechanisms": [
                {
                    "id": "M1",
                    "text": "The fixture supports the mechanism",
                    "epistemic_status": "unresolved",
                }
            ],
            "current_plan": [],
        },
        session_id=uncertainty_session,
        task_id=uncertainty_case,
    ))
    assert downgraded["case"]["stale"] is False
    assert downgraded["replay"]["usable"] is True
    uncertainty_closed = _parsed(handle_function_call(
        "epistemic_model",
        {
            "operation": "close",
            "outcome": "unresolved",
            "summary": "The contrary fixture did not resolve the mechanism",
            "transfer": {"decision": "none"},
        },
        session_id=uncertainty_session,
        task_id=uncertainty_case,
    ))
    assert uncertainty_closed["case"]["status"] == "closed"

    legacy_session = "runtime-legacy-session"
    legacy_case = "runtime-legacy"
    _parsed(handle_function_call(
        "epistemic_model",
        {"operation": "open", "case_id": legacy_case, "top_unknown": "Legacy probe compatibility"},
        session_id=legacy_session,
        task_id=legacy_case,
    ))
    legacy_probe = _parsed(handle_function_call(
        "epistemic_probe",
        {
            "operation": "commit",
            "purpose": "advance",
            "unknown_or_goal": "Whether the legacy path remains usable",
            "tool_name": "read_file",
            "tool_args": {"path": str(ordinary_fixture)},
            "predicted_outcomes": [{"outcome": "ordinary value"}],
            "why_this_action": "Preserve the omitted-claim_id API path",
            "would_change_belief": "The legacy action cannot run",
            "authority_domain": "system_state",
        },
        session_id=legacy_session,
        task_id=legacy_case,
    ))
    assert "claim_id" not in legacy_probe["probe"]
    handle_function_call(
        "read_file",
        {"path": str(ordinary_fixture)},
        session_id=legacy_session,
        task_id=legacy_case,
    )
    legacy_compared = _parsed(handle_function_call(
        "epistemic_probe",
        {
            "operation": "compare",
            "disposition": "match",
            "material": False,
            "rationale": "The omitted-claim_id probe completed on the native path",
        },
        session_id=legacy_session,
        task_id=legacy_case,
    ))
    assert legacy_compared["replay"]["usable"] is True
    legacy_closed = _parsed(handle_function_call(
        "epistemic_model",
        {
            "operation": "close",
            "outcome": "unresolved",
            "summary": "Legacy probe compatibility remains available",
            "transfer": {"decision": "none"},
        },
        session_id=legacy_session,
        task_id=legacy_case,
    ))
    assert legacy_closed["case"]["status"] == "closed"

    events_path = home / "epistemic-harness" / "cases" / case_id / "events.jsonl"
    event_types = [json.loads(line)["type"] for line in events_path.read_text().splitlines()]
    payload = {
        "plugin_tools_available_and_deferred_dispatch_tested": True,
        "ordinary_tool_forwarded": True,
        "committed_result_captured": "probe_observed" in event_types,
        "actual_outer_raw_boundary": native_observed["payload"]["raw_result_provenance"] == "native_handler_result",
        "direct_dispatch_unknown_boundary": direct_observed["payload"]["raw_result_provenance"] == "unknown",
        "direct_dispatch_cannot_resolve": "resolved closure requires" in direct_close["error"],
        "comparison_completed": "probe_compared" in event_types,
        "replay_usable": compared["replay"]["usable"],
        "closed_dormant": True,
        "bound_claim_path": True,
        "uncertainty_downgrade_path": uncertainty_closed["case"]["status"] == "closed",
        "legacy_probe_path": legacy_closed["case"]["status"] == "closed",
        "event_types": event_types,
    }
    print("EPISTEMIC_SMOKE_JSON=" + json.dumps(payload, sort_keys=True))


if __name__ == "__main__":
    main()
