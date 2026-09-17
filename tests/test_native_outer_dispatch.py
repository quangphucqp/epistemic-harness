"""Native frozen-core tests for raw observation provenance and deferred args."""

from __future__ import annotations

import pytest

from native_test_support import (
    HERMES_PREREQUISITE_REASON,
    discover_hermes_core_roots,
    run_native,
)

if not discover_hermes_core_roots():
    pytest.skip(HERMES_PREREQUISITE_REASON, allow_module_level=True)


def test_real_outer_middleware_preserves_raw_failure_and_exactly_once_execution(tmp_path):
    result = run_native(
        tmp_path,
        r'''
import json
import os
from hermes_cli.plugins import PluginContext, PluginManifest, get_plugin_manager
from hermes_cli.middleware import run_tool_execution_middleware
from model_tools import handle_function_call
from tools.registry import registry

manager = get_plugin_manager()
manager.discover_and_load()
transform_modes = {}
executions = {}

def transform(**kw):
    name = kw.get("tool_name")
    if name in transform_modes:
        return transform_modes[name]
    return None

ctx = PluginContext(PluginManifest(name="native-display-fixture", version="0.0.1", description="fixture", source="user"), manager)
ctx.register_hook("transform_tool_result", transform)

def register(name, raw):
    def handler(args, **kwargs):
        executions[name] = executions.get(name, 0) + 1
        return raw
    registry.register(
        name=name,
        toolset="native-fixture",
        schema={"name": name, "description": "fixture", "parameters": {"type": "object", "properties": {"query": {"type": "string"}}, "required": ["query"]}},
        handler=handler,
    )


def call(name, args, session, task, call_id=""):
    return handle_function_call(name, args, session_id=session, task_id=task, tool_call_id=call_id)


def outer(name, args, session, task, call_id):
    return run_tool_execution_middleware(
        name,
        args,
        lambda actual: handle_function_call(
            name,
            actual,
            task_id=task,
            session_id=session,
            tool_call_id=call_id,
            skip_pre_tool_call_hook=True,
            skip_tool_request_middleware=True,
            skip_tool_execution_middleware=True,
        ),
        session_id=session,
        task_id=task,
        tool_call_id=call_id,
    )


def receipt(name, session, task, raw, display):
    register(name, raw)
    transform_modes[name] = display
    call("epistemic_model", {"operation": "open", "case_id": name, "top_unknown": "Whether the fixture result supports M1", "mechanisms": [{"id": "M1", "text": "The fixture supports the claim"}]}, session, task)
    call("epistemic_probe", {"operation": "commit", "purpose": "learn", "unknown_or_goal": "Whether the fixture supports M1", "tool_name": name, "tool_args": {"query": "fixture"}, "predicted_outcomes": ["support"], "why_this_action": "The fixture is the designated source", "would_change_belief": "The fixture is incompatible", "authority_domain": "external_source"}, session, task)
    returned = outer(name, {"query": "fixture"}, session, task, "call-" + name)
    events = [json.loads(line) for line in open(os.environ["HERMES_HOME"] + "/epistemic-harness/cases/" + name + "/events.jsonl")]
    observed = next(event for event in events if event["type"] == "probe_observed")
    exposed = [event for event in events if event["type"] == "probe_exposed"]
    compared = call("epistemic_probe", {"operation": "compare", "disposition": "match", "material": False, "rationale": "Compare the exact boundary receipt", "evidence": {"summary": "The fixture boundary was inspected", "source_ref": "fixture://native", "source_role": "primary", "access_scope": "full", "authority_domain": "external_source"}}, session, task)
    comparison_id = json.loads(compared)["event"]["event_id"]
    revision = call("epistemic_model", {"operation": "revise", "reason": "Bind the observed fixture receipt", "mechanisms": [{"id": "M1", "text": "The fixture supports the claim", "epistemic_status": "grounded", "authority_domain": "external_source", "evidence_event_ids": [exposed[-1]["event_id"] if exposed else observed["event_id"]]}]}, session, task)
    calibration = json.loads(revision)["replay"]
    closed = call("epistemic_model", {"operation": "close", "outcome": "resolved", "summary": "Attempt closure from the captured fixture", "transfer": {"decision": "none"}}, session, task)
    return {"returned": returned, "executions": executions[name], "observed": observed["payload"], "exposed": [event["payload"] for event in exposed], "comparison_id": comparison_id, "replay": calibration, "closed": json.loads(closed)}

failure = receipt("native-raw-failure", "native-raw-failure-session", "native-raw-failure-task", json.dumps({"error": "RAW-EXECUTION-FAILED"}), "TRANSFORMED-REPRESENTATION")
success = receipt("native-raw-success", "native-raw-success-session", "native-raw-success-task", json.dumps({"value": "RAW-EXECUTION-SUCCEEDED"}), "TRANSFORMED-SUCCESS")
display_error = receipt("native-display-error", "native-display-error-session", "native-display-error-task", json.dumps({"value": "RAW-EXECUTION-SUCCEEDED"}), json.dumps({"error": "DISPLAY-ONLY-ERROR"}))
print("RESULT=" + json.dumps({"failure": failure, "success": success, "display_error": display_error}, sort_keys=True))
''',
    )

    failure = result["failure"]
    assert failure["returned"] == "TRANSFORMED-REPRESENTATION"
    assert failure["executions"] == 1
    assert failure["observed"]["result"] == '{"error": "RAW-EXECUTION-FAILED"}'
    assert failure["observed"]["result_status"] == "tool_error"
    assert failure["observed"]["execution_result_status"] == "tool_error"
    assert failure["observed"]["observation_boundary"] == "post_tool_call"
    assert failure["observed"]["raw_result_provenance"] == "native_handler_result"
    assert failure["exposed"]
    assert failure["exposed"][-1]["result"] == "TRANSFORMED-REPRESENTATION"
    assert failure["exposed"][-1]["execution_result_status"] == "tool_error"
    assert failure["closed"]["error"]
    assert "resolved closure requires" in failure["closed"]["error"]

    success = result["success"]
    assert success["returned"] == "TRANSFORMED-SUCCESS"
    assert success["executions"] == 1
    assert success["observed"]["execution_result_status"] == "ok"
    assert success["exposed"][-1]["display_result_status"] == "ok"
    assert success["closed"]["case"]["closure"]["calibration"]["label"] == "supported"

    display_error = result["display_error"]
    assert display_error["returned"] == '{"error": "DISPLAY-ONLY-ERROR"}'
    assert display_error["executions"] == 1
    assert display_error["observed"]["execution_result_status"] == "ok"
    assert display_error["exposed"][-1]["display_result_status"] == "tool_error"
    assert "resolved closure requires" in display_error["closed"]["error"]
