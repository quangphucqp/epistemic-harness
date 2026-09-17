"""Native frozen-core test for provider-safe deferred arguments."""

from __future__ import annotations

import pytest

from native_test_support import (
    HERMES_PREREQUISITE_REASON,
    discover_hermes_core_roots,
    run_native,
)

if not discover_hermes_core_roots():
    pytest.skip(HERMES_PREREQUISITE_REASON, allow_module_level=True)


def test_native_deferred_dispatch_preserves_nested_json_arguments_and_commitment(tmp_path):
    result = run_native(
        tmp_path,
        r'''
import json
from hermes_cli.plugins import get_plugin_manager
from model_tools import handle_function_call
from tools.registry import registry
from tools.schema_sanitizer import sanitize_tool_schemas
from epistemic_harness.schemas import PROBE_SCHEMA

manager = get_plugin_manager()
manager.discover_and_load()
cleaned = sanitize_tool_schemas([{"type": "function", "function": PROBE_SCHEMA}])[0]
assert cleaned["function"]["parameters"]["properties"]["tool_args_json"]["type"] == "string"
received = []
def ordinary(args, **kwargs):
    received.append(args)
    return json.dumps({"received": args}, sort_keys=True)
registry.register(
    name="native-json-tool",
    toolset="native-fixture",
    schema={"name": "native-json-tool", "description": "fixture", "parameters": {"type": "object", "properties": {"payload": {"type": "object", "properties": {"query": {"type": "string"}, "flags": {"type": "array", "items": {"type": "boolean"}}, "nested": {"type": "object", "properties": {"limit": {"type": "integer"}}, "required": ["limit"]}}, "required": ["query", "flags", "nested"]}}, "required": ["payload"]}},
    handler=ordinary,
)

def bridge(name, args, session="deferred-session"):
    raw = {"calls": [{"name": name, "arguments": json.dumps(args, sort_keys=True)}]}
    return handle_function_call("tool_call", raw, session_id=session, task_id="deferred-task")

assert json.loads(bridge("epistemic_model", {"operation": "open", "case_id": "deferred-case", "top_unknown": "Whether the nested payload is preserved"})) ["case"]["status"] == "active"
nested = {"query": "source", "flags": [True, False], "nested": {"limit": 3}}
commit = bridge("epistemic_probe", {"operation": "commit", "purpose": "learn", "unknown_or_goal": "Whether the nested payload is preserved", "tool_name": "native-json-tool", "tool_args_json": json.dumps({"payload": nested}, sort_keys=True), "tool_args": {}, "predicted_outcomes": [{"outcome": "The nested payload is received"}], "why_this_action": "The fixture is the designated ordinary tool", "would_change_belief": "The nested payload is altered", "authority_domain": "system_state"})
commit_obj = json.loads(commit)
assert commit_obj["probe"]["tool_args"] == {"payload": nested}
ordinary_result = bridge("native-json-tool", {"payload": nested})
assert received == [{"payload": nested}]
assert json.loads(ordinary_result)["received"] == {"payload": nested}
compared = json.loads(bridge("epistemic_probe", {"operation": "compare", "disposition": "match", "material": False, "rationale": "The native deferred ordinary tool received the exact nested payload"}))
assert compared["event"]["payload"]["execution_args_match_commitment"] is True
assert compared["replay"]["usable"] is True
print("RESULT=" + json.dumps({"commit": commit_obj, "ordinary": json.loads(ordinary_result), "compare": compared, "received": received}, sort_keys=True))
''',
    )
    assert result["received"] == [
        {"payload": {"query": "source", "flags": [True, False], "nested": {"limit": 3}}}
    ]
    assert result["compare"]["event"]["payload"]["execution_args_match_commitment"] is True
    assert result["compare"]["replay"]["usable"] is True
