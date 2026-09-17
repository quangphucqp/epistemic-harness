"""Native regressions for scoped capture and bounded observer bookkeeping."""

from __future__ import annotations

import pytest

from native_test_support import (
    HERMES_PREREQUISITE_REASON,
    discover_hermes_core_roots,
    run_native,
)

if not discover_hermes_core_roots():
    pytest.skip(HERMES_PREREQUISITE_REASON, allow_module_level=True)


def test_direct_native_dispatch_records_unknown_and_cannot_resolve(tmp_path):
    result = run_native(
        tmp_path,
        r'''
import json
from epistemic_harness import plugin
from epistemic_harness.harness import current_native_capture_state
from hermes_cli.plugins import get_plugin_manager
from model_tools import handle_function_call
from tools.registry import registry

manager = get_plugin_manager()
manager.discover_and_load()
harness = plugin._get_harness()
case_id = "direct-native-unknown"
session_id = "direct-native-unknown-session"
tool_name = "direct-native-unknown-tool"
executions = {"count": 0}

def handler(args, **_kwargs):
    executions["count"] += 1
    return json.dumps({"value": "DIRECT-RAW"})

registry.register(
    name=tool_name,
    toolset="direct-native-unknown-fixture",
    schema={
        "name": tool_name,
        "description": "fixture",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    handler=handler,
)
harness.model(
    {
        "operation": "open",
        "case_id": case_id,
        "top_unknown": "Whether the direct result supports M1",
        "mechanisms": [{"id": "M1", "text": "The direct result supports the claim"}],
    },
    session_id=session_id,
)
harness.probe(
    {
        "operation": "commit",
        "claim_id": "M1",
        "purpose": "learn",
        "unknown_or_goal": "Whether the direct result supports M1",
        "tool_name": tool_name,
        "tool_args": {"query": "fixture"},
        "predicted_outcomes": [{"outcome": "support"}],
        "why_this_action": "The fixture is the designated check",
        "would_change_belief": "The direct result is incompatible",
        "authority_domain": "system_state",
    },
    session_id=session_id,
)
returned = handle_function_call(
    tool_name,
    {"query": "fixture"},
    task_id="direct-native-unknown-task",
    session_id=session_id,
    tool_call_id="direct-native-unknown-call",
)
state_after_return = current_native_capture_state()
events = harness.store.list_events(case_id)
observed = [event for event in events if event["type"] == "probe_observed"]
late = plugin.post_tool_call(
    tool_name=tool_name,
    args={"query": "fixture"},
    result=json.dumps({"value": "LATE-RAW"}),
    session_id=session_id,
    task_id="direct-native-unknown-task",
    tool_call_id="direct-native-unknown-call",
)
compared = harness.probe(
    {
        "operation": "compare",
        "disposition": "match",
        "material": False,
        "rationale": "The direct result was returned but its raw boundary was unavailable",
        "belief_change": "unchanged",
    },
    session_id=session_id,
)
observation_id = observed[0]["event_id"]
harness.model(
    {
        "operation": "revise",
        "reason": "Record the direct result's bounded provenance",
        "mechanisms": [{
            "id": "M1",
            "text": "The direct result supports the claim",
            "epistemic_status": "grounded",
            "authority_domain": "system_state",
            "evidence_event_ids": [observation_id],
        }],
    },
    session_id=session_id,
)
try:
    harness.model(
        {
            "operation": "close",
            "outcome": "resolved",
            "summary": "Attempt closure from the direct result",
            "transfer": {"decision": "none"},
        },
        session_id=session_id,
    )
except Exception as exc:
    close_error = str(exc)
else:
    close_error = None
output = {
    "returned": returned,
    "executions": executions["count"],
    "state_present": state_after_return is not None,
    "observed": [event["payload"] for event in observed],
    "late": late,
    "compared_usable": compared["replay"]["usable"],
    "close_error": close_error,
}
assert output["returned"] == '{"value": "DIRECT-RAW"}'
assert output["executions"] == 1
assert output["state_present"] is False
assert len(output["observed"]) == 1
assert output["observed"][0]["observation_boundary"] == "tool_execution_middleware_return"
assert output["observed"][0]["raw_result_provenance"] == "unknown"
assert output["observed"][0]["result_status"] == "unknown"
assert output["late"] is None
assert output["compared_usable"] is True
assert "resolved closure requires" in output["close_error"]
print("RESULT=" + json.dumps(output, sort_keys=True))
''',
    )
    assert result["returned"] == '{"value": "DIRECT-RAW"}'
    assert result["executions"] == 1
    assert result["state_present"] is False
    assert len(result["observed"]) == 1
    assert result["observed"][0]["observation_boundary"] == "tool_execution_middleware_return"
    assert result["observed"][0]["raw_result_provenance"] == "unknown"
    assert result["observed"][0]["result_status"] == "unknown"
    assert result["late"] is None
    assert result["compared_usable"] is True
    assert "resolved closure requires" in result["close_error"]


def test_outer_native_bookkeeping_fails_open_when_store_lock_is_held(tmp_path):
    result = run_native(
        tmp_path,
        r'''
import contextlib
import json
import subprocess
import sys
import threading
import time

import hermes_cli.plugins as core_plugins
from epistemic_harness import plugin
from hermes_cli.middleware import run_tool_execution_middleware
from hermes_cli.plugins import get_plugin_manager
from model_tools import handle_function_call
from tools.registry import registry

manager = get_plugin_manager()
manager.discover_and_load()
core_plugins._resolve_hook_callback_timeout = lambda: 0.20
case_id = "native-lock-timeout"
session_id = "native-lock-timeout-session"
tool_name = "native-lock-timeout-tool"
handler_ready = threading.Event()
handler_release = threading.Event()
calls = {"count": 0}

def handler(args, **_kwargs):
    calls["count"] += 1
    handler_ready.set()
    if not handler_release.wait(timeout=5):
        raise RuntimeError("fixture handler release timed out")
    return json.dumps({"value": "RAW-LOCK-TIMEOUT"})

registry.register(
    name=tool_name,
    toolset="native-lock-timeout-fixture",
    schema={
        "name": tool_name,
        "description": "fixture",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    handler=handler,
)
harness = plugin._get_harness()
harness.model(
    {"operation": "open", "case_id": case_id, "top_unknown": "Whether the result is usable"},
    session_id=session_id,
)
harness.probe(
    {
        "operation": "commit",
        "purpose": "learn",
        "unknown_or_goal": "Whether the result is usable",
        "tool_name": tool_name,
        "tool_args": {"query": "fixture"},
        "predicted_outcomes": [{"outcome": "usable result"}],
        "why_this_action": "The fixture is the designated action",
        "would_change_belief": "The result is incompatible",
        "authority_domain": "system_state",
    },
    session_id=session_id,
)

def outer_call():
    return run_tool_execution_middleware(
        tool_name,
        {"query": "fixture"},
        lambda actual: handle_function_call(
            tool_name,
            actual,
            task_id="native-lock-timeout-task",
            session_id=session_id,
            tool_call_id="native-lock-timeout-call",
            skip_pre_tool_call_hook=True,
            skip_tool_request_middleware=True,
            skip_tool_execution_middleware=True,
        ),
        session_id=session_id,
        task_id="native-lock-timeout-task",
        tool_call_id="native-lock-timeout-call",
    )

outcome = {}
done = threading.Event()

def invoke():
    try:
        outcome["returned"] = outer_call()
    except BaseException as exc:
        outcome["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        done.set()

host = threading.Thread(target=invoke, name="native-lock-timeout-host", daemon=True)
host.start()
if not handler_ready.wait(timeout=3):
    raise RuntimeError("fixture handler was not reached")
holder_code = (
    "import fcntl,sys\n"
    "with open(sys.argv[1], 'a+') as handle:\n"
    "    fcntl.flock(handle.fileno(), fcntl.LOCK_EX)\n"
    "    print('READY', flush=True)\n"
    "    sys.stdin.readline()\n"
)
holder = subprocess.Popen(
    [sys.executable, "-c", holder_code, str(harness.store.root / ".store.lock")],
    stdin=subprocess.PIPE,
    stdout=subprocess.PIPE,
    stderr=subprocess.PIPE,
    text=True,
)
try:
    if holder.stdout is None or holder.stdout.readline().strip() != "READY":
        raise RuntimeError("lock holder did not become ready")
    handler_release.set()
    returned_while_locked = done.wait(timeout=2.0)
    events_path = harness.store.root / "cases" / case_id / "events.jsonl"
    before_release = events_path.read_text(encoding="utf-8") if events_path.exists() else ""
finally:
    if holder.stdin is not None:
        holder.stdin.write("release\n")
        holder.stdin.flush()
    try:
        holder.wait(timeout=5)
    finally:
        if holder.poll() is None:
            holder.terminate()
            with contextlib.suppress(Exception):
                holder.wait(timeout=3)
if not done.wait(timeout=5):
    raise RuntimeError("outer host did not finish after lock release")
deadline = time.monotonic() + 5
while (manager._hook_running_callbacks or manager._hook_abandoned) and time.monotonic() < deadline:
    time.sleep(0.03)
events = harness.store.list_events(case_id)
observed = [event for event in events if event["type"] == "probe_observed"]
late = plugin.post_tool_call(
    tool_name=tool_name,
    args={"query": "fixture"},
    result=json.dumps({"value": "LATE-RAW"}),
    session_id=session_id,
    task_id="native-lock-timeout-task",
    tool_call_id="native-lock-timeout-call",
)
second = outer_call()
output = {
    "returned_while_locked": returned_while_locked,
    "probe_observed_before_release": any("probe_observed" in line for line in before_release.splitlines()),
    "handler_calls": calls["count"],
    "returned": outcome.get("returned"),
    "error": outcome.get("error"),
    "observed": [event["payload"] for event in observed],
    "late": late,
    "second": second,
}
assert output["returned_while_locked"] is True
assert output["probe_observed_before_release"] is False
assert output["handler_calls"] == 1
assert output["returned"] == '{"value": "RAW-LOCK-TIMEOUT"}'
assert output["error"] is None
assert output["observed"] == []
assert output["late"] is None
assert "EPISTEMIC GATEWAY BLOCKED" in output["second"]
print("RESULT=" + json.dumps(output, sort_keys=True))
''',
    )
    assert result["returned_while_locked"] is True
    assert result["probe_observed_before_release"] is False
    assert result["handler_calls"] == 1
    assert result["returned"] == '{"value": "RAW-LOCK-TIMEOUT"}'
    assert result["error"] is None
    assert result["observed"] == []


def test_late_native_exposure_cannot_rebind_first_result_and_stale_contexts(tmp_path):
    result = run_native(
        tmp_path,
        r'''
import contextvars
import json

from epistemic_harness import plugin
from hermes_cli.middleware import run_tool_execution_middleware
from hermes_cli.plugins import get_plugin_manager
from model_tools import handle_function_call
from tools.registry import registry

manager = get_plugin_manager()
manager.discover_and_load()
harness = plugin._get_harness()


def start_probe(case_id, session_id, tool_name):
    harness.model(
        {"operation": "open", "case_id": case_id, "top_unknown": "Whether the first result supports M1"},
        session_id=session_id,
    )
    harness.probe(
        {
            "operation": "commit",
            "purpose": "learn",
            "unknown_or_goal": "Whether the first result supports M1",
            "tool_name": tool_name,
            "tool_args": {"query": "same"},
            "predicted_outcomes": [{"outcome": "FIRST"}],
            "why_this_action": "The fixture is the designated action",
            "would_change_belief": "The result is not FIRST",
            "authority_domain": "system_state",
        },
        session_id=session_id,
    )


unknown_case = "late-unknown"
unknown_session = "late-unknown-session"
unknown_tool = "late-unknown-tool"
unknown_calls = []
unknown_contexts = []


def unknown_handler(args, **_kwargs):
    unknown_calls.append(args)
    unknown_contexts.append(contextvars.copy_context())
    return json.dumps({"value": "FIRST" if len(unknown_calls) == 1 else "SECOND"})


registry.register(
    name=unknown_tool,
    toolset="native-late-fixture",
    schema={
        "name": unknown_tool,
        "description": "fixture",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    handler=unknown_handler,
)
start_probe(unknown_case, unknown_session, unknown_tool)
saved_hooks = {
    name: manager._hooks.pop(name, None)
    for name in ("post_tool_call", "transform_tool_result")
}
try:
    first_result = handle_function_call(
        unknown_tool,
        {"query": "same"},
        session_id=unknown_session,
        task_id="",
        tool_call_id="",
    )
finally:
    for name, callback in saved_hooks.items():
        if callback is not None:
            manager._hooks[name] = callback

first_probe = harness.store.get_case(unknown_case)["pending_probe"]
first_events = harness.store.list_events(unknown_case)
first_observation = next(
    event for event in first_events if event["type"] == "probe_observed"
)

stale_post = unknown_contexts[0].run(
    plugin.post_tool_call,
    tool_name=unknown_tool,
    args={"query": "same"},
    result=json.dumps({"value": "STALE-RAW"}),
    session_id=unknown_session,
    task_id="",
    tool_call_id="",
)
stale_transform = unknown_contexts[0].run(
    plugin.transform_tool_result,
    tool_name=unknown_tool,
    args={"query": "same"},
    result=json.dumps({"value": "STALE-DISPLAY"}),
    session_id=unknown_session,
    task_id="",
    tool_call_id="",
)
second_result = handle_function_call(
    unknown_tool,
    {"query": "same"},
    session_id=unknown_session,
    task_id="",
    tool_call_id="",
)
second_probe = harness.store.get_case(unknown_case)["pending_probe"]
unknown_events = [
    event
    for event in harness.store.list_events(unknown_case)
    if event["type"] in {"probe_observed", "probe_exposed"}
]

raw_case = "late-raw"
raw_session = "late-raw-session"
raw_tool = "late-raw-tool"
raw_calls = []
raw_contexts = []


def raw_handler(args, **_kwargs):
    raw_calls.append(args)
    raw_contexts.append(contextvars.copy_context())
    return json.dumps({"value": "RAW-FIRST"})


registry.register(
    name=raw_tool,
    toolset="native-late-fixture",
    schema={
        "name": raw_tool,
        "description": "fixture",
        "parameters": {
            "type": "object",
            "properties": {"query": {"type": "string"}},
            "required": ["query"],
        },
    },
    handler=raw_handler,
)
start_probe(raw_case, raw_session, raw_tool)
raw_result = run_tool_execution_middleware(
    raw_tool,
    {"query": "same"},
    lambda actual: handle_function_call(
        raw_tool,
        actual,
        session_id=raw_session,
        task_id="late-raw-task",
        tool_call_id="late-raw-call",
        skip_pre_tool_call_hook=True,
        skip_tool_request_middleware=True,
        skip_tool_execution_middleware=True,
    ),
    session_id=raw_session,
    task_id="late-raw-task",
    tool_call_id="late-raw-call",
)
raw_events_before_stale = harness.store.list_events(raw_case)
raw_direct_late = harness.record_tool_result_exposure(
    tool_name=raw_tool,
    args={"query": "same"},
    result=json.dumps({"value": "RAW-DIRECT-LATE"}),
    session_id=raw_session,
    task_id="late-raw-task",
    tool_call_id="late-raw-call",
)
raw_stale_post = raw_contexts[0].run(
    plugin.post_tool_call,
    tool_name=raw_tool,
    args={"query": "same"},
    result=json.dumps({"value": "RAW-STALE"}),
    session_id=raw_session,
    task_id="late-raw-task",
    tool_call_id="late-raw-call",
)
raw_stale_transform = raw_contexts[0].run(
    plugin.transform_tool_result,
    tool_name=raw_tool,
    args={"query": "same"},
    result=json.dumps({"value": "DISPLAY-STALE"}),
    session_id=raw_session,
    task_id="late-raw-task",
    tool_call_id="late-raw-call",
)
raw_events_after_stale = harness.store.list_events(raw_case)
raw_observations = [
    event for event in raw_events_before_stale if event["type"] == "probe_observed"
]
raw_exposures = [
    event for event in raw_events_before_stale if event["type"] == "probe_exposed"
]
output = {
    "first_result": first_result,
    "second_result": second_result,
    "unknown_calls": len(unknown_calls),
    "unknown_observations": [
        {"type": event["type"], "payload": event["payload"]}
        for event in unknown_events
    ],
    "first_observation_id": first_probe["observation_event_id"],
    "second_observation_id": second_probe["observation_event_id"],
    "first_observation_result": first_observation["payload"]["result"],
    "second_observation_result": second_probe.get("result"),
    "stale_post": stale_post,
    "stale_transform": stale_transform,
    "raw_result": raw_result,
    "raw_calls": len(raw_calls),
    "raw_observations": [event["payload"] for event in raw_observations],
    "raw_exposures": [event["payload"] for event in raw_exposures],
    "raw_direct_late": raw_direct_late,
    "raw_stale_post": raw_stale_post,
    "raw_stale_transform": raw_stale_transform,
    "raw_events_unchanged": raw_events_after_stale == raw_events_before_stale,
}
assert output["second_result"] == '{"value": "SECOND"}'
assert output["unknown_calls"] == 2
assert len(output["unknown_observations"]) == 1
assert output["unknown_observations"][0]["type"] == "probe_observed"
assert output["unknown_observations"][0]["payload"]["result"] == '{"value": "FIRST"}'
assert output["first_observation_id"] == output["second_observation_id"]
assert output["first_observation_result"] == '{"value": "FIRST"}'
assert output["second_observation_result"] is None
assert output["stale_post"] is None
assert output["stale_transform"] is None
assert output["raw_result"] == '{"value": "RAW-FIRST"}'
assert output["raw_calls"] == 1
assert len(output["raw_observations"]) == 1
assert output["raw_observations"][0]["raw_result_provenance"] == "native_handler_result"
assert output["raw_direct_late"] is None
assert output["raw_stale_post"] is None
assert output["raw_stale_transform"] is None
assert output["raw_events_unchanged"] is True
print("RESULT=" + json.dumps(output, sort_keys=True))
''',
    )
    assert result["first_result"] == '{"value": "FIRST"}'
    assert result["second_result"] == '{"value": "SECOND"}'
    assert result["unknown_calls"] == 2
    assert len(result["unknown_observations"]) == 1
    assert result["unknown_observations"][0]["type"] == "probe_observed"
    assert result["unknown_observations"][0]["payload"]["result"] == '{"value": "FIRST"}'
    assert result["first_observation_id"] == result["second_observation_id"]
    assert result["first_observation_result"] == '{"value": "FIRST"}'
    assert result["second_observation_result"] is None
    assert result["stale_post"] is None
    assert result["stale_transform"] is None
    assert result["raw_result"] == '{"value": "RAW-FIRST"}'
    assert result["raw_calls"] == 1
    assert len(result["raw_observations"]) == 1
    assert result["raw_observations"][0]["raw_result_provenance"] == "native_handler_result"
    assert result["raw_direct_late"] is None
    assert result["raw_stale_post"] is None
    assert result["raw_stale_transform"] is None
    assert result["raw_events_unchanged"] is True
