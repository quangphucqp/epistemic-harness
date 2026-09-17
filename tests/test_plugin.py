# AI-assisted contribution; maintained by Epistemic Harness contributors.
"""Standalone Hermes integration contracts for the epistemic plugin."""

from __future__ import annotations

import json
import logging
import sys
import types
from pathlib import Path

import yaml

from epistemic_harness import plugin
from epistemic_harness.harness import EpistemicHarness
from epistemic_harness.schemas import MODEL_SCHEMA


class RecordingContext:
    def __init__(self):
        self.tools = {}
        self.hooks = {}
        self.middleware = []
        self.skills = {}

    def register_tool(self, **kwargs):
        self.tools[kwargs["name"]] = kwargs

    def register_hook(self, name, callback, **kwargs):
        assert kwargs == {}
        self.hooks[name] = callback

    def register_middleware(self, kind, callback, **kwargs):
        self.middleware.append((kind, callback, kwargs))

    def register_skill(self, name, path, description="", frontmatter=None):
        self.skills[name] = {
            "path": path,
            "description": description,
            "frontmatter": dict(frontmatter or {}),
        }

    def get_config(self, key, default=None):
        return default


def _open_payload(case_id: str = "plugin-case") -> dict:
    return {
        "operation": "open",
        "case_id": case_id,
        "decision": "Choose an interpretation",
        "stopping_condition": "An interpretation survives or remains unresolved",
        "state_grounding": [
            {"id": "S1", "text": "Two interpretations", "evidence_event_ids": []}
        ],
        "mechanisms": [
            {"id": "M1", "text": "One mechanism", "evidence_event_ids": []}
        ],
        "alternatives": [{"id": "A1", "text": "Alternative mechanism"}],
        "top_unknown": "Which mechanism fits",
        "current_plan": [
            {"id": "P1", "action": "Inspect evidence", "depends_on": ["M1"]}
        ],
    }


def test_manifest_uses_only_native_hermes_hooks():
    manifest = yaml.safe_load((Path(__file__).parents[1] / "plugin.yaml").read_text())
    assert manifest["version"] == "1.2.0"
    assert manifest["kind"] == "standalone"
    assert manifest["provides_hooks"] == [
        "pre_llm_call",
        "post_tool_call",
        "transform_tool_result",
        "subagent_stop",
        "on_session_reset",
        "on_session_finalize",
    ]


def test_model_schema_keeps_portfolio_and_evidence_quality_fields():
    properties = MODEL_SCHEMA["parameters"]["properties"]
    assert "list" in properties["operation"]["enum"]
    assert properties["status_filter"]["enum"] == ["active", "paused", "closed"]
    assert properties["evidence"]["items"]["required"] == [
        "summary",
        "source_ref",
        "source_role",
        "access_scope",
        "authority_domain",
    ]


def test_plugin_registers_two_tools_five_native_hooks_and_fail_open_middleware(
    tmp_path, monkeypatch
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    plugin.reset_for_tests()
    context = RecordingContext()
    plugin.register(context)

    assert set(context.tools) == {"epistemic_model", "epistemic_probe"}
    assert set(context.hooks) == {
        "pre_llm_call",
        "post_tool_call",
        "transform_tool_result",
        "subagent_stop",
        "on_session_reset",
        "on_session_finalize",
    }
    assert len(context.middleware) == 1
    kind, gateway, options = context.middleware[0]
    assert kind == "tool_execution"
    assert options == {}

    calls = []
    assert gateway(
        tool_name="read_file",
        args={"path": "ordinary.txt"},
        session_id="session-1",
        next_call=lambda args: calls.append(args) or "ordinary",
    ) == "ordinary"
    assert calls == [{"path": "ordinary.txt"}]

    opened = json.loads(context.tools["epistemic_model"]["handler"](
        _open_payload(), session_id="session-1"
    ))
    assert opened["case"]["status"] == "active"
    injected = context.hooks["pre_llm_call"](session_id="session-1")
    assert injected["context"].startswith("[ACTIVE EPISTEMIC CASE")
    assert "ordinary search" in injected["context"].lower()


def test_transform_observer_persists_exposed_representation_without_rewriting(monkeypatch):
    observed = {}

    class Harness:
        def record_tool_result_exposure(self, **kwargs):
            observed.update(kwargs)

    monkeypatch.setattr(plugin, "_session_harness", lambda _session_id: Harness())
    result = plugin.transform_tool_result(
        tool_name="read_file",
        args={"path": "evidence.txt"},
        result="evidence",
        session_id="session-1",
        tool_call_id="call-evidence",
    )
    assert result is None
    assert observed["result"] == "evidence"
    assert observed["tool_call_id"] == "call-evidence"


def test_native_session_reset_pauses_old_case(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    plugin.reset_for_tests()
    context = RecordingContext()
    plugin.register(context)
    context.tools["epistemic_model"]["handler"](
        _open_payload(), session_id="old-session"
    )
    context.hooks["on_session_reset"](
        old_session_id="old-session",
        new_session_id="new-session",
        reason="new_session",
    )
    case = plugin._get_harness().store.get_case("plugin-case")
    assert case["status"] == "paused"


def test_subagent_stop_pauses_only_child_owned_case(tmp_path, monkeypatch):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    plugin.reset_for_tests()
    context = RecordingContext()
    plugin.register(context)
    context.tools["epistemic_model"]["handler"](
        _open_payload(), session_id="child-session"
    )
    context.hooks["subagent_stop"](
        parent_session_id="parent-session",
        child_session_id="child-session",
        child_status="failed",
    )
    case = plugin._get_harness().store.get_case("plugin-case")
    assert case["status"] == "paused"
    assert case["session_id"] == "child-session"


def test_compression_lineage_is_recovered_without_private_core_hooks(tmp_path, monkeypatch):
    harness = EpistemicHarness(tmp_path / "epistemic-harness")
    harness.model(_open_payload("compressed-case"), session_id="parent-session")

    class ReadOnlySessionDB:
        def __init__(self, read_only=False):
            assert read_only is True

        def get_compression_lineage(self, session_id):
            assert session_id == "child-session"
            return ["parent-session", "child-session"]

        def close(self):
            pass

    monkeypatch.setitem(sys.modules, "hermes_state", types.SimpleNamespace(
        SessionDB=ReadOnlySessionDB
    ))
    plugin._adopt_compression_lineage(harness, "child-session")

    assert harness.store.get_active_case("parent-session") is None
    migrated = harness.store.get_active_case("child-session")
    assert migrated["case_id"] == "compressed-case"
    assert harness.store.list_events("compressed-case")[-1]["type"] == "session_migrated"


def test_runtime_diagnostic_and_registration_log_share_release_generation(
    tmp_path, monkeypatch, caplog
):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    plugin.reset_for_tests()
    context = RecordingContext()
    with caplog.at_level(logging.INFO, logger="epistemic_harness.plugin"):
        plugin.register(context)
        diagnostic = json.loads(
            context.tools["epistemic_model"]["handler"](
                {"operation": "list"}, session_id="generation-check"
            )
        )

    generation = "20260916-prospective-inquiry"
    assert diagnostic["runtime"]["generation"] == generation
    assert any(f"generation={generation}" in record.message for record in caplog.records)
