# AI-assisted contribution; maintained by Epistemic Harness contributors.
"""Injection spill-budget ladder tests (E2, plan v10).

The guard's contract: emitted context never exceeds the effective ceiling of
the compose-time configuration snapshot; small cases render byte-identical to
the pre-guard format; every emitted tier preserves case identity plus a
retrieval instruction; an unrepresentable ceiling yields no injection (tier 4)
with the portfolio path named in the logs.
"""

from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import pytest

from epistemic_harness import plugin


@pytest.fixture(autouse=True)
def _reset_budget():
    plugin._INJECTION_BUDGET.clear()
    yield
    plugin._INJECTION_BUDGET.clear()


def _case(**overrides):
    case = {
        "case_id": "budget-case",
        "status": "active",
        "version": 3,
        "stale": False,
        "pending_probe": None,
        "model": {
            "decision": "Pick the surviving account",
            "stopping_condition": "One account survives",
            "state_grounding": [
                {"id": "S1", "text": "Two accounts remain", "evidence_event_ids": []}
            ],
            "mechanisms": [
                {"id": "M1", "text": "Mechanism one", "evidence_event_ids": []}
            ],
            "alternatives": [{"id": "A1", "text": "Task structure explains it"}],
            "top_unknown": "Which account predicts the process",
            "current_plan": [{"id": "P1", "action": "Inspect evidence", "depends_on": ["M1"]}],
            "prior_case_ids": [],
            "retrieved_prior_lessons": [],
            "applied_prior_lessons": [],
        },
    }
    case["model"].update(overrides.pop("model", {}))
    case.update(overrides)
    return case


def _replay():
    return {"usable": True, "timeline": {"valid": True, "events": 12}, "checks_failed": []}


def _json_block(rendered: str) -> dict:
    marker = "Current case:\n"
    assert marker in rendered
    return json.loads(rendered.split(marker, 1)[1])


def test_small_case_renders_byte_identical_to_pre_guard_format():
    case = _case()
    replay = _replay()
    rendered = plugin._render_context(case, replay)
    assert rendered is not None
    expected = (
        f"[ACTIVE EPISTEMIC CASE {case['case_id']}]\n"
        "This bounded case is active. The durable model below controls case-relevant work.\n"
        + plugin._RULES_TEXT
        + "Current case:\n"
        + json.dumps(plugin._compact_case(case, replay), ensure_ascii=False, sort_keys=True, indent=2)
    )
    assert rendered == expected
    assert len(rendered) <= plugin._resolve_injection_ceiling()


def test_oversized_lessons_shed_first_and_replay_survives():
    lessons = [{"lesson": "x" * 8000, "calibration": "supported", "source_case_id": f"c{i}"} for i in range(3)]
    case = _case(model={"retrieved_prior_lessons": lessons})
    rendered = plugin._render_context(case, _replay())
    assert rendered is not None
    assert len(rendered) <= plugin._resolve_injection_ceiling()
    body = _json_block(rendered)
    assert body["retrieved_prior_lessons"] == {
        "shed_for_budget": True,
        "count": 3,
        "retrieve": plugin._RETRIEVAL_HINT,
    }
    # Replay survived tier 1a: it is still the full replay dict.
    assert body["replay"]["timeline"]["events"] == 12


def test_oversized_state_grounding_is_capped_with_marker():
    grounding = [
        {"id": f"S{i}", "text": "claim " + "y" * 3000, "evidence_event_ids": []}
        for i in range(10)
    ]
    case = _case(model={"state_grounding": grounding})
    rendered = plugin._render_context(case, _replay())
    assert rendered is not None
    assert len(rendered) <= plugin._resolve_injection_ceiling()
    body = _json_block(rendered)
    assert "truncated-for-budget" in json.dumps(body["state_grounding"])


def test_pending_probe_is_digested_with_retrieval_path():
    case = _case(
        pending_probe={
            "status": "committed",
            "tool_name": "read_file",
            "purpose": "learn",
            "tool_args": {"path": "z" * 20000},
        }
    )
    case["model"]["state_grounding"] = [
        {"id": "S1", "text": "y" * 8000, "evidence_event_ids": []}
    ]
    rendered = plugin._render_context(case, _replay())
    assert rendered is not None
    assert "z" * 1000 not in rendered
    body = _json_block(rendered)
    probe = body["pending_probe"]
    assert probe["tool_name"] == "read_file"
    assert probe["tool_args_bytes"] > 20000
    assert "epistemic_model" in probe["tool_args_location"]


def test_everything_huge_renders_pointer_form():
    case = _case(
        pending_probe={"status": "committed", "tool_name": "read_file", "tool_args": {"a": "q" * 9000}},
        model={
            "decision": "d" * 9000,
            "stopping_condition": "s" * 9000,
            "state_grounding": [{"id": "S1", "text": "g" * 9000, "evidence_event_ids": []}],
            "mechanisms": [{"id": "M1", "text": "m" * 9000, "evidence_event_ids": []}],
            "alternatives": [{"id": "A1", "text": "a" * 9000}],
            "top_unknown": "u" * 9000,
            "current_plan": [{"id": "P1", "action": "p" * 9000, "depends_on": []}],
        },
    )
    # Force the ceiling between the capped tier and the pointer form.
    plugin._INJECTION_BUDGET["injection_max_chars"] = 1200
    rendered = plugin._render_context(case, _replay())
    assert rendered is not None
    assert len(rendered) <= plugin._resolve_injection_ceiling()
    assert '"budget_exceeded": true' in rendered
    assert "epistemic_model(show)" in rendered
    assert "Current case:" not in rendered


def test_unrepresentable_ceiling_suppresses_injection(caplog):
    plugin._INJECTION_BUDGET["injection_max_chars"] = 10
    rendered = plugin._render_context(_case(), _replay())
    assert rendered is None
    assert any("suppressed" in record.message for record in caplog.records)


def test_invalid_budget_config_falls_back_without_raising():
    plugin._INJECTION_BUDGET["injection_max_chars"] = "not-a-number"
    rendered = plugin._render_context(_case(), _replay())
    assert rendered is not None
    assert plugin._resolve_injection_ceiling() == plugin._DEFAULT_INJECTION_MAX_CHARS


# ------------------------------------------------------------- spill interplay


def _fake_spill_module(config_store: dict):
    tools_pkg = types.ModuleType("tools")
    spill_mod = types.ModuleType("tools.hook_output_spill")
    spill_mod.get_spill_config = lambda: dict(config_store["config"])
    tools_pkg.hook_output_spill = spill_mod
    return tools_pkg, spill_mod


def test_operator_spill_threshold_below_default_tightens_ceiling(monkeypatch):
    store = {"config": {"enabled": True, "max_chars": 2000, "directory": None}}
    tools_pkg, spill_mod = _fake_spill_module(store)
    monkeypatch.setitem(sys.modules, "tools", tools_pkg)
    monkeypatch.setitem(sys.modules, "tools.hook_output_spill", spill_mod)
    assert plugin._resolve_injection_ceiling() == 1000
    case = _case(model={"decision": "w" * 1500})
    rendered = plugin._render_context(case, _replay())
    assert rendered is not None
    assert len(rendered) <= 1000


def test_spill_disabled_leaves_plugin_budget(monkeypatch):
    store = {"config": {"enabled": False, "max_chars": 100, "directory": None}}
    tools_pkg, spill_mod = _fake_spill_module(store)
    monkeypatch.setitem(sys.modules, "tools", tools_pkg)
    monkeypatch.setitem(sys.modules, "tools.hook_output_spill", spill_mod)
    assert plugin._resolve_injection_ceiling() == plugin._DEFAULT_INJECTION_MAX_CHARS


def test_ceiling_follows_config_changes_between_composes(monkeypatch):
    store = {"config": {"enabled": True, "max_chars": 10000, "directory": None}}
    tools_pkg, spill_mod = _fake_spill_module(store)
    monkeypatch.setitem(sys.modules, "tools", tools_pkg)
    monkeypatch.setitem(sys.modules, "tools.hook_output_spill", spill_mod)
    assert plugin._resolve_injection_ceiling() == 9000
    store["config"] = {"enabled": True, "max_chars": 3000, "directory": None}
    assert plugin._resolve_injection_ceiling() == 2000


# ------------------------------------------------------------- real spill integration


_HERMES_REPO = Path.home() / ".hermes" / "hermes-agent"


@pytest.mark.skipif(
    not (_HERMES_REPO / "tools" / "hook_output_spill.py").exists(),
    reason="Hermes repo unavailable",
)
def test_rendered_piece_degrades_to_real_spill_pointer_under_reduced_config(tmp_path):
    """The race's worst case, exercised against the real spill function: a piece
    rendered under one snapshot that Hermes then evaluates under a lower one
    degrades to spill-with-pointer — the documented pre-guard baseline."""
    sys.path.insert(0, str(_HERMES_REPO))
    try:
        from tools.hook_output_spill import spill_if_oversized
    except ImportError:
        pytest.skip("Hermes spill module unavailable")

    rendered = plugin._render_context(_case(), _replay())
    assert rendered is not None
    config = {"enabled": True, "max_chars": 100, "directory": str(tmp_path)}
    spilled = spill_if_oversized(
        rendered, session_id="s-test", source="plugin hook", config=config
    )
    assert spilled != rendered
    assert "budget-case" not in spilled or len(spilled) < len(rendered)
    # The spill replacement names its on-disk location — a retrieval path.
    assert str(tmp_path) in spilled or "spill" in spilled.lower()
