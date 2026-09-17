"""Shared support for native Hermes integration tests."""

from __future__ import annotations

import importlib.util
import json
import os
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest

PLUGIN_ROOT = Path(__file__).resolve().parents[1]
HERMES_PREREQUISITE_REASON = (
    "Hermes Agent core is not installed or importable in the current test interpreter "
    "(hermes_cli is required for native integration tests)"
)


def discover_hermes_core_roots() -> tuple[Path, ...]:
    """Return the import roots selected by this test interpreter."""
    try:
        spec = importlib.util.find_spec("hermes_cli")
    except (ImportError, ModuleNotFoundError):
        return ()
    if spec is None:
        return ()

    locations = spec.submodule_search_locations
    if locations:
        roots = [Path(location).resolve().parent for location in locations]
    elif spec.origin and spec.origin not in {"built-in", "frozen"}:
        roots = [Path(spec.origin).resolve().parent.parent]
    else:
        return ()
    return tuple(dict.fromkeys(roots))


def _require_hermes_core() -> tuple[Path, ...]:
    roots = discover_hermes_core_roots()
    if not roots:
        pytest.skip(HERMES_PREREQUISITE_REASON)
    return roots


def _clean_plugin_export(tmp_path: Path) -> Path:
    """Copy only the installable plugin payload into the test sandbox."""
    export = tmp_path / "plugin-export"
    export.mkdir()
    shutil.copytree(PLUGIN_ROOT / "epistemic_harness", export / "epistemic_harness")
    shutil.copytree(PLUGIN_ROOT / "skills", export / "skills")
    shutil.copy2(PLUGIN_ROOT / "__init__.py", export / "__init__.py")
    shutil.copy2(PLUGIN_ROOT / "plugin.yaml", export / "plugin.yaml")
    return export


def run_native(tmp_path: Path, script: str) -> dict[str, Any]:
    """Run one native fixture in a clean export with isolated Hermes state."""
    core_roots = _require_hermes_core()
    plugin_export = _clean_plugin_export(tmp_path)
    home = tmp_path / "home"
    hermes_home = home / ".hermes"
    (hermes_home / "plugins").mkdir(parents=True)
    shutil.copytree(plugin_export, hermes_home / "plugins" / "epistemic-harness")
    (hermes_home / "config.yaml").write_text(
        json.dumps({"plugins": {"enabled": ["epistemic-harness"]}}),
        encoding="utf-8",
    )

    env = os.environ.copy()
    pythonpath = [*(str(root) for root in core_roots), str(plugin_export)]
    if env.get("PYTHONPATH"):
        pythonpath.append(env["PYTHONPATH"])
    env.update(
        {
            "HOME": str(home),
            "HERMES_HOME": str(hermes_home),
            "PYTHONPATH": os.pathsep.join(pythonpath),
        }
    )
    completed = subprocess.run(
        [sys.executable, "-c", script],
        cwd=str(plugin_export),
        env=env,
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert completed.returncode == 0, completed.stdout + completed.stderr
    try:
        line = next(
            line for line in completed.stdout.splitlines() if line.startswith("RESULT=")
        )
    except StopIteration as exc:
        raise AssertionError(
            "native fixture did not emit RESULT=\n"
            + completed.stdout
            + completed.stderr
        ) from exc
    return json.loads(line.removeprefix("RESULT="))
