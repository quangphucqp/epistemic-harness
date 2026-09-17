# AI-assisted contribution; maintained by Epistemic Harness contributors.
"""User-plugin entry point for the epistemic harness."""

from __future__ import annotations

import sys
from pathlib import Path

if __package__:
    from .epistemic_harness.plugin import register
else:  # Bare-module loading used by pytest and some plugin diagnostics.
    plugin_root = Path(__file__).resolve().parent
    if str(plugin_root) not in sys.path:
        sys.path.insert(0, str(plugin_root))
    from epistemic_harness.plugin import register

__all__ = ["register"]
