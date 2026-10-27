# AI-assisted contribution; maintained by Epistemic Harness contributors.
"""One live wire check for the optional Jev path, using synthetic state only."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from epistemic_harness import jev  # noqa: E402

config = jev.load_config({"enabled": True, "model": "jev-latest"})
state = {
    "case": {
        "top_unknown": "Whether the observed difference survives a matched comparison",
        "decision": "Choose the next investigation method",
        "stopping_condition": "A matched comparison is run, or the question remains unresolved",
        "round": 1,
    },
    "claims": {
        "alternatives": ["The difference is noise", "The difference is caused by the removed line"],
        "state_grounding": ["One output changed after a single line was removed"],
        "mechanisms": [],
    },
    "plan": ["Inspect the changed output"],
    "recent_events": ["case_opened: synthetic wire check"],
    "previous_method": "",
}
candidates = [
    {"method_id": "m1", "text": "Read the changed output closely"},
    {"method_id": "m2", "text": "Run a matched comparison with and without the line"},
    {"method_id": "m3", "text": "Ask a second model for an independent reading"},
]
request = jev.build_request(state, candidates, config)
response = jev.call(request, config=config)
parsed = jev.parse_offer(
    response, offered_ids=[item["method_id"] for item in candidates], config=config
)
print(json.dumps({"model": parsed["model"], "recommended": parsed["recommended_method_id"],
                  "ranking": parsed["ranking"], "continue_probability": parsed["continue_probability"],
                  "confidence": parsed["confidence"], "usage": parsed["usage"]}, indent=2))
