# Manual checks

`check_jev_wire.py` is a manual, live POST check for the optional method-advice
wire contract. It builds a synthetic case state, reads the vendor key through
`epistemic_harness.jev.api_key()`, sends one request, and prints only the parsed
ranking, probability, confidence, model label, and token usage.

It is not imported by the plugin, is not run by the test suite, and is not run by
CI. Use it only when you intend to spend one small hosted call:

```bash
PYTHONDONTWRITEBYTECODE=1 ~/.hermes/hermes-agent/venv/bin/python scripts/check_jev_wire.py
```

The request contains no real case, session, evidence, or project data.
