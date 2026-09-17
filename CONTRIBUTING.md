# Contributing

Epistemic Harness is an AI-assisted contribution maintained by project
contributors. Code, tests, and documentation are not presented as authored by
any individual user. Keep public examples and fixtures generic: do not commit
private profiles, case logs, transcripts, credentials, machine-specific paths,
or evaluation artifacts.

## Development

The runtime dependency is Hermes Agent's public plugin API. The plugin declares
no additional runtime Python dependencies in `plugin.yaml`. Test-only
requirements are `pytest` and `PyYAML` with bounded versions.

Run the standalone tests with the canonical Hermes runner when a Hermes
checkout is available:

```bash
HERMES_PYTHON=/path/to/python \
  bash /path/to/hermes-agent/scripts/run_tests.sh tests -j 2 --file-retries 0
```

For a quick local-only run with test dependencies installed:

```bash
python -m pytest tests -q
```

Validate the registration contract through Hermes before publishing:

```bash
hermes plugins validate /path/to/epistemic-harness
```

Run the native runtime smoke with an installed Hermes CLI. These commands copy
the checkout into an isolated profile, enable it through the public CLI without
granting built-in tool overrides, and pass the exact `HERMES_HOME` to the smoke
script:

```bash
(
  SMOKE_ROOT="$(mktemp -d)"
  export HOME="$SMOKE_ROOT"
  export HERMES_HOME="$SMOKE_ROOT/.hermes"
  mkdir -p "$HERMES_HOME/plugins/epistemic-harness"
  cp -R . "$HERMES_HOME/plugins/epistemic-harness/"
  hermes plugins enable epistemic-harness --no-allow-tool-override
  python tests/runtime_smoke.py "$HERMES_HOME"
)
```

Use a newly created `SMOKE_ROOT` for each run. Do not replace the final
argument with `$HOME`: `runtime_smoke.py` writes fixtures and verifies state
under the exact `HERMES_HOME` it receives. The `hermes` and `python` commands
must come from the same Hermes installation.

Use temporary `HOME`/`HERMES_HOME` directories for tests that exercise state.
Keep verification logs outside the publishable plugin tree. Do not install the
plugin into a live profile, modify Hermes core, or publish from a development
checkout.
