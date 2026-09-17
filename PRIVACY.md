# Privacy notes

Epistemic Harness is local-only. It writes profile-scoped case state under
`$HERMES_HOME/epistemic-harness/` and does not transmit it, index it in a
remote service, or export it automatically.

Stored material can include:

- case decisions, stopping conditions, claims, plans, and closure summaries;
- Timeline event IDs, authority domains, evidence descriptors, and replay
  accounting;
- committed tool names and recursively redacted argument copies;
- captured tool results, with large results placed in local redacted artifacts;
- the local commitment key, lock files, and crash-recovery journals.

Captured source and tool results may remain sensitive despite redaction. Keep
`$HERMES_HOME` access-controlled and avoid putting secrets in case text or tool
arguments. The plugin does not read a credential store and does not add
telemetry or a self-update path.

Compact responses affect only the returned presentation. They do not delete,
truncate, or rewrite the durable case, Timeline, artifacts, or replay data. Use
the returned full-readback call when the complete model-facing receipt is
needed.

To uninstall, remove the plugin package through Hermes. Uninstalling the code
does not delete `$HERMES_HOME/epistemic-harness/`; review, back up, and delete
that directory separately when the local records are no longer needed.
