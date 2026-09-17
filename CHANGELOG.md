# Changelog

## 1.2.0

- Added portable activation and an explicitly registered `epistemic-inquiry`
  workflow skill.
- Added pre-mutation validation and actionable errors for the required
  `operation` field.
- Added opt-in `response_detail: compact` projections to both tools. Full
  receipts remain the default; durable case state and replay behavior are
  unchanged.
- Added standalone-repository metadata, privacy/security guidance, MIT
  licensing, and CI validation.
- Made the competing-dispatch regression assert exactly-once execution rather
  than scheduling-specific wording.
- Added a native `post_tool_call` raw-result observer and explicit unknown provenance
  when the host does not expose a trustworthy raw boundary; display transforms no
  longer replace raw failure status.
- Made `retrieved_prior_lessons` server-managed and `prior_case_ids` open-time-only;
  explicit imports now require closed, replay-valid source cases.
- Added provider-safe `tool_args_json` transport while retaining the legacy
  dictionary argument route.

This release does not claim improved reasoning accuracy. The bounded pilot
that informed the release completed its short tasks with disciplined controls,
but the workflow added overhead and exposed ordinary procedural omissions.

## 1.1.0

- Preserved transaction, provenance, claim-binding, uncertainty, lifecycle,
  and replay-integrity improvements from the prior release.
