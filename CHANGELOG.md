# Changelog

## 1.4.0

- When the optional Jev client is enabled, consultation is now required before
  each committed probe and before closing a case. Following the recommendation
  remains optional; the gate requires a recorded `offer` and `choose`, not
  agreement with the ranking.
- Each consultation carries a `decision_point` of `probe` or `close`. A
  consultation is valid only at the Timeline tip and only for the decision it
  was requested for, so one offer cannot cover several later commitments.
- A probe offer requires `continue_same` or `continue_different`; `stop` does
  not authorize a probe. A close offer requires `stop`; a `continue_*`
  disposition means the case should remain open.
- The plugin records `method_offer_failed` for a verified service failure and
  `method_consultation_excused` for a cooldown or exhausted budget. Either
  record permits the gated decision, except that a 401 or 403 is not treated as
  a temporary failure. Caller errors such as malformed candidates or an invalid
  disposition do not create a waiver. A missing key or invalid configuration
  blocks until fixed or Jev is disabled.
- Plan authority values are now validated against the declared enum, and an
  existing authority tag is retained when a revision rewrites the item without
  repeating it.
- Corrected the privacy receipt wording: `sent_chars` is the sum of included
  field-value characters, not the serialized payload size; `request_chars` is a
  character count, not a UTF-8 byte count. `truncated` now also reports a fixed
  item-count cap. Candidate text is explicitly documented as caller-supplied and
  not authority-filtered.
- Added the schema-sanitizer regression to the Hermes CI job and strengthened
  its assertions for the plan authority enum.

## 1.3.1

- Documented the privacy receipt precisely: `sent_chars` counts the projected
  case fields only and excludes candidate texts and fixed question instructions;
  `request_chars` is the full serialized body.
- Declared `authority_domain` on plan items so a user-authority plan step
  survives schema sanitization and is omitted from hosted advice by default.
- Defined `followed_recommendation` narrowly: it is true only when the chosen
  method equals the recommendation, and always false for `stop`.
- Documented that failed, `busy`, and timed-out attempts consume the per-case
  attempt budget; the token budget uses vendor-reported usage only; and a `busy`
  result counts toward the cooldown.
- Clarified that the four-worker cap applies per loaded plugin module, that the
  caller deadline is not a worker-cancellation guarantee, and that a timed-out
  call can still send its one request.
- Renamed the privacy receipt field `recent_event_types` to the payload key
  `recent_events`.
- Added the missing `timeout_s` setting and the explicit-enable step to the
  README.

## 1.3.0

- Added the optional `epistemic_methods` tool (`offer`, `choose`, `status`) for
  hosted Jev advice on whether to continue an inquiry and which candidate
  investigation method to run next. It is disabled by default, hidden when
  disabled, and cannot open, revise, pause, close, or execute anything.
- Candidate methods are supplied per case; the plugin ships no fixed method
  taxonomy. The judge ranks the offered candidates and estimates whether another
  round would change the decision.
- Added an explicit, redacted projection allowlist, HTTPS-only endpoint
  validation, redirect refusal, response size cap, per-case attempt and token
  budgets, offer expiry, and a consecutive-failure cooldown.
- Recorded method advice as `method_offer_started`, `method_offered`,
  `method_offer_failed`, and `method_chosen` Timeline events. Advice is not
  evidence and does not enter closure calibration.
- Added a best-effort free-text scrubber for common path, URL, DOI, and event-ID
  shapes; excluded plan items that declare a user-authority domain; capped the
  vendor model label; clamped token usage at zero; and enforced a caller-side
  total deadline in addition to the per-read socket timeout.
- The transport now runs open, headers, body, and parse in a daemon worker
  joined at the total deadline; at most four such workers may be in flight per
  loaded plugin module, and a call returns `busy` when all four are held by
  abandoned connections. Timeouts report `reason: timeout` rather than the
  generic transport code. Failed, `busy`, and timed-out attempts consume the
  per-case attempt budget.
- Added `offer_pending` so a second offer cannot invalidate a live unanswered
  one, and made `status` ignore expired or superseded offers.
- Excluded `epistemic_methods` from probe targets and from the middleware probe
  path, and made candidate-input errors report `bad_candidates` rather than
  `bad_config`.
- Fixed duplicate or nested middleware re-entry of the same dispatch: the
  re-entry now runs the inner action once instead of capturing the gateway block
  as the probe observation. The in-flight identity is process-wide, so a wrapper
  on another thread is also recognised; a re-entry whose arguments differ from
  the commitment is blocked rather than executed. A later retry with the same
  tool-call ID remains blocked while a probe is still executing.
- Updated the privacy, security, README, and bundled inquiry guidance for the
  optional outbound path.

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
