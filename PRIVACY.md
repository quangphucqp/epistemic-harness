# Privacy notes

Epistemic Harness is local-only by default. It writes profile-scoped case state
under `$HERMES_HOME/epistemic-harness/`, does not transmit that state, does not
index it in a remote service, and does not export it automatically.

The optional `epistemic_methods` tool is the single exception, and it is disabled
by default. When explicitly enabled, the plugin requires a consultation before
each committed probe and before closing a case, and each `offer` request sends
one bounded, redacted projection of the active case to the configured HTTPS
endpoint (TypeSafe by default). The projection contains the top unknown,
decision, stopping condition, the offer round number, a small number of claim
and plan texts, short recent event types, the previous method, and the candidate
methods supplied by the caller. It excludes
case and session IDs, evidence references and locators, probe arguments or
results, prior lessons, closure text, and artifact paths by the structured
allowlist. Free-text values pass through credential redaction and a best-effort
path, URL, DOI, and event-ID scrubber; a location described in prose can still
survive the pattern filter. Claims and plan items that declare a user-authority
domain are omitted unless `allow_user_domain_state` is set. Candidate method
text is supplied by the caller and is not authority-filtered. Plan authority
values are validated against the declared enum, and an existing tag is retained
when a revision rewrites the item without repeating it.

The `privacy` receipt's `sent_chars` is the sum of included field-value
characters plus the round value. It excludes JSON structure, the candidate
method texts, and the fixed question instructions; `request_chars` is the
character count of the full
serialized request, not its UTF-8 byte size. `truncated` is set when a per-field
character cap, a fixed item-count cap, or the total projected-character budget
drops content.

The vendor key is resolved through Hermes's profile secret scope, with a process
environment fallback for standalone use. The plugin does not read `.env` or a
credential store directly, and a multiplexed host with no active profile scope
fails closed. The endpoint must use HTTPS, redirects are refused, and the
response is capped at 256 KB.

The request and response are not retained by the plugin beyond the local Timeline
event that records the offer: the candidate methods, the recommended method, the
ranking, `confidence`, `continue_probability`, the short model label, token usage,
the privacy receipt, and the expiry time. The vendor's own retention
and logging are governed by the vendor's policy; the plugin cannot guarantee
deletion at the vendor.

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
telemetry or a self-update path. The optional method-advice client is the only
outbound network path; it is inactive unless `settings.jev.enabled` is true and
`TYPESAFE_API_KEY` is present.

Compact responses affect only the returned presentation. They do not delete,
truncate, or rewrite the durable case, Timeline, artifacts, or replay data. Use
the returned full-readback call when the complete model-facing receipt is
needed.

To uninstall, remove the plugin package through Hermes. Uninstalling the code
does not delete `$HERMES_HOME/epistemic-harness/`; review, back up, and delete
that directory separately when the local records are no longer needed.
