# Security

## Scope

Epistemic Harness is an optional local Hermes plugin for Linux and macOS. It
uses POSIX `fcntl` locks for cross-process case and probe coordination. It does
not provide Windows support and refuses to start where those locks are
unavailable rather than weakening the concurrency contract.

The default plugin surface has no network client, telemetry, automatic data
export, self-updater, or credential-store access. The optional
`epistemic_methods` tool adds one outbound HTTPS client for method advice; it is
disabled by default and hidden unless `settings.jev.enabled` is true and
`TYPESAFE_API_KEY` resolves through the profile secret scope. It does not modify
Hermes core.

When the client is enabled, its use is mandatory before each committed probe and
before a case closure. The recommendation is not binding: the gate requires a
recorded offer and disposition, not agreement. A service failure is recorded as
`method_offer_failed` and permits the decision; a 401 or 403 does not, because
it indicates a credential problem. A cooldown or exhausted budget is recorded
as `method_consultation_excused` and also permits it. A missing key or invalid
configuration blocks the decision until the configuration is fixed or Jev is
disabled; `pause` remains available. This gate is enforced by the plugin's case
transitions; it does not authorize or execute an investigation method. It
covers the model-facing commit and close paths; a process that writes the case
store directly is outside this boundary.

## Optional outbound advice

When enabled, `epistemic_methods` sends one POST per `offer` to the configured
HTTPS endpoint. The endpoint is validated as HTTPS at config load and again at
the network boundary, redirects are refused, the caller returns at a hard
caller-side deadline regardless of header or body drip (an abandoned transport
worker may keep reading until its socket closes), and the response is capped at
256 KB. At most four transport workers may be in flight per loaded plugin
module; a multiplexed host can load one module per profile, so the process total
can be higher. A call returns `busy` rather than starting another worker when the
module's four slots are held by abandoned connections; a `busy` result is
recorded as a failure and can trigger the cooldown. A timed-out call can still
send its one request if the transport was already blocked inside DNS or connect;
the deadline bounds the caller, not that abandoned send. An abandoned worker
retains its request object, including the bearer header, until it exits; the
four-worker cap bounds how many can be retained. The client never logs the vendor key, request headers, or an HTTP error
body. A transport, HTTP, malformed-response, or state-change failure is recorded
locally as `method_offer_failed` with a short reason and does not break unrelated
host work. When the mandatory gate applies, a `method_offer_failed` with an
excusable reason permits the probe or closure; a 401 or 403 does not, and
authentication errors do not consume the attempt budget. A cooldown or exhausted
budget is recorded as `method_consultation_excused`. A missing key
or invalid configuration blocks the decision until fixed or Jev is disabled.

The projection is an allowlist, not a sample of the case. It excludes case and
session identifiers, evidence references and locators, probe arguments and
results, prior lessons, closure text, and artifact paths by the structured
allowlist. Free-text values also pass a best-effort path, URL, DOI, and event-ID
scrubber; a location described in prose can still survive the pattern filter.
Treat the endpoint and its operator as a disclosure recipient even when the
projection is small.

Candidate method text is supplied by the caller and is treated as data by the
judge. Do not pass unreviewed third-party text as a candidate: a candidate can
bias the ranking even though the returned choice is validated against the offered
IDs.

## Local state and sensitive material

Profile-scoped state is stored under:

```text
$HERMES_HOME/epistemic-harness/
```

The store can contain case snapshots, hash-linked Timeline events, model
versions, claim text, evidence descriptors, committed tool arguments, and
captured tool results. Large captured results are redacted and stored as local
artifacts by content hash, but redaction is not a guarantee that source or tool
content is safe to disclose. Treat the entire state directory as sensitive.
The commitment key and lock files are also local state.

There is no automatic export. Review and remove the state directory through an
operator-controlled backup and deletion procedure when needed.

## Integrity boundary

The API validates append-only event sequencing, hash-chain consistency,
case-snapshot binding, artifact digests, and transactional recovery. These are
integrity checks, not tamper-proof storage. A local actor with write access to
the complete profile can alter the case files and rewrite the hashes. Do not
represent a valid replay as proof that the files were not edited by such an
actor.

The plugin cannot roll back an external side effect after a capture failure;
it fails open for the host turn and leaves the case available for explicit
inspection or repair. It also cannot semantically fact-check claim prose.

## Reporting

For a suspected vulnerability, use the repository's GitHub security reporting
channel and include the affected release, platform, reproduction steps, and
whether the report involves sensitive local state. Do not attach case logs or
captured source/tool content unless they have been redacted and are necessary
for the report.
