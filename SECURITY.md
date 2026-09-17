# Security

## Scope

Epistemic Harness is an optional local Hermes plugin for Linux and macOS. It
uses POSIX `fcntl` locks for cross-process case and probe coordination. It does
not provide Windows support and refuses to start where those locks are
unavailable rather than weakening the concurrency contract.

The plugin has no network client, telemetry, automatic data export,
self-updater, or credential-store access. It does not modify Hermes core.

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
