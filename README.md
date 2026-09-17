> **Provenance:** AI-assisted contribution maintained by Epistemic Harness contributors.

# Epistemic Harness for Hermes

A bounded, session-scoped implementation of:

```text
MODEL → COMMIT → ACT/ASK → CAPTURE → COMPARE → REVISE → REPLAY → CONTINUE/STOP
```

The plugin is dormant unless an active case is bound to the current Hermes session. Dormant mode injects no prompt context and transparently forwards tools.

## Deliberate activation and response detail

Use the harness after explicit user activation, or after the agent announces a
bounded inquiry with its decision, scope, and stopping condition. Do not
auto-open cases or force every read, search, recall, or code action into a
probe. A meaningful investigative decision is the unit of use. Ordinary tools
remain available outside and inside an active case.

Supply `operation` explicitly on every `epistemic_model` and
`epistemic_probe` call. Missing or invalid operations are rejected before case
recovery or mutation; the error names the valid operations and gives an
explicit example. `response_detail` is optional and defaults to `full` on both
tools. Full remains the default. `compact` is a presentation-only projection:
it omits only retrieved prior-lesson bodies at the two known paths,
`case.model.retrieved_prior_lessons[*].lesson` and
`prior_case_lessons[*].lesson`; those bodies remain available in durable state
and in a full readback. It does not shorten observations, captured results, or
tool schemas. Receipts, event IDs, replay/accounting state, stale indicators,
pending probes, closure calibration, and committed claim snapshots remain
intact. When no lesson body can be omitted, compact metadata can make the
response slightly larger. Compact output is not evidence that the overall task
cost is reduced. When a body is omitted, the response marks the omitted paths
and gives an exact `epistemic_model(show, response_detail=full)` readback call.

The bundled workflow skill is registered through Hermes's public
`ctx.register_skill` API as `epistemic-harness:epistemic-inquiry`; it is not
copied into the user's global skills directory. It teaches open → claim-bound
commit → actual tool read → compare (observation disposition is distinct from
belief change) → revise or non-update → close/readback.

## Installed surfaces

Two model-facing tools:

- `epistemic_model`: `open`, `list`, `show`, `record_evidence`, `revise`, `pause`, `resume`, `close`
- `epistemic_probe`: `commit`, `compare`, `recover`, `discard`

## Distribution

Release 1.2.0 is a standalone MIT plugin for Hermes Agent 0.21 and later
within the declared `<1.0` compatibility range. Supported platforms are Linux
and macOS; POSIX `fcntl` locking is required. The plugin declares no additional
runtime Python dependencies. Validate an unpacked tree with:

```bash
hermes plugins validate /path/to/epistemic-harness
```

The catalog pin is maintained in the Hermes Agent repository, not in this
plugin tree. Updates must come through a reviewed pin change. See
[`SECURITY.md`](SECURITY.md), [`PRIVACY.md`](PRIVACY.md), and
[`CHANGELOG.md`](CHANGELOG.md) for storage, uninstall, and release notes.

One non-critical `tool_execution` middleware captures explicitly committed probes while forwarding ordinary tools. Six native Hermes hooks provide active-case context (with proof-of-life auto-resume), capture the handler result at `post_tool_call` before result transforms when that hook runs inside the active middleware scope, observe the result-transform input boundary, pause unfinished child work, and handle explicit session resets and lifecycle finalizations. The native capture context is finalized and reset on every middleware exit. The actual outer-middleware/inner-dispatch path therefore preserves a proven raw receipt; a lower-level `handle_function_call` whose post event arrives after the middleware returns is recorded at the middleware return with unknown provenance, and a late post event cannot upgrade it. The post-tool observer is correlated by session, task, turn, API request, tool-call, case, probe, tool name, and committed argument digest. Compression continuity is recovered at the next plugin boundary by reading Hermes's already-committed public compression lineage; the plugin does not participate in or patch the core compression transaction. If plugin storage fails, Hermes logs the local failure and continues the underlying turn exactly once.

Lifecycle pause callbacks share a 0.25-second lock-contention budget. If contention exhausts it, the callback fails open and logs an explicit warning that the pause was missed and not persisted; no deferred retry runs. The budget covers lock acquisition only, not arbitrary filesystem stalls.

Native raw, unknown, and exposure bookkeeping is also best-effort: state, cache,
store, and profile locks are acquired nonblocking. When a lock is unavailable,
the action result is returned once and the pending probe may remain for explicit
recover/discard; no background retry or redispatch is performed. The lock budget
does not bound arbitrary filesystem stalls after acquisition.

## Durable state

Profile-scoped state lives at:

```text
$HERMES_HOME/epistemic-harness/cases/<case-id>/
├── case.md
├── events.jsonl
└── archive/
    ├── case-v0001.md
    ├── artifacts/<sha256>.txt
    ├── .probe-execution.lock       # OS-released live-action lease
    └── .pending-transition.json   # transient crash-recovery journal
```

- `case.md`: current human-readable and machine-readable case state
- `events.jsonl`: logically append-only, atomically replaced, fsynced, SHA-256 hash-chained Timeline
- `archive/`: model-version snapshots and redacted large evidence artifacts

A profile-level `.store.lock` serializes state transitions and session claims. A separate per-case OS lease remains held across the actual external action: `recover` is refused while that lease is live, and the operating system releases it if the gateway process crashes. Each case/Timeline transition is write-ahead journaled inside `archive/`; startup or the next store access completes an interrupted transition idempotently before exposing state. New directory entries and atomic replacements are parent-directory fsynced; durability-sync failures abort rather than silently weakening persistence.

Legacy profile-level `compression-boundaries/` intents remain readable for existing data. New compression operations do not create them: the child’s first context/tool/model access follows the durable Hermes session lineage and migrates the case binding after the core has atomically published the child.

## Core invariants

- One active case per Hermes session, including under concurrent open/resume/migration attempts.
- One pending probe at a time.
- If same-name tool calls race for one committed probe, exactly one invocation captures it; siblings receive an ordinary model-visible gateway result rather than terminating the Hermes turn.
- Tool name and keyed HMAC-SHA-256 argument commitment identify an explicitly committed probe; execution-time argument drift is recorded rather than blocked. Durable argument copies are recursively redacted, and the profile-only commitment key prevents a useful plain offline hash oracle.
- `delegate_task` is rejected as a bounded probe because its immediate handle cannot account for later child actions.
- Tool result or error is captured at Hermes's native `post_tool_call` handler-result boundary before `transform_tool_result` when that event occurs inside the active native middleware scope. Capture is finalized and reset on every middleware exit. A lower-level dispatch whose post event arrives after return records the middleware result as unknown provenance; a late callback cannot overwrite or upgrade that observation. The native result-transform observer may add a `probe_exposed` event when the representation available at that boundary differs. Its input is not asserted to be the final model-visible bytes; final exposure is marked unknown when the host provides no later boundary. Tool-call correlation prevents an unrelated or late call from overwriting the observation. Observer bookkeeping is best-effort and nonblocking under state/cache/store/profile lock contention; an unavailable lock can leave a pending probe for explicit recover/discard, with no retry or redispatch.
- Delayed process/delegation results remain available during an active case and are not treated as an explicitly committed probe observation.
- Hermes JSON error envelopes are captured with `result_status: tool_error`, not misreported as success.
- Large evidence is secret-redacted and stored by content hash; replay checks its safe path, existence, byte count, and SHA-256.
- Every committed probe records the claim or unknown, its predicted observation, and `would_change_belief`: the observation that would count against the current judgment.
- Every probe observation needs `match`, `mismatch`, or `unresolved` disposition, or an explicit reasoned `discard`.
- Material mismatch marks the named model items and every transitively dependent plan step stale.
- Stale state reopens only after a revision cites the mismatch event and substantively changes or removes every stale item; metadata and whitespace churn do not qualify.
- Replay validates the entire hash chain, current case-state binding, observation accounting, artifact integrity, authority compatibility, mismatch repair, model event references, unique IDs, and plan dependencies.
- Formal user questions use the existing `clarify` tool and a typed user-authority domain.
- Interrupted execution is never automatically retried; `recover` is permitted only after no process holds the action lease and records that the action may or may not have completed.
- Paused and closed cases inject no context and impose no restrictions while paused; a lifecycle-paused case may later auto-resume on proof of life (see the lifecycle bullet).
- Hermes native reset events pause an unfinished old-session case when the host supplies the old identity, and lifecycle finalization events pause it by reason: `session_boundary` (CLI rotation), TUI/desktop session end (empty reason), and unknown future reasons all auto-pause with a proof-of-life marker; `session_expired` auto-pauses without the marker because expiry is permanent and can fire mid-turn; `new_session` and `shutdown` are skipped (the paired reset owns the former; the latter resumes in place). A lifecycle-paused case whose marker still validates auto-resumes at the next `pre_llm_call` — proof of life — and any deliberate disposition (open, operator pause, close, explicit resume) invalidates older markers on that session through a durable per-session high-water sequence (`sessions.json`, reconciled from pending journals before they are unlinked). Compression lineage adoption is one locked store primitive over the lineage prefix ending at the child: it migrates an active ancestor, or a marker-eligible paused ancestor (kept and re-based only when the child is pristine), and never adopts from a descendant. Resume and branch operations do not import a case into an unrelated session; the portfolio remains available for explicit pause/resume management.
- Injected case context is bounded by an effective ceiling resolved per compose from the plugin's `injection_max_chars` setting (default 9000) tightened under the operator's live `hooks.output_spill.max_chars`. Under pressure the injection sheds in order: JSON indentation, retrieved prior lessons, replay detail, applied prior lessons, per-field caps (committed probe arguments keep a byte count and the durable-store retrieval path), then a minimal pointer form, then no injection at all — never a silently truncated dump.
- Compression child-session rotation remains wholly owned by Hermes. After its atomic parent/child publication, the plugin follows the public compression lineage and migrates the case before case-relevant context or tool work proceeds.

## Opening a case

Use `epistemic_model` with `operation: open`, `case_id`, and `top_unknown`.
Decision, stopping condition, grounding, mechanisms, alternatives, and plan are
optional and receive conservative empty/default values when omitted.

Claims use stable IDs and `evidence_event_ids`. A claim that cites evidence must also declare `authority_domain`; the cited events must belong to the same broad authority class (`user_*` or external/system/empirical). The normalized claim records `grounding_authority_domains`, so attributed user testimony cannot be promoted into an externally grounded proposition. Existing evidence can be supplied through `initial_evidence` with temporary `I1`, `I2`, ... IDs; the harness turns those into Timeline event IDs. Claims with no evidence are explicitly normalized to `epistemic_status: hypothesis`, never silently presented as grounded.

`epistemic_probe` accepts the existing dictionary `tool_args` for compatibility. For a nonempty nested payload through provider schemas, use `tool_args_json` with a JSON object string. It is authoritative when present; a legacy `tool_args` value may be absent, `null`, or `{}`, while a nonempty legacy dictionary must canonically agree. Malformed JSON, duplicate keys, non-object values, and non-finite numbers are rejected before the probe commit. The server stores and hashes the decoded object, so the ordinary action's arguments can be checked against the same commitment.

`retrieved_prior_lessons` is derived by the server from explicitly named, closed, replay-valid source cases and cannot be supplied during open or revise. `prior_case_ids` is open-time-only metadata; revise rejects attempts to change it and directs the agent to open a new bounded case when a new import set is needed. `applied_prior_lessons` remains the separate, agent-authored record.

## Portfolio inspection

Use `epistemic_model` with `operation: list` when the user asks what cases exist, when
auditing unfinished work, or before resuming a case whose ID is not known. The
operation is read-only: it does not update case timestamps, add Timeline events,
change the current session's active case, or inject other sessions' cases.

The default returns at most 50 newest-updated compact summaries across all
statuses. Optional `status_filter` values are `active`, `paused`, or `closed`;
`session_id_filter` matches both current and historical owners; `limit` may be
1–200. Each result includes the case ID, lifecycle status, created/updated times,
one-line decision or top unknown, current and historical session IDs, event
count, pending-probe status, and the existing epistemic-staleness flag. Relay a
human-readable subset rather than dumping the complete JSON portfolio.

`active` means the case is currently bound to one Hermes session. The store
permits at most one active case per session, and pre-LLM context injects only
that session's case. Other sessions' active cases never cross-inject. Paused and
closed cases inject nothing and impose no tool restrictions. Session reset and
subagent termination already pause unfinished cases; compression migrates the
binding. Do not infer abandonment merely from age and do not auto-close or
auto-pause from `list`: use `updated_at` and pending-probe status to identify a
case for explicit review, then `show`, `pause`, `resume`, or `close` as warranted.

The field `epistemically_stale` means a material probe mismatch still requires
consequential model revision. It does not mean "old" or "inactive."

## Probe cycle

1. For a high-value uncertainty, optionally call `epistemic_probe(commit)` with purpose, unknown/goal, tool name/arguments, predicted outcomes, what would change the belief, rationale, and authority domain. Add `claim_id` to bind the probe to one existing `state_grounding`, `mechanisms`, or `alternatives` claim; the server stores an immutable `claim_snapshot` and `model_version`.
2. Call the intended ordinary Hermes tool by itself. Argument drift is recorded rather than blocked; parallel same-name calls cannot share one probe.
3. `epistemic_probe(compare)` with `match`, `mismatch`, or `unresolved`. A claim-bound comparison must also state `belief_change`: `strengthened`, `weakened`, `narrowed`, `unchanged`, or `unresolved`. This is separate from observation disposition.
4. If mismatch is material, call `epistemic_model(revise)` with a consequential update and `addresses_event_ids` containing the mismatch event.
5. If execution arguments did not match the commitment, explain that drift with `argument_drift_reason` at compare, or use `epistemic_probe(discard)` with a reason. A legacy record with no match flag is reported as unknown, never silently matched.
6. Use `epistemic_probe(discard)` with a reason if the commitment is mistaken or stranded.

Ordinary search, recall, file reads, code, and delegation do not require this
cycle and remain available while the case is active.

### A minimal uncertainty-preserving cycle

An unresolved prior can still support a useful check without manufacturing
certainty. Suppose the active case contains claim `M1` with
`epistemic_status: "unresolved"`:

```json
{"operation":"commit","claim_id":"M1","purpose":"learn",
 "unknown_or_goal":"Whether the source supports M1",
 "tool_name":"read_file","tool_args":{"path":"source.txt"},
 "predicted_outcomes":[{"outcome":"support"},{"outcome":"contrary or unavailable"}],
 "would_change_belief":"The source is incompatible with M1",
 "why_this_action":"The source is the designated check",
 "authority_domain":"external_source"}
```

Run the ordinary `read_file` call, then compare it. A successful but
non-discriminating observation can be recorded as
`disposition: "match"` and `belief_change: "unchanged"`; an inaccessible
observation can use `disposition: "unresolved"` and
`belief_change: "unresolved"`. Finish with `epistemic_model(close)` and
`outcome: "unresolved"` when the question is still open. A failed,
unavailable, interrupted, or legacy-status observation records an execution
failure or unknown status, but does not support the external proposition it was
meant to inspect. A search result remains a lead rather than full-text support.

For contrary evidence, compare with `disposition: "mismatch"`,
`belief_change: "weakened"`, and `material: true`, then revise the affected
claim and dependent plan. A legitimate repair may keep the same claim text but
downgrade `grounded` to `hypothesis` or `hypothesis` to `unresolved`, with a
reason and the mismatch event ID. The harness does not automatically mutate a
claim from `belief_change`; the normal explicit `revise` call records the
update. Metadata, whitespace, or a status upgrade alone does not repair stale
state.

The server validates the authoritative on-disk case snapshot before every
existing-case write, including evidence recording, revision, transition,
resume, and crash recovery. This prevents a sibling operation from turning a
corrupt snapshot into a newly hashed state. `epistemic_model(show)` with an
explicit paused or closed `case_id` is read-only, so a parent with another
active case can inspect a settled child without pausing or rebinding either
case. An active case owned by another session remains inaccessible.

Historical calibration boundary: existing closure records are not rewritten or
recalculated. New display-exposure events retain their execution status and point
through same-probe `supersedes_event_id` links to the raw observation. A failed
execution remains non-supporting through every display wrapper; missing, invalid,
or cross-probe links, and an error representation after a successful execution,
are treated as unknown rather than positive source support. When a legacy
observation is encountered in a later calibration, its missing execution status
is treated conservatively as unknown rather than being asserted to have
succeeded.

## Delegated factual inquiry

A focused delegated inquiry may own a case in the worker session, or the
current session may own the case; choose explicitly based on the workflow.
Announce the scope and ownership before opening it. The worker may use ordinary
search, recall, reads, or terminal/code calls and reserve committed probes for
observations where prediction-before-evidence adds value. A final report should
include the case ID when one was used, answer, qualitative confidence, scope or
as-of date, strongest evidence, remaining alternatives, and what would change
the answer.

If a subagent stops while its child session still owns an active case, the
`subagent_stop` hook pauses that case without discarding any pending probe.
Closed child cases are left unchanged. The caller can inspect a named settled
case with `epistemic_model(show)`.

For a formal user question, commit the existing `clarify` tool and use one of:

- `user_preference`
- `user_intent`
- `private_context`
- `user_decision`
- `user_testimony`

`user_testimony` remains attributed testimony and does not become external empirical truth. The harness enforces the declared authority class; it cannot semantically inspect whether arbitrary natural-language claim text truthfully describes that declaration.

## Closure and transfer

`epistemic_model(close)` requires an outcome, summary, and explicit transfer review. A `resolved` closure also requires at least one evidence-linked `grounded` claim. The closure records a derived calibration:

- `hypothesis`: no evidence-linked grounded claim, or every cited probe observation is failed, unavailable, interrupted, or legacy-unknown; cannot close as resolved
- `provisional`: a grounded claim relies only on snippets, abstracts, or legacy search-result leads
- `supported`: a grounded claim cites inspected partial/full text, data, or another direct typed observation
- `mixed`: grounded claims have different support levels, or an ungrounded hypothesis remains among the asserted claims (state_grounding, mechanisms) alongside grounded claims. Unexplored alternatives are the option set and do not by themselves cap the label

None of `provisional`, `mixed`, or `supported` means confirmed, proven, or certain.

Transfer decisions are:

- `none`
- `candidate`
- `explicit_user_correction`
- `repeated_pattern`

`candidate` must cite exactly the case being closed. `explicit_user_correction` additionally cites typed user-authority Timeline events. `repeated_pattern` requires at least two distinct, closed, replay-valid source cases that record the same lesson and scope, and its calibration cannot exceed the weakest source case. Each closure records at most one lesson. When a later case does not explicitly name `prior_case_ids`, the plugin scans replay-valid closed lessons and retrieves at most three with bounded lexical relevance to the new model, including each lesson's calibration. Retrieved lessons remain provenance-bearing candidates in `retrieved_prior_lessons`; they are never auto-applied, never ground a current claim by themselves, and never become stronger merely by repetition. Explicit application remains recorded in `applied_prior_lessons`, while promotion into Mnemosyne or a procedural skill remains an agent action under the existing memory/skill policies.

## Case Lookup and Runtime Diagnostics

Active and lifecycle-paused lookup uses a rebuildable process-local session index.
Each lookup checks case-file identity, size, modification time, and change time;
only changed files are parsed. The index is rebuilt after restart and observes
other processes' writes. It adds no durable cache or new transaction protocol.
Filesystem metadata scans still grow with portfolio size.

An unreadable case snapshot is excluded from unrelated sessions and reported by
ID in `epistemic_model(list).case_errors`. Cached ownership or Timeline session
attribution protects identifiable affected sessions: their lookup reports an error
rather than silently treating the damaged case as absent. A corrupt record with
no recoverable owner remains an explicit portfolio error. Journal and Timeline
integrity validation remain unchanged; this is not permission to ignore a corrupt
pending transaction.

The list tool also returns the responding PID and a code-generation label. The
historical `runtime-attestation.json` file is not current-process evidence and is
not refreshed by this standalone plugin. Runtime smoke tests exercise tool
availability through Hermes's direct/deferred catalog and dispatch the two
control tools through the native deferred-tool route.

## Limitations

Enforcement covers structured case/Timeline state, typed claim citations, and
explicitly committed probe capture. Ordinary tool calls are intentionally not
gated. The plugin does not formally validate ordinary assistant prose or
prevent a model-provider call; an uncited narrative answer remains outside the
executable policy boundary.

The plugin is optional, so capture failure after an external mutation cannot roll back that side effect and does not disable Hermes. The middleware framework returns the already-produced result without executing the tool twice; the affected epistemic case may require explicit inspection or repair.

The Timeline hash chain is tamper-evident, not cryptographically authenticated: a local actor with write access to the complete profile could rewrite both state and hashes. Likewise, authority compatibility and closure calibration are structural contracts, not semantic fact-checking of claim prose. Selective lesson retrieval is deliberately small and lexical; it can miss a conceptually related lesson that uses different language.

This implementation requires POSIX `fcntl` locks. It refuses to start on platforms without them rather than silently weakening cross-process resume and execution-lease guarantees. Replay is integrity-safe but not a linearizable multi-file read snapshot; a concurrent transition can make one diagnostic replay internally older/newer than another, while all writes remain CAS-checked and serialized.
