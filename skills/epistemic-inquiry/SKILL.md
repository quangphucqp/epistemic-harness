---
name: epistemic-inquiry
description: Use for a bounded uncertain inquiry with an explicit decision and stopping condition.
---

# Epistemic inquiry

This skill teaches a bounded workflow for decisions that could change after an
observation:

```text
announce or activate → open → claim-bound commit → actual tool read
→ compare → revise or record non-update → close → full readback
```

It is a procedure aid, not a semantic truth checker. It reduces avoidable drift
in the inquiry procedure; it does not make a claim true, improve reasoning
accuracy by itself, or validate ordinary assistant prose.

The JSON blocks in this skill are schematic payloads, not a complete runnable
sequence. Use each operation as a separate actual tool call and substitute IDs,
paths, and returned values from the active case.

## Activate deliberately

Use the harness only after either:

1. the user explicitly asks for or activates a bounded inquiry; or
2. the agent announces a bounded inquiry before opening it, including the
   decision/question, scope, and stopping condition.

For the second route, announce plainly, for example:

> I am opening a bounded inquiry about whether the source supports this
> mechanism. I will inspect the designated source, compare the observation with
> the prediction, and stop with an unresolved result if it does not discriminate.

Do not auto-open a case, force every read/search/retrieval into a probe, or turn
mechanical work into bookkeeping. Ordinary tools remain available outside and
inside an active case. A meaningful investigative decision is the unit of use.

The bundled skill is registered through Hermes's public plugin API and is loaded
explicitly as `epistemic-harness:epistemic-inquiry`; it is not copied into the
user's global skills directory.

## 1. Open a case

Supply `operation` explicitly on every tool call. Do not infer it from the
payload shape. Preserve any user-provided decision criterion and
`stopping_condition` verbatim, including numeric thresholds; do not paraphrase
or silently round them.

```json
{
  "operation": "open",
  "case_id": "source-mechanism-check",
  "decision": "Use the mechanism only if the source supports the stated threshold of 0.80",
  "stopping_condition": "Stop after the designated source is inspected and the threshold is supported, contradicted, or remains unresolved",
  "top_unknown": "Whether the designated source supports the mechanism",
  "state_grounding": [
    {"id": "S1", "text": "The mechanism is a working hypothesis"}
  ],
  "mechanisms": [
    {"id": "M1", "text": "The source supports the mechanism at the stated threshold"}
  ],
  "alternatives": [
    {"id": "A1", "text": "The source is unavailable or supports a different threshold"}
  ],
  "current_plan": [
    {"id": "P1", "action": "Inspect the designated source", "depends_on": ["M1"]}
  ],
  "response_detail": "compact"
}
```

Compact output is optional presentation only. It keeps receipts, event IDs,
replay/accounting state, stale state, pending probes, closure calibration, and
claim snapshots. If it omits retrieved lesson prose, it marks the omitted field
paths and gives an exact full readback call.

Do not provide `retrieved_prior_lessons`; it is server-managed. To import prior
context, provide `prior_case_ids` only when opening a new case. Each named source
must be closed and replay-valid, including its event chain, snapshot, and captured
artifacts. Invalid explicit sources fail the open without creating a destination
case. During revise, `prior_case_ids` is open-time-only; open a new bounded case
if a different import set is needed. `applied_prior_lessons` is the separate
agent-authored field and may be revised explicitly.

## 2. Commit one claim-bound check

For a high-value uncertainty, bind one probe to the existing claim. Record what
would change the belief and a falsifiable prediction. The server keeps the
claim snapshot; do not provide or invent `claim_snapshot` or `model_version`.

```json
{
  "operation": "commit",
  "claim_id": "M1",
  "purpose": "learn",
  "unknown_or_goal": "Whether the designated source supports M1",
  "tool_name": "read_file",
  "tool_args_json": "{\"path\":\"designated-source.txt\"}",
  "predicted_outcomes": [
    {"outcome": "threshold supported", "meaning": "Retain M1 for this scope"},
    {"outcome": "threshold contradicted or unavailable", "meaning": "Weaken or leave M1 unresolved"}
  ],
  "would_change_belief": "The source contradicts the stated threshold",
  "why_this_action": "The designated source is the direct check",
  "authority_domain": "external_source",
  "response_detail": "full"
}
```

The recommended provider-safe route for nonempty nested arguments is
`tool_args_json`, a string containing one JSON object. The legacy dictionary
`tool_args` remains supported; when both are present, it may be absent, `null`, or
`{}`, while a nonempty dictionary must canonically agree with the decoded JSON.
Malformed JSON, duplicate keys, non-object values, and non-finite numbers are
rejected before commitment. The server hashes and redacts the decoded object.

## 3. Perform the actual tool read

Call the intended ordinary Hermes tool separately. Do not replace it with an
`epistemic_probe` call, and do not assume that a committed probe performs the
action. The native `post_tool_call` boundary captures the handler result before
`transform_tool_result` when it runs inside the active middleware scope; later
returned representations are recorded separately when that boundary is
available. The native capture scope is finalized on every middleware exit. A
lower-level `handle_function_call` whose post event arrives after middleware
return is recorded at the return boundary with unknown provenance, and a late
callback cannot upgrade it. A display-looking success is not raw support.

Native observer bookkeeping is best-effort and nonblocking under state, cache,
store, and profile lock contention. If a lock is unavailable, the action is
still returned once and the pending probe may need explicit `recover` or
`discard`; there is no background retry or redispatch. Lock budgets do not bound
filesystem stalls after a lock has been acquired.

```json
{"path": "designated-source.txt"}
```

The ordinary `read_file` call above is illustrative; use the actual tool and
arguments committed in the case. Argument drift is recorded rather than
blocked. If the arguments differ, explain the drift at compare or discard the
probe. A failed, unavailable, interrupted, or display-only error result is not
positive evidence for the external claim.

## 4. Compare observation and belief change separately

Observation disposition and belief change are different fields. A successful
observation can be `match` while the belief is `unchanged`; inaccessible
material can be `unresolved` with `belief_change: unresolved`.

```json
{
  "operation": "compare",
  "disposition": "match",
  "material": false,
  "rationale": "The source was inspected and is consistent with the prediction, but it does not establish more than this scope",
  "belief_change": "unchanged",
  "evidence": {
    "summary": "The designated source text was inspected",
    "source_ref": "local:designated-source.txt",
    "source_role": "primary",
    "access_scope": "full",
    "authority_domain": "external_source"
  },
  "response_detail": "compact"
}
```

Copy returned Timeline `evidence_event_ids` verbatim into later claims. Never
guess, shorten, or manufacture an event ID.

## 5. Revise after a material mismatch, or record non-update

A material mismatch marks affected claims and dependent plan items stale. Revise
consequentially and cite the exact mismatch event ID. Whitespace/no-op changes
or a status upgrade alone are not a repair. A justified conservative status
downgrade can repair a stale claim when it accurately records what the evidence
now supports—for example, `grounded` to `hypothesis` or `hypothesis` to
`unresolved`—and the revision gives a reason and cites the mismatch event.
Update or remove dependent plan items as needed.

The following is an alternative mismatch branch to the match comparison in
step 4, not a second comparison in the same sequence. First run the actual
compare call and copy its returned `event_id` verbatim. Replace the explicitly
labeled placeholder `<ACTUAL_MISMATCH_EVENT_ID>` below with that returned ID;
never invent an example ID.

```json
{
  "operation": "revise",
  "reason": "The inspected source contradicts the committed threshold, so the mechanism remains unresolved",
  "addresses_event_ids": ["<ACTUAL_MISMATCH_EVENT_ID>"],
  "updates": {
    "mechanisms": [
      {
        "id": "M1",
        "text": "The source does not establish the mechanism at the stated threshold",
        "epistemic_status": "unresolved"
      }
    ],
    "current_plan": []
  },
  "response_detail": "full"
}
```

If the observation is non-discriminating, record the comparison and make an
explicit non-update instead of manufacturing certainty. If a committed probe
is mistaken or stranded, use `operation: discard` with a reason. If execution
was interrupted, `operation: recover` records that the action may or may not
have completed; it is not an automatic retry.

## 6. Close without overstating the result

A case may close as `unresolved`. Closure requires a summary and an explicit
transfer review; choose `none` when there is no reusable lesson. Never call a
supported, provisional, mixed, or unresolved result confirmed, proven, or
certain.

For a mismatch whose repair still leaves the question open:

```json
{
  "operation": "close",
  "outcome": "unresolved",
  "summary": "The source contradicted the committed threshold, and the available evidence does not resolve the mechanism",
  "transfer": {"decision": "none"},
  "response_detail": "compact"
}
```

When compact output marks omitted lesson paths, perform the exact full audit
readback it returns, for example:

```json
{
  "operation": "show",
  "case_id": "source-mechanism-check",
  "response_detail": "full"
}
```

The full readback is the authoritative model-facing receipt. Durable case state,
Timeline events, captured tool results, and replay validation remain unchanged
by response detail.

## Scope and privacy

State is local and profile-scoped. The plugin can redact captured tool results,
but captured source and tool content may still be sensitive. Do not copy case
logs into a report or export them automatically. The plugin has no telemetry,
network request, self-updater, or credential-store access.

AI-assisted contribution; maintained by project contributors.
