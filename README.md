> **Provenance:** AI-assisted contribution maintained by Epistemic Harness contributors. AI-assisted documentation revision by Luftballon, 2026-10-29.

# Epistemic Harness for Hermes

Epistemic Harness helps an AI agent record what it expects **before a selected investigation step**, then compare the result with that expectation and decide whether to change its view. It keeps a local record of the question, selected checks, captured observations, and the agent's revisions. The record is selective, and capture is best-effort.

Suppose an agent is investigating a service failure. It suspects a configuration change caused an error. Without a record of that suspicion, it can read the logs and build an explanation around whatever it happens to find. The harness gives it a way to record its prediction first, then account for what the logs show.

This is useful when an observation could change a decision: debugging, checking a source, comparing explanations, or deciding what information to seek next. It is less useful for routine work that does not involve a meaningful uncertainty.

## What “epistemic” means

*Epistemic* concerns knowledge and how we justify what we believe. Here, the question is: **what did the agent record before a check, what did it observe, and what did it do with the observation?**

The workflow borrows a scientific habit: state an expectation before a test and consider what would count against it. It is similar in spirit to preregistration, but it is not a formal, independent research preregistration. The agent may already have seen relevant evidence, and the same agent can propose the test and interpret the result. The plugin does not establish that a test is sound or that a conclusion is true.

## A worked example

You ask: “Did the latest configuration change cause this service failure?” The agent can open a **case**, a record for this question, and proceed as follows:

1. It records a working hypothesis: the deployment introduced a specific error. It plans to inspect logs covering both before and after the deployment.
2. Before reading the logs, it commits a **probe** linked to that hypothesis. It predicts that the error appears after deployment and does not appear in the inspected logs before it.
3. It calls the ordinary tool to read the logs. The plugin attempts to capture the result for the committed probe.
4. The agent compares the result with its prediction. If the same error appears before deployment, the observation contradicts the prediction that the error was introduced then. The agent records a mismatch and judges whether it is material to the case.
5. If the mismatch is material, the plugin marks the linked claim and dependent plan items as needing revision. The agent must address them before closing the case. It might revise the hypothesis and examine other explanations.
6. The agent summarizes what the logs show and can close the case as unresolved. The earlier error does not rule out the change contributing to the later failure. A matching observation would not prove causation either; an additional check, such as a rollback or a comparison of error rates, could help investigate it.

A stopping condition describes the intended scope of the inquiry. It guides the agent; the plugin does not independently determine whether the stated condition is scientifically appropriate or has been met.

## Install and try it

You need Hermes Agent `>=0.21,<1.0` on Linux or macOS. The case store requires POSIX `fcntl` file locking and refuses to initialize where that facility is unavailable. The plugin has no additional runtime Python dependencies. Install by name from the Hermes plugin catalog:

```bash
hermes plugins install epistemic-harness
hermes plugins enable epistemic-harness
hermes plugins list
```

The install command may offer to enable the plugin; the separate enable command is also available. Check that `epistemic-harness` appears as enabled. If you use the Hermes gateway, restart it to ensure that the tools and hooks are loaded; for a command-line session, start a fresh session. A listing confirms installation and configuration, not that an inquiry has been carried out.

Ask your agent:

> Use Epistemic Harness to investigate whether the configuration change caused the service failure. Record your hypothesis and prediction before checking the logs. Inspect the relevant change and logs, and report an unresolved result if they do not settle the question. Show me the case afterward.

You can explicitly request an inquiry, or the agent can announce one before opening it, stating its question, scope, and stopping condition. The bundled skill `epistemic-harness:epistemic-inquiry` guides the agent through the procedure. Ask the agent to show the case and inspect the Timeline file to see which commitments, observations, and comparisons were recorded. A closed case alone does not establish that a probe was performed.

## How a probe works

A case holds the question, the agent's current working view, alternatives, evidence links, and plan. A probe records a particular planned tool call, expected outcomes, and what would change the agent's view. It may be tied to a specific claim or to an uncertainty.

A probe does **not** execute the tool. After committing it, the agent calls the ordinary Hermes tool separately. The plugin watches for the next call with the committed tool name and attempts to capture its result. A call to a different tool can proceed without being captured as that probe. Concurrent calls using the committed tool cannot share the same probe. The plugin compares the actual arguments with the committed arguments: a difference is recorded, not generally blocked, and the agent must explain it when comparing the result or discard the probe. Capture is best-effort; a failed or interrupted call is not evidence that the external claim is true.

The **agent** judges whether the observation matches the prediction, whether it changes the working view, and whether a mismatch is material. The plugin records those judgments and applies structural rules to the case. It does not independently decide whether the agent's interpretation is correct. A matching result can leave the agent's view unchanged; a failed or uninformative check can leave it unresolved.

The record is selective. Ordinary searches, reads, and other work outside committed probes remain available and need not appear in the case. This is a discipline aid for a cooperating agent, not a complete surveillance log or an adversarial audit of its reasoning.

## What the components do

- **Case state (`epistemic_model`):** opens and maintains a case, including its claims, alternatives, evidence, plan, and conclusion. It also lists, shows, revises, pauses, resumes, and closes cases. Its `record_evidence` operation can add an assessed evidence descriptor from an ordinary tool; a pending probe must first be compared or discarded.
- **Probe (`epistemic_probe`):** records a planned check, then records the agent's comparison of the observation with its prediction. A mistaken or stranded probe can be discarded with a reason; an interrupted execution can be recovered without automatically retrying it.
- **Timeline:** records case events, including probe commitments, observations, comparisons, and revisions. The current case and its history are stored locally.
- **Replay:** reads the recorded case and checks its integrity and consistency. It does **not** rerun tool calls. For example, replay checks whether a recorded material mismatch has been addressed before closure.
- **Optional method consultation (`epistemic_methods`):** asks a hosted TypeSafe/Jev model to rank candidate investigation methods supplied by the agent and estimate whether another round would change the decision. It is off by default. The recommendation is advice, not evidence.

An active case is bound to one Hermes session, and a session can have one active case at a time. Each case can have one pending probe. Paused or closed cases do not inject case context; in certain session-lifecycle situations a paused case can automatically resume. A case can close as unresolved. Closing requires a summary and an explicit review of whether to retain a reusable lesson; the agent can choose `none` if there is no lesson. Closing as resolved also requires an evidence-linked grounded claim. A pending probe or an unrepaired material mismatch can prevent normal pause or closure; use the recovery, discard, or revision operations as appropriate.

When a new case opens without explicitly named prior cases, the plugin may retrieve up to three lexically relevant lessons from eligible earlier closed cases. The agent can instead name particular eligible prior cases. These lessons are suggestions, not validated matches: they are not automatically applied and do not by themselves ground a new claim. An earlier lesson can nevertheless influence an agent, so review its relevance.

## Inspecting the record

Ask the agent to list cases or show a named case. The show operation includes the case, a replay summary, and a brief; the full Timeline is in `events.jsonl`. You can inspect the files in the active Hermes profile. The default profile is usually `~/.hermes`; `$HERMES_HOME` identifies the active profile when set:

```text
$HERMES_HOME/epistemic-harness/
└── cases/<case-id>/
    ├── case.md          # current case state
    ├── events.jsonl     # Timeline events
    └── archive/         # snapshots and larger captured artifacts
```

The full profile-level directory contains additional state, including a commitment key and session records. Do not assume that copying `cases/` alone makes a complete backup. See [Privacy notes](https://github.com/quangphucqp/epistemic-harness/blob/main/PRIVACY.md) and the [technical reference](TECHNICAL.md).

The Timeline uses a hash chain to help detect accidental or partial changes. It is not independently authenticated: a person or process with full write access to the profile can rewrite both the records and their hashes.

## Optional hosted method consultation

The plugin works locally without method consultation. To enable it, configure the following under your Hermes plugin settings and make `TYPESAFE_API_KEY` available through the profile's secret scope:

```yaml
plugins:
  entries:
    epistemic-harness:
      settings:
        jev:
          enabled: true
          model: jev-latest
          max_calls_per_case: 4
          timeout_s: 6
          allow_user_domain_state: false
```

Restart the gateway after changing the setting. When enabled, consultation is required before each committed probe and before closing a case, although the agent may choose a method other than the recommendation. Missing credentials, invalid configuration, or authentication errors block those decisions. Certain recorded service failures, a cooldown, or exhaustion of the attempt budget can permit the decision without advice. A reached input-token limit blocks rather than waives consultation. See the [technical reference](TECHNICAL.md#jev-consultation) for the rules and exceptions.

Each consultation sends a bounded, redacted selection of case text and the proposed methods to the configured HTTPS endpoint. The default TypeSafe endpoint uses your TypeSafe credits. Redaction is best-effort. Candidate method text is not filtered by authority domain, so do not put private information in it. See [Privacy notes](https://github.com/quangphucqp/epistemic-harness/blob/main/PRIVACY.md) before enabling this feature.

## Privacy and limitations

The plugin is local-only by default. Its records may include case text, tool arguments, and captured tool results; redaction does not make them safe to share. There is no plugin telemetry or self-updater. The optional method consultation is the plugin's only outbound network path.

Uninstalling the plugin does not delete its local case data. Review and back up `$HERMES_HOME/epistemic-harness/` before separately removing records you no longer need.

The plugin does not verify a claim, guarantee that the agent chose a good test, prove the agent had not seen the evidence, or validate all ordinary assistant prose. It does not establish that the workflow improves task accuracy. Ordinary unprobed tool calls remain available. Capture failure cannot undo an external side effect, and a case may need recovery or inspection. Replay checks the record but is not a single atomic snapshot during concurrent transitions.

The [technical reference](TECHNICAL.md) describes the tools, capture boundaries, recovery, storage, consultation rules, and further limitations. See also [Security](https://github.com/quangphucqp/epistemic-harness/blob/main/SECURITY.md), [Changelog](https://github.com/quangphucqp/epistemic-harness/blob/main/CHANGELOG.md), and [License](https://github.com/quangphucqp/epistemic-harness/blob/main/LICENSE).
