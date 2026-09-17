# AI-assisted contribution; maintained by Epistemic Harness contributors.
"""JSON schemas for the two epistemic-harness tools."""

# Shared model-facing constants keep both tool schemas aligned.
#
# ``operation`` is deliberately required and is validated again at the plugin
# handler boundary. The second check matters because a malformed request must
# not trigger case recovery or any other store access before it is rejected.
MODEL_OPERATIONS = [
    "open",
    "list",
    "show",
    "record_evidence",
    "revise",
    "pause",
    "resume",
    "close",
]
PROBE_OPERATIONS = ["commit", "compare", "recover", "discard"]
RESPONSE_DETAIL_SCHEMA = {
    "type": "string",
    "enum": ["full", "compact"],
    "default": "full",
    "description": (
        "Optional presentation detail. Defaults to full. Compact is a conservative "
        "projection only; use epistemic_model show with response_detail=full for audit."
    ),
}

# Shared evidence descriptor properties keep the model-facing schemas aligned.
# In particular, Hermes's runtime sanitizer preserves nested keys only when the
# object declares them instead of using an empty ``properties`` mapping.
EVIDENCE_DESCRIPTOR_PROPERTIES = {
    "summary": {"type": "string"},
    "source_ref": {"type": "string"},
    "source_role": {
        "type": "string",
        "enum": ["primary", "synthesis", "secondary", "unknown"],
    },
    "access_scope": {
        "type": "string",
        "enum": ["snippet", "abstract", "partial", "full", "data"],
    },
    "authority_domain": {
        "type": "string",
        "enum": [
            "user_preference",
            "user_intent",
            "private_context",
            "user_decision",
            "user_testimony",
            "external_source",
            "system_state",
            "empirical_test",
        ],
    },
    "locator": {"type": "string"},
    "limitation": {"type": "string"},
}

INITIAL_EVIDENCE_ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        **EVIDENCE_DESCRIPTOR_PROPERTIES,
    },
    # source_role/access_scope are conditional: external sources require them,
    # while typed user/system observations do not.
    "required": ["id", "summary", "source_ref", "authority_domain"],
}

EXTERNAL_EVIDENCE_ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        **EVIDENCE_DESCRIPTOR_PROPERTIES,
        "authority_domain": {
            "type": "string",
            "enum": ["external_source"],
        },
    },
    "required": [
        "summary",
        "source_ref",
        "source_role",
        "access_scope",
        "authority_domain",
    ],
}

# Single source for the claim-item shape used by state_grounding, mechanisms,
# alternatives, and updates: provider function-calling layers strip nested keys
# from objects declared with empty properties, so every claim key the harness
# reads (id, text, evidence_event_ids, authority_domain, epistemic_status) is
# declared here. Declared once to prevent drift between the four use sites.
CLAIM_ITEM_SCHEMA = {
    "type": "object",
    "properties": {
        "id": {"type": "string"},
        "text": {"type": "string"},
        "evidence_event_ids": {"type": "array", "items": {"type": "string"}},
        "authority_domain": {
            "type": "string",
            "enum": ["user_preference", "user_intent", "private_context", "user_decision", "user_testimony", "external_source", "system_state", "empirical_test"],
            "description": "Required for any claim citing evidence_event_ids; must match the cited events' authority class.",
        },
        "epistemic_status": {
            "type": "string",
            "enum": ["grounded", "hypothesis", "unresolved"],
            "description": "Defaults to grounded when evidence is cited, else hypothesis.",
        },
    },
    "required": ["id", "text"],
}

MODEL_SCHEMA = {
    "name": "epistemic_model",
    "description": (
        "Open, list, inspect, record evidence, revise, pause, resume, or close lightweight epistemic cases. "
        "Use after explicit user activation or after the agent announces a bounded inquiry with "
        "its scope and stopping condition; do not auto-open cases or wrap every retrieval. "
        "Opening requires only case_id and top_unknown; richer model fields are optional. "
        "Supply operation on every call and never infer it from the payload. Always supply an explicit operation; ordinary tools remain available. "
        "A material mismatch must be addressed before case closure. Resolved closure "
        "requires an evidence-linked grounded claim and derives claim-level and overall "
        "calibration from what was actually inspected rather than asking the agent for a score. "
        "A show with an explicit paused or closed case_id is read-only and may inspect a settled "
        "child while another case remains active in the caller's session. A delegated focused "
        "inquiry may own a case in the worker session; keep ownership and the bounded scope explicit. "
        "Use unresolved when the evidence does not settle the question."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": MODEL_OPERATIONS,
                "description": (
                    "Required on every call. Choose one explicit operation; never infer it "
                    "from the other fields. Example: {\"operation\":\"show\",\"case_id\":\"case-1\"}."
                ),
            },
            "response_detail": RESPONSE_DETAIL_SCHEMA,
            "status_filter": {
                "type": "string",
                "enum": ["active", "paused", "closed"],
                "description": "For list: return only cases with this lifecycle status.",
            },
            "session_id_filter": {
                "type": "string",
                "description": "For list: match any current or historical owning session ID.",
            },
            "limit": {
                "type": "integer",
                "minimum": 1,
                "maximum": 200,
                "description": "For list: maximum compact summaries to return; defaults to 50.",
            },
            "case_id": {"type": "string", "description": "Stable lowercase case ID."},
            "decision": {"type": "string"},
            "stopping_condition": {"type": "string"},
            "state_grounding": {
                "type": "array",
                "description": (
                    "Claims with stable id, text, and evidence_event_ids. Each item must declare id and text; "
                    "any claim citing evidence must also declare an authority_domain compatible with those events."
                ),
                "items": CLAIM_ITEM_SCHEMA,
            },
            "mechanisms": {
                "type": "array",
                "description": (
                    "Claims with stable id, text, and evidence_event_ids. Each item must declare id and text; "
                    "any claim citing evidence must also declare an authority_domain compatible with those events."
                ),
                "items": CLAIM_ITEM_SCHEMA,
            },
            "alternatives": {
                "type": "array",
                "description": (
                    "Claims with stable id, text, and evidence_event_ids. Each item must declare id and text; "
                    "any claim citing evidence must also declare an authority_domain compatible with those events."
                ),
                "items": CLAIM_ITEM_SCHEMA,
            },
            "top_unknown": {"type": "string"},
            "current_plan": {
                "type": "array",
                "description": "Plan items require id and action; depends_on contains known claim or plan IDs.",
                "items": {
                    "type": "object",
                    "properties": {
                        "id": {"type": "string"},
                        "action": {"type": "string"},
                        "depends_on": {"type": "array", "items": {"type": "string"}},
                    },
                    "required": ["id", "action"],
                },
            },
            "prior_case_ids": {"type": "array", "items": {"type": "string"}},
            "retrieved_prior_lessons": {
                "type": "array",
                "description": (
                    "Read-only lesson candidates selected from relevant, replay-valid "
                    "closed cases. Treat them as provenance-bearing prompts, not truth."
                ),
                "items": {"type": "object"},
            },
            "applied_prior_lessons": {"type": "array", "items": {"type": "string"}},
            "initial_evidence": {
                "type": "array",
                "description": (
                    "Typed observations already available when opening the case. Give each "
                    "an I1-style id, summary, source_ref, authority_domain, and, for external "
                    "sources, source_role and access_scope, plus a limitation for "
                    "snippet, abstract, or partial access. Claims may cite those temporary IDs "
                    "and the harness converts them to Timeline event IDs."
                ),
                "items": INITIAL_EVIDENCE_ITEM_SCHEMA,
            },
            "evidence": {
                "type": "array",
                "description": (
                    "For record_evidence: always pass a JSON array, even for one external "
                    "observation. Do not include recalled, user, or system-state observations. "
                    "Each requires summary, source_ref, source_role (primary, synthesis, "
                    "secondary, unknown), access_scope (snippet, abstract, partial, full, data), "
                    "and authority_domain=external_source. locator is optional; limitation is "
                    "required for snippet, abstract, or partial access. Use returned "
                    "evidence_event_ids verbatim in later claims; never invent or shorten them."
                ),
                "items": EXTERNAL_EVIDENCE_ITEM_SCHEMA,
                "minItems": 1,
            },
            "updates": {
                "type": "object",
                "description": (
                    "Model fields to replace. A revise call may alternatively provide "
                    "those fields directly at the top level."
                ),
                "properties": {
                    "decision": {"type": "string"},
                    "stopping_condition": {"type": "string"},
                    "top_unknown": {"type": "string"},
                    "state_grounding": {"type": "array", "items": CLAIM_ITEM_SCHEMA},
                    "mechanisms": {"type": "array", "items": CLAIM_ITEM_SCHEMA},
                    "alternatives": {"type": "array", "items": CLAIM_ITEM_SCHEMA},
                    "current_plan": {
                        "type": "array",
                        "items": {
                            "type": "object",
                            "properties": {
                                "id": {"type": "string"},
                                "action": {"type": "string"},
                                "depends_on": {"type": "array", "items": {"type": "string"}},
                            },
                            "required": ["id", "action"],
                        },
                    },
                    "prior_case_ids": {
                        "type": "array",
                        "items": {"type": "string"},
                        "description": (
                            "Open-time-only metadata; revise rejects this field. Open a new "
                            "bounded case when a new import set is needed."
                        ),
                    },
                    "retrieved_prior_lessons": {
                        "type": "array",
                        "items": {"type": "object"},
                        "description": (
                            "Server-managed derived metadata; revise rejects caller-supplied "
                            "values."
                        ),
                    },
                    "applied_prior_lessons": {"type": "array", "items": {"type": "string"}},
                },
            },
            "reason": {"type": "string"},
            "addresses_event_ids": {"type": "array", "items": {"type": "string"}},
            "outcome": {
                "type": "string",
                "enum": ["resolved", "unresolved", "stopped", "replaced"],
            },
            "summary": {"type": "string"},
            "transfer": {
                "type": "object",
                "description": (
                    "Explicit closure transfer review. Record at most one compact, reusable "
                    "mechanism or caution. decision is none, candidate, "
                    "explicit_user_correction, or repeated_pattern. Non-none decisions "
                    "also require lesson, scope, and evidence_case_ids. Explicit user "
                    "corrections require typed evidence_event_ids; repeated patterns "
                    "require at least two replay-valid closed source cases. A resolved case "
                    "normally records one candidate lesson; use none when nothing is reusable."
                ),
                "properties": {
                    "decision": {
                        "type": "string",
                        "enum": ["none", "candidate", "explicit_user_correction", "repeated_pattern"],
                    },
                    "lesson": {"type": "string"},
                    "scope": {"type": "string"},
                    "evidence_case_ids": {"type": "array", "items": {"type": "string"}},
                    "evidence_event_ids": {"type": "array", "items": {"type": "string"}},
                    "calibration": {"type": "string"},
                },
                "required": ["decision"],
            },
        },
        "required": ["operation"],
    },
}

PROBE_SCHEMA = {
    "name": "epistemic_probe",
    "description": (
        "Optionally commit one high-value prediction before an external action, then compare "
        "its captured observation. Ordinary tools never require a probe. An optional claim_id "
        "binds a commit to an existing claim and snapshots it server-side; a bound compare "
        "must separately state belief_change. Argument drift is recorded rather than blocked, "
        "but an explicit mismatch needs an explanation at compare or the probe may be discarded. "
        "Use discard to clear a mistaken or stranded probe, "
        "or recover only after an interrupted execution whose live-action lease has ended. "
        "Always supply an explicit operation."
    ),
    "parameters": {
        "type": "object",
        "properties": {
            "operation": {
                "type": "string",
                "enum": PROBE_OPERATIONS,
                "description": (
                    "Required on every call. Choose one explicit operation; never infer it "
                    "from the other fields. Example: {\"operation\":\"commit\",\"tool_name\":\"read_file\"}."
                ),
            },
            "response_detail": RESPONSE_DETAIL_SCHEMA,
            "purpose": {"type": "string", "enum": ["learn", "advance"]},
            "unknown_or_goal": {"type": "string"},
            "tool_name": {"type": "string"},
            "tool_args": {
                "type": "object",
                "description": "Compatibility dictionary route; prefer tool_args_json for nonempty arguments.",
            },
            "tool_args_json": {
                "type": "string",
                "description": (
                    "Recommended provider-safe route for a nonempty tool-argument JSON object. "
                    "The server parses it before committing, rejects malformed JSON, duplicate "
                    "keys, non-object values, and non-finite numbers, and treats it as authoritative."
                ),
            },
            "claim_id": {
                "type": "string",
                "description": (
                    "Optional stable id of one existing state_grounding, mechanisms, "
                    "or alternatives claim. The server snapshots that claim at commit."
                ),
            },
            "would_change_belief": {
                "type": "string",
                "description": (
                    "Required for commit: the observation that would count against the "
                    "current claim or materially redirect judgment."
                ),
            },
            "predicted_outcomes": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": {
                        "outcome": {"type": "string"},
                        "meaning": {
                            "type": "string",
                            "description": "Optional implication for the case.",
                        },
                    },
                    "required": ["outcome"],
                },
            },
            "why_this_action": {"type": "string"},
            "authority_domain": {
                "type": "string",
                "enum": [
                    "user_preference",
                    "user_intent",
                    "private_context",
                    "user_decision",
                    "user_testimony",
                    "external_source",
                    "system_state",
                    "empirical_test",
                ],
            },
            "disposition": {
                "type": "string",
                "enum": ["match", "mismatch", "unresolved"],
            },
            "material": {"type": "boolean"},
            "rationale": {"type": "string"},
            "affected_claim_ids": {"type": "array", "items": {"type": "string"}},
            "belief_change": {
                "type": "string",
                "enum": ["strengthened", "weakened", "narrowed", "unchanged", "unresolved"],
                "description": (
                    "Required when comparing a claim-bound probe. This is the model's "
                    "belief update, separate from match/mismatch observation disposition."
                ),
            },
            "argument_drift_reason": {
                "type": "string",
                "description": "Required to compare an explicitly mismatched execution argument.",
            },
            "reason": {"type": "string"},
            "evidence": {
                "type": "object",
                "description": (
                    "Evidence descriptor for a successful external-source comparison: summary, "
                    "source_ref, source_role, access_scope, optional locator, and a limitation "
                    "when access_scope is snippet, abstract, or partial."
                ),
                "properties": EVIDENCE_DESCRIPTOR_PROPERTIES,
            },
        },
        "required": ["operation"],
    },
}
