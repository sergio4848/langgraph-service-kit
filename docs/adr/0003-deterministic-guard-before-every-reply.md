# ADR 0003: Deterministic output guard before every reply

**Status:** accepted · **Date:** 2026-09-23

## Context

A drafted reply can cite policy text that does not exist, promise a payout, echo a customer's
policy number, or run far too long. Whether the draft came from rules or from a model, none of
these may reach a customer.

## Decision

A `guard` node written in plain code (no model) runs after every draft and checks: every citation
exists in the knowledge base, no banned phrases, no policy-number pattern in the reply, bounded
length, non-empty text. If the guard fails, the graph routes to `escalate`, which returns a fixed
hand-off message and the outcome `escalated`; the offending draft and the issues stay in the state
for logging and evaluation.

## Consequences

- The guard is reproducible and cheap, so it can be strict.
- Guard rejections are counted in the evaluation report and in Prometheus (`outcome="escalated"`),
  which turns "how often does the model misbehave" into a number that can be tracked per model.
- The guard only checks what code can check. Factual accuracy against the snippets is the job of
  the evaluation dataset, not of this node.
