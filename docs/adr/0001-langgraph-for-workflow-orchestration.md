# ADR 0001: LangGraph state graph for the triage workflow

**Status:** accepted · **Date:** 2026-09-23

## Context

The service has to run a multi-step workflow (classify → extract → retrieve → draft → guard) with
two branches: ask a clarifying question when the classification is uncertain, and escalate to a
human when the output guard rejects a draft. The steps must be observable one by one, and the
workflow must be testable without a model.

## Decision

Model the workflow as a `langgraph.graph.StateGraph` with a typed state (`TriageState`). Each step
is a plain Python function that returns only the fields it changed; branching is expressed as
conditional edges; per-node timings are accumulated through an `operator.add` reducer on `trace`.

## Consequences

- The graph is compiled once at startup and invoked per request with `ainvoke`; sync nodes run in a
  worker thread, so the API stays async.
- Adding a step (for example a retrieval re-ranker) is a new node plus two edges; the API does not
  change.
- LangGraph's checkpointers and human-in-the-loop interrupts can be added later without rewriting
  nodes, which is the main reason to prefer a graph over hand-written control flow.
- The dependency is one more library to keep current; pinned loosely (`langgraph>=0.2`) and exercised
  by the test-suite on every commit.
