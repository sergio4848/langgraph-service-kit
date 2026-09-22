# ADR 0002: Backend abstraction with a deterministic implementation

**Status:** accepted · **Date:** 2026-09-23

## Context

Tests, CI and local development should not need an API key, must be fast, and must give the same
result on every run. At the same time the real service has to call a chat model with structured
outputs.

## Decision

Nodes never touch a chat model directly. They call a `Backend` protocol with three methods
(`classify`, `extract`, `draft`) that each return a Pydantic model. Two implementations exist:

- `RuleBackend`: keyword rules and regular expressions. Deterministic, sub-millisecond, no network.
- `OpenAIBackend`: `ChatOpenAI(...).with_structured_output(<PydanticModel>)` against any
  OpenAI-compatible endpoint.

The backend is selected with `AGENT_BACKEND`. The test-suite and CI use `rules`; the Kubernetes
ConfigMap sets `openai`.

## Consequences

- The graph, the guard, the API and the evaluation harness are tested end to end in CI with no
  secrets, and the same evaluation dataset can be run against the model backend to compare numbers.
- The rule backend is a real fallback, not a mock: it is what the service serves when no key is
  configured, and its evaluation score is published so the gap to the model backend is visible.
- Every model call is typed. Prompt changes cannot silently change the API contract.
