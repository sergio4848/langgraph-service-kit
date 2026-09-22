# Agent Service Kit

A reusable FastAPI service for LangGraph agents. It ships a complete example workflow (insurance
customer-message triage) and the parts that are usually missing from agent demos: typed inputs and
outputs for every model call, a deterministic backend so the whole graph runs in tests and CI
without an API key, an output guard, an evaluation harness with published numbers, Prometheus
metrics, a hardened Docker image and Kubernetes manifests.

```
POST /v1/triage {"text": "Water is still leaking from the ceiling, policy HM-4471920"}

START -> classify -> extract -> retrieve -> draft -> guard -> answered
              \-> clarify (low confidence)         \-> escalated (guard failed)
```

## What is in the box

| Concern | Where | Notes |
|---|---|---|
| Workflow | `src/agent_service/graph.py` | `langgraph.StateGraph`, typed state, conditional edges, per-node timings via a reducer |
| Model calls | `src/agent_service/backends.py` | `Backend` protocol; `RuleBackend` (deterministic) and `OpenAIBackend` (`with_structured_output`, any OpenAI-compatible endpoint) |
| Contracts | `src/agent_service/schemas.py` | Pydantic models for every step and for the API |
| Retrieval | `src/agent_service/knowledge/` | Small policy knowledge base, keyword retriever, LangChain `@tool` wrapper |
| Guard | `src/agent_service/guard.py` | Citations must exist, no banned promises, no policy numbers echoed, bounded length |
| API | `src/agent_service/api.py` | `/v1/triage`, `/healthz`, `/readyz`, `/metrics`, request ids, structured logs |
| Evaluation | `src/agent_service/evals/` | 24 labelled messages; accuracy, outcomes, citation rate, guard rejections, latency |
| Delivery | `Dockerfile`, `deploy/k8s/`, `.github/workflows/ci.yml` | Non-root multi-stage image, probes, resource limits, HPA, CI with lint, tests, eval and an image smoke test |
| Decisions | `docs/adr/` | Three ADRs |

## Quickstart

```bash
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -e ".[dev]"
uvicorn agent_service.api:app --reload
curl -s -X POST localhost:8000/v1/triage -H 'content-type: application/json' \
  -d '{"text":"You charged me twice this month, please refund EUR 84,50."}' | python -m json.tool
```

Default backend is `rules`, which needs no key. To run the same graph against a model:

```bash
export AGENT_BACKEND=openai OPENAI_API_KEY=sk-...   # optional: AGENT_OPENAI_BASE_URL, AGENT_OPENAI_MODEL
uvicorn agent_service.api:app
```

Actual response from the rules backend (trimmed only in the reply text):

```json
{
  "request_id": "7f1c2a9e4b3d1c0a",
  "outcome": "answered",
  "classification": {"category": "billing", "urgency": "low", "confidence": 0.9, "rationale": "keywords: charged, refund"},
  "extraction": {"policy_number": null, "incident_date": null, "amount_eur": 84.5, "product": null},
  "reply": "Thank you for contacting us about billing and payments. Premiums are collected by direct debit on the first working day of the month. [KB-004] When a policy is cancelled, the premium for the unused period is refunded pro rata within 10 working days ... [KB-005] We will check the payment records on your policy and confirm the outcome by e-mail.",
  "citations": ["KB-004", "KB-005"],
  "clarification_question": null,
  "guard": {"ok": true, "issues": []},
  "trace": [{"node": "classify", "ms": 0.04}, {"node": "extract", "ms": 0.02}, {"node": "retrieve", "ms": 0.06}, {"node": "draft", "ms": 0.02}, {"node": "guard", "ms": 0.03}]
}
```

Retrieval is category-aware: the classifier's verdict adds topic keywords to the query, which is
why a billing message pulls the payment and refund snippets rather than whatever shares the most
words with the customer's text.

## Tests and evaluation

```bash
ruff check . && pytest            # 14 tests, no network, < 1 s
agent-eval --backend rules        # writes evals/results/rules-<timestamp>.json
agent-eval --backend openai       # same dataset against the model backend
```

Rules backend on the bundled dataset (24 messages, 2026-09-23):

| Metric | Value |
|---|---|
| Category accuracy (answered messages) | 0.727 |
| Urgency accuracy | 0.818 |
| Outcomes | 22 answered, 2 clarify, 0 escalated |
| Answered replies with citations | 22 / 22 |
| Guard rejections | 0 |
| Latency p50 / p95 | 1.2 ms / 2.0 ms |
| Misclassified | e09, e11, e14, e20, e21, e23 |

Those six misses are the point of the harness: they show where keyword rules stop and where a model
backend has to earn its cost. The same command against `--backend openai` produces a comparable
report; keep both in `evals/results/` and diff them when changing prompts or models.

## Operations

- `GET /healthz` liveness, `GET /readyz` readiness (graph compiled, backend name).
- `GET /metrics` Prometheus: `agent_requests_total`, `agent_request_latency_seconds`,
  `agent_node_latency_seconds{node=...}`, `agent_triage_outcomes_total{outcome,category}`.
- Every response carries `x-request-id` (taken from the request header when present).
- Logs are single-line `key=value` records with the request id, route, status and latency.

### Docker

```bash
docker build -t agent-service-kit:local .
docker run --rm -p 8000:8000 -e AGENT_BACKEND=rules agent-service-kit:local
```

### Kubernetes

```bash
docker build -t <your-registry>/agent-service-kit:0.1.0 . && docker push <your-registry>/agent-service-kit:0.1.0
# point deploy/k8s/deployment.yaml `image:` at that tag, then:
kubectl create secret generic agent-service-secrets --from-literal=OPENAI_API_KEY=sk-...
kubectl apply -f deploy/k8s/service-and-config.yaml -f deploy/k8s/deployment.yaml
kubectl port-forward svc/agent-service 8000:80
```

The Deployment runs as a non-root user with a read-only root filesystem, has startup, readiness and
liveness probes on the two health endpoints, CPU/memory requests and limits, a rolling update with
zero unavailable pods, Prometheus scrape annotations and a CPU-based HorizontalPodAutoscaler
(2 to 6 replicas).

## Extending

- New step: add a node function in `graph.py` and wire two edges; the trace and metrics pick it up.
- New model provider: implement the three `Backend` methods; nothing else changes.
- Real retrieval: replace `KnowledgeBase.search` with a vector store call; keep returning `Snippet`s
  so the guard's citation check still works.
- Human in the loop: LangGraph checkpointers and interrupts can be added to the compiled graph
  without touching the nodes (see ADR 0001).

## License

MIT
