"""FastAPI application: one triage endpoint, health probes and Prometheus metrics."""

from __future__ import annotations

import logging
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, Response
from fastapi.responses import PlainTextResponse
from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

from agent_service import __version__
from agent_service.backends import get_backend
from agent_service.config import Settings, get_settings
from agent_service.graph import build_graph, initial_state
from agent_service.knowledge import KnowledgeBase
from agent_service.observability import (
    NODE_LATENCY,
    OUTCOMES,
    REQUEST_LATENCY,
    REQUESTS,
    configure_logging,
)
from agent_service.schemas import TriageRequest, TriageResponse

log = logging.getLogger("agent_service.api")


def create_app(settings: Settings | None = None) -> FastAPI:
    settings = settings or get_settings()
    configure_logging(settings.log_level)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        kb = KnowledgeBase.load_default()
        backend = get_backend(settings)
        app.state.kb = kb
        app.state.backend = backend
        app.state.graph = build_graph(backend, kb, clarify_threshold=settings.clarify_threshold,
                                      max_snippets=settings.max_snippets)
        log.info("startup backend=%s snippets=%d", backend.name, len(kb))
        yield

    app = FastAPI(title="Agent Service Kit", version=__version__, lifespan=lifespan)

    @app.middleware("http")
    async def request_context(request: Request, call_next):
        request_id = request.headers.get("x-request-id") or uuid.uuid4().hex[:16]
        request.state.request_id = request_id
        started = time.perf_counter()
        response: Response = await call_next(request)
        elapsed = time.perf_counter() - started
        route = request.scope.get("route").path if request.scope.get("route") else request.url.path
        REQUESTS.labels(route=route, status=str(response.status_code)).inc()
        REQUEST_LATENCY.labels(route=route).observe(elapsed)
        response.headers["x-request-id"] = request_id
        log.info("request_id=%s method=%s route=%s status=%d ms=%.1f", request_id,
                 request.method, route, response.status_code, elapsed * 1000)
        return response

    @app.get("/healthz", tags=["ops"])
    async def healthz() -> dict:
        return {"status": "ok", "version": __version__}

    @app.get("/readyz", tags=["ops"])
    async def readyz(request: Request) -> dict:
        ready = getattr(request.app.state, "graph", None) is not None
        return {"status": "ready" if ready else "starting", "backend": request.app.state.backend.name}

    @app.get("/metrics", tags=["ops"], response_class=PlainTextResponse)
    async def metrics() -> Response:
        return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)

    @app.post("/v1/triage", response_model=TriageResponse, tags=["triage"])
    async def triage(body: TriageRequest, request: Request) -> TriageResponse:
        graph = request.app.state.graph
        result = await graph.ainvoke(initial_state(body.text, body.customer_id))
        for timing in result.get("trace", []):
            NODE_LATENCY.labels(node=timing.node).observe(timing.ms / 1000)
        classification = result.get("classification")
        OUTCOMES.labels(outcome=result["outcome"],
                        category=classification.category if classification else "none").inc()
        return TriageResponse(
            request_id=request.state.request_id,
            outcome=result["outcome"],
            classification=classification,
            extraction=result.get("extraction"),
            reply=result["reply"],
            citations=result.get("citations", []),
            clarification_question=result.get("clarification_question"),
            guard=result.get("guard"),
            trace=result.get("trace", []),
        )

    return app


app = create_app()
