"""Logging and Prometheus metrics. Kept in one place so nodes and routes stay free of it."""

import logging
import sys

from prometheus_client import Counter, Histogram

REQUESTS = Counter("agent_requests_total", "HTTP requests", ["route", "status"])
REQUEST_LATENCY = Histogram(
    "agent_request_latency_seconds", "End-to-end request latency", ["route"],
    buckets=(0.005, 0.01, 0.025, 0.05, 0.1, 0.25, 0.5, 1, 2.5, 5, 10),
)
NODE_LATENCY = Histogram(
    "agent_node_latency_seconds", "Latency per graph node", ["node"],
    buckets=(0.001, 0.005, 0.01, 0.05, 0.1, 0.5, 1, 2.5, 5),
)
OUTCOMES = Counter("agent_triage_outcomes_total", "Triage outcomes", ["outcome", "category"])


def configure_logging(level: str = "INFO") -> None:
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(
        "ts=%(asctime)s level=%(levelname)s logger=%(name)s %(message)s",
        datefmt="%Y-%m-%dT%H:%M:%S",
    ))
    root = logging.getLogger()
    root.handlers[:] = [handler]
    root.setLevel(level.upper())
