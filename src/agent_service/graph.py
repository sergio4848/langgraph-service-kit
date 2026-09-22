"""The triage workflow as a LangGraph state graph.

    START -> classify -> (confidence >= threshold) -> extract -> retrieve -> draft -> guard -> END
                      \\-> clarify -> END                                          \\-> escalate -> END

Every node is a plain function that reads the state and returns the fields it changed. Node timings
are appended to `trace` through a reducer, so the API can expose per-node latency without any node
knowing about the API.
"""

from __future__ import annotations

import operator
import time
from collections.abc import Callable
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from agent_service import guard as guard_module
from agent_service.backends import Backend
from agent_service.knowledge import KnowledgeBase
from agent_service.schemas import (
    Classification,
    Draft,
    Extraction,
    GuardVerdict,
    NodeTiming,
    Snippet,
)

CATEGORY_QUERY_HINTS: dict[str, str] = {
    "claim": "claim report damage theft accident",
    "policy_change": "change address cover add remove policy",
    "billing": "premium payment direct debit refund cancellation",
    "complaint": "complaint procedure complaints answered",
    "general_question": "",
}

ESCALATION_REPLY = (
    "Thank you for your message. A colleague from our customer service team will review it and"
    " reply personally within one working day."
)


class TriageState(TypedDict, total=False):
    text: str
    customer_id: str | None
    classification: Classification
    extraction: Extraction
    snippets: list[Snippet]
    draft: Draft
    guard: GuardVerdict
    outcome: str
    reply: str
    citations: list[str]
    clarification_question: str | None
    trace: Annotated[list[NodeTiming], operator.add]


def _timed(name: str, fn: Callable[[TriageState], dict]) -> Callable[[TriageState], dict]:
    def wrapper(state: TriageState) -> dict:
        started = time.perf_counter()
        update = fn(state) or {}
        ms = round((time.perf_counter() - started) * 1000, 2)
        return {**update, "trace": [NodeTiming(node=name, ms=ms)]}

    wrapper.__name__ = name
    return wrapper


def build_graph(backend: Backend, kb: KnowledgeBase, *, clarify_threshold: float = 0.6,
                max_snippets: int = 3):
    """Compile the workflow for a given backend and knowledge base."""

    def classify(state: TriageState) -> dict:
        return {"classification": backend.classify(state["text"])}

    def extract(state: TriageState) -> dict:
        return {"extraction": backend.extract(state["text"])}

    def retrieve(state: TriageState) -> dict:
        # Category-aware retrieval: the classifier's verdict steers the keyword query towards the
        # policy topics that belong to that category, which keeps unrelated snippets out of drafts.
        hint = CATEGORY_QUERY_HINTS.get(state["classification"].category, "")
        product = state.get("extraction", Extraction()).product or ""
        query = f"{hint} {product} {state['text']}".strip()
        return {"snippets": kb.search(query, k=max_snippets)}

    def draft(state: TriageState) -> dict:
        d = backend.draft(state["text"], state["classification"], state["extraction"],
                          state.get("snippets", []))
        return {"draft": d}

    def guard(state: TriageState) -> dict:
        verdict = guard_module.check(state["draft"], kb.ids)
        update: dict = {"guard": verdict}
        if verdict.ok:
            update.update(outcome="answered", reply=state["draft"].reply,
                          citations=list(state["draft"].citations))
        return update

    def clarify(state: TriageState) -> dict:
        question = (
            "To help you quickly, could you tell us whether your message is about a claim, a"
            " change to your policy, a payment, or something else, and which policy it concerns?"
        )
        return {"outcome": "clarify", "reply": question, "citations": [],
                "clarification_question": question}

    def escalate(state: TriageState) -> dict:
        return {"outcome": "escalated", "reply": ESCALATION_REPLY, "citations": []}

    def route_after_classify(state: TriageState) -> str:
        return "extract" if state["classification"].confidence >= clarify_threshold else "clarify"

    def route_after_guard(state: TriageState) -> str:
        return END if state["guard"].ok else "escalate"

    g: StateGraph = StateGraph(TriageState)
    g.add_node("classify", _timed("classify", classify))
    g.add_node("extract", _timed("extract", extract))
    g.add_node("retrieve", _timed("retrieve", retrieve))
    g.add_node("draft", _timed("draft", draft))
    g.add_node("guard", _timed("guard", guard))
    g.add_node("clarify", _timed("clarify", clarify))
    g.add_node("escalate", _timed("escalate", escalate))

    g.add_edge(START, "classify")
    g.add_conditional_edges("classify", route_after_classify,
                            {"extract": "extract", "clarify": "clarify"})
    g.add_edge("extract", "retrieve")
    g.add_edge("retrieve", "draft")
    g.add_edge("draft", "guard")
    g.add_conditional_edges("guard", route_after_guard, {END: END, "escalate": "escalate"})
    g.add_edge("clarify", END)
    g.add_edge("escalate", END)
    return g.compile()


def initial_state(text: str, customer_id: str | None = None) -> TriageState:
    return {"text": text, "customer_id": customer_id, "trace": []}
