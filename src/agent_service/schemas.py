"""Pydantic models shared by the graph, the backends and the API. Every LLM call returns one of these."""

from typing import Literal

from pydantic import BaseModel, Field

Category = Literal["claim", "policy_change", "billing", "complaint", "general_question"]
Urgency = Literal["low", "medium", "high"]
Outcome = Literal["answered", "clarify", "escalated"]

CATEGORY_LABELS: dict[str, str] = {
    "claim": "a claim",
    "policy_change": "a change to your policy",
    "billing": "billing and payments",
    "complaint": "a complaint",
    "general_question": "your question",
}


class Classification(BaseModel):
    category: Category
    urgency: Urgency
    confidence: float = Field(ge=0.0, le=1.0)
    rationale: str = ""


class Extraction(BaseModel):
    policy_number: str | None = None
    incident_date: str | None = None
    amount_eur: float | None = None
    product: str | None = None


class Snippet(BaseModel):
    id: str
    title: str
    text: str
    score: float = 0.0


class Draft(BaseModel):
    reply: str
    citations: list[str] = Field(default_factory=list)


class GuardVerdict(BaseModel):
    ok: bool
    issues: list[str] = Field(default_factory=list)


class NodeTiming(BaseModel):
    node: str
    ms: float


class TriageRequest(BaseModel):
    text: str = Field(min_length=5, max_length=4000, description="The customer's message")
    customer_id: str | None = Field(default=None, max_length=64)


class TriageResponse(BaseModel):
    request_id: str
    outcome: Outcome
    classification: Classification | None = None
    extraction: Extraction | None = None
    reply: str
    citations: list[str] = Field(default_factory=list)
    clarification_question: str | None = None
    guard: GuardVerdict | None = None
    trace: list[NodeTiming] = Field(default_factory=list)
