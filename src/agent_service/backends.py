"""Model backends behind the graph.

The graph never calls a chat model directly. It calls one of the methods below and always gets a
Pydantic object back, so the workflow, the API and the tests do not change when the model does.

- RuleBackend: deterministic keyword rules. Used by the test-suite, CI and local development.
- OpenAIBackend: an OpenAI-compatible chat model with structured outputs.
"""

from __future__ import annotations

import re
from typing import Protocol

from agent_service.config import Settings
from agent_service.schemas import (
    CATEGORY_LABELS,
    Classification,
    Draft,
    Extraction,
    Snippet,
)


class Backend(Protocol):
    name: str

    def classify(self, text: str) -> Classification: ...

    def extract(self, text: str) -> Extraction: ...

    def draft(self, text: str, classification: Classification, extraction: Extraction,
              snippets: list[Snippet]) -> Draft: ...


# ---------------------------------------------------------------------------
# Deterministic backend
# ---------------------------------------------------------------------------

_KEYWORDS: dict[str, tuple[str, ...]] = {
    "claim": ("claim", "damage", "damaged", "stolen", "theft", "burglar", "accident", "crash",
              "leak", "leaking", "flood", "fire", "broke", "broken", "hit my car", "lost my"),
    "policy_change": ("change my", "update my", "new address", "moving", "moved", "add my",
                      "remove", "cancel", "renew", "coverage", "cover for", "increase my"),
    "billing": ("invoice", "payment", "charged", "bill", "premium", "refund", "direct debit",
                "paid twice", "debited", "overpaid", "amount"),
    "complaint": ("complaint", "unhappy", "disappointed", "unacceptable", "no one", "nobody",
                  "still waiting", "weeks", "terrible", "frustrat", "escalate"),
}

_HIGH_URGENCY = ("urgent", "asap", "today", "immediately", "emergency", "right now",
                 "water is still", "still leaking", "injur", "hospital", "abroad")

_POLICY_RE = re.compile(r"\b([A-Z]{2,3}-?\d{6,8})\b")
_ISO_DATE_RE = re.compile(r"\b(20\d{2}-\d{2}-\d{2})\b")
_EU_DATE_RE = re.compile(r"\b(\d{1,2}[/.]\d{1,2}[/.]20\d{2})\b")
_AMOUNT_RE = re.compile(r"(?:€|eur\s?)\s?(\d[\d.,]*)|(\d[\d.,]*)\s?(?:euros?|eur)\b", re.I)
_PRODUCTS = ("car", "home", "contents", "travel", "liability", "bike", "boat")


class RuleBackend:
    name = "rules"

    def classify(self, text: str) -> Classification:
        lowered = text.lower()
        hits = {cat: sum(1 for kw in kws if kw in lowered) for cat, kws in _KEYWORDS.items()}
        best_cat, best_hits = max(hits.items(), key=lambda kv: (kv[1], kv[0] == "claim"))
        if best_hits == 0:
            best_cat, confidence = "general_question", 0.4
        else:
            confidence = 0.9 if best_hits >= 2 else 0.7
        if any(kw in lowered for kw in _HIGH_URGENCY):
            urgency = "high"
        elif best_cat in ("claim", "complaint"):
            urgency = "medium"
        else:
            urgency = "low"
        matched = [kw for kw in _KEYWORDS.get(best_cat, ()) if kw in lowered]
        return Classification(category=best_cat, urgency=urgency, confidence=confidence,
                              rationale=f"keywords: {', '.join(matched) or 'none'}")

    def extract(self, text: str) -> Extraction:
        policy = _POLICY_RE.search(text)
        date = _ISO_DATE_RE.search(text) or _EU_DATE_RE.search(text)
        amount = None
        if m := _AMOUNT_RE.search(text):
            raw = (m.group(1) or m.group(2) or "").replace(".", "").replace(",", ".")
            try:
                amount = float(raw)
            except ValueError:
                amount = None
        lowered = text.lower()
        product = next((p for p in _PRODUCTS if p in lowered), None)
        return Extraction(policy_number=policy.group(1) if policy else None,
                          incident_date=date.group(1) if date else None,
                          amount_eur=amount, product=product)

    def draft(self, text: str, classification: Classification, extraction: Extraction,
              snippets: list[Snippet]) -> Draft:
        label = CATEGORY_LABELS[classification.category]
        parts = [f"Thank you for contacting us about {label}."]
        citations: list[str] = []
        for s in snippets[:2]:
            first_sentence = s.text.split(". ")[0].rstrip(".") + "."
            parts.append(f"{first_sentence} [{s.id}]")
            citations.append(s.id)
        next_step = {
            "claim": "To open the claim, reply with the date of the incident and any photos or"
                     " receipts you have; we will confirm the claim number within one working day.",
            "policy_change": "We can make this change for you; please confirm the effective date"
                             " you would like.",
            "billing": "We will check the payment records on your policy and confirm the outcome"
                       " by e-mail.",
            "complaint": "We are sorry about your experience. Your complaint has been registered"
                         " and a case handler will contact you.",
            "general_question": "If this does not answer your question, reply with a few more"
                                " details and we will look into it.",
        }[classification.category]
        parts.append(next_step)
        if classification.urgency == "high":
            parts.append("Because this is urgent, we have flagged it for same-day handling.")
        return Draft(reply=" ".join(parts), citations=citations)


# ---------------------------------------------------------------------------
# OpenAI-compatible backend
# ---------------------------------------------------------------------------

_CLASSIFY_SYSTEM = (
    "You triage messages sent to an insurance company's customer service. Classify the message"
    " into exactly one category (claim, policy_change, billing, complaint, general_question),"
    " estimate urgency (low, medium, high) and give a confidence between 0 and 1. Be conservative"
    " with confidence when the message is vague or could belong to several categories."
)
_EXTRACT_SYSTEM = (
    "Extract structured details from a customer message. Leave a field null when the message does"
    " not state it. Do not guess. amount_eur is a number in euros. incident_date is ISO 8601 when"
    " a full date is given, otherwise the date text as written."
)
_DRAFT_SYSTEM = (
    "You write the reply of an insurance customer-service agent. Use only the policy snippets"
    " provided, cite each fact with its snippet id in square brackets, never invent conditions,"
    " never give legal or medical advice, never promise a payout, and keep the reply under 180"
    " words. Return the snippet ids you used in `citations`."
)


class OpenAIBackend:
    name = "openai"

    def __init__(self, settings: Settings):
        from langchain_openai import ChatOpenAI

        kwargs: dict = {"model": settings.openai_model, "temperature": 0,
                        "timeout": settings.openai_timeout_s, "max_retries": 2}
        if settings.openai_api_key:
            kwargs["api_key"] = settings.openai_api_key
        if settings.openai_base_url:
            kwargs["base_url"] = settings.openai_base_url
        self._llm = ChatOpenAI(**kwargs)

    def _structured(self, schema, system: str, user: str):
        chain = self._llm.with_structured_output(schema)
        return chain.invoke([("system", system), ("human", user)])

    def classify(self, text: str) -> Classification:
        return self._structured(Classification, _CLASSIFY_SYSTEM, text)

    def extract(self, text: str) -> Extraction:
        return self._structured(Extraction, _EXTRACT_SYSTEM, text)

    def draft(self, text: str, classification: Classification, extraction: Extraction,
              snippets: list[Snippet]) -> Draft:
        context = "\n".join(f"[{s.id}] {s.title}: {s.text}" for s in snippets) or "(no snippets)"
        user = (
            f"Category: {classification.category}; urgency: {classification.urgency}.\n"
            f"Extracted details: {extraction.model_dump_json()}\n"
            f"Policy snippets:\n{context}\n\nCustomer message:\n{text}"
        )
        return self._structured(Draft, _DRAFT_SYSTEM, user)


def get_backend(settings: Settings) -> Backend:
    if settings.backend == "openai":
        return OpenAIBackend(settings)
    return RuleBackend()
