"""A small policy knowledge base with a keyword retriever. Replace with a vector store for real data."""

import json
import re
from importlib import resources

from langchain_core.tools import tool

from agent_service.schemas import Snippet

_TOKEN = re.compile(r"[a-z0-9]{3,}")
_STOP = {
    "the", "and", "for", "you", "your", "with", "this", "that", "are", "was", "have", "has",
    "not", "but", "can", "our", "from", "about", "please", "will", "what", "when", "how",
}


def _tokens(text: str) -> set[str]:
    return {t for t in _TOKEN.findall(text.lower()) if t not in _STOP}


class KnowledgeBase:
    def __init__(self, snippets: list[Snippet]):
        self._snippets = snippets
        self._index = {s.id: (s, _tokens(s.title + " " + s.text)) for s in snippets}

    @classmethod
    def load_default(cls) -> "KnowledgeBase":
        raw = resources.files("agent_service.knowledge").joinpath("policies.json").read_text("utf-8")
        return cls([Snippet(**item) for item in json.loads(raw)])

    @property
    def ids(self) -> set[str]:
        return set(self._index)

    def __len__(self) -> int:
        return len(self._snippets)

    def search(self, query: str, k: int = 3) -> list[Snippet]:
        """Rank snippets by weighted keyword overlap; title matches count double."""
        q = _tokens(query)
        if not q:
            return []
        scored: list[Snippet] = []
        for snippet, toks in self._index.values():
            title_toks = _tokens(snippet.title)
            overlap = q & toks
            if not overlap:
                continue
            score = len(overlap) + len(q & title_toks)
            scored.append(snippet.model_copy(update={"score": float(score)}))
        scored.sort(key=lambda s: (-s.score, s.id))
        return scored[:k]


_DEFAULT: KnowledgeBase | None = None


def default_kb() -> KnowledgeBase:
    global _DEFAULT
    if _DEFAULT is None:
        _DEFAULT = KnowledgeBase.load_default()
    return _DEFAULT


@tool
def policy_lookup(query: str) -> str:
    """Look up policy conditions relevant to a customer question. Returns snippets with their ids."""
    hits = default_kb().search(query, k=3)
    return "\n".join(f"[{s.id}] {s.title}: {s.text}" for s in hits) or "No matching policy text."
