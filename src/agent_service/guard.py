"""Deterministic output guard. Runs after every draft, whichever backend produced it.

Model-written text is checked with plain code because the checks must be reproducible and cheap:
citations must exist, no unsupported promises, no policy numbers echoed back, bounded length.
"""

import re

from agent_service.schemas import Draft, GuardVerdict

_BANNED = (
    "guaranteed", "we guarantee", "legal advice", "you will be paid", "will definitely",
    "100%", "sue", "lawyer",
)
_CITATION_RE = re.compile(r"\[(KB-\d{3})\]")
_POLICY_RE = re.compile(r"\b[A-Z]{2,3}-?\d{6,8}\b")
MAX_REPLY_CHARS = 1500


def check(draft: Draft, known_ids: set[str]) -> GuardVerdict:
    issues: list[str] = []
    cited_inline = set(_CITATION_RE.findall(draft.reply))
    for cid in set(draft.citations) | cited_inline:
        if cid not in known_ids:
            issues.append(f"unknown citation {cid}")
    lowered = draft.reply.lower()
    for phrase in _BANNED:
        if phrase in lowered:
            issues.append(f"banned phrase: {phrase}")
    if _POLICY_RE.search(draft.reply):
        issues.append("policy number echoed in reply")
    if len(draft.reply) > MAX_REPLY_CHARS:
        issues.append(f"reply longer than {MAX_REPLY_CHARS} characters")
    if not draft.reply.strip():
        issues.append("empty reply")
    return GuardVerdict(ok=not issues, issues=issues)
