from agent_service import guard
from agent_service.knowledge import KnowledgeBase, policy_lookup
from agent_service.schemas import Draft


def test_search_ranks_the_matching_topic_first():
    kb = KnowledgeBase.load_default()
    hits = kb.search("my direct debit payment failed, is there a late fee?", k=2)
    assert hits and hits[0].id == "KB-004"
    assert hits[0].score >= hits[-1].score


def test_search_with_no_useful_tokens_returns_nothing():
    assert KnowledgeBase.load_default().search("the and for") == []


def test_policy_lookup_tool_returns_ids():
    out = policy_lookup.invoke({"query": "complaint procedure Kifid"})
    assert "[KB-006]" in out


def test_guard_accepts_clean_draft():
    verdict = guard.check(Draft(reply="Report the theft within 48 hours. [KB-001]", citations=["KB-001"]),
                          {"KB-001"})
    assert verdict.ok


def test_guard_flags_each_problem():
    verdict = guard.check(Draft(reply="Guaranteed payout for policy HM-4471920 [KB-042]",
                                citations=["KB-042"]), {"KB-001"})
    assert not verdict.ok
    assert {"unknown citation KB-042", "banned phrase: guaranteed",
            "policy number echoed in reply"} <= set(verdict.issues)
