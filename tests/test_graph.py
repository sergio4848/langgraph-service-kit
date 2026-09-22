from agent_service.backends import RuleBackend
from agent_service.graph import ESCALATION_REPLY, build_graph, initial_state
from agent_service.knowledge import KnowledgeBase
from agent_service.schemas import Draft


def make_graph(backend=None, threshold: float = 0.6):
    return build_graph(backend or RuleBackend(), KnowledgeBase.load_default(),
                       clarify_threshold=threshold)


def test_claim_is_answered_with_citations():
    result = make_graph().invoke(initial_state(
        "Someone stole my laptop from the flat last night, police report 2026-1122. "
        "How do I claim on contents insurance?"))
    assert result["outcome"] == "answered"
    assert result["classification"].category == "claim"
    assert result["citations"], "an answered claim must cite policy text"
    assert all(c.startswith("KB-") for c in result["citations"])
    assert result["guard"].ok


def test_urgent_leak_is_flagged_high():
    result = make_graph().invoke(initial_state(
        "Water is still leaking from the ceiling right now, I need someone today. Policy HM-4471920."))
    assert result["classification"].urgency == "high"
    assert result["extraction"].policy_number == "HM-4471920"
    assert "HM-4471920" not in result["reply"], "policy numbers must not be echoed back"
    assert "same-day" in result["reply"]


def test_vague_message_asks_for_clarification():
    result = make_graph().invoke(initial_state("Hi, I have a question about something."))
    assert result["outcome"] == "clarify"
    assert result["clarification_question"]
    assert [t.node for t in result["trace"]] == ["classify", "clarify"]


def test_guard_failure_escalates():
    class BadDraftBackend(RuleBackend):
        name = "bad"

        def draft(self, text, classification, extraction, snippets):
            return Draft(reply="You will be paid in full, guaranteed. [KB-999]", citations=["KB-999"])

    result = make_graph(BadDraftBackend()).invoke(initial_state(
        "My car was hit in an accident yesterday, what happens next?"))
    assert result["outcome"] == "escalated"
    assert result["reply"] == ESCALATION_REPLY
    assert not result["guard"].ok
    assert any("unknown citation" in i for i in result["guard"].issues)
    assert any("banned phrase" in i for i in result["guard"].issues)


def test_trace_covers_every_node_on_the_happy_path():
    result = make_graph().invoke(initial_state(
        "I am moving to Utrecht on 2026-10-15, please change my address on the home policy."))
    assert [t.node for t in result["trace"]] == ["classify", "extract", "retrieve", "draft", "guard"]
    assert all(t.ms >= 0 for t in result["trace"])
    assert result["extraction"].incident_date == "2026-10-15"
