import pytest

from src.retrieval.hybrid_search import HybridSearch
from src.generation.grounded_answer import GroundedAnswerGenerator


TEST_CASES = [
    (1, "What is the earnings disregard after March 1, 2026?"),
    (2, "What was the earnings disregard before March 1, 2026?"),
    (3, "What is the income threshold for a household of 1 after March 1, 2026?"),
    (4, "What is the reporting deadline for a change occurring on February 28, 2026?"),
    (5, "What is the reporting deadline for a change occurring on March 1, 2026?"),
    (6, "What is the sanction percentage after March 1, 2026?"),
    (7, "What is the sanction percentage before March 1, 2026?"),
    (8, "What changed in the policy on March 1, 2026?"),
    (9, "Which policy rules were amended effective March 1, 2026?"),
    (10, "What is the current reporting deadline under the amended policy?"),
    (11, "What is the current earnings disregard?"),
    (12, "What was the old earnings disregard?"),
    (13, "What is the current income threshold for a household of 1?"),
    (14, "What is the current sanction percentage?"),
    (15, "What is the current reporting deadline for a policy change?"),
    (16, "What policy applies on February 28, 2026?"),
    (17, "What policy applies on March 1, 2026?"),
    (18, "What policy applies after March 1, 2026?"),
    (19, "What policy applies before March 1, 2026?"),
    (20, "Which amendment is effective March 1, 2026?"),
    (21, "Who is eligible for the Household Support Program?"),
    (22, "Does income affect eligibility?"),
    (23, "Can someone owning a car qualify?"),
    (24, "What is the CEO's salary?"),
    (25, "What is the weather in Calder County today?"),
    (26, "Show the policy clause for the earnings disregard."),
    (27, "Show the policy clause that defines the income threshold."),
    (28, "Which clause defines the reporting deadline?"),
    (29, "Which clause defines the sanction percentage?"),
    (30, "What amendment changed the earnings disregard?"),
]


@pytest.fixture(scope="module")
def pipeline():
    search_engine = HybridSearch()
    generator = GroundedAnswerGenerator()
    return search_engine, generator


@pytest.mark.parametrize("test_id, question", TEST_CASES)
def test_policy_question(test_id, question, pipeline):

    search_engine, generator = pipeline

    retrieval_response = search_engine.search(question)

    response = generator.generate(
        question,
        retrieval_response,
    )

    print("\n" + "=" * 70)
    print(f"TEST {test_id}")
    print(f"QUESTION: {question}")
    print(f"ANSWERABLE: {response.get('answerable')}")
    print(f"ANSWER: {response.get('answer')}")
    print(f"CITATIONS: {response.get('citations')}")
    print(f"REASON: {response.get('reason')}")
    print("=" * 70)

    assert response is not None
    assert "answerable" in response
    assert "answer" in response