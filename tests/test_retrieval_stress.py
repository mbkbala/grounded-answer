import pytest

from src.retrieval.hybrid_search import HybridSearch


@pytest.fixture(scope="module")
def searcher():
    return HybridSearch()


# ============================================================
# ADMINISTRATION PARAPHRASES
# ============================================================

@pytest.mark.parametrize(
    "question",
    [
        "Who administers the program?",
        "Which department runs the program?",
        "Who is responsible for administering the program?",
        "What department handles this program?",
        "Who carries out day-to-day administration?",
        "Who manages the Household Support Program?",
        "Which agency administers the Household Support Program?",
        "Who is in charge of the program?",
    ],
)
def test_administration_paraphrases(searcher, question):

    response = searcher.search(question)

    clauses = [
        result["clause"]
        for result in response["results"][:3]
    ]

    assert response["answerable"] is True

    assert "§1.1.2" in clauses, (
        f"\nQuestion: {question}"
        f"\nTop clauses: {clauses}"
    )


# ============================================================
# OUT OF SCOPE
# ============================================================

@pytest.mark.parametrize(
    "question",
    [
        "What is the weather today?",
        "How do I write Python code?",
        "Who won the football match?",
        "What is the capital of France?",
        "Tell me a joke.",
    ],
)
def test_out_of_scope_questions(searcher, question):

    response = searcher.search(question)

    assert response["answerable"] is False

    assert response["results"] == []


# ============================================================
# DIRECT CLAUSE REFERENCES
# ============================================================

@pytest.mark.parametrize(
    "clause_id",
    [
        "§1.1.2",
        "§2.1.2",
        "§2.3.1",
        "§4.1.1",
        "§6.4.1",
        "§10.5.2",
    ],
)
def test_direct_clause_retrieval(searcher, clause_id):

    response = searcher.search(
        f"What does {clause_id} say?"
    )

    assert response["answerable"] is True

    assert response["results"][0]["clause"] == clause_id


# ============================================================
# TEMPORAL POLICY VERSION
# ============================================================

def test_policy_before_amendment(searcher):

    response = searcher.search(
        "What is the income disregard?",
        as_of_date="2026-02-28",
    )

    assert response["policy_version"]["amendments_active"] == 0


def test_policy_on_amendment_date(searcher):

    response = searcher.search(
        "What is the income disregard?",
        as_of_date="2026-03-01",
    )

    assert response["policy_version"]["amendments_active"] >= 1


# ============================================================
# TEMPORAL INCOME DISREGARD
# ============================================================

def test_income_disregard_before_amendment(searcher):

    response = searcher.search(
        "What is the income disregard?",
        as_of_date="2026-02-28",
    )

    clauses = [
        result["clause"]
        for result in response["results"]
    ]

    assert "§6.4.1" in clauses


def test_income_disregard_after_amendment(searcher):

    response = searcher.search(
        "What is the income disregard?",
        as_of_date="2026-03-01",
    )

    clauses = [
        result["clause"]
        for result in response["results"]
    ]

    assert "§6.4.1" in clauses