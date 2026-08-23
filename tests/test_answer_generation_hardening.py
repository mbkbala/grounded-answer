"""
============================================================
GROUNDED POLICY ASSISTANT
ANSWER GENERATION HARDENING TESTS

Stage 2 tests.

Purpose:
    Ensure GroundedAnswerGenerator remains correct when
    retrieval returns noisy, mixed, conflicting, duplicate,
    or misleading results.
============================================================
"""

from src.generation.grounded_answer import GroundedAnswerGenerator


# ============================================================
# HELPER
# ============================================================

def make_retrieval_response(results):
    return {
        "answerable": True,
        "results": results,
    }


def generate(question, results):
    generator = GroundedAnswerGenerator()

    return generator.generate(
        question,
        make_retrieval_response(results),
    )


# ============================================================
# 1. CORRECT CLAUSE VS IRRELEVANT CLAUSE
# ============================================================

def test_correct_clause_beats_irrelevant_clause():

    question = "Who administers the program?"

    results = [
        {
            "clause": "§2.1.2",
            "text": (
                "An applicant must be a resident in "
                "Calder County and meet the eligibility "
                "requirements."
            ),
            "hybrid_score": 0.99,
        },
        {
            "clause": "§1.1.2",
            "text": (
                "The Program is administered by the "
                "Calder County Department of Household "
                "Services."
            ),
            "hybrid_score": 0.70,
        },
    ]

    response = generate(question, results)

    assert response["answerable"] is True
    assert response["citations"] == ["§1.1.2"]
    assert "Calder County Department" in response["answer"]


# ============================================================
# 2. EXCLUSION MUST BEAT ELIGIBILITY
# ============================================================

def test_exclusion_beats_eligibility_with_mixed_results():

    question = (
        "Are people in a correctional facility eligible?"
    )

    results = [
        {
            "clause": "§2.1.2",
            "text": (
                "Applicants must satisfy the eligibility "
                "requirements for the program."
            ),
            "hybrid_score": 0.99,
        },
        {
            "clause": "§4.1.1",
            "text": (
                "Individuals residing in a correctional "
                "facility are excluded from the program."
            ),
            "hybrid_score": 0.60,
        },
    ]

    response = generate(question, results)

    assert response["answerable"] is True
    assert response["citations"] == ["§4.1.1"]
    assert "excluded" in response["answer"].lower()


# ============================================================
# 3. ADMINISTRATION VS ELIGIBILITY
# ============================================================

def test_administration_beats_eligibility():

    question = "Which department administers the program?"

    results = [
        {
            "clause": "§2.1.2",
            "text": (
                "Applicants must be residents of "
                "Calder County."
            ),
            "hybrid_score": 0.99,
        },
        {
            "clause": "§1.1.2",
            "text": (
                "The Program is administered by the "
                "Calder County Department of Household "
                "Services."
            ),
            "hybrid_score": 0.50,
        },
    ]

    response = generate(question, results)

    assert response["citations"] == ["§1.1.2"]


# ============================================================
# 4. DUPLICATE CLAUSES
# ============================================================

def test_duplicate_clause_returns_one_citation():

    question = "Who administers the program?"

    results = [
        {
            "clause": "§1.1.2",
            "text": (
                "The Program is administered by the "
                "Calder County Department of Household "
                "Services."
            ),
            "hybrid_score": 0.90,
        },
        {
            "clause": "§1.1.2",
            "text": (
                "The Program is administered by the "
                "Calder County Department of Household "
                "Services."
            ),
            "hybrid_score": 0.80,
        },
    ]

    response = generate(question, results)

    assert response["answerable"] is True
    assert response["citations"] == ["§1.1.2"]
    assert len(response["citations"]) == 1


# ============================================================
# 5. OVERVIEW CLAUSE MUST NOT WIN
# ============================================================

def test_overview_clause_does_not_beat_real_policy_clause():

    question = "What are the eligibility requirements?"

    results = [
        {
            "clause": "§1.2.1",
            "text": (
                "Part 2 addresses eligibility requirements."
            ),
            "hybrid_score": 0.99,
        },
        {
            "clause": "§2.1.2",
            "text": (
                "An applicant must be a resident in "
                "Calder County and meet the eligibility "
                "requirements."
            ),
            "hybrid_score": 0.60,
        },
    ]

    response = generate(question, results)

    assert response["answerable"] is True
    assert response["citations"] == ["§2.1.2"]


# ============================================================
# 6. WRONG CLAUSE WITH VERY HIGH RETRIEVAL SCORE
# ============================================================

def test_deterministic_policy_rule_beats_high_retrieval_score():

    question = "Who administers the program?"

    results = [
        {
            "clause": "§2.1.2",
            "text": (
                "The program provides assistance to "
                "eligible residents."
            ),
            "hybrid_score": 999.0,
        },
        {
            "clause": "§1.1.2",
            "text": (
                "The Program is administered by the "
                "Calder County Department of Household "
                "Services."
            ),
            "hybrid_score": 0.01,
        },
    ]

    response = generate(question, results)

    assert response["citations"] == ["§1.1.2"]


# ============================================================
# 7. MULTIPLE SENTENCES — RETURN RELEVANT SENTENCE
# ============================================================

def test_multiple_sentences_return_relevant_sentence():

    question = "Who administers the program?"

    results = [
        {
            "clause": "§1.1.2",
            "text": (
                "The Program is administered by the "
                "Calder County Department of Household "
                "Services. "
                "Day-to-day administration is carried "
                "out by caseworkers at four district "
                "offices."
            ),
            "hybrid_score": 0.90,
        },
    ]

    response = generate(question, results)

    assert response["answerable"] is True
    assert response["citations"] == ["§1.1.2"]
    assert "administered" in response["answer"].lower()


# ============================================================
# 8. MISSING CLAUSE TEXT
# ============================================================

def test_missing_clause_text_is_not_used():

    question = "Who administers the program?"

    results = [
        {
            "clause": "§1.1.2",
            "text": "",
            "hybrid_score": 0.99,
        },
        {
            "clause": "§1.1.2",
            "text": (
                "The Program is administered by the "
                "Calder County Department of Household "
                "Services."
            ),
            "hybrid_score": 0.80,
        },
    ]

    response = generate(question, results)

    assert response["answerable"] is True
    assert response["citations"] == ["§1.1.2"]


# ============================================================
# 9. MISSING CLAUSE ID
# ============================================================

def test_missing_clause_id_is_not_used():

    question = "Who administers the program?"

    results = [
        {
            "text": (
                "The Program is administered by the "
                "Calder County Department of Household "
                "Services."
            ),
            "hybrid_score": 0.99,
        },
        {
            "clause": "§1.1.2",
            "text": (
                "The Program is administered by the "
                "Calder County Department of Household "
                "Services."
            ),
            "hybrid_score": 0.80,
        },
    ]

    response = generate(question, results)

    assert response["answerable"] is True
    assert response["citations"] == ["§1.1.2"]


# ============================================================
# 10. COMPLETELY IRRELEVANT RESULTS
# ============================================================

def test_irrelevant_results_return_unknown():

    question = "Who administers the program?"

    results = [
        {
            "clause": "§8.2.1",
            "text": (
                "Applications must be submitted within "
                "thirty days."
            ),
            "hybrid_score": 0.20,
        },
        {
            "clause": "§9.1.1",
            "text": (
                "Payments are issued electronically."
            ),
            "hybrid_score": 0.10,
        },
    ]

    response = generate(question, results)

    assert response["answerable"] is False
    assert "I don't know" in response["answer"]


# ============================================================
# 11. DIRECT CLAUSE REFERENCE MUST WIN
# ============================================================

def test_direct_clause_reference_beats_all_other_results():

    question = "What does §2.1.2 say?"

    results = [
        {
            "clause": "§1.1.2",
            "text": (
                "The Program is administered by the "
                "Calder County Department of Household "
                "Services."
            ),
            "hybrid_score": 0.99,
        },
        {
            "clause": "§2.1.2",
            "text": (
                "An applicant must be a resident in "
                "Calder County."
            ),
            "hybrid_score": 0.10,
        },
    ]

    response = generate(question, results)

    assert response["answerable"] is True
    assert response["citations"] == ["§2.1.2"]


# ============================================================
# 12. ONE RELEVANT CLAUSE AMONG MANY
# ============================================================

def test_one_relevant_clause_among_many():

    question = "Are household resources considered?"

    results = [
        {
            "clause": "§1.1.2",
            "text": (
                "The Program is administered by the "
                "Department."
            ),
            "hybrid_score": 0.95,
        },
        {
            "clause": "§2.1.2",
            "text": (
                "Applicants must be residents of "
                "Calder County."
            ),
            "hybrid_score": 0.90,
        },
        {
            "clause": "§4.1.1",
            "text": (
                "Certain individuals are excluded "
                "from the program."
            ),
            "hybrid_score": 0.85,
        },
        {
            "clause": "§2.4",
            "text": (
                "Household eligibility considers "
                "countable resources."
            ),
            "hybrid_score": 0.30,
        },
    ]

    response = generate(question, results)

    assert response["answerable"] is True
    assert response["citations"] == ["§2.4"]
    assert "countable resources" in response["answer"].lower()