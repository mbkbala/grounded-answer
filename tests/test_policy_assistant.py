import sys
from pathlib import Path

import pytest


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORT APPLICATION
# ============================================================

from src.retrieval.hybrid_search import HybridSearch
from src.answer_generator import generate_answer


# ============================================================
# FIXTURE
# ============================================================

@pytest.fixture(scope="module")
def searcher():
    """
    Create the hybrid search engine once for the complete
    test session.

    Loading SentenceTransformer repeatedly is expensive,
    so we reuse the same searcher.
    """
    return HybridSearch()


# ============================================================
# HELPER
# ============================================================

def run_question(searcher, question):
    """
    Run one complete question through:

        Question
            ↓
        Hybrid Search
            ↓
        Answer Generator
    """

    response = searcher.search(
        question,
        top_k=5
    )

    answer = generate_answer(
        question,
        response
    )

    return response, answer


def get_clause_ids(response):
    """
    Extract clause IDs returned by retrieval.
    """
    return [
        result["clause"]
        for result in response.get("results", [])
    ]


def assert_contains_any(text, keywords):
    """
    Assert that at least one expected keyword exists.
    """
    text_lower = text.lower()

    assert any(
        keyword.lower() in text_lower
        for keyword in keywords
    ), (
        f"None of the expected keywords were found.\n"
        f"Expected one of: {keywords}\n"
        f"Actual answer:\n{text}"
    )


# ============================================================
# TEST 1
# EXCLUSION ELIGIBILITY
# ============================================================

def test_exclusion_eligibility(searcher):

    question = "Who is excluded from eligibility?"

    response, answer = run_question(
        searcher,
        question
    )

    assert response["answerable"] is True

    clauses = get_clause_ids(response)

    assert "§4.1.1" in clauses

    assert_contains_any(
        answer,
        [
            "unexpired sanction",
            "correctional facility",
            "equivalent program",
            "misrepresentation",
        ]
    )

    assert "§4.1.1" in answer


# ============================================================
# TEST 2
# ELIGIBILITY REQUIREMENTS
# ============================================================

def test_eligibility_requirements(searcher):

    question = (
        "What are the eligibility requirements "
        "for assistance?"
    )

    response, answer = run_question(
        searcher,
        question
    )

    assert response["answerable"] is True

    clauses = get_clause_ids(response)

    assert "§2.1.2" in clauses

    assert_contains_any(
        answer,
        [
            "resident",
            "income",
            "resources",
            "application",
            "excluded",
        ]
    )

    assert "§2.1.2" in answer


# ============================================================
# TEST 3
# PROGRAM ADMINISTRATOR
# ============================================================

def test_program_administrator(searcher):

    question = "Who administers the program?"

    response, answer = run_question(
        searcher,
        question
    )

    assert response["answerable"] is True

    clauses = get_clause_ids(response)

    assert "§1.1.2" in clauses

    assert_contains_any(
        answer,
        [
            "Department of Household Services",
            "caseworkers",
            "district offices",
        ]
    )

    assert "§1.1.2" in answer


# ============================================================
# TEST 4
# 17 YEAR OLD
# ============================================================

def test_17_year_old_eligibility(searcher):

    question = (
        "Can I receive assistance if I am 17 years old?"
    )

    response, answer = run_question(
        searcher,
        question
    )

    assert response["answerable"] is True

    clauses = get_clause_ids(response)

    assert "§2.3.1" in clauses
    assert "§2.3.2" in clauses

    assert_contains_any(
        answer,
        [
            "16 or 17",
            "aged 16 or 17",
            "17 may be eligible",
        ]
    )

    assert "§2.3.1" in answer


# ============================================================
# TEST 5
# 18 YEAR OLD
# ============================================================

def test_18_year_old_eligibility(searcher):

    question = (
        "Can I receive assistance if I am 18 years old?"
    )

    response, answer = run_question(
        searcher,
        question
    )

    assert response["answerable"] is True

    clauses = get_clause_ids(response)

    assert "§2.1.2" in clauses

    # IMPORTANT:
    # An 18-year-old falls under the normal 18+ rule.
    # The answer must NOT incorrectly describe them
    # as a 16/17-year-old.

    assert "17 may be eligible" not in answer
    assert "16 or 17" not in answer

    assert_contains_any(
        answer,
        [
            "18",
            "18 or over",
            "eligible",
        ]
    )

    assert "§2.1.2" in answer


# ============================================================
# TEST 6
# 16 YEAR OLD
# ============================================================

def test_16_year_old_eligibility(searcher):

    question = (
        "Can a 16 year old receive assistance?"
    )

    response, answer = run_question(
        searcher,
        question
    )

    assert response["answerable"] is True

    clauses = get_clause_ids(response)

    assert "§2.3.1" in clauses

    assert_contains_any(
        answer,
        [
            "16 or 17",
            "aged 16 or 17",
            "16-year-old",
        ]
    )

    assert "§2.3.1" in answer


# ============================================================
# TEST 7
# WEATHER
# ============================================================

def test_weather_outside_policy(searcher):

    question = (
        "What is the weather in Calder County?"
    )

    response, answer = run_question(
        searcher,
        question
    )

    assert response["answerable"] is False

    assert_contains_any(
        answer,
        [
            "I don't know",
            "outside the scope",
        ]
    )

    # It must NOT answer with a random policy clause.
    assert "§1.1.2" not in answer


# ============================================================
# TEST 8
# PYTHON QUESTION
# ============================================================

def test_python_question_outside_policy(searcher):

    question = (
        "Write a Python program to calculate factorial"
    )

    response, answer = run_question(
        searcher,
        question
    )

    assert response["answerable"] is False

    assert_contains_any(
        answer,
        [
            "I don't know",
            "outside the scope",
        ]
    )


# ============================================================
# TEST 9
# GENERAL KNOWLEDGE
# ============================================================

def test_general_knowledge_outside_policy(searcher):

    question = "Who is the president of India?"

    response, answer = run_question(
        searcher,
        question
    )

    assert response["answerable"] is False

    assert_contains_any(
        answer,
        [
            "I don't know",
            "outside the scope",
            "does not provide",
        ]
    )


# ============================================================
# TEST 10
# DIRECT CLAUSE REFERENCE
# ============================================================

def test_direct_clause_reference(searcher):

    question = "What does §2.1.2 say?"

    response, answer = run_question(
        searcher,
        question
    )

    assert response["answerable"] is True

    clauses = get_clause_ids(response)

    assert "§2.1.2" in clauses

    assert "§2.1.2" in answer

    assert_contains_any(
        answer,
        [
            "resident",
            "income",
            "resources",
            "application",
            "excluded",
        ]
    )


# ============================================================
# TEST 11
# RESIDENCE REQUIREMENT
# ============================================================

def test_residence_requirement(searcher):

    question = (
        "Do I need to live in Calder County to qualify?"
    )

    response, answer = run_question(
        searcher,
        question
    )

    assert response["answerable"] is True

    clauses = get_clause_ids(response)

    assert "§2.1.2" in clauses

    assert_contains_any(
        answer,
        [
            "resident of Calder County",
            "residence requirements",
        ]
    )

    assert "§2.1.2" in answer


# ============================================================
# TEST 12
# INCOME REQUIREMENT
# ============================================================

def test_income_requirement(searcher):

    question = (
        "Is there an income limit for assistance?"
    )

    response, answer = run_question(
        searcher,
        question
    )

    assert response["answerable"] is True

    clauses = get_clause_ids(response)

    assert "§2.1.2" in clauses

    assert_contains_any(
        answer,
        [
            "income",
            "countable income",
            "threshold",
        ]
    )

    # The answer should not accidentally cite
    # an unrelated definitions clause.
    assert "§1.3.3" not in answer


# ============================================================
# TEST 13
# RESOURCE REQUIREMENT
# ============================================================

def test_resource_requirement(searcher):

    question = (
        "Are my resources considered when determining eligibility?"
    )

    response, answer = run_question(
        searcher,
        question
    )

    assert response["answerable"] is True

    clauses = get_clause_ids(response)

    assert "§2.1.2" in clauses

    assert_contains_any(
        answer,
        [
            "resources",
            "countable resources",
        ]
    )

    assert "§2.1.2" in answer


# ============================================================
# TEST 14
# HOUSEHOLD ELIGIBILITY
# ============================================================

def test_household_eligibility(searcher):

    question = (
        "Is eligibility assessed for the household?"
    )

    response, answer = run_question(
        searcher,
        question
    )

    assert response["answerable"] is True

    clauses = get_clause_ids(response)

    assert_contains_any(
        " ".join(clauses),
        [
            "§2.1.3",
            "§2.1.2",
        ]
    )

    assert_contains_any(
        answer,
        [
            "household",
            "eligibility",
        ]
    )


# ============================================================
# TEST 15
# CASEWORKER ADMINISTRATION
# ============================================================

def test_caseworker_administration(searcher):

    question = (
        "Who handles the day-to-day administration "
        "of the program?"
    )

    response, answer = run_question(
        searcher,
        question
    )

    assert response["answerable"] is True

    clauses = get_clause_ids(response)

    assert "§1.1.2" in clauses

    assert_contains_any(
        answer,
        [
            "caseworkers",
            "day-to-day administration",
            "district offices",
        ]
    )

    assert "§1.1.2" in answer


# ============================================================
# TEST 16
# SANCTION EXCLUSION
# ============================================================

def test_sanction_exclusion(searcher):

    question = (
        "Can someone with an unexpired sanction "
        "receive assistance?"
    )

    response, answer = run_question(
        searcher,
        question
    )

    assert response["answerable"] is True

    clauses = get_clause_ids(response)

    assert "§4.1.1" in clauses

    assert_contains_any(
        answer,
        [
            "unexpired sanction",
            "excluded",
            "not eligible",
        ]
    )

    assert "§4.1.1" in answer


# ============================================================
# TEST 17
# CORRECTIONAL FACILITY
# ============================================================

def test_correctional_facility_exclusion(searcher):

    question = (
        "Is someone detained in a correctional facility eligible?"
    )

    response, answer = run_question(
        searcher,
        question
    )

    assert response["answerable"] is True

    clauses = get_clause_ids(response)

    assert "§4.1.1" in clauses

    assert_contains_any(
        answer,
        [
            "correctional facility",
            "excluded",
            "not eligible",
        ]
    )

    assert "§4.1.1" in answer


# ============================================================
# TEST 18
# APPLICATION REQUIREMENT
# ============================================================

def test_application_requirement(searcher):

    question = (
        "Do I need to submit an application?"
    )

    response, answer = run_question(
        searcher,
        question
    )

    assert response["answerable"] is True

    clauses = get_clause_ids(response)

    assert "§2.1.2" in clauses

    assert_contains_any(
        answer,
        [
            "valid application",
            "application",
            "submit",
        ]
    )

    assert "§2.1.2" in answer


# ============================================================
# TEST 19
# UNKNOWN POLICY TOPIC
# ============================================================

def test_unknown_policy_topic(searcher):

    question = (
        "Does the program provide free cars?"
    )

    response, answer = run_question(
        searcher,
        question
    )

    assert response["answerable"] is False

    assert_contains_any(
        answer,
        [
            "I don't know",
            "outside the scope",
            "does not provide",
        ]
    )


# ============================================================
# TEST 20
# RANDOM QUESTION
# ============================================================

def test_random_question(searcher):

    question = (
        "What is the capital of France?"
    )

    response, answer = run_question(
        searcher,
        question
    )

    assert response["answerable"] is False

    assert_contains_any(
        answer,
        [
            "I don't know",
            "outside the scope",
        ]
    )


# ============================================================
# ADDITIONAL SAFETY TESTS
# ============================================================

def test_weather_never_returns_policy_answer(searcher):

    questions = [
        "What is the weather in Calder County?",
        "Will it rain in Calder County?",
        "What is today's temperature?",
    ]

    for question in questions:

        response, answer = run_question(
            searcher,
            question
        )

        assert response["answerable"] is False

        assert "§1.1.2" not in answer


def test_18_year_old_does_not_use_minor_exception(searcher):

    question = (
        "Can I receive assistance if I am 18 years old?"
    )

    response, answer = run_question(
        searcher,
        question
    )

    assert response["answerable"] is True

    assert "§2.1.2" in answer

    # Prevent the exact bug shown in your test report.
    assert "§2.3.1" not in answer
    assert "16 or 17" not in answer
    assert "17 may be eligible" not in answer


def test_17_year_old_uses_minor_exception(searcher):

    question = (
        "Can I receive assistance if I am 17 years old?"
    )

    response, answer = run_question(
        searcher,
        question
    )

    assert response["answerable"] is True

    assert "§2.3.1" in answer

    assert_contains_any(
        answer,
        [
            "16 or 17",
            "17 may be eligible",
        ]
    )


def test_direct_clause_reference_is_exact(searcher):

    question = "What does §2.1.2 say?"

    response, answer = run_question(
        searcher,
        question
    )

    assert response["answerable"] is True

    clauses = get_clause_ids(response)

    # The requested clause must be present.
    assert "§2.1.2" in clauses

    # Do not allow another clause to be cited instead.
    assert "📄 Policy citation: §2.1.1" not in answer


# ============================================================
# TEST COUNT
# ============================================================

def test_suite_loaded():

    """
    Sanity check that pytest is discovering the test file.
    """
    assert True