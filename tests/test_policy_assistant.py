# ============================================================
# GROUNDED POLICY ASSISTANT
# TEST SUITE
# ============================================================

import json
from pathlib import Path

import pytest

from src.retrieval.hybrid_search import HybridSearch
from src.answer_generator import generate_answer
from src.ingestion.amendment_parser import parse_amendment_file
from src.reasoning.temporal_policy import TemporalPolicy


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

CLAUSES_FILE = (
    PROJECT_ROOT
    / "data"
    / "clauses.json"
)

AMENDMENTS_FILE = (
    PROJECT_ROOT
    / "data"
    / "amendments.json"
)

AMENDMENT_FILE = (
    PROJECT_ROOT
    / "Data pack"
    / "Amendment No. 2026-01.md"
)


# ============================================================
# SEARCH ENGINE FIXTURE
# ============================================================

@pytest.fixture(scope="module")
def searcher():

    return HybridSearch()


# ============================================================
# TEMPORAL POLICY FIXTURE
# ============================================================

@pytest.fixture(scope="module")
def temporal_policy():

    return TemporalPolicy(
        AMENDMENTS_FILE
    )


# ============================================================
# BASIC DATA TESTS
# ============================================================

def test_clauses_file_exists():

    assert CLAUSES_FILE.exists(), (
        "data/clauses.json does not exist."
    )


def test_clauses_file_is_valid():

    with CLAUSES_FILE.open(
        "r",
        encoding="utf-8"
    ) as file:

        clauses = json.load(file)

    assert isinstance(
        clauses,
        list
    )

    assert len(clauses) > 0


def test_amendments_file_exists():

    assert AMENDMENTS_FILE.exists(), (
        "data/amendments.json does not exist."
    )


def test_amendments_file_is_valid():

    with AMENDMENTS_FILE.open(
        "r",
        encoding="utf-8"
    ) as file:

        data = json.load(file)

    assert "amendments" in data

    assert isinstance(
        data["amendments"],
        list
    )


# ============================================================
# AMENDMENT SOURCE TESTS
# ============================================================

def test_amendment_source_exists():

    assert AMENDMENT_FILE.exists(), (
        "Amendment No. 2026-01.md does not exist."
    )


def test_amendment_parser_reads_source():

    amendment = parse_amendment_file(
        AMENDMENT_FILE
    )

    assert amendment is not None

    assert (
        "2026-01"
        in amendment["amendment_id"]
    )

    assert amendment["title"] != ""

    assert amendment["text"] != ""


def test_amendment_effective_date():

    amendment = parse_amendment_file(
        AMENDMENT_FILE
    )

    assert (
        amendment["effective_date"]
        == "2026-03-01"
    )


def test_amendment_extracts_clause_references():

    amendment = parse_amendment_file(
        AMENDMENT_FILE
    )

    clauses = amendment["clauses"]

    assert "§6.4.1" in clauses
    assert "§4.3.2" in clauses
    assert "§9.1.4" in clauses
    assert "§6.6.1" in clauses
    assert "§10.5.2" in clauses
    assert "§10.5.3" in clauses


# ============================================================
# TEMPORAL POLICY TESTS
# ============================================================

def test_amendment_is_not_active_before_effective_date(
    temporal_policy
):

    active = temporal_policy.get_active_amendments(
        "2026-02-28"
    )

    amendment_ids = [
        item["amendment_id"]
        for item in active
    ]

    assert "2026-01" not in amendment_ids


def test_amendment_is_active_on_effective_date(
    temporal_policy
):

    active = temporal_policy.get_active_amendments(
        "2026-03-01"
    )

    amendment_ids = [
        item["amendment_id"]
        for item in active
    ]

    assert "2026-01" in amendment_ids


def test_amendment_remains_active_after_effective_date(
    temporal_policy
):

    active = temporal_policy.get_active_amendments(
        "2026-08-23"
    )

    amendment_ids = [
        item["amendment_id"]
        for item in active
    ]

    assert "2026-01" in amendment_ids


def test_latest_amendment_before_effective_date(
    temporal_policy
):

    latest = temporal_policy.get_latest_amendment(
        "2026-02-28"
    )

    assert latest is None


def test_latest_amendment_after_effective_date(
    temporal_policy
):

    latest = temporal_policy.get_latest_amendment(
        "2026-03-01"
    )

    assert latest is not None

    assert (
        latest["amendment_id"]
        == "2026-01"
    )


# ============================================================
# CLAUSE HISTORY TESTS
# ============================================================

def test_clause_history_for_earnings_disregard(
    temporal_policy
):

    history = temporal_policy.get_clause_history(
        "§6.4.1"
    )

    assert len(history) >= 1

    amendment_ids = [
        item["amendment_id"]
        for item in history
    ]

    assert "2026-01" in amendment_ids


def test_clause_history_for_sanctions(
    temporal_policy
):

    history = temporal_policy.get_clause_history(
        "§10.5.2"
    )

    assert len(history) >= 1

    amendment_ids = [
        item["amendment_id"]
        for item in history
    ]

    assert "2026-01" in amendment_ids


def test_clause_history_for_reporting_deadline(
    temporal_policy
):

    history = temporal_policy.get_clause_history(
        "§4.3.2"
    )

    assert len(history) >= 1

    assert any(
        item["amendment_id"]
        == "2026-01"
        for item in history
    )


# ============================================================
# POLICY VERSION TESTS
# ============================================================

def test_policy_version_before_amendment(
    temporal_policy
):

    version = temporal_policy.get_policy_version(
        "2026-02-28"
    )

    assert (
        version["amendments_active"]
        == 0
    )

    assert (
        version["latest_amendment"]
        is None
    )


def test_policy_version_after_amendment(
    temporal_policy
):

    version = temporal_policy.get_policy_version(
        "2026-03-01"
    )

    assert (
        version["amendments_active"]
        >= 1
    )

    assert (
        version["latest_amendment"]
        is not None
    )

    assert (
        version["latest_amendment"]
        ["amendment_id"]
        == "2026-01"
    )


# ============================================================
# TEMPORAL EXPLANATION TEST
# ============================================================

def test_temporal_explanation(
    temporal_policy
):

    explanation = temporal_policy.explain(
        "2026-03-01"
    )

    assert isinstance(
        explanation,
        str
    )

    assert "2026-03-01" in explanation

    assert (
        "Amendment No. 2026-01"
        in explanation
        or "2026-01"
        in explanation
    )


# ============================================================
# SEARCH ENGINE TESTS
# ============================================================

def test_search_engine_loads(
    searcher
):

    assert searcher is not None

    assert len(
        searcher.clauses
    ) > 0


# ============================================================
# ADMINISTRATION
# ============================================================

def test_administration_question(
    searcher
):

    question = (
        "Who administers the program?"
    )

    response = searcher.search(
        question
    )

    assert response["answerable"] is True

    clauses = [
        result["clause"]
        for result in response["results"]
    ]

    assert "§1.1.2" in clauses


def test_administration_answer(
    searcher
):

    question = (
        "Who administers the program?"
    )

    response = searcher.search(
        question
    )

    answer = generate_answer(
        question,
        response
    )

    assert "§1.1.2" in answer

    assert (
        "administered"
        in answer.lower()
        or
        "administration"
        in answer.lower()
    )


# ============================================================
# ELIGIBILITY
# ============================================================

def test_general_eligibility(
    searcher
):

    question = (
        "What are the eligibility requirements?"
    )

    response = searcher.search(
        question
    )

    assert response["answerable"] is True

    clauses = [
        result["clause"]
        for result in response["results"]
    ]

    assert "§2.1.2" in clauses


def test_eligibility_answer_is_grounded(
    searcher
):

    question = (
        "What are the eligibility requirements?"
    )

    response = searcher.search(
        question
    )

    answer = generate_answer(
        question,
        response
    )

    assert "§2.1.2" in answer


# ============================================================
# AGE TESTS
# ============================================================

def test_age_18(
    searcher
):

    question = (
        "Can an 18 years old person qualify?"
    )

    response = searcher.search(
        question
    )

    assert response["answerable"] is True

    clauses = [
        result["clause"]
        for result in response["results"]
    ]

    assert "§2.1.2" in clauses

    assert "§2.3.1" not in clauses


def test_minor_age(
    searcher
):

    question = (
        "Can a 17 years old person qualify?"
    )

    response = searcher.search(
        question
    )

    assert response["answerable"] is True

    clauses = [
        result["clause"]
        for result in response["results"]
    ]

    assert "§2.3.1" in clauses


# ============================================================
# EXCLUSION TESTS
# ============================================================

def test_correctional_facility_exclusion(
    searcher
):

    question = (
        "Is someone detained in a correctional "
        "facility eligible?"
    )

    response = searcher.search(
        question
    )

    assert response["answerable"] is True

    clauses = [
        result["clause"]
        for result in response["results"]
    ]

    assert "§4.1.1" in clauses


def test_sanction_question(
    searcher
):

    question = (
        "Can someone subject to a sanction "
        "receive assistance?"
    )

    response = searcher.search(
        question
    )

    assert response["answerable"] is True

    clauses = [
        result["clause"]
        for result in response["results"]
    ]

    assert "§4.1.1" in clauses


# ============================================================
# OUT-OF-SCOPE TEST
# ============================================================

def test_outside_policy_question(
    searcher
):

    question = (
        "What is the weather today?"
    )

    response = searcher.search(
        question
    )

    assert response["answerable"] is False

    assert (
        len(response["results"])
        == 0
    )


def test_programming_question_is_outside_policy(
    searcher
):

    question = (
        "How do I write Python code?"
    )

    response = searcher.search(
        question
    )

    assert response["answerable"] is False


# ============================================================
# EMPTY QUESTION
# ============================================================

def test_empty_question(
    searcher
):

    response = searcher.search(
        ""
    )

    assert response["answerable"] is False

    assert (
        len(response["results"])
        == 0
    )


# ============================================================
# DIRECT CLAUSE REFERENCE
# ============================================================

def test_direct_clause_reference(
    searcher
):

    question = (
        "What does §2.1.2 say?"
    )

    response = searcher.search(
        question
    )

    assert response["answerable"] is True

    assert (
        response["results"][0]["clause"]
        == "§2.1.2"
    )


# ============================================================
# REFUSAL / UNKNOWN QUESTION
# ============================================================

def test_unsupported_question(
    searcher
):

    question = (
        "What is the favorite color of the "
        "program administrator?"
    )

    response = searcher.search(
        question
    )

    # The system should not confidently
    # invent an answer.

    assert response["answerable"] is False


def test_refusal_contains_dont_know(
    searcher
):

    question = (
        "What is the favorite color of the "
        "program administrator?"
    )

    response = searcher.search(
        question
    )

    answer = generate_answer(
        question,
        response
    )

    assert (
        "I don't know"
        in answer
    )


# ============================================================
# GROUNDING TESTS
# ============================================================

def test_answer_contains_policy_citation(
    searcher
):

    question = (
        "Who administers the program?"
    )

    response = searcher.search(
        question
    )

    answer = generate_answer(
        question,
        response
    )

    assert "Policy citation" in answer

    assert "§1.1.2" in answer


def test_answer_only_uses_retrieved_clause(
    searcher
):

    question = (
        "Who administers the program?"
    )

    response = searcher.search(
        question
    )

    retrieved_clauses = {
        result["clause"]
        for result in response["results"]
    }

    answer = generate_answer(
        question,
        response
    )

    # The authoritative answer must be
    # based on §1.1.2.

    assert "§1.1.2" in retrieved_clauses

    assert "§1.1.2" in answer


# ============================================================
# AMENDMENT CONTENT TESTS
# ============================================================

def test_amendment_contains_earnings_change():

    text = AMENDMENT_FILE.read_text(
        encoding="utf-8"
    )

    assert "$120 per month" in text

    assert "$175 per month" in text

    assert "§6.4.1(a)" in text


def test_amendment_contains_reporting_change():

    text = AMENDMENT_FILE.read_text(
        encoding="utf-8"
    )

    assert "10 calendar days" in text

    assert "14 calendar days" in text

    assert "§4.3.2" in text

    assert "§9.1.4" in text


def test_amendment_contains_income_thresholds():

    text = AMENDMENT_FILE.read_text(
        encoding="utf-8"
    )

    assert "$1,225" in text
    assert "$1,650" in text
    assert "$2,075" in text
    assert "$2,500" in text
    assert "$2,925" in text


def test_amendment_contains_sanction_change():

    text = AMENDMENT_FILE.read_text(
        encoding="utf-8"
    )

    assert "20 per cent" in text

    assert "15 per cent" in text

    assert "10.5.3A" in text


# ============================================================
# TRANSITIONAL PROVISION TESTS
# ============================================================

def test_transitional_provision_exists():

    text = AMENDMENT_FILE.read_text(
        encoding="utf-8"
    )

    assert "Transitional provision" in text

    assert (
        "determination made on or after 1 March 2026"
        in text
    )

    assert (
        "change of circumstances occurring on or after 1 March 2026"
        in text
    )


def test_cross_date_provision_exists():

    text = AMENDMENT_FILE.read_text(
        encoding="utf-8"
    )

    assert (
        "spanning 1 March 2026"
        in text
    )

    assert "§7.4.3" in text


# ============================================================
# FULL PIPELINE SMOKE TEST
# ============================================================

def test_full_grounded_pipeline(
    searcher
):

    question = (
        "Who administers the program?"
    )

    response = searcher.search(
        question
    )

    answer = generate_answer(
        question,
        response
    )

    assert response is not None

    assert response["answerable"] is True

    assert len(
        response["results"]
    ) > 0

    assert isinstance(
        answer,
        str
    )

    assert len(
        answer.strip()
    ) > 0

    assert "§1.1.2" in answer