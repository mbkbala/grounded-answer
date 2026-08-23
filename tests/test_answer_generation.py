# ============================================================
# GROUNDED POLICY ASSISTANT
# ANSWER GENERATION TESTS
# ============================================================

from src.generation.grounded_answer import (
    GroundedAnswerGenerator,
    build_grounded_answer,
    extract_relevant_sentence,
)


# ============================================================
# TEST DATA
# ============================================================

ADMINISTRATION_RESULT = {
    "clause": "§1.1.2",
    "text": (
        "The Program is administered by the "
        "Calder County Department of Household Services "
        "(\"the Department\"). "
        "Day-to-day administration is carried out by "
        "caseworkers at the four district offices."
    ),
    "hybrid_score": 0.98,
}


ELIGIBILITY_RESULT = {
    "clause": "§2.1.2",
    "text": (
        "To qualify for the Household Support Program, "
        "a household must satisfy the eligibility requirements."
    ),
    "hybrid_score": 0.95,
}


EXCLUSION_RESULT = {
    "clause": "§4.1.1",
    "text": (
        "A person who is confined in a correctional facility "
        "is excluded from receiving Household Support benefits."
    ),
    "hybrid_score": 0.94,
}


# ============================================================
# BASIC ANSWER GENERATION
# ============================================================

def test_single_clause_answer():
    generator = GroundedAnswerGenerator()

    retrieval_response = {
        "answerable": True,
        "results": [
            ADMINISTRATION_RESULT,
        ],
    }

    response = generator.generate(
        "Who administers the program?",
        retrieval_response,
    )

    assert response["answerable"] is True
    assert response["citations"] == ["§1.1.2"]
    assert "administered" in response["answer"].lower()


# ============================================================
# ONE CLAUSE ONLY
# ============================================================

def test_multiple_results_returns_one_clause():
    generator = GroundedAnswerGenerator()

    retrieval_response = {
        "answerable": True,
        "results": [
            ADMINISTRATION_RESULT,
            ELIGIBILITY_RESULT,
            EXCLUSION_RESULT,
        ],
    }

    response = generator.generate(
        "Who administers the program?",
        retrieval_response,
    )

    assert response["answerable"] is True

    # Only ONE citation should be returned.
    assert len(response["citations"]) == 1

    # Administration must win.
    assert response["citations"][0] == "§1.1.2"


# ============================================================
# EXCLUSION MUST BEAT GENERAL ELIGIBILITY
# ============================================================

def test_exclusion_beats_eligibility():
    generator = GroundedAnswerGenerator()

    retrieval_response = {
        "answerable": True,
        "results": [
            ELIGIBILITY_RESULT,
            EXCLUSION_RESULT,
        ],
    }

    response = generator.generate(
        "Are people in a correctional facility eligible?",
        retrieval_response,
    )

    assert response["answerable"] is True

    assert response["citations"] == ["§4.1.1"]

    assert "correctional" in response["answer"].lower()


# ============================================================
# DIRECT CLAUSE REFERENCE
# ============================================================

def test_direct_clause_reference():
    generator = GroundedAnswerGenerator()

    retrieval_response = {
        "answerable": True,
        "results": [
            ADMINISTRATION_RESULT,
            ELIGIBILITY_RESULT,
        ],
    }

    response = generator.generate(
        "What does §2.1.2 say?",
        retrieval_response,
    )

    assert response["answerable"] is True
    assert response["citations"] == ["§2.1.2"]


# ============================================================
# EMPTY QUESTION
# ============================================================

def test_empty_question():
    generator = GroundedAnswerGenerator()

    response = generator.generate(
        "",
        {
            "answerable": True,
            "results": [ADMINISTRATION_RESULT],
        },
    )

    assert response["answerable"] is False
    assert "don't know" in response["answer"].lower()
    assert response["citations"] == []


# ============================================================
# NO RETRIEVAL RESPONSE
# ============================================================

def test_missing_retrieval_response():
    generator = GroundedAnswerGenerator()

    response = generator.generate(
        "Who administers the program?",
        None,
    )

    assert response["answerable"] is False
    assert response["citations"] == []


# ============================================================
# RETRIEVAL SAYS OUTSIDE POLICY
# ============================================================

def test_retrieval_marks_question_unanswerable():
    generator = GroundedAnswerGenerator()

    response = generator.generate(
        "What is the capital of France?",
        {
            "answerable": False,
            "reason": "Question is outside policy scope.",
            "results": [],
        },
    )

    assert response["answerable"] is False
    assert response["citations"] == []
    assert "don't know" in response["answer"].lower()


# ============================================================
# NO USABLE EVIDENCE
# ============================================================

def test_no_usable_evidence():
    generator = GroundedAnswerGenerator()

    response = generator.generate(
        "Who administers the program?",
        {
            "answerable": True,
            "results": [
                {
                    "clause": "§1.1.2",
                    "text": "",
                }
            ],
        },
    )

    assert response["answerable"] is False
    assert response["citations"] == []


# ============================================================
# SENTENCE EXTRACTION
# ============================================================

def test_extract_relevant_sentence():
    text = (
        "The Program is administered by the Calder County "
        "Department of Household Services. "
        "Day-to-day administration is carried out by "
        "caseworkers at the four district offices."
    )

    sentence = extract_relevant_sentence(
        "Who administers the program?",
        text,
    )

    assert "administered" in sentence.lower()
    assert "caseworkers" not in sentence.lower()


# ============================================================
# GROUNDING
# ============================================================

def test_answer_contains_only_selected_clause():
    generator = GroundedAnswerGenerator()

    retrieval_response = {
        "answerable": True,
        "results": [
            ADMINISTRATION_RESULT,
            ELIGIBILITY_RESULT,
            EXCLUSION_RESULT,
        ],
    }

    response = generator.generate(
        "Who administers the program?",
        retrieval_response,
    )

    assert response["citations"] == ["§1.1.2"]

    # The answer should not combine unrelated clauses.
    assert "eligibility requirements" not in response["answer"].lower()
    assert "correctional facility" not in response["answer"].lower()