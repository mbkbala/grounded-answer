"""
============================================================
GROUNDED POLICY ASSISTANT
FULL PIPELINE TEST SUITE

Tests:

    Question
       ↓
    HybridSearch
       ↓
    GroundedAnswerGenerator
       ↓
    Final Answer
       ↓
    Citations
       ↓
    Answerability

Run with pytest:

    python -m pytest tests/test_pipeline.py -v

Run the complete test suite:

    python -m pytest -v
============================================================
"""

import sys
from pathlib import Path


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORT COMPONENTS
# ============================================================

from src.retrieval.hybrid_search import HybridSearch
from src.generation.grounded_answer import GroundedAnswerGenerator


# ============================================================
# TEST CASES
# ============================================================

TEST_CASES = [

    # --------------------------------------------------------
    # 1
    # --------------------------------------------------------

    {
        "name": "Administration",
        "question": "Who administers the program?",
        "expected_answerable": True,
        "expected_clause": "§1.1.2",
        "expected_text": "Calder County Department of Household Services",
    },

    # --------------------------------------------------------
    # 2
    # --------------------------------------------------------

    {
        "name": "General eligibility",
        "question": "What are the eligibility requirements?",
        "expected_answerable": True,
        "expected_clause": "§2.1.2",
        "expected_text": "resident in Calder County",
    },

    # --------------------------------------------------------
    # 3
    # --------------------------------------------------------

    {
        "name": "Age 18 eligibility",
        "question": "Can an 18 year old qualify for the program?",
        "expected_answerable": True,
        "expected_clause": "§2.1.2",
    },

    # --------------------------------------------------------
    # 4
    # --------------------------------------------------------

    {
        "name": "Minor eligibility - age 17",
        "question": "Can a 17 year old receive assistance?",
        "expected_answerable": True,
        "expected_clause": "§2.3.1",
        "expected_text": "16 or 17",
    },

    # --------------------------------------------------------
    # 5
    # --------------------------------------------------------

    {
        "name": "Minor eligibility - age 16",
        "question": "Can a 16 year old receive assistance?",
        "expected_answerable": True,
        "expected_clause": "§2.3.1",
        "expected_text": "16 or 17",
    },

    # --------------------------------------------------------
    # 6
    # --------------------------------------------------------

    {
        "name": "Residence requirement",
        "question": "Do I need to live in the county to qualify?",
        "expected_answerable": True,
        "expected_clause": "§2.1.2",
        "expected_text": "resident in Calder County",
    },

    # --------------------------------------------------------
    # 7
    # --------------------------------------------------------

    {
        "name": "Income requirement",
        "question": "Is there an income requirement?",
        "expected_answerable": True,
        "expected_clause": "§2.1.2",
        "expected_text": "countable income",
    },

    # --------------------------------------------------------
    # 8
    # --------------------------------------------------------

    {
        "name": "Household resources",
        "question": "Are household resources considered?",
        "expected_answerable": True,
        "expected_clause": "§2.4",
        "expected_text": "countable resources",
    },

    # --------------------------------------------------------
    # 9
    # --------------------------------------------------------

    {
        "name": "Application",
        "question": "How do I apply for assistance?",
        "expected_answerable": True,
        "expected_clause": "§2.1.2",
    },

    # --------------------------------------------------------
    # 10
    # --------------------------------------------------------

    {
        "name": "General exclusion",
        "question": "Who is excluded from the program?",
        "expected_answerable": True,
        "expected_clause": "§4.1.1",
    },

    # --------------------------------------------------------
    # 11
    # --------------------------------------------------------

    {
        "name": "Correctional facility exclusion",
        "question": "Are people in a correctional facility eligible?",
        "expected_answerable": True,
        "expected_clause": "§4.1.1",
    },

    # --------------------------------------------------------
    # 12
    # --------------------------------------------------------

    {
        "name": "Detention exclusion",
        "question": "Can someone who is detained receive benefits?",
        "expected_answerable": True,
        "expected_clause": "§4.1.1",
    },

    # --------------------------------------------------------
    # 13
    # --------------------------------------------------------

    {
        "name": "Sanction exclusion",
        "question": "Can a person under a sanction receive assistance?",
        "expected_answerable": True,
        "expected_clause": "§4.1.1",
    },

    # --------------------------------------------------------
    # 14
    # --------------------------------------------------------

    {
        "name": "Direct clause reference",
        "question": "What does §2.1.2 say?",
        "expected_answerable": True,
        "expected_clause": "§2.1.2",
    },

    # --------------------------------------------------------
    # 15
    # --------------------------------------------------------

    {
        "name": "Direct administration clause",
        "question": "Explain §1.1.2",
        "expected_answerable": True,
        "expected_clause": "§1.1.2",
    },

    # --------------------------------------------------------
    # 16
    # --------------------------------------------------------

    {
        "name": "General policy question",
        "question": "What does the policy manual say about household support?",
        "expected_answerable": True,
        "expected_clause": None,
    },

    # --------------------------------------------------------
    # 17
    # --------------------------------------------------------

    {
        "name": "Outside policy - weather",
        "question": "What is the weather today?",
        "expected_answerable": False,
        "expected_clause": None,
    },

    # --------------------------------------------------------
    # 18
    # --------------------------------------------------------

    {
        "name": "Outside policy - programming",
        "question": "How do I write a Python program?",
        "expected_answerable": False,
        "expected_clause": None,
    },

    # --------------------------------------------------------
    # 19
    # --------------------------------------------------------

    {
        "name": "Outside policy - football",
        "question": "Who won the football match?",
        "expected_answerable": False,
        "expected_clause": None,
    },

    # --------------------------------------------------------
    # 20
    # --------------------------------------------------------

    {
        "name": "Empty question",
        "question": "",
        "expected_answerable": False,
        "expected_clause": None,
    },
]


# ============================================================
# HELPERS
# ============================================================

def get_citations(response):
    """
    Safely retrieve citations from final answer.
    """

    citations = response.get("citations", [])

    if not citations:
        return []

    return [
        str(citation)
        for citation in citations
    ]


def get_answer(response):
    """
    Safely retrieve final answer text.
    """

    return str(
        response.get("answer", "")
    ).strip()


# ============================================================
# RUN FULL PIPELINE
# ============================================================

def run_tests():

    print("=" * 70)
    print("GROUNDED POLICY ASSISTANT - FULL PIPELINE TEST")
    print("=" * 70)

    # --------------------------------------------------------
    # Load retrieval engine
    # --------------------------------------------------------

    print("\nLoading HybridSearch...")

    try:

        searcher = HybridSearch()

    except Exception as exc:

        print("\nERROR: Could not load HybridSearch.")
        print(f"Reason: {exc}")

        return False

    # --------------------------------------------------------
    # Load answer generator
    # --------------------------------------------------------

    print("\nLoading GroundedAnswerGenerator...")

    try:

        generator = GroundedAnswerGenerator()

    except Exception as exc:

        print("\nERROR: Could not load GroundedAnswerGenerator.")
        print(f"Reason: {exc}")

        return False

    print("\nPipeline ready.")

    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------

    total = len(TEST_CASES)

    passed = 0
    failed = 0

    failures = []

    # ========================================================
    # RUN ALL TEST CASES
    # ========================================================

    print("\n")
    print("=" * 70)
    print("RUNNING ALL TEST CASES")
    print("=" * 70)

    for index, test in enumerate(
        TEST_CASES,
        start=1
    ):

        question = test["question"]

        print("\n" + "-" * 70)

        print(
            f"TEST {index:02d}: "
            f"{test['name']}"
        )

        print(
            f"Question: {question!r}"
        )

        try:

            # ------------------------------------------------
            # STEP 1
            # RETRIEVAL
            # ------------------------------------------------

            retrieval_response = searcher.search(
                question
            )

            # ------------------------------------------------
            # STEP 2
            # GENERATION
            # ------------------------------------------------

            final_response = generator.generate(
                question,
                retrieval_response
            )

            # ------------------------------------------------
            # ACTUAL VALUES
            # ------------------------------------------------

            actual_answerable = final_response.get(
                "answerable",
                False
            )

            actual_answer = get_answer(
                final_response
            )

            actual_citations = get_citations(
                final_response
            )

            # ------------------------------------------------
            # EXPECTED VALUES
            # ------------------------------------------------

            expected_answerable = test[
                "expected_answerable"
            ]

            expected_clause = test.get(
                "expected_clause"
            )

            expected_text = test.get(
                "expected_text"
            )

            # ------------------------------------------------
            # CHECK ANSWERABILITY
            # ------------------------------------------------

            answerable_ok = (
                actual_answerable
                == expected_answerable
            )

            # ------------------------------------------------
            # CHECK CITATION
            # ------------------------------------------------

            citation_ok = True

            if (
                expected_answerable
                and expected_clause is not None
            ):

                citation_ok = (
                    expected_clause
                    in actual_citations
                )

            # ------------------------------------------------
            # CHECK ANSWER CONTENT
            # ------------------------------------------------

            text_ok = True

            if (
                expected_answerable
                and expected_text
            ):

                text_ok = (
                    expected_text.lower()
                    in actual_answer.lower()
                )

            # ------------------------------------------------
            # UNKNOWN RESPONSE CHECK
            # ------------------------------------------------

            unknown_ok = True

            if not expected_answerable:

                unknown_ok = (
                    actual_answerable is False
                    and
                    (
                        "I don't know"
                        in actual_answer
                    )
                )

            # ------------------------------------------------
            # FINAL TEST RESULT
            # ------------------------------------------------

            test_passed = (
                answerable_ok
                and citation_ok
                and text_ok
                and unknown_ok
            )

            # ------------------------------------------------
            # PRINT RESULT
            # ------------------------------------------------

            print(
                f"Expected answerable : "
                f"{expected_answerable}"
            )

            print(
                f"Actual answerable   : "
                f"{actual_answerable}"
            )

            print(
                f"Citations           : "
                f"{actual_citations}"
            )

            if expected_clause:

                print(
                    f"Expected clause     : "
                    f"{expected_clause}"
                )

            if expected_text:

                print(
                    f"Expected text       : "
                    f"{expected_text!r}"
                )

            print(
                f"Answer              : "
                f"{actual_answer[:250]}"
            )

            # ------------------------------------------------
            # PASS
            # ------------------------------------------------

            if test_passed:

                print(
                    "RESULT              : PASS"
                )

                passed += 1

            # ------------------------------------------------
            # FAIL
            # ------------------------------------------------

            else:

                print(
                    "RESULT              : FAIL"
                )

                failed += 1

                failures.append(
                    {
                        "test": index,
                        "name": test["name"],
                        "question": question,
                        "expected_answerable":
                            expected_answerable,
                        "actual_answerable":
                            actual_answerable,
                        "expected_clause":
                            expected_clause,
                        "actual_citations":
                            actual_citations,
                        "expected_text":
                            expected_text,
                        "actual_answer":
                            actual_answer,
                        "answerable_ok":
                            answerable_ok,
                        "citation_ok":
                            citation_ok,
                        "text_ok":
                            text_ok,
                        "unknown_ok":
                            unknown_ok,
                        "reason":
                            final_response.get(
                                "reason"
                            ),
                    }
                )

        except Exception as exc:

            print(
                "RESULT              : ERROR"
            )

            print(
                f"Error               : {exc}"
            )

            failed += 1

            failures.append(
                {
                    "test": index,
                    "name": test["name"],
                    "question": question,
                    "error": str(exc),
                }
            )

    # ========================================================
    # SUMMARY
    # ========================================================

    print("\n")

    print("=" * 70)
    print("FULL PIPELINE TEST SUMMARY")
    print("=" * 70)

    print(
        f"Total tests : {total}"
    )

    print(
        f"Passed      : {passed}"
    )

    print(
        f"Failed      : {failed}"
    )

    accuracy = (
        passed / total * 100
        if total
        else 0
    )

    print(
        f"Accuracy    : {accuracy:.1f}%"
    )

    # ========================================================
    # FAILURE DETAILS
    # ========================================================

    if failures:

        print("\n")

        print("=" * 70)
        print("FAILED TESTS")
        print("=" * 70)

        for failure in failures:

            print("\n")

            print(
                f"TEST {failure['test']:02d}: "
                f"{failure['name']}"
            )

            print(
                f"Question: "
                f"{failure['question']!r}"
            )

            if "error" in failure:

                print(
                    f"ERROR: "
                    f"{failure['error']}"
                )

                continue

            print(
                f"Expected answerable: "
                f"{failure['expected_answerable']}"
            )

            print(
                f"Actual answerable: "
                f"{failure['actual_answerable']}"
            )

            print(
                f"Expected clause: "
                f"{failure['expected_clause']}"
            )

            print(
                f"Actual citations: "
                f"{failure['actual_citations']}"
            )

            print(
                f"Expected text: "
                f"{failure['expected_text']}"
            )

            print(
                f"Actual answer: "
                f"{failure['actual_answer']}"
            )

            print(
                f"Answerable check: "
                f"{failure['answerable_ok']}"
            )

            print(
                f"Citation check: "
                f"{failure['citation_ok']}"
            )

            print(
                f"Text check: "
                f"{failure['text_ok']}"
            )

            print(
                f"Unknown check: "
                f"{failure['unknown_ok']}"
            )

            print(
                f"Reason: "
                f"{failure['reason']}"
            )

    # ========================================================
    # FINAL STATUS
    # ========================================================

    print("\n")

    print("=" * 70)

    if failed == 0:

        print(
            "ALL FULL PIPELINE TESTS PASSED"
        )

    else:

        print(
            f"{failed} TEST(S) NEED ATTENTION"
        )

    print("=" * 70)

    return failed == 0


# ============================================================
# PYTEST ENTRY POINT
# ============================================================

def test_full_pipeline():
    """
    Pytest entry point.

    The complete pipeline suite contains 20 test cases.
    run_tests() performs the individual checks and returns
    True only when every case passes.
    """

    assert run_tests(), "Full pipeline test suite failed"


# ============================================================
# OPTIONAL DIRECT EXECUTION
# ============================================================

if __name__ == "__main__":

    success = run_tests()

    sys.exit(
        0 if success else 1
    )