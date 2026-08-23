"""
============================================================
GROUNDED POLICY ASSISTANT
RETRIEVAL TEST SUITE
============================================================

Tests the HybridSearch retrieval engine using carefully
selected policy and out-of-scope questions.

Run all tests:
    python -m pytest -v

Run only retrieval tests:
    python -m pytest tests/test_retrieval.py -v

Run this file directly:
    python tests/test_retrieval.py
============================================================
"""

import sys
from pathlib import Path


# ============================================================
# PROJECT ROOT
# ============================================================

# tests/test_retrieval.py
#       ↑
# parents[0] = tests
# parents[1] = project root
PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORT RETRIEVAL ENGINE
# ============================================================

from src.retrieval.hybrid_search import HybridSearch


# ============================================================
# TEST CASES
# ============================================================

TEST_CASES = [

    # --------------------------------------------------------
    # 1. Administration
    # --------------------------------------------------------

    {
        "name": "Administration",
        "question": "Who administers the program?",
        "expected_answerable": True,
        "expected_clause": "§1.1.2",
    },

    # --------------------------------------------------------
    # 2. General eligibility
    # --------------------------------------------------------

    {
        "name": "General eligibility",
        "question": "What are the eligibility requirements?",
        "expected_answerable": True,
        "expected_clause": "§2.1.2",
    },

    # --------------------------------------------------------
    # 3. Age 18
    # --------------------------------------------------------

    {
        "name": "Age 18 eligibility",
        "question": "Can an 18 year old qualify for the program?",
        "expected_answerable": True,
        "expected_clause": "§2.1.2",
    },

    # --------------------------------------------------------
    # 4. Age 17
    # --------------------------------------------------------

    {
        "name": "Minor eligibility",
        "question": "Can a 17 year old receive assistance?",
        "expected_answerable": True,
        "expected_clause": "§2.3.1",
    },

    # --------------------------------------------------------
    # 5. Age 16
    # --------------------------------------------------------

    {
        "name": "Age 16 eligibility",
        "question": "Is someone who is 16 years old eligible?",
        "expected_answerable": True,
        "expected_clause": "§2.3.1",
    },

    # --------------------------------------------------------
    # 6. Residence
    # --------------------------------------------------------

    {
        "name": "Residence requirement",
        "question": "Do I need to live in the county to qualify?",
        "expected_answerable": True,
        "expected_clause": "§2.1.2",
    },

    # --------------------------------------------------------
    # 7. Income
    # --------------------------------------------------------

    {
        "name": "Income requirement",
        "question": "Is there an income requirement?",
        "expected_answerable": True,
        "expected_clause": "§2.1.2",
    },

    # --------------------------------------------------------
    # 8. Resources
    # --------------------------------------------------------

    {
        "name": "Resource requirement",
        "question": "Are household resources considered?",
        "expected_answerable": True,
        "expected_clause_prefix": "§2.4",
    },

    # --------------------------------------------------------
    # 9. Application
    # --------------------------------------------------------

    {
        "name": "Application",
        "question": "How do I apply for assistance?",
        "expected_answerable": True,
        "expected_clause": "§2.1.2",
    },

    # --------------------------------------------------------
    # 10. General exclusion
    # --------------------------------------------------------

    {
        "name": "General exclusion",
        "question": "Who is excluded from the program?",
        "expected_answerable": True,
        "expected_clause": "§4.1.1",
    },

    # --------------------------------------------------------
    # 11. Correctional facility
    # --------------------------------------------------------

    {
        "name": "Correctional facility exclusion",
        "question": "Are people in a correctional facility eligible?",
        "expected_answerable": True,
        "expected_clause": "§4.1.1",
    },

    # --------------------------------------------------------
    # 12. Detention
    # --------------------------------------------------------

    {
        "name": "Detention exclusion",
        "question": "Can someone who is detained receive benefits?",
        "expected_answerable": True,
        "expected_clause": "§4.1.1",
    },

    # --------------------------------------------------------
    # 13. Sanction
    # --------------------------------------------------------

    {
        "name": "Sanction exclusion",
        "question": "Can a person under a sanction receive assistance?",
        "expected_answerable": True,
        "expected_clause": "§4.1.1",
    },

    # --------------------------------------------------------
    # 14. Direct clause reference
    # --------------------------------------------------------

    {
        "name": "Direct clause reference",
        "question": "What does §2.1.2 say?",
        "expected_answerable": True,
        "expected_clause": "§2.1.2",
    },

    # --------------------------------------------------------
    # 15. Direct administration clause
    # --------------------------------------------------------

    {
        "name": "Direct administration clause",
        "question": "Explain §1.1.2",
        "expected_answerable": True,
        "expected_clause": "§1.1.2",
    },

    # --------------------------------------------------------
    # 16. General policy question
    # --------------------------------------------------------

    {
        "name": "Policy manual",
        "question": "What does the policy manual say about household support?",
        "expected_answerable": True,
        "expected_clause": None,
    },

    # --------------------------------------------------------
    # 17. Outside policy - weather
    # --------------------------------------------------------

    {
        "name": "Outside policy - weather",
        "question": "What is the weather today?",
        "expected_answerable": False,
        "expected_clause": None,
    },

    # --------------------------------------------------------
    # 18. Outside policy - programming
    # --------------------------------------------------------

    {
        "name": "Outside policy - programming",
        "question": "How do I write a Python program?",
        "expected_answerable": False,
        "expected_clause": None,
    },

    # --------------------------------------------------------
    # 19. Outside policy - football
    # --------------------------------------------------------

    {
        "name": "Outside policy - football",
        "question": "Who won the football match?",
        "expected_answerable": False,
        "expected_clause": None,
    },

    # --------------------------------------------------------
    # 20. Empty question
    # --------------------------------------------------------

    {
        "name": "Empty question",
        "question": "",
        "expected_answerable": False,
        "expected_clause": None,
    },
]


# ============================================================
# HELPER
# ============================================================

def get_clause_ids(response):
    """
    Extract clause IDs from retrieval results.
    """

    if not isinstance(response, dict):
        return []

    results = response.get("results", [])

    if not isinstance(results, list):
        return []

    return [
        result.get("clause")
        for result in results
        if isinstance(result, dict)
        and result.get("clause")
    ]


# ============================================================
# CLAUSE MATCHING
# ============================================================

def check_clause_match(
    actual_clauses,
    expected_clause=None,
    expected_clause_prefix=None,
):
    """
    Check whether the expected clause was retrieved.

    Supports:

        exact:
            §2.1.2

        prefix:
            §2.4

    Prefix matching allows:

        §2.4.1
        §2.4.2
        §2.4.3
        etc.
    """

    # --------------------------------------------------------
    # No clause requirement
    # --------------------------------------------------------

    if (
        expected_clause is None
        and expected_clause_prefix is None
    ):
        return True

    # --------------------------------------------------------
    # Exact clause
    # --------------------------------------------------------

    if expected_clause is not None:
        return expected_clause in actual_clauses

    # --------------------------------------------------------
    # Prefix clause
    # --------------------------------------------------------

    if expected_clause_prefix is not None:

        return any(
            isinstance(clause, str)
            and clause.startswith(expected_clause_prefix)
            for clause in actual_clauses
        )

    return False


# ============================================================
# SINGLE TEST EXECUTION
# ============================================================

def execute_test(searcher, test):
    """
    Execute one retrieval test case.

    Returns:
        passed: bool
        details: dict
    """

    question = test["question"]

    response = searcher.search(question)

    if not isinstance(response, dict):
        raise TypeError(
            "HybridSearch.search() must return a dictionary."
        )

    # --------------------------------------------------------
    # Actual values
    # --------------------------------------------------------

    actual_answerable = response.get(
        "answerable",
        False,
    )

    actual_clauses = get_clause_ids(response)

    # --------------------------------------------------------
    # Expected values
    # --------------------------------------------------------

    expected_answerable = test[
        "expected_answerable"
    ]

    expected_clause = test.get(
        "expected_clause"
    )

    expected_clause_prefix = test.get(
        "expected_clause_prefix"
    )

    # --------------------------------------------------------
    # Answerability check
    # --------------------------------------------------------

    answerable_ok = (
        actual_answerable
        == expected_answerable
    )

    # --------------------------------------------------------
    # Clause check
    # --------------------------------------------------------

    clause_ok = True

    if expected_answerable:

        clause_ok = check_clause_match(
            actual_clauses=actual_clauses,
            expected_clause=expected_clause,
            expected_clause_prefix=expected_clause_prefix,
        )

    # --------------------------------------------------------
    # Final status
    # --------------------------------------------------------

    passed = (
        answerable_ok
        and clause_ok
    )

    details = {
        "response": response,
        "actual_answerable": actual_answerable,
        "actual_clauses": actual_clauses,
        "expected_answerable": expected_answerable,
        "expected_clause": expected_clause,
        "expected_clause_prefix": expected_clause_prefix,
        "answerable_ok": answerable_ok,
        "clause_ok": clause_ok,
        "passed": passed,
    }

    return passed, details


# ============================================================
# RUN TESTS
# ============================================================

def run_tests():

    print("=" * 70)
    print("GROUNDED POLICY ASSISTANT - RETRIEVAL TEST SUITE")
    print("=" * 70)

    # --------------------------------------------------------
    # Load search engine
    # --------------------------------------------------------

    print("\nLoading HybridSearch...")

    try:

        searcher = HybridSearch()

    except Exception as exc:

        print("\nERROR: Could not load HybridSearch.")
        print(f"Reason: {exc}")

        return False

    # --------------------------------------------------------
    # Counters
    # --------------------------------------------------------

    passed = 0
    failed = 0

    failures = []

    # ========================================================
    # RUN EACH TEST
    # ========================================================

    for index, test in enumerate(
        TEST_CASES,
        start=1,
    ):

        question = test["question"]

        print("-" * 70)

        print(
            f"TEST {index:02d}: "
            f"{test['name']}"
        )

        print(
            f"Question: {question!r}"
        )

        try:

            test_passed, details = execute_test(
                searcher,
                test,
            )

            # ------------------------------------------------
            # Print results
            # ------------------------------------------------

            print(
                f"Expected answerable : "
                f"{details['expected_answerable']}"
            )

            print(
                f"Actual answerable   : "
                f"{details['actual_answerable']}"
            )

            print(
                f"Retrieved clauses   : "
                f"{details['actual_clauses']}"
            )

            if details["expected_clause"] is not None:

                print(
                    f"Expected clause     : "
                    f"{details['expected_clause']}"
                )

            if details["expected_clause_prefix"] is not None:

                print(
                    f"Expected clause     : "
                    f"{details['expected_clause_prefix']}.x"
                )

            # ------------------------------------------------
            # PASS
            # ------------------------------------------------

            if test_passed:

                print("RESULT              : PASS")

                passed += 1

            # ------------------------------------------------
            # FAIL
            # ------------------------------------------------

            else:

                print("RESULT              : FAIL")

                failed += 1

                failures.append(
                    {
                        "test": index,
                        "name": test["name"],
                        "question": question,
                        "expected_answerable":
                            details["expected_answerable"],
                        "actual_answerable":
                            details["actual_answerable"],
                        "expected_clause":
                            details["expected_clause"],
                        "expected_clause_prefix":
                            details["expected_clause_prefix"],
                        "actual_clauses":
                            details["actual_clauses"],
                        "reason":
                            details["response"].get(
                                "reason"
                            ),
                    }
                )

        # ====================================================
        # ERROR
        # ====================================================

        except Exception as exc:

            print("RESULT              : ERROR")

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

    total = len(TEST_CASES)

    accuracy = (
        (passed / total) * 100
        if total
        else 0
    )

    print("\n")
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    print(f"Total tests : {total}")
    print(f"Passed      : {passed}")
    print(f"Failed      : {failed}")
    print(f"Accuracy    : {accuracy:.1f}%")

    # ========================================================
    # FAILURE DETAILS
    # ========================================================

    if failures:

        print("\n")
        print("=" * 70)
        print("FAILED TESTS")
        print("=" * 70)

        for failure in failures:

            print()

            print(
                f"TEST {failure['test']:02d}: "
                f"{failure['name']}"
            )

            print(
                f"Question: "
                f"{failure['question']!r}"
            )

            # ------------------------------------------------
            # Runtime error
            # ------------------------------------------------

            if "error" in failure:

                print(
                    f"Error: "
                    f"{failure['error']}"
                )

                continue

            # ------------------------------------------------
            # Answerability
            # ------------------------------------------------

            print(
                f"Expected answerable: "
                f"{failure['expected_answerable']}"
            )

            print(
                f"Actual answerable: "
                f"{failure['actual_answerable']}"
            )

            # ------------------------------------------------
            # Clause
            # ------------------------------------------------

            if failure.get("expected_clause") is not None:

                print(
                    f"Expected clause: "
                    f"{failure['expected_clause']}"
                )

            if failure.get(
                "expected_clause_prefix"
            ) is not None:

                print(
                    f"Expected clause prefix: "
                    f"{failure['expected_clause_prefix']}"
                )

            print(
                f"Actual clauses: "
                f"{failure['actual_clauses']}"
            )

            # ------------------------------------------------
            # Reason
            # ------------------------------------------------

            if failure.get("reason"):

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

        print("ALL TESTS PASSED")

    else:

        print(
            f"{failed} TEST(S) NEED ATTENTION"
        )

    print("=" * 70)

    return failed == 0


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    success = run_tests()

    sys.exit(
        0 if success else 1
    )