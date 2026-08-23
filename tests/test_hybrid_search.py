# ============================================================
# GROUNDED POLICY ASSISTANT
# HYBRID RETRIEVAL ENGINE TESTS
# ============================================================

import sys
from pathlib import Path
from pprint import pprint


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORT SEARCH ENGINE
# ============================================================

from src.retrieval.hybrid_search import HybridSearch


# ============================================================
# TEST CONFIGURATION
# ============================================================

TOP_K = 5


# ============================================================
# TEST CASES
# ============================================================

TEST_CASES = [

    # --------------------------------------------------------
    # 1. GENERAL ELIGIBILITY
    # --------------------------------------------------------

    {
        "name": "General eligibility",
        "question": "What are the eligibility requirements?",
        "expected_type": "eligibility",
        "expected_clauses": ["§2.1.2", "§2.1.1"],
        "should_be_answerable": True,
    },

    # --------------------------------------------------------
    # 2. AGE 18
    # --------------------------------------------------------

    {
        "name": "Age 18 eligibility",
        "question": "Can someone who is 18 years old qualify?",
        "expected_type": "age_18",
        "expected_clauses": ["§2.1.2"],
        "should_be_answerable": True,
    },

    # --------------------------------------------------------
    # 3. MINOR
    # --------------------------------------------------------

    {
        "name": "Minor eligibility",
        "question": "Can a 17 year old receive assistance?",
        "expected_type": "age_minor",
        "expected_clauses": [
            "§2.1.2",
            "§2.3.1",
            "§2.3.2",
        ],
        "should_be_answerable": True,
    },

    # --------------------------------------------------------
    # 4. RESIDENCE
    # --------------------------------------------------------

    {
        "name": "Residence requirement",
        "question": "Do I need to live in Calder County to qualify?",
        "expected_type": "residence",
        "expected_clauses": ["§2.1.2"],
        "should_be_answerable": True,
    },

    # --------------------------------------------------------
    # 5. INCOME
    # --------------------------------------------------------

    {
        "name": "Income requirement",
        "question": "What are the income requirements?",
        "expected_type": "income",
        "expected_clauses": ["§2.1.2"],
        "should_be_answerable": True,
    },

    # --------------------------------------------------------
    # 6. INCOME DISREGARD
    # --------------------------------------------------------

    {
        "name": "Income disregard",
        "question": "What income is disregarded?",
        "expected_type": "income",
        "expected_clauses": ["§6.4.1"],
        "should_be_answerable": True,
    },

    # --------------------------------------------------------
    # 7. RESOURCES
    # --------------------------------------------------------

    {
        "name": "Resource requirements",
        "question": "What resources are considered?",
        "expected_type": "resources",
        "expected_clauses": ["§2.4"],
        "should_be_answerable": True,
    },

    # --------------------------------------------------------
    # 8. APPLICATION
    # --------------------------------------------------------

    {
        "name": "Application process",
        "question": "How do I apply for the program?",
        "expected_type": "application",
        "expected_clauses": ["§2.1.2"],
        "should_be_answerable": True,
    },

    # --------------------------------------------------------
    # 9. ADMINISTRATION
    # --------------------------------------------------------

    {
        "name": "Program administration",
        "question": "Who administers the program?",
        "expected_type": "administration",
        "expected_clauses": ["§1.1.2"],
        "should_be_answerable": True,
    },

    # --------------------------------------------------------
    # 10. CORRECTIONAL FACILITY
    # --------------------------------------------------------

    {
        "name": "Correctional facility exclusion",
        "question": "Can someone in a correctional facility receive assistance?",
        "expected_type": "correctional_exclusion",
        "expected_clauses": ["§4.1.1"],
        "should_be_answerable": True,
    },

    # --------------------------------------------------------
    # 11. SANCTION
    # --------------------------------------------------------

    {
        "name": "Sanction exclusion",
        "question": "Are people under a sanction excluded?",
        "expected_type": "sanction_exclusion",
        "expected_clauses": ["§4.1.1"],
        "should_be_answerable": True,
    },

    # --------------------------------------------------------
    # 12. GENERAL EXCLUSION
    # --------------------------------------------------------

    {
        "name": "General exclusion",
        "question": "Who is excluded from the program?",
        "expected_type": "exclusion",
        "expected_clauses": ["§4.1.1"],
        "should_be_answerable": True,
    },

    # --------------------------------------------------------
    # 13. EXACT CLAUSE
    # --------------------------------------------------------

    {
        "name": "Exact clause reference",
        "question": "What does §2.4 say?",
        "expected_type": "clause_reference",
        "expected_clauses": ["§2.4"],
        "should_be_answerable": True,
    },

    # --------------------------------------------------------
    # 14. EXACT CHILD CLAUSE
    # --------------------------------------------------------

    {
        "name": "Exact child clause",
        "question": "Explain §2.1.2",
        "expected_type": "clause_reference",
        "expected_clauses": ["§2.1.2"],
        "should_be_answerable": True,
    },

    # --------------------------------------------------------
    # 15. OUTSIDE POLICY
    # --------------------------------------------------------

    {
        "name": "Outside policy - weather",
        "question": "What is the weather today?",
        "expected_type": "outside_policy",
        "expected_clauses": [],
        "should_be_answerable": False,
    },

    # --------------------------------------------------------
    # 16. OUTSIDE POLICY - PROGRAMMING
    # --------------------------------------------------------

    {
        "name": "Outside policy - programming",
        "question": "How do I write Python code?",
        "expected_type": "outside_policy",
        "expected_clauses": [],
        "should_be_answerable": False,
    },

    # --------------------------------------------------------
    # 17. EMPTY QUESTION
    # --------------------------------------------------------

    {
        "name": "Empty question",
        "question": "",
        "expected_type": None,
        "expected_clauses": [],
        "should_be_answerable": False,
    },

    # --------------------------------------------------------
    # 18. NATURAL LANGUAGE QUESTION
    # --------------------------------------------------------

    {
        "name": "Natural language eligibility",
        "question": "I live in the county and want financial assistance. What do I need?",
        "expected_type": "residence",
        "expected_clauses": ["§2.1.2"],
        "should_be_answerable": True,
    },
]


# ============================================================
# HELPER
# ============================================================

def get_result_clauses(response):
    return [
        result.get("clause")
        for result in response.get("results", [])
    ]


# ============================================================
# RUN TESTS
# ============================================================

def run_tests():

    print("=" * 70)
    print("GROUNDED POLICY ASSISTANT - HYBRID RETRIEVAL TESTS")
    print("=" * 70)

    print("\nLoading HybridSearch...\n")

    searcher = HybridSearch()

    passed = 0
    failed = 0

    for index, test in enumerate(TEST_CASES, start=1):

        print("\n" + "-" * 70)
        print(f"TEST {index}: {test['name']}")
        print("-" * 70)

        question = test["question"]

        print(f"Question: {question}")

        try:

            response = searcher.search(
                question,
                top_k=TOP_K
            )

            actual_answerable = response.get(
                "answerable",
                False
            )

            actual_type = response.get(
                "question_type"
            )

            actual_clauses = get_result_clauses(
                response
            )

            print(
                f"Expected type : {test['expected_type']}"
            )

            print(
                f"Actual type   : {actual_type}"
            )

            print(
                f"Expected answerable : "
                f"{test['should_be_answerable']}"
            )

            print(
                f"Actual answerable   : "
                f"{actual_answerable}"
            )

            print(
                f"Retrieved clauses   : "
                f"{actual_clauses}"
            )

            # ------------------------------------------------
            # CHECK ANSWERABILITY
            # ------------------------------------------------

            answerable_ok = (
                actual_answerable
                == test["should_be_answerable"]
            )

            # ------------------------------------------------
            # CHECK QUESTION TYPE
            # ------------------------------------------------

            type_ok = (
                test["expected_type"] is None
                or actual_type == test["expected_type"]
            )

            # ------------------------------------------------
            # CHECK EXPECTED CLAUSES
            # ------------------------------------------------

            clauses_ok = all(
                clause in actual_clauses
                for clause in test["expected_clauses"]
            )

            # ------------------------------------------------
            # FINAL RESULT
            # ------------------------------------------------

            if (
                answerable_ok
                and type_ok
                and clauses_ok
            ):

                print("STATUS: ✅ PASS")
                passed += 1

            else:

                print("STATUS: ❌ FAIL")
                failed += 1

                print("\nFull response:")
                pprint(response)

        except Exception as exc:

            print("STATUS: ❌ ERROR")
            print(f"Error: {exc}")

            failed += 1

    # ========================================================
    # SUMMARY
    # ========================================================

    total = len(TEST_CASES)

    print("\n")
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    print(f"Total tests : {total}")
    print(f"Passed      : {passed}")
    print(f"Failed      : {failed}")

    if total:
        accuracy = (
            passed / total
        ) * 100
    else:
        accuracy = 0

    print(
        f"Pass rate   : {accuracy:.2f}%"
    )

    print("=" * 70)

    if failed == 0:
        print(
            "\n🎉 ALL RETRIEVAL TESTS PASSED!"
        )
    else:
        print(
            "\n⚠️ Some tests failed."
        )
        print(
            "Review the failed cases before submission."
        )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":
    run_tests()