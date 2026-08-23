
"""
GROUNDed POLICY ASSISTANT
CLI END-TO-END TEST SUITE

Run:
    python tests/test_cli.py

IMPORTANT:
Change CLI_COMMAND below to the exact command you currently use
to run your policy assistant from the terminal.

Example:
    CLI_COMMAND = ["python", "main.py"]
"""

import subprocess
import sys
from pathlib import Path


# ============================================================
# CONFIGURATION
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

# CHANGE THIS ONLY IF YOUR CLI ENTRY POINT IS DIFFERENT
CLI_COMMAND = [sys.executable, "main.py"]


# ============================================================
# TEST RESULT TRACKING
# ============================================================

passed = 0
failed = 0
results = []


# ============================================================
# CLI RUNNER
# ============================================================

def run_cli(question: str, timeout: int = 30):
    """
    Run the real CLI application and return its output.
    """

    try:
        process = subprocess.run(
            CLI_COMMAND,
            input=question + "\n",
            text=True,
            capture_output=True,
            cwd=PROJECT_ROOT,
            timeout=timeout,
        )

        output = (
            (process.stdout or "")
            + "\n"
            + (process.stderr or "")
        ).strip()

        return process.returncode, output

    except subprocess.TimeoutExpired:
        return -1, "TIMEOUT"

    except Exception as exc:
        return -1, f"ERROR: {exc}"


# ============================================================
# ASSERTION
# ============================================================

def check(
    test_id: str,
    name: str,
    question: str,
    expected=None,
    forbidden=None,
):
    global passed, failed

    return_code, output = run_cli(question)

    success = True
    reasons = []

    # Application must not crash
    if return_code != 0:
        success = False
        reasons.append(f"CLI exited with code {return_code}")

    if not output:
        success = False
        reasons.append("No output returned")

    # Expected strings
    if expected:
        for value in expected:
            if value.lower() not in output.lower():
                success = False
                reasons.append(f"Missing expected text: {value}")

    # Forbidden strings
    if forbidden:
        for value in forbidden:
            if value.lower() in output.lower():
                success = False
                reasons.append(f"Found forbidden text: {value}")

    if success:
        passed += 1
        status = "PASS"
        results.append((test_id, name, True))
    else:
        failed += 1
        status = "FAIL"
        results.append((test_id, name, False))

    print(f"[{status}] {test_id} - {name}")

    if not success:
        for reason in reasons:
            print(f"       -> {reason}")

        print("\n       OUTPUT:")
        print("       " + output.replace("\n", "\n       "))

    return success


# ============================================================
# TEST SUITE
# ============================================================

def run_tests():

    print()
    print("=" * 70)
    print("GROUNDED POLICY ASSISTANT - CLI TEST SUITE")
    print("=" * 70)
    print()

    # --------------------------------------------------------
    # 1. BASIC RETRIEVAL
    # --------------------------------------------------------

    check(
        "T01",
        "Basic leave policy retrieval",
        "What is the leave policy?",
    )

    check(
        "T02",
        "Annual leave retrieval",
        "How many days of annual leave are allowed?",
    )

    check(
        "T03",
        "Work from home retrieval",
        "What is the work from home policy?",
    )

    check(
        "T04",
        "Travel policy retrieval",
        "What is the travel reimbursement policy?",
    )

    check(
        "T05",
        "Notice period retrieval",
        "What is the resignation notice period?",
    )

    # --------------------------------------------------------
    # 2. HYBRID RETRIEVAL / PARAPHRASED QUERIES
    # --------------------------------------------------------

    check(
        "T06",
        "Semantic leave retrieval",
        "How much time off can an employee take?",
    )

    check(
        "T07",
        "Semantic WFH retrieval",
        "Can employees work remotely?",
    )

    check(
        "T08",
        "Semantic travel retrieval",
        "Can I get money back for business travel?",
    )

    check(
        "T09",
        "Semantic notice retrieval",
        "How long before leaving the company must I inform them?",
    )

    # --------------------------------------------------------
    # 3. CLAUSE RETRIEVAL
    # --------------------------------------------------------

    check(
        "T10",
        "Specific clause retrieval",
        "What does clause 4.2 say?",
    )

    check(
        "T11",
        "Section retrieval",
        "Explain section 5.",
    )

    # --------------------------------------------------------
    # 4. CURRENT POLICY
    # --------------------------------------------------------

    check(
        "T12",
        "Current policy resolution",
        "What is the current leave policy?",
    )

    # --------------------------------------------------------
    # 5. HISTORICAL / TEMPORAL POLICY
    # --------------------------------------------------------

    check(
        "T13",
        "Historical policy - 2023",
        "What was the leave policy in 2023?",
    )

    check(
        "T14",
        "Historical policy - 2024",
        "What was the leave policy in 2024?",
    )

    check(
        "T15",
        "Historical policy by exact date",
        "What policy was effective on 2024-06-01?",
    )

    check(
        "T16",
        "Historical policy by exact date",
        "What policy was effective on 2025-01-01?",
    )

    # --------------------------------------------------------
    # 6. AMENDMENT TESTS
    # --------------------------------------------------------

    check(
        "T17",
        "Amendment detection",
        "What changed in the leave policy?",
    )

    check(
        "T18",
        "Amendment explanation",
        "Which amendment changed the leave policy?",
    )

    # --------------------------------------------------------
    # 7. DATE BOUNDARY TESTS
    # --------------------------------------------------------

    check(
        "T19",
        "Date before amendment",
        "What policy applied on 2024-12-31?",
    )

    check(
        "T20",
        "Date at amendment",
        "What policy applied on 2025-01-01?",
    )

    check(
        "T21",
        "Date after amendment",
        "What policy applied on 2025-01-02?",
    )

    # --------------------------------------------------------
    # 8. GROUNDING / UNKNOWN QUESTIONS
    # --------------------------------------------------------

    check(
        "T22",
        "Unknown policy rejection",
        "What is the company's policy on moon travel?",
    )

    check(
        "T23",
        "Unsupported information rejection",
        "What is the CEO's favorite food?",
    )

    check(
        "T24",
        "Out of scope rejection",
        "Who will win the cricket match?",
    )

    # --------------------------------------------------------
    # 9. INVALID INPUTS
    # --------------------------------------------------------

    check(
        "T25",
        "Greeting does not crash",
        "hello",
    )

    check(
        "T26",
        "Question mark does not crash",
        "?",
    )

    check(
        "T27",
        "Numeric input does not crash",
        "123456",
    )

    check(
        "T28",
        "Special characters do not crash",
        "@#$%^&*",
    )

    # --------------------------------------------------------
    # 10. DATE FORMAT TESTS
    # --------------------------------------------------------

    check(
        "T29",
        "ISO date query",
        "What policy applied on 2025-01-01?",
    )

    check(
        "T30",
        "Natural language date query",
        "What policy applied on January 1 2025?",
    )

    # --------------------------------------------------------
    # 11. SOURCE / GROUNDING
    # --------------------------------------------------------

    check(
        "T31",
        "Source identification",
        "What policy document supports the leave policy?",
    )

    # --------------------------------------------------------
    # FINAL
    # --------------------------------------------------------

    print()
    print("=" * 70)
    print("TEST SUMMARY")
    print("=" * 70)

    total = passed + failed

    print(f"TOTAL : {total}")
    print(f"PASS  : {passed}")
    print(f"FAIL  : {failed}")

    if total:
        percentage = (passed / total) * 100
        print(f"SCORE : {percentage:.1f}%")

    print("=" * 70)

    if failed == 0:
        print("ALL TESTS PASSED")
        print("CLI ENGINE IS READY FOR UI TESTING")
    else:
        print("TESTS FAILED")
        print("FIX THE FAILURES BEFORE MOVING TO UI")

    print("=" * 70)

    return failed == 0


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    success = run_tests()
    sys.exit(0 if success else 1)
