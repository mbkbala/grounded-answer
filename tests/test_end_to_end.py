# ============================================================
# GROUNDED POLICY ASSISTANT
# END-TO-END CLI / SURPRISE CHALLENGE TESTS
# ============================================================

import sys
from pathlib import Path


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# IMPORTS
# ============================================================

from src.reasoning.policy_version import PolicyVersion
from src.reasoning.temporal_policy import TemporalPolicy
from src.retrieval.hybrid_search import HybridSearch


# ============================================================
# TEST HELPERS
# ============================================================

TOTAL = 0
PASSED = 0
FAILED = 0


def check(name, condition, details=""):
    global TOTAL, PASSED, FAILED

    TOTAL += 1

    if condition:
        PASSED += 1
        print(f"PASS: {name}")
    else:
        FAILED += 1
        print(f"FAIL: {name}")

        if details:
            print(f"      {details}")


def safe_search(search_engine, question):
    """
    Run HybridSearch safely.

    Returns:
        result
        None if an exception occurs
    """
    try:
        return search_engine.search(question)
    except Exception as exc:
        print(f"      Search error: {exc}")
        return None


# ============================================================
# HEADER
# ============================================================

print("=" * 70)
print("GROUNDED POLICY ASSISTANT - END-TO-END CLI TESTS")
print("=" * 70)


# ============================================================
# LOAD COMPONENTS
# ============================================================

print("\nLoading policy engine...")

try:
    temporal_policy = TemporalPolicy()

    policy_version = PolicyVersion(
        temporal_policy
    )

    print("Policy engine loaded successfully.")

except Exception as exc:

    print(f"ERROR: Policy engine failed to load: {exc}")

    raise SystemExit(1)


print("\nLoading HybridSearch...")

try:

    search = HybridSearch()

    print("HybridSearch loaded successfully.")

except Exception as exc:

    print(f"ERROR: HybridSearch failed to load: {exc}")

    raise SystemExit(1)


print("\nAll components loaded successfully.")


# ============================================================
# 1. BASIC RETRIEVAL
# ============================================================

print("\n" + "-" * 70)
print("SECTION 1: BASIC POLICY RETRIEVAL")
print("-" * 70)

question = "What are the eligibility requirements?"

result = safe_search(
    search,
    question
)

check(
    "Basic eligibility question returns results",
    bool(result),
    "No retrieval results returned."
)


# ============================================================
# 2. OUTSIDE-POLICY SAFETY
# ============================================================

print("\n" + "-" * 70)
print("SECTION 2: OUTSIDE-POLICY SAFETY")
print("-" * 70)

question = "How do I write Python code?"

result = safe_search(
    search,
    question
)

check(
    "Outside-policy question handled safely",
    result is not None,
    "Search crashed or returned None."
)


# ============================================================
# 3. EXACT CLAUSE REFERENCE
# ============================================================

print("\n" + "-" * 70)
print("SECTION 3: EXACT CLAUSE RETRIEVAL")
print("-" * 70)

question = "What does §2.4 say?"

result = safe_search(
    search,
    question
)

check(
    "Exact clause §2.4 can be retrieved",
    bool(result),
    "§2.4 was not retrieved."
)


# ============================================================
# 4. SURPRISE CHALLENGE
# ============================================================

print("\n" + "-" * 70)
print("SECTION 4: SURPRISE CHALLENGE / NEW AMENDMENT")
print("-" * 70)

print("\nTesting newly inserted clause: §10.5.3A")

try:

    history = temporal_policy.get_clause_history(
        "§10.5.3A"
    )

    check(
        "New clause §10.5.3A exists in amendment history",
        bool(history),
        "No amendment history found for §10.5.3A."
    )

    if history:

        latest = history[-1]

        check(
            "§10.5.3A amendment effective date is 2026-03-01",
            latest.get("effective_date") == "2026-03-01",
            f"Found: {latest.get('effective_date')}"
        )

except Exception as exc:

    check(
        "New clause §10.5.3A exists in amendment history",
        False,
        str(exc)
    )


# ============================================================
# 5. SURPRISE CHALLENGE - TEMPORAL RESOLUTION
# ============================================================

print("\nTesting temporal applicability of §10.5.3A")

try:

    result = policy_version.resolve(
        "§10.5.3A",
        determination_date="2026-03-01"
    )

    check(
        "§10.5.3A resolves successfully",
        result is not None,
        "No resolution result."
    )

    if result is not None:

        check(
            "§10.5.3A is recognized as amended",
            result.get("status") == "amended_rule",
            f"Actual status: {result.get('status')}"
        )

        check(
            "§10.5.3A amendment applies on 2026-03-01",
            result.get("amendment_applies") is True,
            f"Actual value: {result.get('amendment_applies')}"
        )

except Exception as exc:

    check(
        "§10.5.3A resolves successfully",
        False,
        str(exc)
    )


# ============================================================
# 6. EVENT-DATE TEMPORAL RULE
# ============================================================

print("\n" + "-" * 70)
print("SECTION 5: EVENT-DATE TEMPORAL REASONING")
print("-" * 70)

try:

    # --------------------------------------------------------
    # Before amendment
    # --------------------------------------------------------

    result = policy_version.resolve(
        "§4.3.2",
        determination_date="2026-03-10",
        event_date="2026-02-25"
    )

    check(
        "Pre-amendment event uses original §4.3.2 rule",
        result.get("status") == "original_rule",
        f"Actual status: {result.get('status')}"
    )

    # --------------------------------------------------------
    # After amendment
    # --------------------------------------------------------

    result = policy_version.resolve(
        "§4.3.2",
        determination_date="2026-03-10",
        event_date="2026-03-05"
    )

    check(
        "Post-amendment event uses amended §4.3.2 rule",
        result.get("status") == "amended_rule",
        f"Actual status: {result.get('status')}"
    )

except Exception as exc:

    check(
        "Event-date temporal reasoning works",
        False,
        str(exc)
    )


# ============================================================
# 7. DETERMINATION-DATE TEMPORAL RULE
# ============================================================

print("\n" + "-" * 70)
print("SECTION 6: DETERMINATION-DATE TEMPORAL REASONING")
print("-" * 70)

try:

    # --------------------------------------------------------
    # Before amendment
    # --------------------------------------------------------

    result = policy_version.resolve(
        "§6.4.1",
        determination_date="2026-02-28"
    )

    check(
        "Pre-amendment determination uses original §6.4.1",
        result.get("status") == "original_rule",
        f"Actual status: {result.get('status')}"
    )

    # --------------------------------------------------------
    # Amendment effective date
    # --------------------------------------------------------

    result = policy_version.resolve(
        "§6.4.1",
        determination_date="2026-03-01"
    )

    check(
        "Determination on amendment date uses amended §6.4.1",
        result.get("status") == "amended_rule",
        f"Actual status: {result.get('status')}"
    )

except Exception as exc:

    check(
        "Determination-date temporal reasoning works",
        False,
        str(exc)
    )


# ============================================================
# 8. AMENDMENT BOUNDARY
# ============================================================

print("\n" + "-" * 70)
print("SECTION 7: EXACT AMENDMENT BOUNDARY")
print("-" * 70)

try:

    # --------------------------------------------------------
    # Exact effective date
    # --------------------------------------------------------

    result = policy_version.resolve(
        "§4.3.2",
        determination_date="2026-03-01",
        event_date="2026-03-01"
    )

    check(
        "Exact amendment date is treated as amended",
        result.get("status") == "amended_rule",
        f"Actual status: {result.get('status')}"
    )

    # --------------------------------------------------------
    # One day before
    # --------------------------------------------------------

    result = policy_version.resolve(
        "§4.3.2",
        determination_date="2026-03-01",
        event_date="2026-02-28"
    )

    check(
        "One day before amendment remains original",
        result.get("status") == "original_rule",
        f"Actual status: {result.get('status')}"
    )

except Exception as exc:

    check(
        "Amendment boundary works",
        False,
        str(exc)
    )


# ============================================================
# 9. TRANSITIONAL RULE
# ============================================================

print("\n" + "-" * 70)
print("SECTION 8: TRANSITIONAL PERIOD")
print("-" * 70)

try:

    result = policy_version.resolve(
        "§7.4.3",
        determination_date="2026-03-10",
        event_date="2026-02-25"
    )

    check(
        "Cross-amendment period is detected as transitional",
        result.get("status") == "transitional",
        f"Actual status: {result.get('status')}"
    )

except Exception as exc:

    check(
        "Cross-amendment period is detected as transitional",
        False,
        str(exc)
    )


# ============================================================
# 10. NATURAL LANGUAGE SURPRISE QUESTION
# ============================================================

print("\n" + "-" * 70)
print("SECTION 9: NATURAL LANGUAGE SURPRISE TEST")
print("-" * 70)

question = (
    "Can a sanction be imposed if someone failed to report "
    "a change that would have increased their award?"
)

result = safe_search(
    search,
    question
)

check(
    "Natural-language surprise question retrieves policy",
    bool(result),
    "No relevant clauses returned."
)


# ============================================================
# 11. NEW AMENDMENT SEARCHABILITY
# ============================================================

print("\n" + "-" * 70)
print("SECTION 10: NEW AMENDMENT SEARCHABILITY")
print("-" * 70)

question = (
    "What happens when a failure to report "
    "would have increased the award?"
)

result = safe_search(
    search,
    question
)

check(
    "New amendment content is searchable",
    bool(result),
    "No results returned for new amendment content."
)


# ============================================================
# 12. CLAUSE NORMALIZATION
# ============================================================

print("\n" + "-" * 70)
print("SECTION 11: CLAUSE NORMALIZATION")
print("-" * 70)

try:

    references = [
        "§4.3.2",
        "4.3.2",
        "Â§4.3.2",
        "Ã‚Â§4.3.2",
    ]

    normalized = []

    for reference in references:

        history = temporal_policy.get_clause_history(
            reference
        )

        normalized.append(
            bool(history)
        )

    check(
        "Clause references normalize consistently",
        all(normalized),
        f"Results: {normalized}"
    )

except Exception as exc:

    check(
        "Clause references normalize consistently",
        False,
        str(exc)
    )


# ============================================================
# 13. MISSING DATE SAFETY
# ============================================================

print("\n" + "-" * 70)
print("SECTION 12: MISSING DATE SAFETY")
print("-" * 70)

try:

    result = policy_version.resolve(
        "§4.3.2",
        determination_date=None,
        event_date="2026-03-05"
    )

    check(
        "Missing determination date handled safely",
        result is not None,
        "Resolution crashed with missing determination date."
    )

except Exception as exc:

    check(
        "Missing determination date handled safely",
        False,
        str(exc)
    )


try:

    result = policy_version.resolve(
        "§4.3.2",
        determination_date="2026-03-10",
        event_date=None
    )

    check(
        "Missing event date handled safely",
        result is not None,
        "Resolution crashed with missing event date."
    )

except Exception as exc:

    check(
        "Missing event date handled safely",
        False,
        str(exc)
    )


# ============================================================
# FINAL SUMMARY
# ============================================================

print("\n" + "=" * 70)
print("END-TO-END TEST SUMMARY")
print("=" * 70)

print(f"TOTAL : {TOTAL}")
print(f"PASSED: {PASSED}")
print(f"FAILED: {FAILED}")

if FAILED == 0:

    print("\n🎉 ALL END-TO-END CLI TESTS PASSED!")

    print(
        "The following functionality is working:"
    )

    print("  ✓ Policy retrieval")
    print("  ✓ Exact clause retrieval")
    print("  ✓ Outside-policy safety")
    print("  ✓ Amendment history")
    print("  ✓ New amendment clause detection")
    print("  ✓ Surprise challenge")
    print("  ✓ Temporal reasoning")
    print("  ✓ Event-date reasoning")
    print("  ✓ Determination-date reasoning")
    print("  ✓ Amendment boundary handling")
    print("  ✓ Transitional-period handling")
    print("  ✓ Natural-language retrieval")
    print("  ✓ New amendment searchability")
    print("  ✓ Clause normalization")
    print("  ✓ Missing-date safety")

    print(
        "\nCLI validation is complete. "
        "You can proceed to Streamlit app testing."
    )

else:

    print("\n⚠️ SOME END-TO-END TESTS FAILED!")

    print(
        "Fix the failing components before moving to Streamlit."
    )


print("=" * 70)


# ============================================================
# EXIT STATUS
# ============================================================

if FAILED > 0:
    raise SystemExit(1)