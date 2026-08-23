# ============================================================
# POLICY VERSION / TEMPORAL RULE TESTS
# ============================================================

from src.reasoning.policy_version import PolicyVersion


def test_event_date_before_amendment():
    """
    §4.3.2:
    Change occurred before 2026-03-01.
    Even if determination is after amendment,
    OLD 10-day rule should apply.
    """

    policy = PolicyVersion()

    result = policy.resolve(
        "§4.3.2",
        determination_date="2026-03-10",
        event_date="2026-02-25",
    )

    assert result["status"] == "original_rule"
    assert result["amendment_applies"] is False

    print("PASS: §4.3.2 pre-amendment event")


def test_event_date_after_amendment():
    """
    §4.3.2:
    Change occurred after 2026-03-01.
    NEW 14-day rule should apply.
    """

    policy = PolicyVersion()

    result = policy.resolve(
        "§4.3.2",
        determination_date="2026-03-10",
        event_date="2026-03-05",
    )

    assert result["status"] == "amended_rule"
    assert result["amendment_applies"] is True

    print("PASS: §4.3.2 post-amendment event")


def test_determination_date_before_amendment():
    """
    §6.4.1:
    Determination before amendment => old $120 disregard.
    """

    policy = PolicyVersion()

    result = policy.resolve(
        "§6.4.1",
        determination_date="2026-02-25",
    )

    assert result["status"] == "original_rule"
    assert result["amendment_applies"] is False

    print("PASS: §6.4.1 pre-amendment determination")


def test_determination_date_after_amendment():
    """
    §6.4.1:
    Determination after amendment => new $175 disregard.
    """

    policy = PolicyVersion()

    result = policy.resolve(
        "§6.4.1",
        determination_date="2026-03-10",
    )

    assert result["status"] == "amended_rule"
    assert result["amendment_applies"] is True

    print("PASS: §6.4.1 post-amendment determination")


def test_income_threshold_before_amendment():
    """
    §6.6.1:
    Determination before March 1 => original threshold.
    """

    policy = PolicyVersion()

    result = policy.resolve(
        "§6.6.1",
        determination_date="2026-02-28",
    )

    assert result["status"] == "original_rule"
    assert result["amendment_applies"] is False

    print("PASS: §6.6.1 pre-amendment")


def test_income_threshold_after_amendment():
    """
    §6.6.1:
    Determination on/after March 1 => new thresholds.
    """

    policy = PolicyVersion()

    result = policy.resolve(
        "§6.6.1",
        determination_date="2026-03-01",
    )

    assert result["status"] == "amended_rule"
    assert result["amendment_applies"] is True

    print("PASS: §6.6.1 amendment effective date")


def test_sanction_percentage_before_amendment():
    """
    §10.5.2:
    Old sanction percentage applies before amendment.
    """

    policy = PolicyVersion()

    result = policy.resolve(
        "§10.5.2",
        determination_date="2026-02-28",
    )

    assert result["status"] == "original_rule"
    assert result["amendment_applies"] is False

    print("PASS: §10.5.2 pre-amendment")


def test_sanction_percentage_after_amendment():
    """
    §10.5.2:
    New 15% rule applies on/after March 1.
    """

    policy = PolicyVersion()

    result = policy.resolve(
        "§10.5.2",
        determination_date="2026-03-01",
    )

    assert result["status"] == "amended_rule"
    assert result["amendment_applies"] is True

    print("PASS: §10.5.2 post-amendment")


def test_new_clause_10_5_3A():
    """
    §10.5.3A is introduced by the amendment.

    The resolver must recognize the clause as amended
    rather than treating it as an unknown standard clause.
    """

    policy = PolicyVersion()

    result = policy.resolve(
        "§10.5.3A",
        determination_date="2026-03-01",
    )

    assert result["status"] == "amended_rule"
    assert result["amendment_applies"] is True

    print("PASS: §10.5.3A new amendment clause")


def test_reporting_date_boundary():
    """
    Boundary test:
    Event occurring exactly on 2026-03-01
    should use the NEW rule.
    """

    policy = PolicyVersion()

    result = policy.resolve(
        "§4.3.2",
        determination_date="2026-03-01",
        event_date="2026-03-01",
    )

    assert result["status"] == "amended_rule"
    assert result["amendment_applies"] is True

    print("PASS: §4.3.2 exact amendment boundary")


def test_reporting_date_one_day_before():
    """
    Event one day before amendment should use OLD rule.
    """

    policy = PolicyVersion()

    result = policy.resolve(
        "§4.3.2",
        determination_date="2026-03-01",
        event_date="2026-02-28",
    )

    assert result["status"] == "original_rule"
    assert result["amendment_applies"] is False

    print("PASS: §4.3.2 one-day-before boundary")


def test_missing_event_date():
    """
    §4.3.2 requires event date.
    Missing event date should not guess.
    """

    policy = PolicyVersion()

    result = policy.resolve(
        "§4.3.2",
        determination_date="2026-03-10",
    )

    assert result["status"] == "date_required"
    assert result["amendment_applies"] is None

    print("PASS: missing event date handled safely")


def test_missing_determination_date():
    """
    §6.4.1 requires determination date.
    Missing date should not guess.
    """

    policy = PolicyVersion()

    result = policy.resolve(
        "§6.4.1",
    )

    assert result["status"] == "date_required"
    assert result["amendment_applies"] is None

    print("PASS: missing determination date handled safely")


def test_transitional_period():
    """
    §7.4.3:
    Event/period begins before March 1 and determination
    occurs after March 1.

    This should be recognized as transitional.
    """

    policy = PolicyVersion()

    result = policy.resolve(
        "§7.4.3",
        determination_date="2026-03-10",
        event_date="2026-02-25",
    )

    assert result["status"] == "transitional"
    assert result["amendment_applies"] is None

    print("PASS: §7.4.3 transitional period")


def test_exact_clause_normalization():
    """
    Clause references should work with or without
    minor formatting differences.
    """

    policy = PolicyVersion()

    result = policy.resolve(
        "2.4",
        determination_date="2026-03-10",
    )

    assert result["clause"] == "§2.4"

    print("PASS: clause normalization")


if __name__ == "__main__":

    tests = [
        test_event_date_before_amendment,
        test_event_date_after_amendment,
        test_determination_date_before_amendment,
        test_determination_date_after_amendment,
        test_income_threshold_before_amendment,
        test_income_threshold_after_amendment,
        test_sanction_percentage_before_amendment,
        test_sanction_percentage_after_amendment,
        test_new_clause_10_5_3A,
        test_reporting_date_boundary,
        test_reporting_date_one_day_before,
        test_missing_event_date,
        test_missing_determination_date,
        test_transitional_period,
        test_exact_clause_normalization,
    ]

    print("=" * 70)
    print("GROUNDED POLICY ASSISTANT - TEMPORAL POLICY TESTS")
    print("=" * 70)

    passed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except AssertionError as e:
            print(f"FAIL: {test.__name__}")
            print(f"      {e}")
        except Exception as e:
            print(f"ERROR: {test.__name__}")
            print(f"       {e}")

    print()
    print("=" * 70)
    print(f"TOTAL : {len(tests)}")
    print(f"PASSED: {passed}")
    print(f"FAILED: {len(tests) - passed}")
    print("=" * 70)

    if passed == len(tests):
        print("🎉 ALL TEMPORAL TESTS PASSED!")
    else:
        print("⚠️ SOME TEMPORAL TESTS FAILED!")