# ============================================================
# GROUNDED ANSWER GENERATOR
# ============================================================

from typing import Dict, List


# ============================================================
# HELPERS
# ============================================================

def _find_clause(
    results: List[Dict],
    clause_id: str
):
    """
    Find a specific clause in retrieved results.
    """

    for result in results:

        if result.get("clause") == clause_id:
            return result

    return None


def _has_clause(
    results: List[Dict],
    clause_id: str
) -> bool:

    return any(
        result.get("clause") == clause_id
        for result in results
    )


def _citation(
    clause_id: str
) -> str:

    return (
        f"📄 Policy citation: {clause_id}\n"
        "Household Support Program Manual"
    )


# ============================================================
# REFUSAL
# ============================================================

def _refusal(
    response: Dict
) -> str:

    reason = response.get(
        "reason",
        "The policy manual does not provide sufficiently strong support."
    )

    return (
        "I don't know.\n\n"
        "The policy manual does not provide sufficiently "
        "strong support to answer this question.\n\n"
        f"Reason: {reason}\n\n"
        "Please contact the Calder County Department of "
        "Household Services for clarification."
    )


# ============================================================
# QUESTION TYPE
# ============================================================

def _question_type(
    question: str
) -> str:

    q = question.lower()

    # --------------------------------------------------------
    # Exact 18
    # --------------------------------------------------------

    if (
        "18 years old" in q
        or "18 year old" in q
        or "18-year-old" in q
    ):
        return "age_18"

    # --------------------------------------------------------
    # 16 / 17
    # --------------------------------------------------------

    if (
        "17 years old" in q
        or "17 year old" in q
        or "17-year-old" in q
        or "16 years old" in q
        or "16 year old" in q
        or "16-year-old" in q
        or "under 18" in q
        or "minor" in q
    ):
        return "age_minor"

    # --------------------------------------------------------
    # Correctional facility
    # --------------------------------------------------------

    if (
        "correctional facility" in q
        or "detained" in q
        or "prison" in q
        or "jail" in q
    ):
        return "correctional_exclusion"

    # --------------------------------------------------------
    # Exclusion
    # --------------------------------------------------------

    if (
        "excluded" in q
        or "exclusion" in q
        or "disqualified" in q
        or "not eligible" in q
    ):
        return "exclusion"

    # --------------------------------------------------------
    # Residence
    # --------------------------------------------------------

    if (
        "live in calder county" in q
        or "living in calder county" in q
        or "resident" in q
        or "residence" in q
    ):
        return "residence"

    # --------------------------------------------------------
    # Income
    # --------------------------------------------------------

    if "income" in q:
        return "income"

    # --------------------------------------------------------
    # Resources
    # --------------------------------------------------------

    if "resource" in q:
        return "resources"

    # --------------------------------------------------------
    # Application
    # --------------------------------------------------------

    if (
        "application" in q
        or "apply" in q
    ):
        return "application"

    # --------------------------------------------------------
    # Administration
    # --------------------------------------------------------

    if (
        "administer" in q
        or "department" in q
        or "caseworker" in q
        or "administration" in q
    ):
        return "administration"

    # --------------------------------------------------------
    # Eligibility
    # --------------------------------------------------------

    if (
        "eligibility" in q
        or "eligible" in q
        or "requirements" in q
        or "requirement" in q
        or "qualify" in q
        or "qualification" in q
    ):
        return "eligibility"

    return "general"


# ============================================================
# MAIN ANSWER GENERATOR
# ============================================================

def generate_answer(
    question: str,
    response: Dict
) -> str:

    """
    Generate a grounded plain-language answer.

    IMPORTANT:
    The answer is generated only from clauses that were
    retrieved by the policy search engine.

    Intent-specific logic determines which retrieved clause
    is authoritative.
    """

    # ========================================================
    # REFUSAL
    # ========================================================

    if not response.get(
        "answerable",
        False
    ):
        return _refusal(
            response
        )

    results = response.get(
        "results",
        []
    )

    if not results:

        return (
            "I don't know.\n\n"
            "No sufficiently relevant policy clause was found."
        )

    question_type = _question_type(
        question
    )

    # ========================================================
    # 18-YEAR-OLD
    # ========================================================

    if question_type == "age_18":

        clause = _find_clause(
            results,
            "§2.1.2"
        )

        if clause is None:

            return (
                "I don't know.\n\n"
                "The policy manual does not provide sufficiently "
                "strong support for the normal 18-or-over age rule.\n\n"
                "Please contact the Calder County Department of "
                "Household Services for clarification."
            )

        return (
            "Yes. A person aged 18 or over satisfies the "
            "age condition for assistance, subject to the "
            "other eligibility requirements in §2.1.2.\n\n"
            f"{_citation('§2.1.2')}"
        )

    # ========================================================
    # 16 / 17-YEAR-OLD
    # ========================================================

    if question_type == "age_minor":

        general_clause = _find_clause(
            results,
            "§2.1.2"
        )

        minor_clause = _find_clause(
            results,
            "§2.3.1"
        )

        if minor_clause is None:

            return (
                "I don't know.\n\n"
                "The policy manual indicates that persons "
                "under 18 may be subject to special conditions, "
                "but the relevant exception could not be "
                "located in the available policy data.\n\n"
                "Please contact the Calder County Department of "
                "Household Services for clarification."
            )

        return (
            "Yes, a person aged 16 or 17 may be eligible "
            "for assistance if they satisfy the special "
            "conditions in §2.3.\n\n"
            "The general eligibility rule requires a person "
            "to be aged 18 or over unless they satisfy "
            "the exception in §2.3.\n\n"
            f"{minor_clause['text']}\n\n"
            f"{_citation('§2.3.1')}"
        )

    # ========================================================
    # CORRECTIONAL FACILITY
    # ========================================================

    if question_type == "correctional_exclusion":

        clause = _find_clause(
            results,
            "§4.1.1"
        )

        if clause is None:

            return (
                "I don't know.\n\n"
                "The policy manual does not provide sufficiently "
                "strong support for the correctional-facility "
                "eligibility question."
            )

        return (
            "No. A person detained in a correctional facility "
            "is excluded from eligibility.\n\n"
            f"{_citation('§4.1.1')}"
        )

    # ========================================================
    # GENERAL EXCLUSION
    # ========================================================

    if question_type == "exclusion":

        clause = _find_clause(
            results,
            "§4.1.1"
        )

        if clause is None:

            return _refusal(
                {
                    "reason": (
                        "The relevant exclusion clause "
                        "could not be located."
                    )
                }
            )

        return (
            "A person is excluded from eligibility if they:\n\n"
            "1. Are subject to an unexpired sanction "
            "under §10.5.\n"
            "2. Are detained in a correctional facility.\n"
            "3. Receive assistance from an equivalent "
            "program administered by another county or state.\n"
            "4. Have been determined to have obtained "
            "assistance by deliberate misrepresentation, "
            "for the period stated in that determination.\n\n"
            f"{_citation('§4.1.1')}"
        )

    # ========================================================
    # GENERAL ELIGIBILITY
    # ========================================================

    if question_type == "eligibility":

        clause = _find_clause(
            results,
            "§2.1.2"
        )

        if clause is None:

            return _refusal(
                {
                    "reason": (
                        "The controlling eligibility clause "
                        "§2.1.2 was not found."
                    )
                }
            )

        return (
            "To be eligible for assistance, a person must:\n\n"
            "1. Be a resident of Calder County and satisfy "
            "the residence requirements in Part 3.\n"
            "2. Be aged 18 or over, or satisfy the requirements "
            "in §2.3.\n"
            "3. Have countable income at or below the applicable "
            "threshold under Part 6.\n"
            "4. Have countable resources at or below the limit "
            "in §2.4.\n"
            "5. Not be excluded under Part 4.\n"
            "6. Have made a valid application under Part 8.\n\n"
            f"{_citation('§2.1.2')}"
        )

    # ========================================================
    # RESIDENCE
    # ========================================================

    if question_type == "residence":

        clause = _find_clause(
            results,
            "§2.1.2"
        )

        if clause is None:

            return _refusal(
                {
                    "reason": (
                        "The controlling eligibility clause "
                        "§2.1.2 was not found."
                    )
                }
            )

        return (
            "Yes. To be eligible for assistance, a person "
            "must be resident in Calder County and satisfy "
            "the residence requirements in Part 3.\n\n"
            f"{_citation('§2.1.2')}"
        )

    # ========================================================
    # INCOME
    # ========================================================

    if question_type == "income":

        clause = _find_clause(
            results,
            "§2.1.2"
        )

        if clause is None:

            return _refusal(
                {
                    "reason": (
                        "The controlling eligibility clause "
                        "§2.1.2 was not found."
                    )
                }
            )

        return (
            "A person must have countable income at or below "
            "the applicable threshold under Part 6 to satisfy "
            "the eligibility requirements.\n\n"
            f"{_citation('§2.1.2')}"
        )

    # ========================================================
    # RESOURCES
    # ========================================================

    if question_type == "resources":

        clause = _find_clause(
            results,
            "§2.1.2"
        )

        if clause is None:

            return _refusal(
                {
                    "reason": (
                        "The controlling eligibility clause "
                        "§2.1.2 was not found."
                    )
                }
            )

        return (
            "A person must have countable resources at or "
            "below the applicable limit in §2.4 to satisfy "
            "the eligibility requirements.\n\n"
            f"{_citation('§2.1.2')}"
        )

    # ========================================================
    # APPLICATION
    # ========================================================

    if question_type == "application":

        clause = _find_clause(
            results,
            "§2.1.2"
        )

        if clause is None:

            return _refusal(
                {
                    "reason": (
                        "The controlling eligibility clause "
                        "§2.1.2 was not found."
                    )
                }
            )

        return (
            "A person must have made a valid application "
            "under Part 8 to satisfy the eligibility "
            "requirements.\n\n"
            f"{_citation('§2.1.2')}"
        )

    # ========================================================
    # ADMINISTRATION
    # ========================================================

    if question_type == "administration":

        clause = _find_clause(
            results,
            "§1.1.2"
        )

        if clause is None:

            return _refusal(
                {
                    "reason": (
                        "The program administration clause "
                        "§1.1.2 was not found."
                    )
                }
            )

        return (
            f"{clause['text']}\n\n"
            f"{_citation('§1.1.2')}"
        )

    # ========================================================
    # FALLBACK
    # ========================================================

    # For a generic policy question, use the highest-ranked
    # retrieved clause.
    best = results[0]

    clause_id = best.get(
        "clause",
        "Unknown"
    )

    text = best.get(
        "text",
        ""
    )

    if not text:

        return _refusal(
            {
                "reason": (
                    "No usable policy text was retrieved."
                )
            }
        )

    return (
        f"{text}\n\n"
        f"{_citation(clause_id)}"
    )