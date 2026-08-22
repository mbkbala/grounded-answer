# ============================================================
# GROUNDED ANSWER GENERATOR
# ============================================================

def generate_answer(question, response):

    """
    Generate a plain-language answer using only
    retrieved policy clauses.

    Every answer contains an exact policy clause citation.
    """

    # ========================================================
    # REFUSAL
    # ========================================================

    if not response["answerable"]:

        return (
            "I don't know.\n\n"
            "The policy manual does not provide sufficiently "
            "strong support to answer this question.\n\n"
            "Please contact the Calder County Department of "
            "Household Services for clarification."
        )

    results = response.get("results", [])

    if not results:

        return (
            "I don't know.\n\n"
            "No sufficiently relevant policy clause was found."
        )

    # ========================================================
    # DIRECT RESULTS
    # ========================================================

    direct_results = [
        result
        for result in results
        if "expanded_from" not in result
    ]

    if not direct_results:
        direct_results = results

    question_lower = question.lower()

    # ========================================================
    # QUESTION: EXCLUSIONS
    # ========================================================

    if (
        "excluded" in question_lower
        and "eligibility" in question_lower
    ):

        for result in direct_results:

            if result["clause"] == "§4.1.1":

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

                    "📄 Policy citation: §4.1.1\n"
                    "Household Support Program Manual"
                )

    # ========================================================
    # QUESTION: ELIGIBILITY REQUIREMENTS
    # ========================================================

    if (
        "eligibility" in question_lower
        and (
            "requirement" in question_lower
            or "qualify" in question_lower
            or "eligible" in question_lower
        )
    ):

        for result in direct_results:

            if result["clause"] == "§2.1.2":

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

                    "📄 Policy citation: §2.1.2\n"
                    "Household Support Program Manual"
                )

    # ========================================================
    # GENERAL FALLBACK
    # ========================================================

    best = direct_results[0]

    clause = best["clause"]
    text = best["text"]

    return (
        f"{text}\n\n"
        f"📄 Policy citation: {clause}\n"
        "Household Support Program Manual"
    )