# ============================================================
# GROUNDED POLICY ASSISTANT
# POLICY VERSION / TEMPORAL RULE RESOLVER
# ============================================================

from datetime import date
from typing import Dict, Optional

from src.reasoning.temporal_policy import (
    TemporalPolicy,
    normalize_clause_reference,
    parse_date,
)


# ============================================================
# POLICY VERSION ENGINE
# ============================================================

class PolicyVersion:

    # ========================================================
    # AMENDMENT EFFECTIVE DATE
    # ========================================================

    DEFAULT_AMENDMENT_DATE = "2026-03-01"

    # ========================================================
    # CLAUSE TEMPORAL RULE TYPES
    # ========================================================

    # These clauses are governed by the date the underlying
    # change/circumstance occurred.
    EVENT_DATE_CLAUSES = {
        "§4.3.2",
        "§9.1.4",
    }

    # These clauses are governed by the determination date.
    DETERMINATION_DATE_CLAUSES = {
        "§6.4.1",
        "§6.6.1",
        "§10.5.2",
    }

    # New protection / applicability rule.
    DETERMINATION_OR_AMENDMENT_CLAUSES = {
        "§10.5.3A",
    }

    # Transitional rule.
    TRANSITIONAL_CLAUSES = {
        "§7.4.3",
    }

    # ========================================================
    # INIT
    # ========================================================

    def __init__(
        self,
        temporal_policy: Optional[TemporalPolicy] = None,
    ):

        self.temporal_policy = (
            temporal_policy
            if temporal_policy is not None
            else TemporalPolicy()
        )

    # ========================================================
    # DATE HELPERS
    # ========================================================

    @staticmethod
    def _parse_optional_date(
        value: Optional[str],
    ) -> Optional[date]:

        if not value:
            return None

        try:
            return parse_date(value)

        except (ValueError, TypeError):
            return None

    # ========================================================
    # GET AMENDMENT DATE
    # ========================================================

    def get_amendment_effective_date(
        self,
        clause_id: str,
    ) -> Optional[str]:
        """
        Find the effective date of the amendment affecting
        the requested clause.
        """

        normalized_clause = normalize_clause_reference(
            clause_id
        )

        history = self.temporal_policy.get_clause_history(
            normalized_clause
        )

        if not history:
            return None

        valid_history = []

        for amendment in history:

            effective_date = amendment.get(
                "effective_date"
            )

            if not effective_date:
                continue

            if self._parse_optional_date(
                effective_date
            ) is None:
                continue

            valid_history.append(amendment)

        if not valid_history:
            return None

        latest = max(
            valid_history,
            key=lambda item: item.get(
                "effective_date",
                "0000-00-00"
            )
        )

        return latest.get("effective_date")

    # ========================================================
    # DETERMINE TEMPORAL RULE TYPE
    # ========================================================

    def get_rule_type(
        self,
        clause_id: str,
    ) -> str:
        """
        Determine which date controls applicability.

        Returns one of:

            event_date
            determination_date
            determination_or_amendment
            transitional
            standard
        """

        normalized_clause = normalize_clause_reference(
            clause_id
        )

        if normalized_clause in self.EVENT_DATE_CLAUSES:
            return "event_date"

        if (
            normalized_clause
            in self.DETERMINATION_DATE_CLAUSES
        ):
            return "determination_date"

        if (
            normalized_clause
            in self.DETERMINATION_OR_AMENDMENT_CLAUSES
        ):
            return "determination_or_amendment"

        if normalized_clause in self.TRANSITIONAL_CLAUSES:
            return "transitional"

        return "standard"

    # ========================================================
    # GET APPLICABILITY DATE
    # ========================================================

    def get_applicability_date(
        self,
        clause_id: str,
        determination_date: Optional[str] = None,
        event_date: Optional[str] = None,
    ) -> Optional[str]:
        """
        Select the date that controls amendment applicability.
        """

        rule_type = self.get_rule_type(
            clause_id
        )

        if rule_type == "event_date":
            return event_date

        if rule_type == "determination_date":
            return determination_date

        if rule_type == "determination_or_amendment":
            return determination_date

        if rule_type == "transitional":

            # Transitional rules can depend on both dates.
            # The resolution itself is handled separately.
            return (
                event_date
                or determination_date
            )

        # Standard policy behavior.
        return determination_date or event_date

    # ========================================================
    # RESOLVE
    # ========================================================

    def resolve(
        self,
        clause_id: str,
        determination_date: Optional[str] = None,
        event_date: Optional[str] = None,
    ) -> Dict:
        """
        Determine which temporal version of a clause applies.

        Example:

            §4.3.2
            determination_date = 2026-03-10
            event_date = 2026-02-25

        Result:

            Original rule applies because the event occurred
            before 2026-03-01.
        """

        normalized_clause = normalize_clause_reference(
            clause_id
        )

        rule_type = self.get_rule_type(
            normalized_clause
        )

        applicability_date = (
            self.get_applicability_date(
                normalized_clause,
                determination_date,
                event_date,
            )
        )

        amendment_date = (
            self.get_amendment_effective_date(
                normalized_clause
            )
        )

        # ----------------------------------------------------
        # No amendment affecting this clause.
        # ----------------------------------------------------

        if amendment_date is None:

            return {
                "clause": normalized_clause,
                "rule_type": rule_type,
                "determination_date": determination_date,
                "event_date": event_date,
                "applicable_date": applicability_date,
                "amendment_effective_date": None,
                "amendment_applies": False,
                "status": "original_rule",
                "reason": (
                    "No amendment affecting this clause "
                    "was found."
                ),
            }

        # ----------------------------------------------------
        # Missing required date.
        # ----------------------------------------------------

        if applicability_date is None:

            return {
                "clause": normalized_clause,
                "rule_type": rule_type,
                "determination_date": determination_date,
                "event_date": event_date,
                "applicable_date": None,
                "amendment_effective_date": amendment_date,
                "amendment_applies": None,
                "status": "date_required",
                "reason": (
                    f"This clause is governed by "
                    f"{rule_type}, but the required date "
                    f"was not provided."
                ),
            }

        applicable = self._parse_optional_date(
            applicability_date
        )

        effective = self._parse_optional_date(
            amendment_date
        )

        if applicable is None or effective is None:

            return {
                "clause": normalized_clause,
                "rule_type": rule_type,
                "determination_date": determination_date,
                "event_date": event_date,
                "applicable_date": applicability_date,
                "amendment_effective_date": amendment_date,
                "amendment_applies": None,
                "status": "invalid_date",
                "reason": (
                    "The dates required for temporal "
                    "resolution could not be interpreted."
                ),
            }

        # ----------------------------------------------------
        # Transitional rule
        # ----------------------------------------------------

        if rule_type == "transitional":

            return self._resolve_transitional(
                normalized_clause,
                determination_date,
                event_date,
                amendment_date,
            )

        # ----------------------------------------------------
        # Normal temporal comparison
        # ----------------------------------------------------

        amendment_applies = (
            applicable >= effective
        )

        if amendment_applies:

            status = "amended_rule"

            reason = (
                f"The applicable {rule_type.replace('_', ' ')} "
                f"({applicability_date}) is on or after "
                f"the amendment effective date "
                f"({amendment_date})."
            )

        else:

            status = "original_rule"

            reason = (
                f"The applicable {rule_type.replace('_', ' ')} "
                f"({applicability_date}) is before "
                f"the amendment effective date "
                f"({amendment_date})."
            )

        return {
            "clause": normalized_clause,
            "rule_type": rule_type,
            "determination_date": determination_date,
            "event_date": event_date,
            "applicable_date": applicability_date,
            "amendment_effective_date": amendment_date,
            "amendment_applies": amendment_applies,
            "status": status,
            "reason": reason,
        }

    # ========================================================
    # TRANSITIONAL RESOLUTION
    # ========================================================

    def _resolve_transitional(
        self,
        clause_id: str,
        determination_date: Optional[str],
        event_date: Optional[str],
        amendment_date: str,
    ) -> Dict:
        """
        Handle provisions whose applicability spans a
        transition period.

        Current behavior:
        - If both dates exist and the event begins before
          the amendment while determination occurs after it,
          mark as transitional.
        - Otherwise use the available controlling date.
        """

        determination = self._parse_optional_date(
            determination_date
        )

        event = self._parse_optional_date(
            event_date
        )

        effective = self._parse_optional_date(
            amendment_date
        )

        if (
            determination is not None
            and event is not None
            and effective is not None
            and event < effective
            and determination >= effective
        ):

            return {
                "clause": clause_id,
                "rule_type": "transitional",
                "determination_date": determination_date,
                "event_date": event_date,
                "applicable_date": None,
                "amendment_effective_date": amendment_date,
                "amendment_applies": None,
                "status": "transitional",
                "reason": (
                    "The relevant period spans the amendment "
                    "effective date. Transitional treatment "
                    "must be applied."
                ),
            }

        # If we do not have a spanning period, use the
        # determination date when available.
        applicable_date = (
            determination_date
            or event_date
        )

        applicable = self._parse_optional_date(
            applicable_date
        )

        if applicable is None or effective is None:

            return {
                "clause": clause_id,
                "rule_type": "transitional",
                "determination_date": determination_date,
                "event_date": event_date,
                "applicable_date": applicable_date,
                "amendment_effective_date": amendment_date,
                "amendment_applies": None,
                "status": "date_required",
                "reason": (
                    "Additional dates are required to resolve "
                    "the transitional provision."
                ),
            }

        amendment_applies = (
            applicable >= effective
        )

        return {
            "clause": clause_id,
            "rule_type": "transitional",
            "determination_date": determination_date,
            "event_date": event_date,
            "applicable_date": applicable_date,
            "amendment_effective_date": amendment_date,
            "amendment_applies": amendment_applies,
            "status": (
                "amended_rule"
                if amendment_applies
                else "original_rule"
            ),
            "reason": (
                "The transitional clause does not span "
                "the amendment date, so the applicable "
                "date determines the rule."
            ),
        }

    # ========================================================
    # RESOLVE MULTIPLE CLAUSES
    # ========================================================

    def resolve_clauses(
        self,
        clause_ids,
        determination_date: Optional[str] = None,
        event_date: Optional[str] = None,
    ) -> Dict[str, Dict]:
        """
        Resolve temporal applicability for multiple clauses.
        """

        results = {}

        for clause_id in clause_ids:

            normalized = normalize_clause_reference(
                clause_id
            )

            results[normalized] = self.resolve(
                normalized,
                determination_date=determination_date,
                event_date=event_date,
            )

        return results


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":

    policy = TemporalPolicy()

    version = PolicyVersion(policy)

    print("\n=== §4.3.2 BEFORE AMENDMENT ===")

    result = version.resolve(
        "§4.3.2",
        determination_date="2026-03-10",
        event_date="2026-02-25",
    )

    print(result)

    print("\n=== §4.3.2 AFTER AMENDMENT ===")

    result = version.resolve(
        "§4.3.2",
        determination_date="2026-03-10",
        event_date="2026-03-05",
    )

    print(result)

    print("\n=== §6.4.1 BEFORE AMENDMENT ===")

    result = version.resolve(
        "§6.4.1",
        determination_date="2026-02-25",
    )

    print(result)

    print("\n=== §6.4.1 AFTER AMENDMENT ===")

    result = version.resolve(
        "§6.4.1",
        determination_date="2026-03-10",
    )

    print(result)