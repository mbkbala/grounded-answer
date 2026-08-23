# ============================================================
# GROUNDED POLICY ASSISTANT
# TEMPORAL POLICY REASONING
# ============================================================

import json
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional


# ============================================================
# DATE HELPERS
# ============================================================

def parse_date(value: str) -> date:
    """
    Convert YYYY-MM-DD string to a date object.
    """
    return date.fromisoformat(value)


def normalize_clause_reference(clause_id: str) -> str:
    """
    Normalize clause references.

    Handles:
        §2.1.2
        2.1.2
        Â§2.1.2
        Ã‚Â§2.1.2
    """

    if not clause_id:
        return ""

    clause_id = str(clause_id).strip()

    # Fix common UTF-8 mojibake forms.
    clause_id = clause_id.replace("Ã‚Â§", "§")
    clause_id = clause_id.replace("Â§", "§")

    if not clause_id.startswith("§"):
        clause_id = "§" + clause_id

    return clause_id


# ============================================================
# TEMPORAL POLICY
# ============================================================

class TemporalPolicy:

    def __init__(
        self,
        amendments_file: Optional[str | Path] = None
    ):

        project_root = (
            Path(__file__)
            .resolve()
            .parents[2]
        )

        if amendments_file is None:
            amendments_file = (
                project_root
                / "data"
                / "amendments.json"
            )

        self.amendments_file = Path(
            amendments_file
        )

        self.amendments: List[Dict] = []

        self._load_amendments()

    # ========================================================
    # LOAD AMENDMENTS
    # ========================================================

    def _load_amendments(self) -> None:

        if not self.amendments_file.exists():
            self.amendments = []
            return

        try:
            with self.amendments_file.open(
                "r",
                encoding="utf-8"
            ) as file:

                data = json.load(file)

        except (OSError, json.JSONDecodeError):
            self.amendments = []
            return

        if isinstance(data, dict):
            amendments = data.get(
                "amendments",
                []
            )
        elif isinstance(data, list):
            amendments = data
        else:
            amendments = []

        if not isinstance(amendments, list):
            amendments = []

        self.amendments = amendments

        # Sort oldest -> newest.
        self.amendments.sort(
            key=lambda item: (
                item.get("effective_date")
                or "9999-12-31"
            )
        )

    # ========================================================
    # ACTIVE AMENDMENTS
    # ========================================================

    def get_active_amendments(
        self,
        as_of: str
    ) -> List[Dict]:
        """
        Return all amendments effective on or before
        the requested date.
        """

        target_date = parse_date(as_of)

        active = []

        for amendment in self.amendments:

            effective_date = amendment.get(
                "effective_date"
            )

            if not effective_date:
                continue

            try:
                amendment_date = parse_date(
                    effective_date
                )

            except (ValueError, TypeError):
                continue

            if amendment_date <= target_date:
                active.append(amendment)

        return active

    # ========================================================
    # LATEST AMENDMENT
    # ========================================================

    def get_latest_amendment(
        self,
        as_of: str
    ) -> Optional[Dict]:
        """
        Return the latest amendment effective
        on or before the requested date.
        """

        active = self.get_active_amendments(
            as_of
        )

        if not active:
            return None

        return max(
            active,
            key=lambda item: (
                item.get(
                    "effective_date",
                    "0000-00-00"
                )
            )
        )

    # ========================================================
    # CLAUSE HISTORY
    # ========================================================

    def get_clause_history(
        self,
        clause_id: str,
        as_of: Optional[str] = None
    ) -> List[Dict]:
        """
        Return amendment history for a specific clause.

        Example:
            get_clause_history("§2.1.2")

        If as_of is supplied, only amendments effective
        on or before that date are returned.
        """

        requested_clause = (
            normalize_clause_reference(
                clause_id
            )
        )

        results = []

        for amendment in self.amendments:

            effective_date = amendment.get(
                "effective_date"
            )

            if not effective_date:
                continue

            # ------------------------------------------------
            # Respect requested date
            # ------------------------------------------------

            if as_of:

                try:

                    amendment_date = parse_date(
                        effective_date
                    )

                    target_date = parse_date(
                        as_of
                    )

                    if amendment_date > target_date:
                        continue

                except (ValueError, TypeError):
                    continue

            # ------------------------------------------------
            # Check direct clause list
            # ------------------------------------------------

            clauses = amendment.get(
                "clauses",
                []
            )

            normalized_clauses = {
                normalize_clause_reference(
                    clause
                )
                for clause in clauses
            }

            if requested_clause in normalized_clauses:

                results.append(amendment)

                continue

            # ------------------------------------------------
            # Check individual changes
            # ------------------------------------------------

            changes = amendment.get(
                "changes",
                []
            )

            for change in changes:

                if not isinstance(change, dict):
                    continue

                change_clauses = change.get(
                    "clauses",
                    []
                )

                normalized_change_clauses = {
                    normalize_clause_reference(
                        clause
                    )
                    for clause in change_clauses
                }

                if requested_clause in normalized_change_clauses:

                    results.append(amendment)

                    break

        return results

    # ========================================================
    # GET CLAUSE STATUS
    # ========================================================

    def get_clause_status(
        self,
        clause_id: str,
        as_of: str
    ) -> Dict:
        """
        Determine whether a clause has amendment history
        as of a particular date.
        """

        history = self.get_clause_history(
            clause_id,
            as_of
        )

        latest = None

        if history:
            latest = max(
                history,
                key=lambda item: (
                    item.get(
                        "effective_date",
                        "0000-00-00"
                    )
                )
            )

        return {
            "clause": normalize_clause_reference(
                clause_id
            ),
            "as_of": as_of,
            "has_history": bool(history),
            "history_count": len(history),
            "latest_amendment": latest,
            "history": history
        }

    # ========================================================
    # POLICY VERSION
    # ========================================================

    def get_policy_version(
        self,
        as_of: str
    ) -> Dict:
        """
        Return the policy state applicable to a date.
        """

        active = self.get_active_amendments(
            as_of
        )

        latest = (
            self.get_latest_amendment(
                as_of
            )
            if active
            else None
        )

        return {
            "as_of": as_of,
            "amendments_active": len(active),
            "latest_amendment": latest,
            "amendments": active
        }

    # ========================================================
    # EXPLAIN
    # ========================================================

    def explain(
        self,
        as_of: str
    ) -> str:
        """
        Produce a human-readable policy version summary.
        """

        version = self.get_policy_version(
            as_of
        )

        if not version["amendments"]:

            return (
                f"No amendments recorded as effective "
                f"on or before {as_of}."
            )

        lines = [
            f"Policy version as of {as_of}:"
        ]

        for amendment in version["amendments"]:

            title = amendment.get(
                "title",
                "Unknown amendment"
            )

            effective_date = amendment.get(
                "effective_date",
                "unknown"
            )

            lines.append(
                f"- {title} "
                f"(effective {effective_date})"
            )

        return "\n".join(lines)


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":

    policy = TemporalPolicy()

    print(
        policy.explain(
            "2026-08-23"
        )
    )