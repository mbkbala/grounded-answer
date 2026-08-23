# ============================================================
# GROUNDED POLICY ASSISTANT
# TEMPORAL POLICY REASONING
# ============================================================

import json
import re

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
        §10.5.3A
        10.5.3A
    """

    if not clause_id:
        return ""

    clause_id = str(clause_id).strip()

    # --------------------------------------------------------
    # Fix common UTF-8 mojibake
    # --------------------------------------------------------

    clause_id = clause_id.replace("Ã‚Â§", "§")
    clause_id = clause_id.replace("Â§", "§")

    # --------------------------------------------------------
    # Remove accidental surrounding whitespace
    # --------------------------------------------------------

    clause_id = clause_id.strip()

    # --------------------------------------------------------
    # Add section symbol if missing
    # --------------------------------------------------------

    if not clause_id.startswith("§"):
        clause_id = "§" + clause_id

    return clause_id


# ============================================================
# EXTRACT CLAUSE REFERENCES FROM TEXT
# ============================================================

def extract_clause_references(text: str) -> List[str]:
    """
    Extract clause references from amendment text.

    Examples detected:

        §10.5.3A
        10.5.3A
        §6.4.1(a)
        10.5.3A A sanction must not...

    The function intentionally supports alphabetic suffixes
    because amendments may introduce clauses such as:

        10.5.3A
        10.5.3B
        10.5.3C
    """

    if not text:
        return []

    text = str(text)

    # --------------------------------------------------------
    # Match optional section symbol followed by:
    #
    # number.number
    # number.number.number
    # optional alphabetic suffix
    #
    # Examples:
    #   2.1
    #   2.1.2
    #   10.5.3A
    # --------------------------------------------------------

    pattern = r"(?:§\s*)?\d+(?:\.\d+)+(?:[A-Za-z]+)?"

    matches = re.findall(
        pattern,
        text
    )

    results = []

    for match in matches:

        normalized = normalize_clause_reference(
            match
        )

        if normalized and normalized not in results:
            results.append(normalized)

    return results


# ============================================================
# TEMPORAL POLICY
# ============================================================

class TemporalPolicy:

    def __init__(
        self,
        amendments_file: Optional[str | Path] = None
    ):

        # ----------------------------------------------------
        # Project root
        # ----------------------------------------------------

        project_root = (
            Path(__file__)
            .resolve()
            .parents[2]
        )

        # ----------------------------------------------------
        # Default amendments file
        # ----------------------------------------------------

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
        """
        Load amendment data from JSON.

        The loader is deliberately tolerant of incomplete
        amendment metadata.

        A clause may be declared in:

        1. amendment["clauses"]
        2. change["clauses"]
        3. amendment/change text

        This is important for newly inserted clauses such as
        §10.5.3A.
        """

        if not self.amendments_file.exists():

            self.amendments = []

            return

        try:

            with self.amendments_file.open(
                "r",
                encoding="utf-8"
            ) as file:

                data = json.load(file)

        except (
            OSError,
            json.JSONDecodeError
        ):

            self.amendments = []

            return

        # ----------------------------------------------------
        # Support both:
        #
        # {
        #     "amendments": [...]
        # }
        #
        # and:
        #
        # [...]
        # ----------------------------------------------------

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

        # ----------------------------------------------------
        # Normalize amendment metadata
        # ----------------------------------------------------

        for amendment in self.amendments:

            if not isinstance(amendment, dict):
                continue

            self._augment_amendment_clause_metadata(
                amendment
            )

        # ----------------------------------------------------
        # Sort oldest -> newest
        # ----------------------------------------------------

        self.amendments.sort(
            key=lambda item: (
                item.get(
                    "effective_date",
                    "9999-12-31"
                )
                if isinstance(item, dict)
                else "9999-12-31"
            )
        )

    # ========================================================
    # AUGMENT AMENDMENT CLAUSE METADATA
    # ========================================================

    def _augment_amendment_clause_metadata(
        self,
        amendment: Dict
    ) -> None:
        """
        Build a reliable internal clause list.

        This fixes incomplete amendment metadata without
        modifying amendments.json itself.

        For example:

        change:
            "After §10.5.3, insert —
             10.5.3A A sanction must not..."

        will automatically cause:

            §10.5.3A

        to be recognized as affected by the amendment.
        """

        discovered = set()

        # ----------------------------------------------------
        # 1. Existing amendment-level clauses
        # ----------------------------------------------------

        amendment_clauses = amendment.get(
            "clauses",
            []
        )

        if isinstance(amendment_clauses, list):

            for clause in amendment_clauses:

                normalized = (
                    normalize_clause_reference(
                        clause
                    )
                )

                if normalized:
                    discovered.add(normalized)

        # ----------------------------------------------------
        # 2. Inspect changes
        # ----------------------------------------------------

        changes = amendment.get(
            "changes",
            []
        )

        if isinstance(changes, list):

            for change in changes:

                if not isinstance(change, dict):
                    continue

                # --------------------------------------------
                # Explicit change clauses
                # --------------------------------------------

                change_clauses = change.get(
                    "clauses",
                    []
                )

                if isinstance(
                    change_clauses,
                    list
                ):

                    for clause in change_clauses:

                        normalized = (
                            normalize_clause_reference(
                                clause
                            )
                        )

                        if normalized:
                            discovered.add(
                                normalized
                            )

                # --------------------------------------------
                # Extract clauses from change text
                # --------------------------------------------

                change_text = change.get(
                    "text",
                    ""
                )

                for clause in extract_clause_references(
                    change_text
                ):

                    discovered.add(clause)

        # ----------------------------------------------------
        # 3. Extract clauses from complete amendment text
        # ----------------------------------------------------

        amendment_text = amendment.get(
            "text",
            ""
        )

        for clause in extract_clause_references(
            amendment_text
        ):

            discovered.add(clause)

        # ----------------------------------------------------
        # Save augmented metadata
        #
        # Keep original order where possible, then append
        # newly discovered clauses.
        # ----------------------------------------------------

        existing_order = []

        if isinstance(
            amendment_clauses,
            list
        ):

            for clause in amendment_clauses:

                normalized = (
                    normalize_clause_reference(
                        clause
                    )
                )

                if normalized:
                    existing_order.append(
                        normalized
                    )

        for clause in sorted(discovered):

            if clause not in existing_order:
                existing_order.append(clause)

        amendment["_normalized_clauses"] = (
            existing_order
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

        target_date = parse_date(
            as_of
        )

        active = []

        for amendment in self.amendments:

            if not isinstance(
                amendment,
                dict
            ):
                continue

            effective_date = amendment.get(
                "effective_date"
            )

            if not effective_date:
                continue

            try:

                amendment_date = parse_date(
                    effective_date
                )

            except (
                ValueError,
                TypeError
            ):

                continue

            if amendment_date <= target_date:

                active.append(
                    amendment
                )

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

        The search checks:

        1. Normalized amendment clause metadata
        2. Explicit change clause metadata
        3. Amendment text
        4. Change text

        This allows newly inserted clauses such as
        §10.5.3A to be discovered automatically.
        """

        requested_clause = (
            normalize_clause_reference(
                clause_id
            )
        )

        results = []

        # ----------------------------------------------------
        # Optional target date
        # ----------------------------------------------------

        target_date = None

        if as_of:

            try:

                target_date = parse_date(
                    as_of
                )

            except (
                ValueError,
                TypeError
            ):

                return []

        # ----------------------------------------------------
        # Search amendments
        # ----------------------------------------------------

        for amendment in self.amendments:

            if not isinstance(
                amendment,
                dict
            ):
                continue

            effective_date = amendment.get(
                "effective_date"
            )

            if not effective_date:
                continue

            # ------------------------------------------------
            # Respect requested date
            # ------------------------------------------------

            if target_date is not None:

                try:

                    amendment_date = parse_date(
                        effective_date
                    )

                except (
                    ValueError,
                    TypeError
                ):

                    continue

                if amendment_date > target_date:
                    continue

            found = False

            # ------------------------------------------------
            # 1. Check augmented amendment metadata
            # ------------------------------------------------

            normalized_clauses = amendment.get(
                "_normalized_clauses",
                []
            )

            if isinstance(
                normalized_clauses,
                list
            ):

                if requested_clause in normalized_clauses:

                    found = True

            # ------------------------------------------------
            # 2. Check change-level clauses
            # ------------------------------------------------

            if not found:

                changes = amendment.get(
                    "changes",
                    []
                )

                if isinstance(
                    changes,
                    list
                ):

                    for change in changes:

                        if not isinstance(
                            change,
                            dict
                        ):
                            continue

                        change_clauses = change.get(
                            "clauses",
                            []
                        )

                        if not isinstance(
                            change_clauses,
                            list
                        ):
                            continue

                        normalized_change_clauses = {
                            normalize_clause_reference(
                                clause
                            )
                            for clause
                            in change_clauses
                        }

                        if (
                            requested_clause
                            in normalized_change_clauses
                        ):

                            found = True
                            break

            # ------------------------------------------------
            # 3. Search complete amendment text
            # ------------------------------------------------

            if not found:

                amendment_text = amendment.get(
                    "text",
                    ""
                )

                extracted = (
                    extract_clause_references(
                        amendment_text
                    )
                )

                if requested_clause in extracted:

                    found = True

            # ------------------------------------------------
            # 4. Search change text
            # ------------------------------------------------

            if not found:

                changes = amendment.get(
                    "changes",
                    []
                )

                if isinstance(
                    changes,
                    list
                ):

                    for change in changes:

                        if not isinstance(
                            change,
                            dict
                        ):
                            continue

                        change_text = change.get(
                            "text",
                            ""
                        )

                        extracted = (
                            extract_clause_references(
                                change_text
                            )
                        )

                        if (
                            requested_clause
                            in extracted
                        ):

                            found = True
                            break

            # ------------------------------------------------
            # Add amendment to history
            # ------------------------------------------------

            if found:

                results.append(
                    amendment
                )

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
            "has_history": bool(
                history
            ),
            "history_count": len(
                history
            ),
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
            "amendments_active": len(
                active
            ),
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

        return "\n".join(
            lines
        )


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

    print(
        "\n=== NEW CLAUSE TEST ==="
    )

    history = policy.get_clause_history(
        "§10.5.3A"
    )

    print(
        f"History count: {len(history)}"
    )

    for amendment in history:

        print(
            amendment.get(
                "amendment_id"
            ),
            amendment.get(
                "effective_date"
            )
        )