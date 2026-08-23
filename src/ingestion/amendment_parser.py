# ============================================================
# GROUNDED POLICY ASSISTANT
# AMENDMENT PARSER
# ============================================================

import json
import re
from pathlib import Path
from typing import Dict, List, Optional


# ============================================================
# HELPERS
# ============================================================

def normalize_clause_reference(clause: str) -> str:
    """
    Normalize clause references.

    Handles:
        §6.4.1
        Â§6.4.1
        § 6.4.1
    """

    clause = clause.strip()

    # Fix common UTF-8 / Windows mojibake
    clause = clause.replace("Â§", "§")

    # Remove whitespace after section symbol
    clause = re.sub(r"§\s+", "§", clause)

    # Ensure section symbol
    if not clause.startswith("§"):
        clause = "§" + clause

    return clause


def extract_clause_references(text: str) -> List[str]:
    """
    Extract policy clause references.

    Examples:
        §1.2.3
        §6.4.1
        §4.3.2
        §9.1.4
        §6.6.1
        §10.5.2
        §10.5.3
        §7.4.3
    """

    # Supports both:
    #   §6.4.1
    #   Â§6.4.1
    #   § 6.4.1
    pattern = r"(?:Â§|§)\s*\d+(?:\.\d+){1,3}"

    matches = re.findall(pattern, text)

    normalized = []

    for match in matches:

        clause = normalize_clause_reference(match)

        if clause not in normalized:
            normalized.append(clause)

    return normalized


# ============================================================
# EFFECTIVE DATE
# ============================================================

def extract_effective_date(text: str) -> Optional[str]:
    """
    Extract the effective date from an amendment.

    Supported formats:

        **Effective:** 1 March 2026
        **Effective:** 2026-03-01
        Effective: 1 March 2026
        Effective Date: 1 March 2026
        Effective Date: March 1, 2026

    Returns:

        YYYY-MM-DD

    Example:

        2026-03-01
    """

    # --------------------------------------------------------
    # Remove Markdown emphasis around the label.
    #
    # This converts:
    #
    # **Effective:** 1 March 2026
    #
    # into something easier for regex to process.
    # --------------------------------------------------------

    cleaned = text.replace("*", "")

    # Also fix possible mojibake.
    cleaned = cleaned.replace("Â§", "§")

    # --------------------------------------------------------
    # YYYY-MM-DD
    #
    # Example:
    # Effective: 2026-03-01
    # Effective Date: 2026-03-01
    # --------------------------------------------------------

    match = re.search(
        r"Effective(?:\s+Date)?\s*:\s*"
        r"(\d{4}-\d{2}-\d{2})",
        cleaned,
        re.IGNORECASE,
    )

    if match:
        return match.group(1)

    # --------------------------------------------------------
    # Day Month Year
    #
    # Example:
    # Effective: 1 March 2026
    # Effective Date: 1 March 2026
    # --------------------------------------------------------

    match = re.search(
        r"Effective(?:\s+Date)?\s*:\s*"
        r"(\d{1,2})\s+"
        r"(January|February|March|April|May|June|July|August|"
        r"September|October|November|December)"
        r"\s+(\d{4})",
        cleaned,
        re.IGNORECASE,
    )

    if match:

        day = int(match.group(1))
        month_name = match.group(2).lower()
        year = int(match.group(3))

        months = {
            "january": 1,
            "february": 2,
            "march": 3,
            "april": 4,
            "may": 5,
            "june": 6,
            "july": 7,
            "august": 8,
            "september": 9,
            "october": 10,
            "november": 11,
            "december": 12,
        }

        month = months[month_name]

        return f"{year:04d}-{month:02d}-{day:02d}"

    # --------------------------------------------------------
    # Month Day, Year
    #
    # Example:
    # Effective: March 1, 2026
    # Effective Date: March 1, 2026
    # --------------------------------------------------------

    match = re.search(
        r"Effective(?:\s+Date)?\s*:\s*"
        r"(January|February|March|April|May|June|July|August|"
        r"September|October|November|December)"
        r"\s+(\d{1,2}),\s+(\d{4})",
        cleaned,
        re.IGNORECASE,
    )

    if match:

        month_name = match.group(1).lower()
        day = int(match.group(2))
        year = int(match.group(3))

        months = {
            "january": 1,
            "february": 2,
            "march": 3,
            "april": 4,
            "may": 5,
            "june": 6,
            "july": 7,
            "august": 8,
            "september": 9,
            "october": 10,
            "november": 11,
            "december": 12,
        }

        month = months[month_name]

        return f"{year:04d}-{month:02d}-{day:02d}"

    # --------------------------------------------------------
    # Nothing found
    # --------------------------------------------------------

    return None


# ============================================================
# AMENDMENT ID
# ============================================================

def extract_amendment_id(
    text: str,
    file_path: Path
) -> str:
    """
    Extract amendment ID.

    Example:

        Amendment No. 2026-01

    Returns:

        2026-01
    """

    match = re.search(
        r"Amendment\s+No\.\s*"
        r"([0-9]{4}-[0-9]{2})",
        text,
        re.IGNORECASE,
    )

    if match:
        return match.group(1)

    return file_path.stem


# ============================================================
# TITLE
# ============================================================

def extract_title(
    text: str,
    file_path: Path
) -> str:
    """
    Extract amendment title.

    Example:

        ## Amendment No. 2026-01
    """

    match = re.search(
        r"^##\s+(Amendment\s+No\.\s+.+)$",
        text,
        re.MULTILINE | re.IGNORECASE,
    )

    if match:
        return match.group(1).strip()

    return file_path.stem


# ============================================================
# CHANGE EXTRACTION
# ============================================================

def extract_changes(
    text: str
) -> List[Dict]:
    """
    Extract individual amendment changes.

    Examples:

        1.1
        2.1
        2.2
        3.1
        4.1
        4.2
        5.1
        5.2
        5.3
    """

    changes = []

    # --------------------------------------------------------
    # Find every numbered amendment paragraph.
    #
    # We look for:
    #
    # **1.1**
    # **2.1**
    # **2.2**
    #
    # and capture until the next numbered paragraph.
    # --------------------------------------------------------

    pattern = re.compile(
        r"\*\*(\d+\.\d+)\*\*\s*(.*?)(?="
        r"\n\s*\*\*\d+\.\d+\*\*"
        r"|\Z)",
        re.DOTALL,
    )

    matches = pattern.finditer(text)

    for match in matches:

        change_number = match.group(1)

        change_text = match.group(2).strip()

        # Reconstruct complete text
        full_text = (
            f"**{change_number}** "
            f"{change_text}"
        )

        clauses = extract_clause_references(
            full_text
        )

        changes.append(
            {
                "change_number": change_number,
                "clauses": clauses,
                "text": full_text,
            }
        )

    return changes


# ============================================================
# PARSE MARKDOWN AMENDMENT
# ============================================================

def parse_amendment_file(
    file_path: str | Path
) -> Dict:
    """
    Parse one amendment markdown file.

    Extracts:

        - amendment ID
        - title
        - effective date
        - status
        - clause references
        - individual changes
        - source file
        - raw text
    """

    path = Path(file_path)

    if not path.exists():
        raise FileNotFoundError(
            f"Amendment file not found: {path}"
        )

    # --------------------------------------------------------
    # Read markdown
    # --------------------------------------------------------

    text = path.read_text(
        encoding="utf-8"
    )

    # --------------------------------------------------------
    # Extract metadata
    # --------------------------------------------------------

    amendment_id = extract_amendment_id(
        text,
        path
    )

    title = extract_title(
        text,
        path
    )

    effective_date = extract_effective_date(
        text
    )

    clauses = extract_clause_references(
        text
    )

    changes = extract_changes(
        text
    )

    # --------------------------------------------------------
    # Build amendment object
    # --------------------------------------------------------

    return {
        "amendment_id": amendment_id,
        "title": title,
        "effective_date": effective_date,
        "status": "active",
        "source_file": str(path),
        "clauses": clauses,
        "changes": changes,
        "text": text,
    }


# ============================================================
# SAVE PARSED AMENDMENT
# ============================================================

def save_amendment(
    amendment: Dict,
    output_file: str | Path
) -> None:
    """
    Save parsed amendment into amendments.json.

    Existing amendment with the same ID is replaced.
    """

    output_path = Path(
        output_file
    )

    # --------------------------------------------------------
    # Load existing data
    # --------------------------------------------------------

    if output_path.exists():

        with output_path.open(
            "r",
            encoding="utf-8"
        ) as file:

            data = json.load(file)

    else:

        data = {
            "amendments": []
        }

    # --------------------------------------------------------
    # Get amendments list
    # --------------------------------------------------------

    amendments = data.setdefault(
        "amendments",
        []
    )

    amendment_id = amendment.get(
        "amendment_id"
    )

    # --------------------------------------------------------
    # Remove previous version
    # --------------------------------------------------------

    amendments = [
        item
        for item in amendments
        if item.get("amendment_id")
        != amendment_id
    ]

    # --------------------------------------------------------
    # Add new version
    # --------------------------------------------------------

    amendments.append(
        amendment
    )

    data["amendments"] = amendments

    # --------------------------------------------------------
    # Ensure directory exists
    # --------------------------------------------------------

    output_path.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # --------------------------------------------------------
    # Save JSON
    # --------------------------------------------------------

    with output_path.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            data,
            file,
            indent=2,
            ensure_ascii=False
        )


# ============================================================
# STANDALONE EXECUTION
# ============================================================

if __name__ == "__main__":

    # --------------------------------------------------------
    # Project root
    # --------------------------------------------------------

    project_root = (
        Path(__file__)
        .resolve()
        .parents[2]
    )

    # --------------------------------------------------------
    # Amendment source
    # --------------------------------------------------------

    amendment_file = (
        project_root
        / "Data pack"
        / "Amendment No. 2026-01.md"
    )

    # --------------------------------------------------------
    # JSON output
    # --------------------------------------------------------

    output_file = (
        project_root
        / "data"
        / "amendments.json"
    )

    # --------------------------------------------------------
    # Parse
    # --------------------------------------------------------

    amendment = parse_amendment_file(
        amendment_file
    )

    # --------------------------------------------------------
    # Display
    # --------------------------------------------------------

    print(
        json.dumps(
            amendment,
            indent=2,
            ensure_ascii=False
        )
    )

    # --------------------------------------------------------
    # Save
    # --------------------------------------------------------

    save_amendment(
        amendment,
        output_file
    )

    print(
        "\nAmendment saved successfully."
    )