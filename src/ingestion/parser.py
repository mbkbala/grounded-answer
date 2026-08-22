import json
import re
from pathlib import Path


# Where our policy manual is located
POLICY_FILE = Path("Data pack/policy-manual.md")

# Where we will save the extracted clauses
OUTPUT_FILE = Path("data/clauses.json")


# This recognizes lines such as:
# **1.1.1** Some policy text
# **2.4.1** Some other policy text
CLAUSE_PATTERN = re.compile(
    r"^\*\*(\d+\.\d+\.\d+)\*\*\s*(.*)$"
)


def parse_policy_manual():

    clauses = []

    current_clause = None
    current_text = []

    # Open the policy manual
    with POLICY_FILE.open(
        "r",
        encoding="utf-8"
    ) as file:

        # Read every line
        for line in file:

            line = line.rstrip()

            # Check whether this line starts a new clause
            match = CLAUSE_PATTERN.match(line)

            if match:

                # Save the previous clause
                if current_clause is not None:

                    clauses.append({
                        "clause": f"§{current_clause}",
                        "text": " ".join(current_text).strip()
                    })

                # Get the new clause number
                current_clause = match.group(1)

                # Get the text after the clause number
                current_text = [match.group(2)]

            else:
            
                # Ignore Markdown headings
                # such as:
                # ## 1.2 Structure of this manual
                if line.startswith("#"):
                    continue
            
                # If we are already inside a clause,
                # add continuation text to that clause.
                if current_clause is not None:
            
                    if line.strip():
                        current_text.append(
                            line.strip()
                        )

        # Save the final clause
        if current_clause is not None:

            clauses.append({
                "clause": f"§{current_clause}",
                "text": " ".join(current_text).strip()
            })

    return clauses


def main():

    clauses = parse_policy_manual()

    # Make sure the data folder exists
    OUTPUT_FILE.parent.mkdir(
        parents=True,
        exist_ok=True
    )

    # Save clauses as JSON
    with OUTPUT_FILE.open(
        "w",
        encoding="utf-8"
    ) as file:

        json.dump(
            clauses,
            file,
            indent=2,
            ensure_ascii=False
        )

    print(
        f"Parsed {len(clauses)} policy clauses."
    )

    print(
        f"Saved to: {OUTPUT_FILE}"
    )

    print("\nFirst 5 clauses:\n")

    for clause in clauses[:5]:

        print(
            f"{clause['clause']}: "
            f"{clause['text']}"
        )


if __name__ == "__main__":
    main()
