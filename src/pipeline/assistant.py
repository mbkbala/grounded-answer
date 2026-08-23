# ============================================================
# GROUNDED POLICY ASSISTANT
# END-TO-END ASSISTANT PIPELINE
# ============================================================

import re
from typing import Dict, Optional

from src.retrieval.hybrid_search import HybridSearch
from src.generation.grounded_answer import GroundedAnswerGenerator


def needs_temporal_dates(question: str) -> bool:
    """Return whether a question needs date-based policy resolution."""

    return bool(
        re.search(
            r"\b(?:date|dated|before|after|effective|earnings?|"
            r"report(?:ing)?|deadline|sanction|claim|period|change)\b",
            question.lower(),
        )
    )


# ============================================================
# GROUNDED POLICY ASSISTANT
# ============================================================

class GroundedPolicyAssistant:
    """
    End-to-end policy assistant.

    Pipeline:

        User Question
              ↓
        HybridSearch
              ↓
        Retrieved Evidence
              ↓
        GroundedAnswerGenerator
              ↓
        Final Answer + Citation
    """

    def __init__(self):

        print("=" * 70)
        print("INITIALIZING GROUNDED POLICY ASSISTANT")
        print("=" * 70)

        # ----------------------------------------------------
        # Retrieval engine
        # ----------------------------------------------------

        print("\nLoading retrieval engine...")

        self.searcher = HybridSearch()

        # ----------------------------------------------------
        # Answer generator
        # ----------------------------------------------------

        print("Loading grounded answer generator...")

        self.generator = GroundedAnswerGenerator()

        print("\nAssistant ready.")

    # ========================================================
    # ASK QUESTION
    # ========================================================

    def ask(
        self,
        question: str,
        determination_date: Optional[str] = None,
        event_date: Optional[str] = None,
    ) -> Dict:
        """
        Process a user question through the complete pipeline.
        """

        # ----------------------------------------------------
        # Validate question
        # ----------------------------------------------------

        if not question or not question.strip():

            return {
                "answerable": False,
                "answer": (
                    "Please enter a valid question."
                ),
                "citations": [],
                "sources": [],
                "reason": "Empty question."
            }

        question = question.strip()

        # ----------------------------------------------------
        # STEP 1: RETRIEVAL
        # ----------------------------------------------------

        retrieval_response = self.searcher.search(
            question,
            determination_date=determination_date,
            event_date=event_date,
        )

        # ----------------------------------------------------
        # STEP 2: GROUNDED GENERATION
        # ----------------------------------------------------

        answer_response = self.generator.generate(
            question,
            retrieval_response
        )

        # ----------------------------------------------------
        # STEP 3: COMBINE PIPELINE INFORMATION
        # ----------------------------------------------------

        return {
            "question": question,

            "answerable": answer_response.get(
                "answerable",
                False
            ),

            "answer": answer_response.get(
                "answer",
                ""
            ),

            "citations": answer_response.get(
                "citations",
                []
            ),

            "sources": answer_response.get(
                "sources",
                []
            ),

            "reason": answer_response.get(
                "reason",
                ""
            ),

            "temporal": retrieval_response.get(
                "temporal",
                {},
            ),

            # Keep retrieval information for debugging
            # and future UI/API use.
            "retrieval": retrieval_response
        }


# ============================================================
# DISPLAY RESPONSE
# ============================================================

def display_response(
    response: Dict
):
    """
    Display a clean user-facing response.
    """

    print("\n")
    print("=" * 70)
    print("ANSWER")
    print("=" * 70)

    print(
        response.get(
            "answer",
            "No answer available."
        )
    )

    # --------------------------------------------------------
    # Citations
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("CITATIONS")
    print("=" * 70)

    citations = response.get(
        "citations",
        []
    )

    if citations:

        for citation in citations:

            print(f"- {citation}")

    else:

        print("No supporting clause found.")

    # --------------------------------------------------------
    # Status
    # --------------------------------------------------------

    print("\n" + "=" * 70)
    print("STATUS")
    print("=" * 70)

    print(
        f"Answerable : "
        f"{response.get('answerable', False)}"
    )

    print(
        f"Reason     : "
        f"{response.get('reason', '')}"
    )


# ============================================================
# INTERACTIVE MODE
# ============================================================

def interactive_mode():

    assistant = GroundedPolicyAssistant()

    print("\n")
    print("=" * 70)
    print("GROUNDED POLICY ASSISTANT")
    print("=" * 70)

    print(
        "\nAsk questions about the policy manual."
    )

    print(
        "Type 'exit' or 'quit' to stop."
    )

    while True:

        try:

            question = input(
                "\nEnter your question: "
            ).strip()

        except (
            KeyboardInterrupt,
            EOFError
        ):

            print("\n\nExiting...")
            break

        # ----------------------------------------------------
        # Exit
        # ----------------------------------------------------

        if question.lower() in {
            "exit",
            "quit"
        }:

            print("\nGoodbye!")
            break

        # ----------------------------------------------------
        # Empty question
        # ----------------------------------------------------

        if not question:

            print(
                "\nPlease enter a question."
            )

            continue

        determination_date = None
        event_date = None

        if needs_temporal_dates(question):
            print("Dates use YYYY-MM-DD.")

            determination_date = input(
                "Determination date (optional): "
            ).strip() or None

            event_date = input(
                "Change/event date (optional): "
            ).strip() or None

        # ----------------------------------------------------
        # Run complete pipeline
        # ----------------------------------------------------

        try:

            response = assistant.ask(
                question,
                determination_date=determination_date,
                event_date=event_date,
            )

            display_response(
                response
            )

        except Exception as exc:

            print("\n" + "=" * 70)
            print("ERROR")
            print("=" * 70)

            print(
                f"Something went wrong: {exc}"
            )


# ============================================================
# MAIN
# ============================================================

if __name__ == "__main__":

    interactive_mode()