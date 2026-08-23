# ============================================================
# GROUNDED POLICY ASSISTANT
# RELEVANCE CHECKER
# ============================================================

from typing import Dict, List


class RelevanceChecker:
    """
    Checks whether retrieved policy evidence is strong enough
    to support an answer.

    This prevents weak semantic matches from being treated
    as valid evidence.
    """

    def __init__(
        self,
        semantic_threshold: float = 0.30,
        minimum_hybrid_score: float = 0.70,
        minimum_keyword_score: float = 0.15
    ):
        self.semantic_threshold = semantic_threshold
        self.minimum_hybrid_score = minimum_hybrid_score
        self.minimum_keyword_score = minimum_keyword_score

    def is_relevant(
        self,
        results: List[Dict]
    ) -> bool:
        """
        Return True only when the strongest result has
        sufficient retrieval support.
        """

        if not results:
            return False

        valid_results = [
            result
            for result in results
            if result.get("clause")
        ]

        if not valid_results:
            return False

        best = max(
            valid_results,
            key=lambda result: float(
                result.get("hybrid_score", 0.0)
            )
        )

        semantic_score = float(
            best.get("semantic_score", 0.0)
        )

        hybrid_score = float(
            best.get("hybrid_score", 0.0)
        )

        keyword_score = float(
            best.get("keyword_score", 0.0)
        )

        bm25_score = float(
            best.get("bm25_score", 0.0)
        )

        # ----------------------------------------------------
        # Strong semantic + hybrid evidence
        # ----------------------------------------------------

        if (
            semantic_score >= self.semantic_threshold
            and hybrid_score >= self.minimum_hybrid_score
        ):
            return True

        # ----------------------------------------------------
        # Strong lexical evidence can compensate for slightly
        # weaker semantic similarity.
        # ----------------------------------------------------

        if (
            bm25_score >= 5.0
            and keyword_score >= 0.75
            and hybrid_score >= 0.60
        ):
            return True

        return False