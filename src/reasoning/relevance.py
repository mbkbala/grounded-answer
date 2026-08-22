class RelevanceChecker:

    def __init__(
        self,
        semantic_threshold=0.30,
        minimum_hybrid_score=0.70
    ):
        self.semantic_threshold = semantic_threshold
        self.minimum_hybrid_score = minimum_hybrid_score

    def is_relevant(self, results):

        if not results:
            return False

        best = results[0]

        semantic_score = best["semantic_score"]
        hybrid_score = best["hybrid_score"]

        # Very weak semantic similarity means
        # the question is probably outside the policy.
        if semantic_score < self.semantic_threshold:
            return False

        # Also require reasonable overall retrieval.
        if hybrid_score < self.minimum_hybrid_score:
            return False

        return True