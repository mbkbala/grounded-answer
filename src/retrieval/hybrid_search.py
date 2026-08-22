import json
import re
from pathlib import Path

import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


# ============================================================
# CONFIGURATION
# ============================================================

CLAUSES_FILE = Path("data/clauses.json")

MODEL_NAME = "all-MiniLM-L6-v2"

# Semantic threshold
SEMANTIC_THRESHOLD = 0.30

# Final hybrid threshold
HYBRID_THRESHOLD = 0.45

# Strong lexical evidence can compensate for lower semantic score
STRONG_BM25_THRESHOLD = 5.0
STRONG_KEYWORD_THRESHOLD = 0.75
STRONG_HYBRID_THRESHOLD = 0.60


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize(text):
    return re.findall(r"\b[a-zA-Z0-9]+\b", text.lower())


# ============================================================
# KEYWORD OVERLAP
# ============================================================

def keyword_overlap(question, text):

    question_tokens = set(tokenize(question))
    text_tokens = set(tokenize(text))

    if not question_tokens:
        return 0.0

    important_words = {
        "eligibility",
        "eligible",
        "requirements",
        "requirement",
        "condition",
        "conditions",
        "qualify",
        "qualification",
        "residence",
        "resident",
        "income",
        "resources",
        "assistance",
        "application",
        "excluded",
        "exclusion",
        "age",
        "household",
        "sanction",
        "detained",
        "correctional",
        "equivalent",
        "misrepresentation",
    }

    question_important = question_tokens & important_words
    text_important = text_tokens & important_words

    if not question_important:
        return 0.0

    overlap = question_important & text_important

    return len(overlap) / len(question_important)


# ============================================================
# CLAUSE REFERENCES
# ============================================================

def extract_clause_references(text):

    # Matches:
    # §2.1.2
    # §2.4
    # §10.5

    return re.findall(
        r"§\d+(?:\.\d+){1,2}",
        text
    )


# ============================================================
# HYBRID SEARCH
# ============================================================

class HybridSearch:

    def __init__(self):

        # ----------------------------------------------------
        # Load policy clauses
        # ----------------------------------------------------

        with CLAUSES_FILE.open(
            "r",
            encoding="utf-8"
        ) as file:

            self.clauses = json.load(file)

        self.clause_lookup = {
            clause["clause"]: clause
            for clause in self.clauses
        }

        # ----------------------------------------------------
        # BM25
        # ----------------------------------------------------

        self.tokenized_clauses = [
            tokenize(clause["text"])
            for clause in self.clauses
        ]

        self.bm25 = BM25Okapi(
            self.tokenized_clauses
        )

        # ----------------------------------------------------
        # Semantic model
        # ----------------------------------------------------

        print("Loading semantic model...")

        self.model = SentenceTransformer(
            MODEL_NAME
        )

        texts = [
            clause["text"]
            for clause in self.clauses
        ]

        print("Creating semantic index...")

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(
            dimension
        )

        self.index.add(embeddings)

        print(
            f"Loaded {len(self.clauses)} policy clauses."
        )

    # ========================================================
    # SECTION BOOST
    # ========================================================

    def get_section_boost(
        self,
        question,
        clause_id
    ):

        question_lower = question.lower()

        # ----------------------------------------------------
        # Eligibility intent
        # ----------------------------------------------------

        eligibility_words = {
            "eligibility",
            "eligible",
            "requirements",
            "requirement",
            "qualify",
            "qualification",
            "conditions",
            "excluded",
            "exclusion",
        }

        if not any(
            word in question_lower
            for word in eligibility_words
        ):

            return 0.0

        # ----------------------------------------------------
        # Exclusion questions
        # ----------------------------------------------------

        exclusion_words = {
            "excluded",
            "exclusion",
            "disqualified",
            "disqualify",
            "not eligible",
        }

        is_exclusion_question = any(
            word in question_lower
            for word in exclusion_words
        )

        if is_exclusion_question:

            # Part 4 is the main exclusion section
            if clause_id.startswith("§4."):

                if clause_id == "§4.1.1":
                    return 0.35

                return 0.15

            # Part 2 references exclusions
            if clause_id.startswith("§2."):

                return 0.05

            return 0.0

        # ----------------------------------------------------
        # General eligibility question
        # ----------------------------------------------------

        if clause_id.startswith("§2."):

            # Direct conditions
            if clause_id == "§2.1.2":
                return 0.30

            # Main eligibility rule
            if clause_id == "§2.1.1":
                return 0.25

            # Other Part 2 clauses
            return 0.10

        # ----------------------------------------------------
        # Residence requirements
        # ----------------------------------------------------

        if clause_id.startswith("§3."):

            return 0.08

        # ----------------------------------------------------
        # Exclusions
        # ----------------------------------------------------

        if clause_id.startswith("§4."):

            return 0.08

        return 0.0

    # ========================================================
    # SEARCH
    # ========================================================

    def search(
        self,
        question,
        top_k=5
    ):

        # ----------------------------------------------------
        # 1. BM25
        # ----------------------------------------------------

        query_tokens = tokenize(question)

        bm25_scores = self.bm25.get_scores(
            query_tokens
        )

        # ----------------------------------------------------
        # 2. Semantic similarity
        # ----------------------------------------------------

        query_embedding = self.model.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        semantic_scores, _ = self.index.search(
            query_embedding,
            len(self.clauses)
        )

        semantic_scores = semantic_scores[0]

        # ----------------------------------------------------
        # 3. Normalize BM25
        # ----------------------------------------------------

        max_bm25 = max(bm25_scores)

        if max_bm25 > 0:

            normalized_bm25 = [
                score / max_bm25
                for score in bm25_scores
            ]

        else:

            normalized_bm25 = [
                0.0
                for _ in bm25_scores
            ]

        # ----------------------------------------------------
        # 4. Calculate hybrid scores
        # ----------------------------------------------------

        combined_results = []

        for i, clause in enumerate(
            self.clauses
        ):

            keyword_score = keyword_overlap(
                question,
                clause["text"]
            )

            section_boost = self.get_section_boost(
                question,
                clause["clause"]
            )

            # ------------------------------------------------
            # Hybrid score
            # ------------------------------------------------

            hybrid_score = (

                0.25 * normalized_bm25[i]

                + 0.60 * semantic_scores[i]

                + 0.15 * keyword_score

                + section_boost
            )

            result = clause.copy()

            result["bm25_score"] = float(
                bm25_scores[i]
            )

            result["semantic_score"] = float(
                semantic_scores[i]
            )

            result["keyword_score"] = float(
                keyword_score
            )

            result["section_boost"] = float(
                section_boost
            )

            result["hybrid_score"] = float(
                hybrid_score
            )

            combined_results.append(
                result
            )

        # ----------------------------------------------------
        # 5. Rank
        # ----------------------------------------------------

        combined_results.sort(
            key=lambda x: x["hybrid_score"],
            reverse=True
        )

        # ----------------------------------------------------
        # 6. REFUSAL GATE
        # ----------------------------------------------------

        if not combined_results:
            return {
                "answerable": False,
                "reason": "No clauses found.",
                "results": []
            }

        best = combined_results[0]

        semantic_ok = (
            best["semantic_score"]
            >= SEMANTIC_THRESHOLD
        )

        hybrid_ok = (
            best["hybrid_score"]
            >= HYBRID_THRESHOLD
        )

        # Strong lexical evidence can compensate
        # for lower semantic similarity.
        strong_lexical_support = (
            best["bm25_score"] >= 5.0
            and best["keyword_score"] >= 0.75
            and best["hybrid_score"] >= 0.60
        )

        # Final answerability decision
        answerable = (
            hybrid_ok
            and
            (
                semantic_ok
                or
                strong_lexical_support
            )
        )

        print(
            "DEBUG GATE:",
            "semantic_ok=", semantic_ok,
            "hybrid_ok=", hybrid_ok,
            "strong_lexical_support=", strong_lexical_support,
            "answerable=", answerable
        )

        if not answerable:
            return {
                "answerable": False,
                "reason": (
                    "The question does not have "
                    "sufficiently strong support in "
                    "the policy manual."
                ),
                "results": combined_results[:top_k]
            }

        # ----------------------------------------------------
        # Strong lexical evidence
        #
        # This prevents correct policy clauses from being
        # rejected merely because semantic similarity is low.
        # ----------------------------------------------------

        strong_lexical_support = (

            best["bm25_score"]
            >= STRONG_BM25_THRESHOLD

            and

            best["keyword_score"]
            >= STRONG_KEYWORD_THRESHOLD

            and

            best["hybrid_score"]
            >= STRONG_HYBRID_THRESHOLD
        )

        # ----------------------------------------------------
        # Final answerability decision
        # ----------------------------------------------------

        answerable = (

            hybrid_ok

            and

            (
                semantic_ok
                or
                strong_lexical_support
            )
        )

        if not answerable:

            return {

                "answerable": False,

                "reason": (
                    "The question does not have "
                    "sufficiently strong support in "
                    "the policy manual."
                ),

                "results": combined_results[:top_k]
            }

        # ----------------------------------------------------
        # 7. Expand referenced clauses
        # ----------------------------------------------------

        expanded_results = []

        added_clauses = set()

        for result in combined_results[:top_k]:

            clause_id = result["clause"]

            # ------------------------------------------------
            # Add original result
            # ------------------------------------------------

            if clause_id not in added_clauses:

                expanded_results.append(
                    result
                )

                added_clauses.add(
                    clause_id
                )

            # ------------------------------------------------
            # Find references
            # ------------------------------------------------

            references = extract_clause_references(
                result["text"]
            )

            for ref in references:

                if ref not in self.clause_lookup:
                    continue

                if ref in added_clauses:
                    continue

                referenced_clause = (
                    self.clause_lookup[ref].copy()
                )

                referenced_clause[
                    "bm25_score"
                ] = 0.0

                referenced_clause[
                    "semantic_score"
                ] = 0.0

                referenced_clause[
                    "keyword_score"
                ] = 0.0

                referenced_clause[
                    "section_boost"
                ] = 0.0

                referenced_clause[
                    "hybrid_score"
                ] = (
                    result["hybrid_score"]
                    * 0.95
                )

                referenced_clause[
                    "expanded_from"
                ] = result["clause"]

                expanded_results.append(
                    referenced_clause
                )

                added_clauses.add(
                    ref
                )

        # ----------------------------------------------------
        # 8. Return
        # ----------------------------------------------------

        return {

            "answerable": True,

            "reason": (
                "Relevant policy support found."
            ),

            "results": expanded_results[:top_k]
        }


# ============================================================
# MAIN
# ============================================================

def main():

    searcher = HybridSearch()

    print(
        "\n" + "=" * 60
    )

    print(
        "HYBRID POLICY SEARCH"
    )

    print(
        "=" * 60
    )

    while True:

        question = input(
            "\nEnter your question: "
        ).strip()

        if not question:
            continue

        if question.lower() in {
            "exit",
            "quit"
        }:
            break

        response = searcher.search(
            question,
            top_k=5
        )

        # ====================================================
        # REFUSAL
        # ====================================================

        if not response["answerable"]:

            print(
                "\n" + "-" * 60
            )

            print(
                "REFUSAL: The policy manual does not "
                "provide sufficiently strong support "
                "for this question."
            )

            print(
                "\nReason:",
                response["reason"]
            )

            print(
                "\nBest candidate for debugging:"
            )

            if response["results"]:

                result = response["results"][0]

                print(
                    "-" * 60
                )

                print(
                    f"Clause: "
                    f"{result['clause']}"
                )

                print(
                    f"BM25: "
                    f"{result['bm25_score']:.3f}"
                )

                print(
                    f"Semantic: "
                    f"{result['semantic_score']:.3f}"
                )

                print(
                    f"Keyword: "
                    f"{result['keyword_score']:.3f}"
                )

                print(
                    f"Section Boost: "
                    f"{result['section_boost']:.3f}"
                )

                print(
                    f"Hybrid: "
                    f"{result['hybrid_score']:.3f}"
                )

                print(
                    f"Text: "
                    f"{result['text']}"
                )

            continue

        # ====================================================
        # NORMAL RESULT
        # ====================================================

        print(
            "\nTop matching clauses:\n"
        )

        for result in response["results"]:

            print(
                "-" * 60
            )

            print(
                f"Clause: "
                f"{result['clause']}"
            )

            print(
                f"BM25: "
                f"{result['bm25_score']:.3f}"
            )

            print(
                f"Semantic: "
                f"{result['semantic_score']:.3f}"
            )

            print(
                f"Keyword: "
                f"{result['keyword_score']:.3f}"
            )

            print(
                f"Section Boost: "
                f"{result['section_boost']:.3f}"
            )

            print(
                f"Hybrid: "
                f"{result['hybrid_score']:.3f}"
            )

            if "expanded_from" in result:

                print(
                    f"Expanded from: "
                    f"{result['expanded_from']}"
                )

            print(
                f"Text: "
                f"{result['text']}"
            )


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()