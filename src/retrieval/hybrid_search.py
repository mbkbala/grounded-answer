# ============================================================
# GROUNDED POLICY ASSISTANT
# HYBRID RETRIEVAL ENGINE
# ============================================================

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Set

import faiss
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer


# ============================================================
# PROJECT PATH
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[2]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# CONFIGURATION
# ============================================================

CLAUSES_FILE = PROJECT_ROOT / "data" / "clauses.json"

MODEL_NAME = "all-MiniLM-L6-v2"

# Retrieval thresholds
SEMANTIC_THRESHOLD = 0.30
HYBRID_THRESHOLD = 0.45

# Lexical fallback
STRONG_BM25_THRESHOLD = 5.0
STRONG_KEYWORD_THRESHOLD = 0.75
STRONG_HYBRID_THRESHOLD = 0.60


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize(text: str) -> List[str]:
    """
    Simple normalized tokenizer.
    """
    return re.findall(
        r"\b[a-zA-Z0-9]+\b",
        text.lower()
    )


# ============================================================
# CLAUSE REFERENCE EXTRACTION
# ============================================================

def extract_clause_references(text: str) -> List[str]:
    """
    Extract policy references such as:

        §2.1.2
        §2.4
        §10.5
    """

    return re.findall(
        r"§\d+(?:\.\d+){1,2}",
        text
    )


# ============================================================
# QUESTION INTENT
# ============================================================

def get_question_type(question: str) -> str:
    """
    Determine the most specific policy intent.

    Order matters.
    Specific intents must be detected before
    generic eligibility intent.
    """

    q = question.lower().strip()

    # --------------------------------------------------------
    # DIRECT CLAUSE REFERENCE
    # --------------------------------------------------------

    if re.search(r"§\d+(?:\.\d+){1,2}", question):
        return "clause_reference"

    # --------------------------------------------------------
    # WEATHER / OUTSIDE POLICY
    # --------------------------------------------------------

    outside_policy_terms = {
        "weather",
        "temperature",
        "forecast",
        "rain",
        "rainfall",
        "climate",
        "python",
        "programming",
        "code",
        "capital",
        "president",
        "football",
        "movie",
        "music",
    }

    if any(term in q for term in outside_policy_terms):
        return "outside_policy"

    # --------------------------------------------------------
    # CORRECTIONAL FACILITY
    # --------------------------------------------------------

    if (
        "correctional facility" in q
        or "detained" in q
        or "detention" in q
        or "jail" in q
        or "prison" in q
    ):
        return "correctional_exclusion"

    # --------------------------------------------------------
    # SANCTION
    # --------------------------------------------------------

    if "sanction" in q:
        return "sanction_exclusion"

    # --------------------------------------------------------
    # GENERAL EXCLUSION
    # --------------------------------------------------------

    exclusion_terms = {
        "excluded",
        "exclusion",
        "disqualified",
        "disqualify",
        "not eligible",
        "ineligible",
    }

    if any(term in q for term in exclusion_terms):
        return "exclusion"

    # --------------------------------------------------------
    # AGE
    # --------------------------------------------------------

    # Exact 18 must be checked BEFORE generic age.
    if re.search(r"\b18\s*(?:years?\s*old|year-old)\b", q):
        return "age_18"

    if re.search(r"\b17\s*(?:years?\s*old|year-old)\b", q):
        return "age_minor"

    if re.search(r"\b16\s*(?:years?\s*old|year-old)\b", q):
        return "age_minor"

    if "under 18" in q:
        return "age_minor"

    if "minor" in q:
        return "age_minor"

    # Generic age question
    if "age" in q or "aged" in q:
        return "age"

    # --------------------------------------------------------
    # RESIDENCE
    # --------------------------------------------------------

    residence_terms = {
        "residence",
        "resident",
        "live",
        "living",
        "lives",
        "stay",
        "staying",
        "county",
    }

    if (
        any(term in q for term in residence_terms)
        and (
            "qualify" in q
            or "eligible" in q
            or "assistance" in q
            or "program" in q
            or "requirement" in q
            or "need to" in q
        )
    ):
        return "residence"

    # --------------------------------------------------------
    # INCOME
    # --------------------------------------------------------

    if "income" in q:
        return "income"

    # --------------------------------------------------------
    # RESOURCES
    # --------------------------------------------------------

    if (
        "resource" in q
        or "resources" in q
    ):
        return "resources"

    # --------------------------------------------------------
    # APPLICATION
    # --------------------------------------------------------

    if (
        "application" in q
        or "apply" in q
        or "application requirement" in q
    ):
        return "application"

    # --------------------------------------------------------
    # ADMINISTRATION
    # --------------------------------------------------------

    if (
        "administer" in q
        or "administers" in q
        or "administration" in q
        or "department" in q
        or "caseworker" in q
        or "caseworker" in q
    ):
        return "administration"

    # --------------------------------------------------------
    # GENERAL ELIGIBILITY
    # --------------------------------------------------------

    eligibility_terms = {
        "eligibility",
        "eligible",
        "requirements",
        "requirement",
        "qualify",
        "qualification",
        "conditions",
    }

    if any(term in q for term in eligibility_terms):
        return "eligibility"

    # --------------------------------------------------------
    # GENERAL POLICY
    # --------------------------------------------------------

    policy_terms = {
        "assistance",
        "program",
        "benefit",
        "benefits",
        "household",
        "recipient",
        "payment",
        "support",
        "calder",
        "county",
        "policy",
        "manual",
    }

    if any(term in q for term in policy_terms):
        return "general"

    return "outside_policy"


# ============================================================
# POLICY SCOPE GATE
# ============================================================

def is_policy_question(question: str) -> bool:
    """
    Determine whether the question belongs to the
    Household Support Program policy domain.
    """

    question_type = get_question_type(question)

    if question_type == "outside_policy":
        return False

    return True


# ============================================================
# KEYWORD OVERLAP
# ============================================================

def keyword_overlap(
    question: str,
    text: str
) -> float:
    """
    Calculate overlap between important policy terms.
    """

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
        "apply",
        "excluded",
        "exclusion",
        "disqualified",
        "sanction",
        "detained",
        "correctional",
        "facility",
        "equivalent",
        "misrepresentation",
        "administered",
        "administer",
        "department",
        "caseworker",
        "program",
        "award",
        "recipient",
        "payment",
        "support",
        "household",
        "age",
        "aged",
        "minor",
        "county",
        "live",
        "living",
        "18",
        "17",
        "16",
    }

    question_important = (
        question_tokens & important_words
    )

    text_important = (
        text_tokens & important_words
    )

    if not question_important:
        return 0.0

    overlap = (
        question_important & text_important
    )

    return (
        len(overlap)
        / len(question_important)
    )


# ============================================================
# HYBRID SEARCH
# ============================================================

class HybridSearch:

    def __init__(self):

        # ----------------------------------------------------
        # Load clauses
        # ----------------------------------------------------

        if not CLAUSES_FILE.exists():
            raise FileNotFoundError(
                f"Policy clauses file not found: "
                f"{CLAUSES_FILE}"
            )

        with CLAUSES_FILE.open(
            "r",
            encoding="utf-8"
        ) as file:
            self.clauses = json.load(file)

        if not self.clauses:
            raise ValueError(
                "No policy clauses were loaded."
            )

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
        question: str,
        clause_id: str
    ) -> float:

        question_type = get_question_type(
            question
        )

        # ----------------------------------------------------
        # EXACT CLAUSE REFERENCE
        # ----------------------------------------------------

        if question_type == "clause_reference":

            references = extract_clause_references(
                question
            )

            if clause_id in references:
                return 1.00

            return 0.0

        # ----------------------------------------------------
        # CORRECTIONAL EXCLUSION
        # ----------------------------------------------------

        if question_type == "correctional_exclusion":

            if clause_id == "§4.1.1":
                return 0.70

            if clause_id.startswith("§4."):
                return 0.20

            return 0.0

        # ----------------------------------------------------
        # SANCTION EXCLUSION
        # ----------------------------------------------------

        if question_type == "sanction_exclusion":

            if clause_id == "§4.1.1":
                return 0.55

            if clause_id.startswith("§4."):
                return 0.20

            return 0.0

        # ----------------------------------------------------
        # GENERAL EXCLUSION
        # ----------------------------------------------------

        if question_type == "exclusion":

            if clause_id == "§4.1.1":
                return 0.60

            if clause_id.startswith("§4."):
                return 0.20

            if clause_id.startswith("§2."):
                return 0.05

            return 0.0

        # ----------------------------------------------------
        # AGE 18
        # ----------------------------------------------------

        if question_type == "age_18":

            # Normal rule MUST dominate.
            if clause_id == "§2.1.2":
                return 0.70

            if clause_id == "§2.1.1":
                return 0.25

            # Do not boost §2.3 for exactly 18.
            return 0.0

        # ----------------------------------------------------
        # AGE 16 / 17
        # ----------------------------------------------------

        if question_type == "age_minor":

            if clause_id == "§2.3.1":
                return 0.70

            if clause_id == "§2.3.2":
                return 0.35

            if clause_id == "§2.1.2":
                return 0.45

            if clause_id == "§2.1.1":
                return 0.15

            return 0.0

        # ----------------------------------------------------
        # GENERIC AGE
        # ----------------------------------------------------

        if question_type == "age":

            if clause_id == "§2.1.2":
                return 0.45

            if clause_id == "§2.3.1":
                return 0.40

            if clause_id == "§2.3.2":
                return 0.20

            return 0.0

        # ----------------------------------------------------
        # RESIDENCE
        # ----------------------------------------------------

        if question_type == "residence":

            # Controlling eligibility clause
            if clause_id == "§2.1.2":
                return 0.65

            # Supporting residence clauses
            if clause_id.startswith("§3."):
                return 0.20

            return 0.0

        # ----------------------------------------------------
        # INCOME
        # ----------------------------------------------------

        if question_type == "income":

            # §2.1.2 establishes income as an eligibility
            # condition. Part 6 provides calculation details.
            if clause_id == "§2.1.2":
                return 0.55

            if clause_id.startswith("§6."):
                return 0.20

            return 0.0

        # ----------------------------------------------------
        # RESOURCES
        # ----------------------------------------------------

        if question_type == "resources":

            if clause_id == "§2.1.2":
                return 0.55

            if clause_id == "§2.4":
                return 0.35

            if clause_id.startswith("§2.4."):
                return 0.25

            return 0.0

        # ----------------------------------------------------
        # APPLICATION
        # ----------------------------------------------------

        if question_type == "application":

            if clause_id == "§2.1.2":
                return 0.60

            if clause_id.startswith("§8."):
                return 0.20

            return 0.0

        # ----------------------------------------------------
        # ADMINISTRATION
        # ----------------------------------------------------

        if question_type == "administration":

            if clause_id == "§1.1.2":
                return 0.70

            if clause_id.startswith("§1."):
                return 0.20

            return 0.0

        # ----------------------------------------------------
        # GENERAL ELIGIBILITY
        # ----------------------------------------------------

        if question_type == "eligibility":

            if clause_id == "§2.1.2":
                return 0.60

            if clause_id == "§2.1.1":
                return 0.30

            if clause_id.startswith("§2."):
                return 0.10

            if clause_id.startswith("§3."):
                return 0.05

            if clause_id.startswith("§4."):
                return 0.05

            return 0.0

        return 0.0

    # ========================================================
    # FORCE CLAUSE
    # ========================================================

    def _make_forced_result(
        self,
        clause_id: str,
        expanded_from: str = None,
        score: float = 1.0
    ):

        clause = self.clause_lookup.get(
            clause_id
        )

        if clause is None:
            return None

        result = clause.copy()

        result["bm25_score"] = 0.0
        result["semantic_score"] = 0.0
        result["keyword_score"] = 0.0
        result["section_boost"] = score
        result["hybrid_score"] = score

        if expanded_from:
            result["expanded_from"] = (
                expanded_from
            )

        return result

    # ========================================================
    # SEARCH
    # ========================================================

    def search(
        self,
        question: str,
        top_k: int = 5
    ) -> Dict:

        question = question.strip()

        # ----------------------------------------------------
        # Empty question
        # ----------------------------------------------------

        if not question:
            return {
                "answerable": False,
                "reason": "Empty question.",
                "results": []
            }

        # ----------------------------------------------------
        # POLICY SCOPE GATE
        # ----------------------------------------------------

        if not is_policy_question(question):

            return {
                "answerable": False,
                "reason": (
                    "The question is outside the scope "
                    "of the policy manual."
                ),
                "results": []
            }

        question_type = get_question_type(
            question
        )

        # ----------------------------------------------------
        # DIRECT CLAUSE REFERENCE
        # ----------------------------------------------------

        if question_type == "clause_reference":

            references = extract_clause_references(
                question
            )

            exact_results = []

            for ref in references:

                result = self._make_forced_result(
                    ref,
                    score=1.0
                )

                if result:
                    exact_results.append(
                        result
                    )

            if exact_results:

                return {
                    "answerable": True,
                    "reason": (
                        "Exact policy clause reference found."
                    ),
                    "results": exact_results[:top_k]
                }

        # ----------------------------------------------------
        # BM25
        # ----------------------------------------------------

        query_tokens = tokenize(
            question
        )

        bm25_scores = self.bm25.get_scores(
            query_tokens
        )

        # ----------------------------------------------------
        # Semantic similarity
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
        # Normalize BM25
        # ----------------------------------------------------

        max_bm25 = max(
            bm25_scores
        )

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
        # Hybrid scoring
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
        # Rank
        # ----------------------------------------------------

        combined_results.sort(
            key=lambda item: item["hybrid_score"],
            reverse=True
        )

        if not combined_results:

            return {
                "answerable": False,
                "reason": "No policy clauses found.",
                "results": []
            }

        # ====================================================
        # INTENT-AWARE RESULT CONSTRUCTION
        # ====================================================

        selected = []
        selected_ids: Set[str] = set()

        def add_result(result):

            if result is None:
                return

            clause_id = result["clause"]

            if clause_id in selected_ids:
                return

            selected.append(result)
            selected_ids.add(clause_id)

        # ----------------------------------------------------
        # AGE 18
        # ----------------------------------------------------

        if question_type == "age_18":

            # §2.1.2 is controlling.
            add_result(
                self._make_forced_result(
                    "§2.1.2",
                    score=1.0
                )
            )

            # Do NOT include §2.3 for exact 18.
            # This prevents the minor exception from
            # contaminating the answer.

            # Add useful normal eligibility support.
            for result in combined_results:

                if result["clause"] == "§2.1.1":
                    add_result(result)
                    break

        # ----------------------------------------------------
        # AGE 16 / 17
        # ----------------------------------------------------

        elif question_type == "age_minor":

            # General rule
            add_result(
                self._make_forced_result(
                    "§2.1.2",
                    score=1.0
                )
            )

            # Minor exception
            add_result(
                self._make_forced_result(
                    "§2.3.1",
                    expanded_from="§2.1.2",
                    score=0.95
                )
            )

            # Supervisor referral
            add_result(
                self._make_forced_result(
                    "§2.3.2",
                    expanded_from="§2.3.1",
                    score=0.85
                )
            )

        # ----------------------------------------------------
        # RESIDENCE
        # ----------------------------------------------------

        elif question_type == "residence":

            # Controlling clause MUST be present.
            add_result(
                self._make_forced_result(
                    "§2.1.2",
                    score=1.0
                )
            )

            # Add best Part 3 clauses.
            for result in combined_results:

                if (
                    result["clause"].startswith("§3.")
                ):
                    add_result(result)

                    if len(selected) >= top_k:
                        break

        # ----------------------------------------------------
        # CORRECTIONAL FACILITY
        # ----------------------------------------------------

        elif question_type == "correctional_exclusion":

            # Specific exclusion clause MUST be first.
            add_result(
                self._make_forced_result(
                    "§4.1.1",
                    score=1.0
                )
            )

            # General eligibility support.
            for result in combined_results:

                if result["clause"] == "§2.1.2":
                    add_result(result)
                    break

        # ----------------------------------------------------
        # SANCTION
        # ----------------------------------------------------

        elif question_type == "sanction_exclusion":

            add_result(
                self._make_forced_result(
                    "§4.1.1",
                    score=1.0
                )
            )

            for result in combined_results:

                if result["clause"] == "§2.1.2":
                    add_result(result)
                    break

        # ----------------------------------------------------
        # GENERAL EXCLUSION
        # ----------------------------------------------------

        elif question_type == "exclusion":

            add_result(
                self._make_forced_result(
                    "§4.1.1",
                    score=1.0
                )
            )

            for result in combined_results:

                if result["clause"].startswith("§4."):
                    add_result(result)

                    if len(selected) >= top_k:
                        break

        # ----------------------------------------------------
        # INCOME
        # ----------------------------------------------------

        elif question_type == "income":

            # Controlling eligibility rule first.
            add_result(
                self._make_forced_result(
                    "§2.1.2",
                    score=1.0
                )
            )

            # Calculation rules from Part 6.
            for result in combined_results:

                if result["clause"].startswith("§6."):
                    add_result(result)

                    if len(selected) >= top_k:
                        break

        # ----------------------------------------------------
        # RESOURCES
        # ----------------------------------------------------

        elif question_type == "resources":

            add_result(
                self._make_forced_result(
                    "§2.1.2",
                    score=1.0
                )
            )

            add_result(
                self._make_forced_result(
                    "§2.4",
                    expanded_from="§2.1.2",
                    score=0.90
                )
            )

            for result in combined_results:

                if result["clause"].startswith("§2.4."):
                    add_result(result)

                    if len(selected) >= top_k:
                        break

        # ----------------------------------------------------
        # APPLICATION
        # ----------------------------------------------------

        elif question_type == "application":

            add_result(
                self._make_forced_result(
                    "§2.1.2",
                    score=1.0
                )
            )

            for result in combined_results:

                if result["clause"].startswith("§8."):
                    add_result(result)

                    if len(selected) >= top_k:
                        break

        # ----------------------------------------------------
        # ADMINISTRATION
        # ----------------------------------------------------

        elif question_type == "administration":

            add_result(
                self._make_forced_result(
                    "§1.1.2",
                    score=1.0
                )
            )

            for result in combined_results:

                if result["clause"].startswith("§1."):
                    add_result(result)

                    if len(selected) >= top_k:
                        break

        # ----------------------------------------------------
        # GENERAL ELIGIBILITY
        # ----------------------------------------------------

        elif question_type == "eligibility":

            add_result(
                self._make_forced_result(
                    "§2.1.2",
                    score=1.0
                )
            )

            add_result(
                self._make_forced_result(
                    "§2.1.1",
                    score=0.90
                )
            )

            for result in combined_results:

                add_result(result)

                if len(selected) >= top_k:
                    break

        # ----------------------------------------------------
        # GENERAL POLICY QUESTION
        # ----------------------------------------------------

        else:

            for result in combined_results:

                add_result(result)

                if len(selected) >= top_k:
                    break

        # ====================================================
        # ANSWERABILITY
        # ====================================================

        # If an intent-specific rule produced authoritative
        # clauses, we can trust that result set.
        intent_specific_types = {
            "age_18",
            "age_minor",
            "residence",
            "correctional_exclusion",
            "sanction_exclusion",
            "exclusion",
            "income",
            "resources",
            "application",
            "administration",
            "eligibility",
        }

        if question_type in intent_specific_types:

            if selected:

                return {
                    "answerable": True,
                    "reason": (
                        "Relevant policy support found."
                    ),
                    "results": selected[:top_k]
                }

        # ----------------------------------------------------
        # Generic answerability gate
        # ----------------------------------------------------

        best = combined_results[0]

        semantic_ok = (
            best["semantic_score"]
            >= SEMANTIC_THRESHOLD
        )

        hybrid_ok = (
            best["hybrid_score"]
            >= HYBRID_THRESHOLD
        )

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

        answerable = (
            hybrid_ok
            and
            (
                semantic_ok
                or strong_lexical_support
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

        return {
            "answerable": True,
            "reason": (
                "Relevant policy support found."
            ),
            "results": selected[:top_k]
        }


# ============================================================
# OPTIONAL STANDALONE TEST
# ============================================================

if __name__ == "__main__":

    searcher = HybridSearch()

    while True:

        question = input(
            "\nEnter your question: "
        ).strip()

        if question.lower() in {
            "exit",
            "quit"
        }:
            break

        response = searcher.search(
            question
        )

        print(
            json.dumps(
                response,
                indent=2,
                ensure_ascii=False
            )
        )