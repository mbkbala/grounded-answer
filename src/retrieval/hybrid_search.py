# ============================================================
# GROUNDED POLICY ASSISTANT
# HYBRID RETRIEVAL ENGINE
# ============================================================

import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Set, Optional

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
# OPTIONAL TEMPORAL POLICY SUPPORT
# ============================================================

try:
    from src.reasoning.temporal_policy import TemporalPolicy
except ImportError:
    TemporalPolicy = None


# ============================================================
# CONFIGURATION
# ============================================================

CLAUSES_FILE = PROJECT_ROOT / "data" / "clauses.json"

MODEL_NAME = "all-MiniLM-L6-v2"


# ============================================================
# RETRIEVAL THRESHOLDS
# ============================================================

SEMANTIC_THRESHOLD = 0.30
HYBRID_THRESHOLD = 0.45

STRONG_BM25_THRESHOLD = 5.0
STRONG_KEYWORD_THRESHOLD = 0.75
STRONG_HYBRID_THRESHOLD = 0.60


# ============================================================
# TOKENIZATION
# ============================================================

def tokenize(text: str) -> List[str]:
    """
    Normalize text into lowercase alphanumeric tokens.
    """

    if not text:
        return []

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

    if not text:
        return []

    return re.findall(
        r"§\d+(?:\.\d+){1,2}",
        text
    )


# ============================================================
# CLAUSE ID NORMALIZATION
# ============================================================

def normalize_clause_id(clause_id: str) -> str:
    """
    Normalize clause IDs.

    Examples:
        2.4       -> §2.4
        §2.4      -> §2.4
        2.1.2     -> §2.1.2
        §2.1.2    -> §2.1.2
    """

    if not clause_id:
        return ""

    clause_id = str(clause_id).strip()

    if not clause_id.startswith("§"):
        clause_id = "§" + clause_id

    return clause_id


# ============================================================
# GET PARENT SECTION
# ============================================================

def get_parent_section(clause_id: str) -> Optional[str]:
    """
    Convert a child clause into its parent section.

    Examples:
        §2.4.1 -> §2.4
        §2.4.2 -> §2.4
        §2.4.3 -> §2.4
    """

    normalized = normalize_clause_id(clause_id)

    match = re.match(
        r"^(§\d+\.\d+)\.\d+$",
        normalized
    )

    if match:
        return match.group(1)

    return None


# ============================================================
# QUESTION INTENT
# ============================================================

def get_question_type(question: str) -> str:
    """
    Determine the most specific policy intent.

    Specific intents are checked before generic intents.
    """

    q = question.lower().strip()

    # ========================================================
    # DIRECT CLAUSE REFERENCE
    # ========================================================

    if re.search(
        r"§\d+(?:\.\d+){1,2}",
        question
    ):
        return "clause_reference"

    # ========================================================
    # OUTSIDE POLICY
    # ========================================================

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

    question_tokens = set(tokenize(q))

    if question_tokens.intersection(outside_policy_terms):
        return "outside_policy"

    # ========================================================
    # CORRECTIONAL FACILITY
    # ========================================================

    if (
        "correctional facility" in q
        or re.search(r"\bdetained\b", q)
        or re.search(r"\bdetention\b", q)
        or re.search(r"\bjail\b", q)
        or re.search(r"\bprison\b", q)
    ):
        return "correctional_exclusion"

    # ========================================================
    # SANCTION
    # ========================================================

    if re.search(r"\bsanction\b", q):
        return "sanction_exclusion"

    # ========================================================
    # GENERAL EXCLUSION
    # ========================================================

    exclusion_terms = {
        "excluded",
        "exclusion",
        "disqualified",
        "disqualify",
        "not eligible",
        "ineligible",
    }

    if any(
        term in q
        for term in exclusion_terms
    ):
        return "exclusion"

    # ========================================================
    # AGE 18
    # ========================================================

    if re.search(
        r"\b18\s+(?:years?\s+old|year-old)\b",
        q
    ):
        return "age_18"

    # Also handle:
    # "age 18"
    # "at 18"
    # "18 years"

    if re.search(
        r"\bage\s+18\b",
        q
    ):
        return "age_18"

    if re.search(
        r"\bat\s+18\b",
        q
    ):
        return "age_18"

    # ========================================================
    # AGE 17
    # ========================================================

    if re.search(
        r"\b17\s+(?:years?\s+old|year-old)\b",
        q
    ):
        return "age_minor"

    if re.search(
        r"\bage\s+17\b",
        q
    ):
        return "age_minor"

    # ========================================================
    # AGE 16
    # ========================================================

    if re.search(
        r"\b16\s+(?:years?\s+old|year-old)\b",
        q
    ):
        return "age_minor"

    if re.search(
        r"\bage\s+16\b",
        q
    ):
        return "age_minor"

    # ========================================================
    # MINOR / UNDER 18
    # ========================================================

    if re.search(
        r"\bunder\s+18\b",
        q
    ):
        return "age_minor"

    if re.search(
        r"\bminor\b",
        q
    ):
        return "age_minor"

    # ========================================================
    # GENERIC AGE
    # ========================================================

    if re.search(
        r"\bage\b",
        q
    ):
        return "age"

    if re.search(
        r"\baged\b",
        q
    ):
        return "age"

    # ========================================================
    # RESIDENCE
    # ========================================================

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
        any(
            term in question_tokens
            for term in residence_terms
        )
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

    # ========================================================
    # INCOME
    # ========================================================

    if re.search(
        r"\bincome\b",
        q
    ):
        return "income"

    # ========================================================
    # RESOURCES
    # ========================================================

    if (
        re.search(r"\bresource\b", q)
        or re.search(r"\bresources\b", q)
    ):
        return "resources"

    # ========================================================
    # APPLICATION
    # ========================================================

    if (
        re.search(r"\bapplication\b", q)
        or re.search(r"\bapply\b", q)
        or "submit an application" in q
    ):
        return "application"

    # ========================================================
    # ADMINISTRATION
    # ========================================================

    administration_phrases = {
        "administer",
        "administers",
        "administered",
        "administering",
        "administration",
        "department",
        "caseworker",
        "manages",
        "manage",
        "manager",
        "agency",
        "agencies",
        "runs",
        "run",
        "handles",
        "handle",
        "responsible",
    }

    if any(
        phrase in question_tokens
        for phrase in administration_phrases
        if " " not in phrase and "-" not in phrase
    ):
        return "administration"

    if (
        "in charge" in q
        or "day-to-day" in q
        or "day to day" in q
    ):
        return "administration"

    # ========================================================
    # GENERAL ELIGIBILITY
    # ========================================================

    eligibility_terms = {
        "eligibility",
        "eligible",
        "requirements",
        "requirement",
        "qualify",
        "qualification",
        "conditions",
    }

    if any(
        term in question_tokens
        for term in eligibility_terms
    ):
        return "eligibility"

    # ========================================================
    # GENERAL POLICY
    # ========================================================

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

    if question_tokens.intersection(policy_terms):
        return "general"

    # ========================================================
    # DEFAULT
    # ========================================================

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

    return question_type != "outside_policy"


# ============================================================
# KEYWORD OVERLAP
# ============================================================

def keyword_overlap(
    question: str,
    text: str
) -> float:
    """
    Calculate overlap between important question terms
    and important terms in the clause.
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
        "administering",
        "administration",
        "department",
        "caseworker",
        "agency",
        "agencies",
        "manage",
        "manages",
        "manager",
        "runs",
        "run",
        "handles",
        "handle",
        "responsible",
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

        # ====================================================
        # LOAD CLAUSES
        # ====================================================

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

        # ====================================================
        # NORMALIZE CLAUSE IDs
        # ====================================================

        for clause in self.clauses:
            if "clause" in clause:
                clause["clause"] = normalize_clause_id(
                    clause["clause"]
                )

        # ====================================================
        # CLAUSE LOOKUP
        # ====================================================

        self.clause_lookup = {
            clause["clause"]: clause
            for clause in self.clauses
            if "clause" in clause
        }

        # ====================================================
        # PARENT SECTION INDEX
        # ====================================================

        self.section_children: Dict[str, List[dict]] = {}

        for clause in self.clauses:

            clause_id = clause["clause"]

            parent = get_parent_section(
                clause_id
            )

            if parent:
                self.section_children.setdefault(
                    parent,
                    []
                ).append(clause)

        # ====================================================
        # BM25
        # ====================================================

        self.tokenized_clauses = [
            tokenize(clause["text"])
            for clause in self.clauses
        ]

        self.bm25 = BM25Okapi(
            self.tokenized_clauses
        )

        # ====================================================
        # SEMANTIC MODEL
        # ====================================================

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

        # ====================================================
        # TEMPORAL POLICY ENGINE
        # ====================================================

        self.temporal_policy = None

        if TemporalPolicy is not None:

            try:
                self.temporal_policy = TemporalPolicy()

                print(
                    "Temporal policy engine loaded."
                )

            except Exception as exc:

                print(
                    "Warning: Temporal policy engine "
                    f"could not be loaded: {exc}"
                )

    # ========================================================
    # TEMPORAL POLICY
    # ========================================================

    def get_policy_version(
        self,
        as_of_date: Optional[str] = None
    ) -> Optional[dict]:

        if self.temporal_policy is None:
            return None

        if as_of_date is None:
            return None

        try:

            if hasattr(
                self.temporal_policy,
                "get_policy_version"
            ):

                version = (
                    self.temporal_policy
                    .get_policy_version(
                        as_of_date
                    )
                )

                if isinstance(version, dict):
                    return version

            active_amendments = []

            if hasattr(
                self.temporal_policy,
                "get_active_amendments"
            ):

                active_amendments = (
                    self.temporal_policy
                    .get_active_amendments(
                        as_of_date
                    )
                )

            if not isinstance(
                active_amendments,
                list
            ):
                active_amendments = []

            latest = None

            if hasattr(
                self.temporal_policy,
                "get_latest_amendment"
            ):

                latest = (
                    self.temporal_policy
                    .get_latest_amendment(
                        as_of_date
                    )
                )

            explanation = None

            if hasattr(
                self.temporal_policy,
                "explain"
            ):

                explanation = (
                    self.temporal_policy
                    .explain(
                        as_of_date
                    )
                )

            return {
                "as_of_date": as_of_date,
                "amendments_active": len(
                    active_amendments
                ),
                "active_amendments": active_amendments,
                "latest_amendment": latest,
                "explanation": explanation,
            }

        except Exception:
            return None

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

        clause_id = normalize_clause_id(
            clause_id
        )

        # ====================================================
        # EXACT CLAUSE REFERENCE
        # ====================================================

        if question_type == "clause_reference":

            references = extract_clause_references(
                question
            )

            normalized_refs = [
                normalize_clause_id(ref)
                for ref in references
            ]

            if clause_id in normalized_refs:
                return 1.00

            return 0.0

        # ====================================================
        # CORRECTIONAL EXCLUSION
        # ====================================================

        if question_type == "correctional_exclusion":

            if clause_id == "§4.1.1":
                return 0.70

            if clause_id.startswith("§4."):
                return 0.20

            return 0.0

        # ====================================================
        # SANCTION EXCLUSION
        # ====================================================

        if question_type == "sanction_exclusion":

            if clause_id == "§4.1.1":
                return 0.55

            if clause_id.startswith("§4."):
                return 0.20

            return 0.0

        # ====================================================
        # GENERAL EXCLUSION
        # ====================================================

        if question_type == "exclusion":

            if clause_id == "§4.1.1":
                return 0.60

            if clause_id.startswith("§4."):
                return 0.20

            if clause_id.startswith("§2."):
                return 0.05

            return 0.0

        # ====================================================
        # AGE 18
        # ====================================================

        if question_type == "age_18":

            if clause_id == "§2.1.2":
                return 0.70

            if clause_id == "§2.1.1":
                return 0.25

            return 0.0

        # ====================================================
        # AGE MINOR
        # ====================================================

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

        # ====================================================
        # GENERIC AGE
        # ====================================================

        if question_type == "age":

            if clause_id == "§2.1.2":
                return 0.45

            if clause_id == "§2.3.1":
                return 0.40

            if clause_id == "§2.3.2":
                return 0.20

            return 0.0

        # ====================================================
        # RESIDENCE
        # ====================================================

        if question_type == "residence":

            if clause_id == "§2.1.2":
                return 0.65

            if clause_id.startswith("§3."):
                return 0.20

            return 0.0

        # ====================================================
        # INCOME
        # ====================================================

        if question_type == "income":

            if (
                "disregard" in question.lower()
                and clause_id == "§6.4.1"
            ):
                return 0.95

            if clause_id == "§2.1.2":
                return 0.55

            if clause_id.startswith("§6."):
                return 0.20

            return 0.0

        # ====================================================
        # RESOURCES
        # ====================================================

        if question_type == "resources":

            if clause_id == "§2.4":
                return 0.80

            if clause_id == "§2.1.2":
                return 0.55

            if clause_id.startswith("§2.4."):
                return 0.35

            return 0.0

        # ====================================================
        # APPLICATION
        # ====================================================

        if question_type == "application":

            if clause_id == "§2.1.2":
                return 0.60

            if clause_id.startswith("§8."):
                return 0.20

            return 0.0

        # ====================================================
        # ADMINISTRATION
        # ========================================================

        if question_type == "administration":

            if clause_id == "§1.1.2":
                return 0.70

            if clause_id.startswith("§1."):
                return 0.20

            return 0.0

        # ====================================================
        # GENERAL ELIGIBILITY
        # ====================================================

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
    # MAKE FORCED RESULT
    # ========================================================

    def _make_forced_result(
        self,
        clause_id: str,
        expanded_from: Optional[str] = None,
        score: float = 1.0
    ):

        clause_id = normalize_clause_id(
            clause_id
        )

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
            result["expanded_from"] = expanded_from

        return result

    # ========================================================
    # BUILD SYNTHETIC SECTION RESULT
    # ========================================================

    def _make_section_result(
        self,
        section_id: str,
        score: float = 1.0
    ):

        section_id = normalize_clause_id(
            section_id
        )

        real_section = self.clause_lookup.get(
            section_id
        )

        if real_section is not None:

            result = real_section.copy()

            result["bm25_score"] = 0.0
            result["semantic_score"] = 0.0
            result["keyword_score"] = 0.0
            result["section_boost"] = score
            result["hybrid_score"] = score

            return result

        children = self.section_children.get(
            section_id,
            []
        )

        if not children:
            return None

        children = sorted(
            children,
            key=lambda item: item["clause"]
        )

        combined_text_parts = []

        for child in children:

            child_id = child["clause"]
            child_text = child.get(
                "text",
                ""
            )

            combined_text_parts.append(
                f"{child_id}: {child_text}"
            )

        combined_text = "\n".join(
            combined_text_parts
        )

        return {
            "clause": section_id,
            "text": combined_text,
            "synthetic_section": True,
            "section_children": [
                child["clause"]
                for child in children
            ],
            "bm25_score": 0.0,
            "semantic_score": 0.0,
            "keyword_score": 0.0,
            "section_boost": score,
            "hybrid_score": score,
        }

    # ========================================================
    # SEARCH
    # ========================================================

    def search(
        self,
        question: str,
        top_k: int = 5,
        as_of_date: Optional[str] = None
    ) -> Dict:

        question = question.strip()

        # ====================================================
        # EMPTY QUESTION
        # ====================================================

        if not question:

            return {
                "answerable": False,
                "reason": "Empty question.",
                "results": []
            }

        # ====================================================
        # POLICY SCOPE GATE
        # ====================================================

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

        # ====================================================
        # TEMPORAL POLICY INFORMATION
        # ====================================================

        policy_version = self.get_policy_version(
            as_of_date
        )

        # ====================================================
        # DIRECT CLAUSE REFERENCE
        # ====================================================

        if question_type == "clause_reference":

            references = extract_clause_references(
                question
            )

            exact_results = []

            for ref in references:

                normalized_ref = normalize_clause_id(
                    ref
                )

                result = self._make_forced_result(
                    normalized_ref,
                    score=1.0
                )

                if result is None:

                    result = self._make_section_result(
                        normalized_ref,
                        score=1.0
                    )

                if result:
                    exact_results.append(result)

            if exact_results:

                response = {
                    "answerable": True,
                    "reason": (
                        "Exact policy clause reference found."
                    ),
                    "results": exact_results[:top_k]
                }

                if policy_version:
                    response["policy_version"] = (
                        policy_version
                    )

                return response

            return {
                "answerable": False,
                "reason": (
                    "The referenced policy clause "
                    "could not be found."
                ),
                "results": []
            }

        # ====================================================
        # BM25
        # ====================================================

        query_tokens = tokenize(
            question
        )

        bm25_scores = self.bm25.get_scores(
            query_tokens
        )

        # ====================================================
        # SEMANTIC SIMILARITY
        # ====================================================

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

        # ====================================================
        # NORMALIZE BM25
        # ====================================================

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

        # ====================================================
        # HYBRID SCORING
        # ====================================================

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

        # ====================================================
        # RANK
        # ====================================================

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

            clause_id = normalize_clause_id(
                result["clause"]
            )

            if clause_id in selected_ids:
                return

            selected.append(result)

            selected_ids.add(
                clause_id
            )

        # ====================================================
        # AGE 18
        # ====================================================

        if question_type == "age_18":

            add_result(
                self._make_forced_result(
                    "§2.1.2",
                    score=1.0
                )
            )

            for result in combined_results:

                if result["clause"] == "§2.1.1":

                    add_result(result)
                    break

        # ====================================================
        # AGE MINOR
        # ====================================================

        elif question_type == "age_minor":

            add_result(
                self._make_forced_result(
                    "§2.1.2",
                    score=1.0
                )
            )

            add_result(
                self._make_forced_result(
                    "§2.3.1",
                    expanded_from="§2.1.2",
                    score=0.95
                )
            )

            add_result(
                self._make_forced_result(
                    "§2.3.2",
                    expanded_from="§2.3.1",
                    score=0.85
                )
            )

        # ====================================================
        # RESIDENCE
        # ====================================================

        elif question_type == "residence":

            add_result(
                self._make_forced_result(
                    "§2.1.2",
                    score=1.0
                )
            )

            for result in combined_results:

                if result["clause"].startswith("§3."):

                    add_result(result)

                    if len(selected) >= top_k:
                        break

        # ====================================================
        # CORRECTIONAL EXCLUSION
        # ====================================================

        elif question_type == "correctional_exclusion":

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

        # ====================================================
        # SANCTION
        # ====================================================

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

        # ====================================================
        # GENERAL EXCLUSION
        # ====================================================

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

        # ====================================================
        # INCOME
        # ====================================================

        elif question_type == "income":

            if "disregard" in question.lower():

                add_result(
                    self._make_forced_result(
                        "§6.4.1",
                        score=1.0
                    )
                )

            add_result(
                self._make_forced_result(
                    "§2.1.2",
                    score=0.90
                )
            )

            for result in combined_results:

                if result["clause"].startswith("§6."):

                    add_result(result)

                    if len(selected) >= top_k:
                        break

        # ====================================================
        # RESOURCES
        # ====================================================

        elif question_type == "resources":

            resource_section = (
                self._make_section_result(
                    "§2.4",
                    score=1.0
                )
            )

            add_result(
                resource_section
            )

            add_result(
                self._make_forced_result(
                    "§2.1.2",
                    expanded_from="§2.4",
                    score=0.90
                )
            )

            for result in combined_results:

                clause_id = result["clause"]

                if clause_id.startswith("§2.4."):

                    add_result(result)

                    if len(selected) >= top_k:
                        break

        # ====================================================
        # APPLICATION
        # ====================================================

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

        # ====================================================
        # ADMINISTRATION
        # ====================================================

        elif question_type == "administration":

            administration_clause = (
                self._make_forced_result(
                    "§1.1.2",
                    score=1.0
                )
            )

            add_result(
                administration_clause
            )

            for result in combined_results:

                if result["clause"].startswith("§1."):

                    add_result(result)

                    if len(selected) >= top_k:
                        break

            # =================================================
            # FINAL SAFETY GUARANTEE
            # =================================================

            administration_id = "§1.1.2"

            if administration_id not in selected_ids:

                fallback = None

                for result in combined_results:

                    if (
                        normalize_clause_id(
                            result["clause"]
                        )
                        == administration_id
                    ):

                        fallback = result
                        break

                if fallback is not None:

                    selected.insert(
                        0,
                        fallback
                    )

                    selected_ids.add(
                        administration_id
                    )

        # ====================================================
        # GENERAL ELIGIBILITY
        # ====================================================

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

        # ====================================================
        # GENERAL POLICY
        # ====================================================

        else:

            for result in combined_results:

                add_result(result)

                if len(selected) >= top_k:
                    break

        # ====================================================
        # ANSWERABILITY
        # ====================================================

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

        # ====================================================
        # IMPORTANT:
        # Intent-aware forced clauses are trusted only when
        # there is at least some retrieval support.
        #
        # This prevents a random question from becoming
        # answerable merely because an intent handler forced
        # a clause.
        # ====================================================

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

        intent_has_support = (
            best["semantic_score"]
            >= SEMANTIC_THRESHOLD
            or best["bm25_score"]
            >= STRONG_BM25_THRESHOLD
            or best["keyword_score"]
            >= STRONG_KEYWORD_THRESHOLD
        )

        if question_type in intent_specific_types:

            answerable = (
                bool(selected)
                and intent_has_support
            )

        else:

            answerable = (
                hybrid_ok
                and
                (
                    semantic_ok
                    or strong_lexical_support
                )
            )

        # ====================================================
        # NOT ANSWERABLE
        # ====================================================

        if not answerable:

            response = {
                "answerable": False,
                "reason": (
                    "The question does not have "
                    "sufficiently strong support in "
                    "the policy manual."
                ),
                "results": combined_results[:top_k]
            }

            if policy_version:

                response["policy_version"] = (
                    policy_version
                )

            return response

        # ====================================================
        # ANSWERABLE
        # ====================================================

        response = {
            "answerable": True,
            "reason": (
                "Relevant policy support found."
            ),
            "results": selected[:top_k]
        }

        if policy_version:

            response["policy_version"] = (
                policy_version
            )

        return response


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