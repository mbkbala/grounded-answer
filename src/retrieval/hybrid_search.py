# ============================================================
# GROUNDED POLICY ASSISTANT
# HYBRID RETRIEVAL ENGINE
# TEMPORAL POLICY-AWARE RETRIEVAL
# ============================================================

import json
import re
import sys
from datetime import date
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
# TEMPORAL POLICY SUPPORT
# ============================================================

try:
    from src.reasoning.policy_version import PolicyVersion
except ImportError:
    PolicyVersion = None


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
        str(text).lower()
    )


# ============================================================
# DATE VALIDATION
# ============================================================

def validate_date(value: Optional[str]) -> Optional[str]:
    """
    Validate and normalize an ISO date.

    Expected:
        YYYY-MM-DD
    """

    if value is None:
        return None

    value = str(value).strip()

    if not value:
        return None

    try:
        parsed = date.fromisoformat(value)
        return parsed.isoformat()

    except ValueError:
        raise ValueError(
            f"Invalid date '{value}'. "
            f"Expected format YYYY-MM-DD."
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

    # Fix common UTF-8 mojibake.
    clause_id = clause_id.replace("Ã‚Â§", "§")
    clause_id = clause_id.replace("Â§", "§")

    # Remove spaces after section symbol.
    clause_id = re.sub(
        r"^§\s+",
        "§",
        clause_id
    )

    # Remove section / clause prefixes.
    clause_id = re.sub(
        r"^(section|clause)\s+",
        "",
        clause_id,
        flags=re.IGNORECASE
    )

    if not clause_id.startswith("§"):
        clause_id = "§" + clause_id

    return clause_id


# ============================================================
# CLAUSE REFERENCE EXTRACTION
# ============================================================

def extract_clause_references(text: str) -> List[str]:
    """
    Extract explicit policy clause references.

    Supported:
        §2.1.2
        §2.4
        2.1.2
        section 2.1.2
        clause 2.4
    """

    if not text:
        return []

    references = []

    # Explicit § reference.
    references.extend(
        re.findall(
            r"§\s*\d+(?:\.\d+){1,2}",
            text
        )
    )

    # Section reference.
    references.extend(
        re.findall(
            r"\bsection\s+(\d+(?:\.\d+){1,2})\b",
            text,
            flags=re.IGNORECASE
        )
    )

    # Clause reference.
    references.extend(
        re.findall(
            r"\bclause\s+(\d+(?:\.\d+){1,2})\b",
            text,
            flags=re.IGNORECASE
        )
    )

    normalized = []

    for reference in references:

        reference = str(reference).strip()

        reference = normalize_clause_id(
            reference
        )

        if reference not in normalized:
            normalized.append(reference)

    return normalized


# ============================================================
# PARENT SECTION
# ============================================================

def get_parent_section(
    clause_id: str
) -> Optional[str]:
    """
    Convert child clause to parent section.

    §2.4.1 -> §2.4
    §2.4.2 -> §2.4
    """

    normalized = normalize_clause_id(
        clause_id
    )

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
    """

    q = question.lower().strip()

    # ========================================================
    # DIRECT CLAUSE REFERENCE
    # ========================================================

    if extract_clause_references(question):
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
        "recipe",
        "cricket",
        "stock",
        "bitcoin",
        "celebrity",
    }

    question_tokens = set(
        tokenize(q)
    )

    if question_tokens.intersection(
        outside_policy_terms
    ):
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
        or re.search(r"\bincarcerated\b", q)
    ):
        return "correctional_exclusion"

    # ========================================================
    # SANCTION
    # ========================================================

    if re.search(
        r"\bsanction(?:s|ed)?\b",
        q
    ):
        return "sanction_exclusion"

    # ========================================================
    # GENERAL EXCLUSION
    # ========================================================

    exclusion_terms = {
        "excluded",
        "exclusion",
        "disqualified",
        "disqualify",
        "ineligible",
        "not eligible",
    }

    if any(
        term in q
        for term in exclusion_terms
    ):
        return "exclusion"

    # ========================================================
    # AGE 18
    # ========================================================

    if (
        re.search(r"\bage\s*18\b", q)
        or re.search(r"\bat\s+18\b", q)
        or re.search(r"\b18\s+years?\s+old\b", q)
        or re.search(r"\b18-year-old\b", q)
        or re.search(r"\b18-year old\b", q)
    ):
        return "age_18"

    # ========================================================
    # AGE 17
    # ========================================================

    if (
        re.search(r"\bage\s*17\b", q)
        or re.search(r"\b17\s+years?\s+old\b", q)
        or re.search(r"\b17-year-old\b", q)
        or re.search(r"\b17-year old\b", q)
    ):
        return "age_minor"

    # ========================================================
    # AGE 16
    # ========================================================

    if (
        re.search(r"\bage\s*16\b", q)
        or re.search(r"\b16\s+years?\s+old\b", q)
        or re.search(r"\b16-year-old\b", q)
        or re.search(r"\b16-year old\b", q)
    ):
        return "age_minor"

    # ========================================================
    # MINOR / UNDER 18
    # ========================================================

    if (
        re.search(r"\bunder\s+18\b", q)
        or re.search(r"\bminor\b", q)
        or re.search(r"\bminor-age\b", q)
        or re.search(r"\bunderage\b", q)
    ):
        return "age_minor"

    # ========================================================
    # GENERIC AGE
    # ========================================================

    if (
        re.search(r"\bage\b", q)
        or re.search(r"\baged\b", q)
        or re.search(r"\bhow old\b", q)
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
            or "resident" in q
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

    if re.search(
        r"\bresources?\b",
        q
    ):
        return "resources"

    # ========================================================
    # APPLICATION
    # ========================================================

    if (
        re.search(r"\bapplication\b", q)
        or re.search(r"\bapply\b", q)
        or "submit an application" in q
        or "how do i apply" in q
    ):
        return "application"

    # ========================================================
    # ADMINISTRATION
    # ========================================================

    administration_terms = {
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
        term in question_tokens
        for term in administration_terms
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

    if question_tokens.intersection(
        policy_terms
    ):
        return "general"

    return "outside_policy"


# ============================================================
# POLICY SCOPE
# ============================================================

def is_policy_question(
    question: str
) -> bool:
    """
    Determine whether question belongs to
    Household Support Program policy domain.
    """

    return (
        get_question_type(question)
        != "outside_policy"
    )


# ============================================================
# KEYWORD OVERLAP
# ============================================================

def keyword_overlap(
    question: str,
    text: str
) -> float:
    """
    Calculate overlap of policy-relevant terms.
    """

    question_tokens = set(
        tokenize(question)
    )

    text_tokens = set(
        tokenize(text)
    )

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
        question_tokens
        & important_words
    )

    text_important = (
        text_tokens
        & important_words
    )

    if not question_important:
        return 0.0

    overlap = (
        question_important
        & text_important
    )

    return (
        len(overlap)
        / len(question_important)
    )


# ============================================================
# HYBRID SEARCH ENGINE
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

        if not isinstance(
            self.clauses,
            list
        ):
            raise ValueError(
                "clauses.json must contain a list of clauses."
            )

        if not self.clauses:
            raise ValueError(
                "No policy clauses were loaded."
            )

        # ====================================================
        # NORMALIZE CLAUSES
        # ====================================================

        normalized_clauses = []

        for clause in self.clauses:

            if not isinstance(
                clause,
                dict
            ):
                continue

            if "clause" not in clause:
                continue

            clause_copy = clause.copy()

            clause_copy["clause"] = (
                normalize_clause_id(
                    clause_copy["clause"]
                )
            )

            clause_copy.setdefault(
                "text",
                ""
            )

            normalized_clauses.append(
                clause_copy
            )

        self.clauses = normalized_clauses

        if not self.clauses:
            raise ValueError(
                "No valid policy clauses were found."
            )

        # ====================================================
        # CLAUSE LOOKUP
        # ====================================================

        self.clause_lookup = {
            clause["clause"]: clause
            for clause in self.clauses
        }

        # ====================================================
        # PARENT SECTION INDEX
        # ====================================================

        self.section_children: Dict[
            str,
            List[dict]
        ] = {}

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
        # BM25 INDEX
        # ====================================================

        self.tokenized_clauses = [
            tokenize(
                clause.get(
                    "text",
                    ""
                )
            )
            for clause in self.clauses
        ]

        self.bm25 = BM25Okapi(
            self.tokenized_clauses
        )

        # ====================================================
        # SEMANTIC MODEL
        # ====================================================

        print(
            "Loading semantic model..."
        )

        self.model = SentenceTransformer(
            MODEL_NAME
        )

        texts = [
            clause.get(
                "text",
                ""
            )
            for clause in self.clauses
        ]

        print(
            "Creating semantic index..."
        )

        embeddings = self.model.encode(
            texts,
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        dimension = embeddings.shape[1]

        self.index = faiss.IndexFlatIP(
            dimension
        )

        self.index.add(
            embeddings
        )

        print(
            f"Loaded {len(self.clauses)} "
            f"policy clauses."
        )

        # ====================================================
        # POLICY VERSION ENGINE
        # ====================================================

        self.policy_version = None

        if PolicyVersion is not None:

            try:

                self.policy_version = (
                    PolicyVersion()
                )

                print(
                    "Policy version engine loaded."
                )

            except Exception as exc:

                print(
                    "Warning: Policy version engine "
                    f"could not be loaded: {exc}"
                )

        # ====================================================
        # LEGACY TEMPORAL ENGINE
        # ====================================================

        self.temporal_policy = None

        if self.policy_version is None:

            if TemporalPolicy is not None:

                try:

                    self.temporal_policy = (
                        TemporalPolicy()
                    )

                    print(
                        "Legacy temporal policy engine "
                        "loaded."
                    )

                except Exception as exc:

                    print(
                        "Warning: Temporal policy engine "
                        f"could not be loaded: {exc}"
                    )

    # ========================================================
    # POLICY VERSION
    # ========================================================

    def get_policy_version(
        self,
        as_of_date: Optional[str] = None
    ) -> Optional[dict]:
        """
        Return policy version applicable to a date.
        """

        if not as_of_date:
            return None

        as_of_date = validate_date(
            as_of_date
        )

        if self.policy_version is not None:

            try:

                if hasattr(
                    self.policy_version,
                    "get_policy_version"
                ):

                    result = (
                        self.policy_version
                        .get_policy_version(
                            as_of_date
                        )
                    )

                    if isinstance(
                        result,
                        dict
                    ):
                        return result

            except Exception:
                pass

        if self.temporal_policy is not None:

            try:

                if hasattr(
                    self.temporal_policy,
                    "get_policy_version"
                ):

                    result = (
                        self.temporal_policy
                        .get_policy_version(
                            as_of_date
                        )
                    )

                    if isinstance(
                        result,
                        dict
                    ):
                        return result

            except Exception:
                pass

        return None

    # ========================================================
    # CLAUSE TEMPORAL STATUS
    # ========================================================

    def get_clause_temporal_status(
        self,
        clause_id: str,
        determination_date: Optional[str] = None,
        event_date: Optional[str] = None,
        rule_type: Optional[str] = None
    ) -> Optional[dict]:
        """
        Determine temporal applicability of a clause.
        """

        clause_id = normalize_clause_id(
            clause_id
        )

        determination_date = validate_date(
            determination_date
        )

        event_date = validate_date(
            event_date
        )

        # ====================================================
        # NEW ENGINE
        # ====================================================

        if self.policy_version is not None:

            try:

                if hasattr(
                    self.policy_version,
                    "determine_clause_version"
                ):

                    result = (
                        self.policy_version
                        .determine_clause_version(
                            clause_id=clause_id,
                            determination_date=(
                                determination_date
                            ),
                            event_date=event_date,
                            rule_type=rule_type
                        )
                    )

                    if isinstance(
                        result,
                        dict
                    ):
                        return result

            except TypeError:

                try:

                    result = (
                        self.policy_version
                        .determine_clause_version(
                            clause_id,
                            determination_date,
                            event_date,
                            rule_type
                        )
                    )

                    if isinstance(
                        result,
                        dict
                    ):
                        return result

                except Exception:
                    pass

            except Exception:
                pass

            # Alternative API.
            try:

                if hasattr(
                    self.policy_version,
                    "get_clause_version"
                ):

                    result = (
                        self.policy_version
                        .get_clause_version(
                            clause_id=clause_id,
                            determination_date=(
                                determination_date
                            ),
                            event_date=event_date,
                            rule_type=rule_type
                        )
                    )

                    if isinstance(
                        result,
                        dict
                    ):
                        return result

            except Exception:
                pass

        # ====================================================
        # LEGACY ENGINE
        # ====================================================

        if self.temporal_policy is not None:

            try:

                if hasattr(
                    self.temporal_policy,
                    "determine_clause_version"
                ):

                    result = (
                        self.temporal_policy
                        .determine_clause_version(
                            clause_id=clause_id,
                            determination_date=(
                                determination_date
                            ),
                            event_date=event_date,
                            rule_type=rule_type
                        )
                    )

                    if isinstance(
                        result,
                        dict
                    ):
                        return result

            except Exception:
                pass

        return None

    # ========================================================
    # APPLY TEMPORAL METADATA
    # ========================================================

    def _apply_temporal_metadata(
        self,
        result: dict,
        determination_date: Optional[str] = None,
        event_date: Optional[str] = None,
        rule_type: Optional[str] = None
    ) -> dict:

        if not result:
            return result

        result = result.copy()

        clause_id = normalize_clause_id(
            result.get(
                "clause",
                ""
            )
        )

        temporal_status = (
            self.get_clause_temporal_status(
                clause_id=clause_id,
                determination_date=(
                    determination_date
                ),
                event_date=event_date,
                rule_type=rule_type
            )
        )

        if temporal_status:

            result["temporal_status"] = (
                temporal_status
            )

            if "status" in temporal_status:

                result["policy_rule_status"] = (
                    temporal_status["status"]
                )

            if "amendment_applies" in temporal_status:

                result["amendment_applies"] = (
                    temporal_status[
                        "amendment_applies"
                    ]
                )

            if "applicable_date" in temporal_status:

                result["applicable_date"] = (
                    temporal_status[
                        "applicable_date"
                    ]
                )

            if "amendment_effective_date" in temporal_status:

                result["amendment_effective_date"] = (
                    temporal_status[
                        "amendment_effective_date"
                    ]
                )

            if "reason" in temporal_status:

                result["temporal_reason"] = (
                    temporal_status["reason"]
                )

            result["policy_version"] = (
                "amended"
                if temporal_status.get(
                    "amendment_applies",
                    False
                )
                else "original"
            )

        return result

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
        # DIRECT CLAUSE
        # ====================================================

        if question_type == "clause_reference":

            references = (
                extract_clause_references(
                    question
                )
            )

            if clause_id in references:
                return 1.00

            return 0.0

        # ====================================================
        # CORRECTIONAL
        # ====================================================

        if question_type == "correctional_exclusion":

            if clause_id == "§4.1.1":
                return 0.70

            if clause_id.startswith("§4."):
                return 0.20

            return 0.0

        # ====================================================
        # SANCTION
        # ====================================================

        if question_type == "sanction_exclusion":

            if clause_id == "§4.1.1":
                return 0.55

            if clause_id.startswith("§4."):
                return 0.20

            return 0.0

        # ====================================================
        # EXCLUSION
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
        # ====================================================

        if question_type == "administration":

            if clause_id == "§1.1.2":
                return 0.70

            if clause_id.startswith("§1."):
                return 0.20

            return 0.0

        # ====================================================
        # ELIGIBILITY
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
    # FORCED RESULT
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

            result["expanded_from"] = (
                expanded_from
            )

        return result

    # ========================================================
    # SECTION RESULT
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

            combined_text_parts.append(
                f'{child["clause"]}: '
                f'{child.get("text", "")}'
            )

        return {
            "clause": section_id,
            "text": "\n".join(
                combined_text_parts
            ),
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
    # BUILD COMMON RESPONSE
    # ========================================================

    @staticmethod
    def _add_temporal_request_metadata(
        response: dict,
        determination_date: Optional[str],
        event_date: Optional[str],
        rule_type: Optional[str],
        policy_version: Optional[dict]
    ) -> dict:

        if determination_date:
            response[
                "determination_date"
            ] = determination_date

        if event_date:
            response[
                "event_date"
            ] = event_date

        if rule_type:
            response[
                "rule_type"
            ] = rule_type

        if policy_version:
            response[
                "policy_version"
            ] = policy_version

        return response

    # ========================================================
    # SEARCH
    # ========================================================

    def search(
        self,
        question: str,
        top_k: int = 5,
        as_of_date: Optional[str] = None,
        event_date: Optional[str] = None,
        determination_date: Optional[str] = None,
        rule_type: Optional[str] = None
    ) -> Dict:

        question = str(
            question or ""
        ).strip()

        # ====================================================
        # TOP K
        # ====================================================

        try:
            top_k = int(top_k)
        except (
            TypeError,
            ValueError
        ):
            top_k = 5

        top_k = max(
            1,
            top_k
        )

        # ====================================================
        # DATES
        # ====================================================

        if determination_date is None:
            determination_date = as_of_date

        determination_date = validate_date(
            determination_date
        )

        event_date = validate_date(
            event_date
        )

        # ====================================================
        # RULE TYPE
        # ====================================================

        if rule_type is not None:

            rule_type = str(
                rule_type
            ).strip().lower()

            allowed_rule_types = {
                "event_date",
                "determination_date"
            }

            if rule_type not in allowed_rule_types:

                raise ValueError(
                    "rule_type must be either "
                    "'event_date' or "
                    "'determination_date'."
                )

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

        if not is_policy_question(
            question
        ):

            return {
                "answerable": False,
                "reason": (
                    "The question is outside the scope "
                    "of the policy manual."
                ),
                "question_type": "outside_policy",
                "results": []
            }

        question_type = get_question_type(
            question
        )

        # ====================================================
        # POLICY VERSION
        # ====================================================

        policy_version = (
            self.get_policy_version(
                as_of_date=(
                    as_of_date
                    or determination_date
                )
            )
        )

        # ====================================================
        # DIRECT CLAUSE REFERENCE
        # ====================================================

        if question_type == "clause_reference":

            references = (
                extract_clause_references(
                    question
                )
            )

            exact_results = []

            for ref in references:

                result = (
                    self._make_forced_result(
                        ref,
                        score=1.0
                    )
                )

                if result is None:

                    result = (
                        self._make_section_result(
                            ref,
                            score=1.0
                        )
                    )

                if result:

                    result = (
                        self._apply_temporal_metadata(
                            result,
                            determination_date=(
                                determination_date
                            ),
                            event_date=event_date,
                            rule_type=rule_type
                        )
                    )

                    exact_results.append(
                        result
                    )

            if exact_results:

                response = {
                    "answerable": True,
                    "reason": (
                        "Exact policy clause reference found."
                    ),
                    "question_type": question_type,
                    "results": exact_results[:top_k]
                }

                return self._add_temporal_request_metadata(
                    response,
                    determination_date,
                    event_date,
                    rule_type,
                    policy_version
                )

            return {
                "answerable": False,
                "reason": (
                    "The referenced policy clause "
                    "could not be found."
                ),
                "question_type": question_type,
                "results": []
            }

        # ====================================================
        # BM25
        # ====================================================

        query_tokens = tokenize(
            question
        )

        if query_tokens:

            bm25_scores = (
                self.bm25.get_scores(
                    query_tokens
                )
            )

        else:

            bm25_scores = [
                0.0
                for _ in self.clauses
            ]

        # ====================================================
        # SEMANTIC
        # ====================================================

        query_embedding = self.model.encode(
            [question],
            convert_to_numpy=True,
            normalize_embeddings=True
        )

        semantic_scores, _ = (
            self.index.search(
                query_embedding,
                len(self.clauses)
            )
        )

        semantic_scores = semantic_scores[0]

        # ====================================================
        # NORMALIZE BM25
        # ====================================================

        max_bm25 = (
            max(bm25_scores)
            if len(bm25_scores)
            else 0.0
        )

        if max_bm25 > 0:

            normalized_bm25 = [
                max(
                    0.0,
                    float(score)
                ) / max_bm25
                for score in bm25_scores
            ]

        else:

            normalized_bm25 = [
                0.0
                for _ in bm25_scores
            ]

        # ====================================================
        # HYBRID SCORE
        # ====================================================

        combined_results = []

        for i, clause in enumerate(
            self.clauses
        ):

            keyword_score = keyword_overlap(
                question,
                clause.get(
                    "text",
                    ""
                )
            )

            section_boost = (
                self.get_section_boost(
                    question,
                    clause["clause"]
                )
            )

            semantic_score = max(
                0.0,
                float(
                    semantic_scores[i]
                )
            )

            hybrid_score = (
                0.25
                * normalized_bm25[i]
                + 0.60
                * semantic_score
                + 0.15
                * keyword_score
                + section_boost
            )

            result = clause.copy()

            result["bm25_score"] = float(
                bm25_scores[i]
            )

            result["semantic_score"] = float(
                semantic_score
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
            key=lambda item: item[
                "hybrid_score"
            ],
            reverse=True
        )

        if not combined_results:

            return {
                "answerable": False,
                "reason": "No policy clauses found.",
                "question_type": question_type,
                "results": []
            }

        # ====================================================
        # INTENT-AWARE SELECTION
        # ====================================================

        selected = []

        selected_ids: Set[str] = set()

        def add_result(result):

            if result is None:
                return

            clause_id = normalize_clause_id(
                result.get(
                    "clause",
                    ""
                )
            )

            if not clause_id:
                return

            if clause_id in selected_ids:
                return

            selected.append(
                result
            )

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
        # MINOR
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

                if result["clause"].startswith(
                    "§3."
                ):

                    add_result(result)

                    if len(selected) >= top_k:
                        break

        # ====================================================
        # CORRECTIONAL
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

                if result["clause"].startswith(
                    "§4."
                ):

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

                if result["clause"].startswith(
                    "§6."
                ):

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

                if result["clause"].startswith(
                    "§2.4."
                ):

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

                if result["clause"].startswith(
                    "§8."
                ):

                    add_result(result)

                    if len(selected) >= top_k:
                        break

        # ====================================================
        # ADMINISTRATION
        # ====================================================

        elif question_type == "administration":

            add_result(
                self._make_forced_result(
                    "§1.1.2",
                    score=1.0
                )
            )

            for result in combined_results:

                if result["clause"].startswith(
                    "§1."
                ):

                    add_result(result)

                    if len(selected) >= top_k:
                        break

        # ====================================================
        # ELIGIBILITY
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
        # GENERAL
        # ====================================================

        else:

            for result in combined_results:

                add_result(result)

                if len(selected) >= top_k:
                    break

        # ====================================================
        # TEMPORAL METADATA ON SELECTED
        # ====================================================

        temporal_selected = []

        for result in selected:

            result = (
                self._apply_temporal_metadata(
                    result,
                    determination_date=(
                        determination_date
                    ),
                    event_date=event_date,
                    rule_type=rule_type
                )
            )

            temporal_selected.append(
                result
            )

        selected = temporal_selected

        # ====================================================
        # TEMPORAL METADATA ON RANKED RESULTS
        # ====================================================

        temporal_combined = []

        for result in combined_results:

            result = (
                self._apply_temporal_metadata(
                    result,
                    determination_date=(
                        determination_date
                    ),
                    event_date=event_date,
                    rule_type=rule_type
                )
            )

            temporal_combined.append(
                result
            )

        combined_results = temporal_combined

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
            semantic_ok
            or
            best["bm25_score"]
            >= STRONG_BM25_THRESHOLD
            or
            best["keyword_score"]
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
                and (
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
                "question_type": question_type,
                "results": combined_results[:top_k]
            }

            return self._add_temporal_request_metadata(
                response,
                determination_date,
                event_date,
                rule_type,
                policy_version
            )

        # ====================================================
        # ANSWERABLE
        # ====================================================

        response = {
            "answerable": True,
            "reason": (
                "Relevant policy support found."
            ),
            "question_type": question_type,
            "results": selected[:top_k]
        }

        return self._add_temporal_request_metadata(
            response,
            determination_date,
            event_date,
            rule_type,
            policy_version
        )


# ============================================================
# STANDALONE TEST
# ============================================================

if __name__ == "__main__":

    try:

        searcher = HybridSearch()

        print(
            "\n============================================"
        )

        print(
            "Grounded Policy Assistant"
        )

        print(
            "Temporal Hybrid Retrieval Engine"
        )

        print(
            "============================================"
        )

        print(
            "Type 'exit' or 'quit' to stop."
        )

        while True:

            question = input(
                "\nEnter your question: "
            ).strip()

            if question.lower() in {
                "exit",
                "quit"
            }:
                break

            try:

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

            except Exception as exc:

                print(
                    json.dumps(
                        {
                            "answerable": False,
                            "reason": (
                                f"Search error: {exc}"
                            ),
                            "results": []
                        },
                        indent=2,
                        ensure_ascii=False
                    )
                )

    except Exception as exc:

        print(
            json.dumps(
                {
                    "answerable": False,
                    "reason": (
                        f"Initialization error: {exc}"
                    ),
                    "results": []
                },
                indent=2,
                ensure_ascii=False
            )
        )