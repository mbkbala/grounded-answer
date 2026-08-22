import sys
from pathlib import Path

import streamlit as st


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parents[1]

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# PROJECT IMPORTS
# ============================================================

from src.retrieval.hybrid_search import HybridSearch
from src.answer_generator import generate_answer

import streamlit as st

from src.retrieval.hybrid_search import HybridSearch
from src.answer_generator import generate_answer


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Grounded Policy Assistant",
    page_icon="P",
    layout="wide",
    initial_sidebar_state="expanded",
)


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    """
    <style>

    /* Main page */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 3rem;
        max-width: 1200px;
    }

    /* Header */
    .main-title {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 5px;
    }

    .subtitle {
        font-size: 17px;
        color: #666666;
        margin-bottom: 25px;
    }

    /* Answer card */
    .answer-card {
        padding: 22px;
        border-radius: 12px;
        border: 1px solid #dddddd;
        margin-top: 15px;
        margin-bottom: 15px;
    }

    /* Citation */
    .citation-card {
        padding: 14px 18px;
        border-radius: 10px;
        border: 1px solid #dddddd;
        margin-top: 15px;
    }

    /* Evidence */
    .evidence-card {
        padding: 15px;
        border-radius: 10px;
        border: 1px solid #dddddd;
        margin-top: 10px;
    }

    /* Small text */
    .small-text {
        color: #777777;
        font-size: 14px;
    }

    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# LOAD SEARCH ENGINE
# ============================================================

@st.cache_resource
def load_search_engine():
    """
    Load the hybrid search engine only once.

    Streamlit reruns the script whenever the user interacts
    with the page, so caching prevents the semantic model
    from being loaded repeatedly.
    """
    return HybridSearch()


with st.spinner("Loading policy assistant..."):
    searcher = load_search_engine()


# ============================================================
# SIDEBAR
# ============================================================

with st.sidebar:

    st.title("Policy Assistant")

    st.markdown(
        """
        ### About

        This assistant answers questions using the
        **Household Support Program Policy Manual**.

        The system uses:

        - BM25 lexical retrieval
        - Semantic similarity
        - Keyword overlap
        - Policy-section boosting
        - Grounded answer generation
        - Exact policy clause citations
        """
    )

    st.divider()

    st.markdown("### Grounding")

    st.success(
        "Answers are restricted to the policy evidence "
        "retrieved from the manual."
    )

    st.divider()

    st.caption(
        "Grounded Answer RAG System"
    )


# ============================================================
# HEADER
# ============================================================

st.markdown(
    '<div class="main-title">Grounded Policy Assistant</div>',
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="subtitle">
    Ask questions about the Household Support Program policy manual.
    Answers are grounded in policy evidence and include exact
    policy clause citations.
    </div>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# EXAMPLE QUESTIONS
# ============================================================

st.markdown("### Example questions")

example_columns = st.columns(4)

example_questions = [
    "Who administers the program?",
    "Who is eligible for assistance?",
    "Can a 17 year old receive assistance?",
    "Who is excluded from eligibility?",
]

for column, example in zip(
    example_columns,
    example_questions
):
    with column:
        if st.button(
            example,
            use_container_width=True
        ):
            st.session_state["question"] = example


# ============================================================
# QUESTION INPUT
# ============================================================

question = st.text_area(
    "Ask a question",
    value=st.session_state.get(
        "question",
        ""
    ),
    placeholder="Example: Who administers the program?",
    height=100,
)


# ============================================================
# ASK BUTTON
# ============================================================

ask = st.button(
    "Ask Question",
    type="primary",
    use_container_width=True,
)


# ============================================================
# PROCESS QUESTION
# ============================================================

if ask:

    question = question.strip()

    if not question:

        st.warning(
            "Please enter a question before clicking Ask Question."
        )

    else:

        # ----------------------------------------------------
        # RETRIEVAL
        # ----------------------------------------------------

        with st.spinner(
            "Searching the policy manual..."
        ):

            response = searcher.search(
                question
            )

        # ----------------------------------------------------
        # ANSWER GENERATION
        # ----------------------------------------------------

        with st.spinner(
            "Generating grounded answer..."
        ):

            answer = generate_answer(
                question,
                response
            )

        st.divider()

        # ====================================================
        # ANSWER SECTION
        # ====================================================

        st.markdown("## Answer")

        if response.get(
            "answerable",
            False
        ):

            st.success(
                "Supported by the policy manual"
            )

        else:

            st.warning(
                "Policy support is not sufficient"
            )

        st.markdown(
            '<div class="answer-card">',
            unsafe_allow_html=True,
        )

        st.markdown(answer)

        st.markdown(
            "</div>",
            unsafe_allow_html=True,
        )

        # ====================================================
        # RESULTS
        # ====================================================

        results = response.get(
            "results",
            []
        )

        # ====================================================
        # POLICY EVIDENCE
        # ====================================================

        if results:

            st.divider()

            st.markdown("## Policy Evidence")

            st.caption(
                "The following clauses were retrieved by the "
                "hybrid search engine."
            )

            for index, result in enumerate(
                results,
                start=1
            ):

                clause_id = result.get(
                    "clause",
                    "Unknown"
                )

                text = result.get(
                    "text",
                    ""
                )

                hybrid_score = result.get(
                    "hybrid_score",
                    0.0
                )

                bm25_score = result.get(
                    "bm25_score",
                    0.0
                )

                semantic_score = result.get(
                    "semantic_score",
                    0.0
                )

                keyword_score = result.get(
                    "keyword_score",
                    0.0
                )

                section_boost = result.get(
                    "section_boost",
                    0.0
                )

                expanded_from = result.get(
                    "expanded_from"
                )

                # --------------------------------------------
                # Evidence expander
                # --------------------------------------------

                title = (
                    f"{index}. {clause_id} "
                    f"| Hybrid: {hybrid_score:.3f}"
                )

                with st.expander(title):

                    st.markdown(
                        "### Policy text"
                    )

                    st.write(text)

                    st.divider()

                    # ----------------------------------------
                    # Retrieval scores
                    # ----------------------------------------

                    st.markdown(
                        "### Retrieval scores"
                    )

                    col1, col2, col3 = st.columns(3)

                    with col1:

                        st.metric(
                            "BM25",
                            f"{bm25_score:.3f}"
                        )

                    with col2:

                        st.metric(
                            "Semantic",
                            f"{semantic_score:.3f}"
                        )

                    with col3:

                        st.metric(
                            "Hybrid",
                            f"{hybrid_score:.3f}"
                        )

                    col4, col5 = st.columns(2)

                    with col4:

                        st.metric(
                            "Keyword overlap",
                            f"{keyword_score:.3f}"
                        )

                    with col5:

                        st.metric(
                            "Section boost",
                            f"{section_boost:.3f}"
                        )

                    # ----------------------------------------
                    # Expansion information
                    # ----------------------------------------

                    if expanded_from:

                        st.info(
                            f"This clause was included as "
                            f"supporting evidence expanded from "
                            f"{expanded_from}."
                        )

        # ====================================================
        # RETRIEVAL DETAILS
        # ====================================================

        st.divider()

        with st.expander(
            "Retrieval Details"
        ):

            st.write(
                "**Answerable:**",
                response.get(
                    "answerable",
                    False
                )
            )

            st.write(
                "**Reason:**",
                response.get(
                    "reason",
                    "Not available"
                )
            )

            st.write(
                "**Question:**",
                question
            )

            st.write(
                "**Clauses retrieved:**",
                len(results)
            )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "Grounded Policy Assistant | "
    "Hybrid Retrieval + Grounded Answer Generation"
)