# ============================================================
# GROUNDED POLICY ASSISTANT
# CALDER COUNTY
# "PUBLIC RECORD" VISUAL IDENTITY — STREAMLIT UI
# ============================================================

import sys
from datetime import date
from pathlib import Path

import streamlit as st


# ============================================================
# PROJECT ROOT
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent

if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


# ============================================================
# PROJECT IMPORTS
# ============================================================

from src.retrieval.hybrid_search import HybridSearch
from src.generation.grounded_answer import GroundedAnswerGenerator


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="Calder County | Grounded Policy Assistant",
    page_icon="🏛️",
    layout="wide",
    initial_sidebar_state="collapsed",
)


# ============================================================
# SESSION STATE
# ============================================================

if "question" not in st.session_state:
    st.session_state.question = ""

if "response" not in st.session_state:
    st.session_state.response = None

if "dark_mode" not in st.session_state:
    st.session_state.dark_mode = False


# ============================================================
# THEME TOKENS
#
# "Public Record" direction: cool paper + ink-navy text,
# a brass/gold accent standing in for an official seal, and
# two ink-stamp colors (approval green / caution rust) for
# the grounded / not-grounded verdict.
# ============================================================

if st.session_state.dark_mode:

    background = "#141b24"
    card = "#1c2530"
    input_bg = "#202a36"

    text = "#eef1ea"
    secondary = "#93a3ac"

    border = "#2c3a48"
    border_hover = "#c99a5b"

    brand = "#c99a5b"
    brand_soft = "rgba(201, 154, 91, 0.14)"

    shadow = "0 1px 2px rgba(0, 0, 0, 0.45), 0 10px 28px rgba(0, 0, 0, 0.4)"
    shadow_soft = "0 1px 2px rgba(0, 0, 0, 0.35)"

    success_text = "#8ccb9b"
    success_bg = "rgba(140, 203, 155, 0.12)"
    success_border = "rgba(140, 203, 155, 0.4)"

    warning_text = "#e2986e"
    warning_bg = "rgba(226, 152, 110, 0.12)"
    warning_border = "rgba(226, 152, 110, 0.4)"

    ledger_line = "rgba(238, 241, 234, 0.05)"

else:

    background = "#eef1ea"
    card = "#fbfbf7"
    input_bg = "#ffffff"

    text = "#1c2b39"
    secondary = "#5b6b73"

    border = "#d7ddd0"
    border_hover = "#a9793a"

    brand = "#a9793a"
    brand_soft = "rgba(169, 121, 58, 0.10)"

    shadow = "0 1px 2px rgba(28, 43, 57, 0.05), 0 10px 28px rgba(28, 43, 57, 0.07)"
    shadow_soft = "0 1px 3px rgba(28, 43, 57, 0.06)"

    success_text = "#2f5233"
    success_bg = "rgba(47, 82, 51, 0.08)"
    success_border = "rgba(47, 82, 51, 0.28)"

    warning_text = "#9c4221"
    warning_bg = "rgba(156, 66, 33, 0.08)"
    warning_border = "rgba(156, 66, 33, 0.28)"

    ledger_line = "rgba(28, 43, 57, 0.035)"


# ============================================================
# CUSTOM CSS
# ============================================================

st.markdown(
    f"""
<style>

@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@400;500;600;700&family=IBM+Plex+Sans:wght@400;500;600;700;800&family=Source+Serif+4:opsz,wght@8..60,600;8..60,700&display=swap');

:root {{
    --font-display: 'Source Serif 4', Georgia, serif;
    --font-body: 'IBM Plex Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    --font-mono: 'IBM Plex Mono', 'Courier New', monospace;
}}


/* ------------------------------------------------------------
   BASE
------------------------------------------------------------ */

html, body, [class*="css"] {{
    font-family: var(--font-body);
}}

.stApp {{
    background-color: {background};
    background-image: repeating-linear-gradient(
        {ledger_line},
        {ledger_line} 1px,
        transparent 1px,
        transparent 29px
    );
    color: {text};
}}

.main .block-container {{
    max-width: 860px;
    padding-top: 1.4rem;
    padding-bottom: 4rem;
}}

h1, h2, h3, h4 {{
    color: {text} !important;
    font-family: var(--font-display);
    letter-spacing: -0.01em;
}}

p {{
    line-height: 1.65;
}}

[data-testid="stCaptionContainer"] {{
    color: {secondary} !important;
    line-height: 1.55;
}}

::selection {{
    background-color: {brand_soft};
}}

*:focus-visible {{
    outline: 2px solid {brand};
    outline-offset: 2px;
}}


/* ------------------------------------------------------------
   MOTION (respects reduced-motion)
------------------------------------------------------------ */

@media (prefers-reduced-motion: no-preference) {{

    @keyframes fade-slide-up {{
        from {{ opacity: 0; transform: translateY(10px); }}
        to   {{ opacity: 1; transform: translateY(0); }}
    }}

    @keyframes stamp-press {{
        0%   {{ transform: rotate(-3deg) scale(1.5); opacity: 0; }}
        55%  {{ transform: rotate(-3deg) scale(0.94); opacity: 1; }}
        100% {{ transform: rotate(-3deg) scale(1);    opacity: 1; }}
    }}

    .st-key-answer_card {{
        animation: fade-slide-up 0.45s cubic-bezier(.2,.7,.3,1) both;
    }}

    [class*="st-key-citation_card_"] {{
        animation: fade-slide-up 0.4s cubic-bezier(.2,.7,.3,1) both;
    }}

    .status-badge {{
        animation: stamp-press 0.5s cubic-bezier(.2,1.4,.4,1) both;
    }}
}}


/* ------------------------------------------------------------
   LETTERHEAD / HEADER
------------------------------------------------------------ */

.app-header-title {{
    display: flex;
    align-items: center;
    gap: 10px;
    font-family: var(--font-body);
    font-size: 1.02rem;
    font-weight: 800;
    letter-spacing: 0.03em;
    margin: 0;
    color: {text};
}}

.app-header-subtitle {{
    margin-top: 2px;
    margin-left: 40px;
    font-size: 0.83rem;
    color: {secondary};
}}

.letterhead-rule {{
    height: 3px;
    margin: 14px 0 22px 0;
    border-radius: 2px;
    background: linear-gradient(
        90deg,
        {brand} 0%,
        {brand} 45%,
        transparent 45%,
        transparent 55%,
        {brand} 55%,
        {brand} 100%
    );
    opacity: 0.55;
}}

.st-key-site_header .stButton > button {{
    min-height: 38px;
    font-size: 0.8rem;
    padding: 0 10px;
    border-radius: 999px;
}}


/* ------------------------------------------------------------
   BUTTONS (default)
------------------------------------------------------------ */

.stButton > button {{
    width: 100%;
    min-height: 44px;

    border-radius: 10px;

    border: 1px solid {border};
    background-color: {card};
    color: {text};

    font-family: var(--font-body);
    font-weight: 600;
    font-size: 0.92rem;

    box-shadow: {shadow_soft};

    transition:
        border-color 0.15s ease,
        background-color 0.15s ease,
        box-shadow 0.15s ease,
        transform 0.12s ease;
}}

.stButton > button:hover {{
    border-color: {border_hover};
    color: {brand};
    background-color: {brand_soft};
    transform: translateY(-1px);
    box-shadow: {shadow};
}}

.stButton > button:active {{
    transform: translateY(0px);
}}

.stButton > button p {{
    font-weight: 600;
    font-size: 0.92rem;
}}

button[kind="primary"] {{
    background-color: {brand} !important;
    color: {card} !important;
    border: none !important;
    box-shadow: {shadow_soft};
}}

button[kind="primary"]:hover {{
    opacity: 0.92;
    color: {card} !important;
    background-color: {brand} !important;
    box-shadow: {shadow};
}}

button[kind="primary"] p {{
    font-weight: 700 !important;
}}


/* ------------------------------------------------------------
   FAQ CHIP ROW (scoped to the hero card's chip container)
------------------------------------------------------------ */

.st-key-faq_chips .stButton > button {{
    border-radius: 999px;
    min-height: 40px;
    font-size: 0.8rem;
    font-weight: 600;
    padding: 0 12px;
    background-color: transparent;
    border: 1px solid {border};
}}

.st-key-faq_chips .stButton > button:hover {{
    background-color: {brand_soft};
    border-color: {brand};
}}


/* ------------------------------------------------------------
   INPUT
------------------------------------------------------------ */

div[data-testid="stTextInput"] input {{
    background-color: {input_bg} !important;
    color: {text} !important;

    border: 1px solid {border};
    border-radius: 10px;

    min-height: 48px;

    font-family: var(--font-body);
    font-size: 15px;

    padding-left: 16px;
    padding-right: 16px;

    box-shadow: {shadow_soft};

    transition: border-color 0.15s ease, box-shadow 0.15s ease;
}}

div[data-testid="stTextInput"] input::placeholder {{
    color: {secondary};
}}

div[data-testid="stTextInput"] input:focus {{
    border-color: {border_hover};
    box-shadow: 0 0 0 3px {brand_soft};
    outline: none;
}}


/* ------------------------------------------------------------
   HERO / INTAKE CARD
------------------------------------------------------------ */

.st-key-hero_card {{
    position: relative;
    overflow: hidden;
}}

.hero-watermark {{
    position: absolute;
    top: -30px;
    right: -30px;
    opacity: 0.06;
    pointer-events: none;
}}

.hero-block {{
    text-align: center;
    position: relative;
    margin-bottom: 4px;
}}

.hero-block h2 {{
    font-size: 1.65rem;
    font-weight: 700;
    margin-bottom: 0.4rem;
}}

.hero-block .hero-caption {{
    max-width: 520px;
    margin: 0 auto;
    color: {secondary};
    font-size: 0.92rem;
    line-height: 1.6;
}}


/* ------------------------------------------------------------
   SECTION LABELS (eyebrow with brass tick)
------------------------------------------------------------ */

.section-label {{
    display: flex;
    align-items: center;
    gap: 9px;
    font-family: var(--font-body);
    font-size: 0.85rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    color: {text};
    margin-bottom: 3px;
}}

.section-label::before {{
    content: "";
    width: 3px;
    height: 15px;
    border-radius: 2px;
    background-color: {brand};
    display: inline-block;
    flex-shrink: 0;
}}

.section-sub {{
    font-size: 0.85rem;
    color: {secondary};
    margin-bottom: 0.7rem;
    margin-left: 12px;
}}


/* ------------------------------------------------------------
   CONTAINERS / CARDS
------------------------------------------------------------ */

div[data-testid="stVerticalBlockBorderWrapper"] {{
    background-color: {card};
    border-color: {border};
    border-radius: 12px;
    box-shadow: {shadow_soft};
}}


/* ------------------------------------------------------------
   EXPANDER
------------------------------------------------------------ */

details {{
    background-color: {card};
    border: 1px solid {border};
    border-radius: 10px;
    box-shadow: {shadow_soft};
}}

details summary {{
    color: {text};
    font-weight: 600;
    padding: 0.4rem 0.2rem;
}}


/* ------------------------------------------------------------
   DIVIDER
------------------------------------------------------------ */

hr {{
    border-color: {border};
    margin: 1.4rem 0;
}}


/* ------------------------------------------------------------
   ALERTS
------------------------------------------------------------ */

div[data-testid="stAlert"] {{
    border-radius: 10px;
    box-shadow: {shadow_soft};
}}


/* ------------------------------------------------------------
   ANSWER VERDICT — INK STAMP
------------------------------------------------------------ */

.status-badge {{
    display: inline-flex;
    align-items: center;
    justify-content: center;
    gap: 6px;
    width: 100%;
    padding: 8px 14px;
    border-radius: 6px;
    border: 2px solid currentColor;
    font-family: var(--font-mono);
    font-size: 0.7rem;
    font-weight: 700;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    transform: rotate(-3deg);
    position: relative;
    white-space: nowrap;
}}

.status-badge::after {{
    content: "";
    position: absolute;
    inset: 3px;
    border: 1px solid currentColor;
    border-radius: 4px;
    opacity: 0.45;
    pointer-events: none;
}}

.status-badge.success {{
    background-color: {success_bg};
    color: {success_text};
}}

.status-badge.warning {{
    background-color: {warning_bg};
    color: {warning_text};
}}


/* ------------------------------------------------------------
   CITATION / EXHIBIT TAG
------------------------------------------------------------ */

.citation-tag {{
    display: inline-block;
    padding: 5px 10px;
    border-radius: 6px;
    border: 1px solid {border};
    background-color: {brand_soft};
    color: {brand};
    font-family: var(--font-mono);
    font-weight: 700;
    font-size: 0.82rem;
    text-align: center;
    white-space: nowrap;
}}


/* ------------------------------------------------------------
   FOOTER
------------------------------------------------------------ */

.app-footer {{
    text-align: center;
    color: {secondary};
    font-family: var(--font-mono);
    font-size: 0.72rem;
    letter-spacing: 0.04em;
    text-transform: uppercase;
}}

</style>
""",
    unsafe_allow_html=True,
)


# ============================================================
# SEAL ICONS (inline SVG, purely decorative — no new copy)
# ============================================================

# NOTE: built as single-line strings on purpose (no embedded
# newlines). A multi-line HTML string passed through st.markdown
# can get misread by the Markdown parser as an indented code
# block instead of raw HTML, especially once it carries the
# leading whitespace of nested Python code — flattening to one
# line sidesteps that entirely.

SEAL_SMALL = (
    f'<svg width="26" height="26" viewBox="0 0 40 40" fill="none" xmlns="http://www.w3.org/2000/svg" style="flex-shrink:0;">'
    f'<circle cx="20" cy="20" r="18" stroke="{brand}" stroke-width="1.4"/>'
    f'<circle cx="20" cy="20" r="13.5" stroke="{brand}" stroke-width="1" stroke-dasharray="2 3"/>'
    f'<polygon points="20,9 24.5,16.5 15.5,16.5" fill="{brand}"/>'
    f'<rect x="12.5" y="17.5" width="2.3" height="10" fill="{brand}"/>'
    f'<rect x="16.9" y="17.5" width="2.3" height="10" fill="{brand}"/>'
    f'<rect x="21.2" y="17.5" width="2.3" height="10" fill="{brand}"/>'
    f'<rect x="25.5" y="17.5" width="2.3" height="10" fill="{brand}"/>'
    f'<rect x="11" y="28" width="18" height="2.2" fill="{brand}"/>'
    f'</svg>'
)

SEAL_WATERMARK = (
    f'<svg class="hero-watermark" width="220" height="220" viewBox="0 0 40 40" fill="none" xmlns="http://www.w3.org/2000/svg">'
    f'<circle cx="20" cy="20" r="18" stroke="{text}" stroke-width="0.8"/>'
    f'<circle cx="20" cy="20" r="13.5" stroke="{text}" stroke-width="0.6" stroke-dasharray="1.6 2.4"/>'
    f'<polygon points="20,9 24.5,16.5 15.5,16.5" fill="{text}"/>'
    f'<rect x="12.5" y="17.5" width="2.3" height="10" fill="{text}"/>'
    f'<rect x="16.9" y="17.5" width="2.3" height="10" fill="{text}"/>'
    f'<rect x="21.2" y="17.5" width="2.3" height="10" fill="{text}"/>'
    f'<rect x="25.5" y="17.5" width="2.3" height="10" fill="{text}"/>'
    f'<rect x="11" y="28" width="18" height="2.2" fill="{text}"/>'
    f'</svg>'
)


# ============================================================
# LOAD PIPELINE
# ============================================================

@st.cache_resource
def load_pipeline():

    search_engine = HybridSearch()

    generator = GroundedAnswerGenerator()

    return search_engine, generator


# ============================================================
# INITIALIZE PIPELINE
# ============================================================

try:

    with st.spinner("Loading policy assistant..."):

        search_engine, generator = load_pipeline()

except Exception as exc:

    st.error(
        "Unable to initialize the Grounded Policy Assistant."
    )

    st.code(str(exc))

    st.stop()


# ============================================================
# HEADER (letterhead)
# ============================================================

with st.container(key="site_header"):

    header_left, header_right = st.columns(
        [11, 1.4],
        vertical_alignment="center",
    )

    with header_left:

        st.markdown(
            f'<p class="app-header-title">{SEAL_SMALL} CALDER COUNTY</p>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<p class="app-header-subtitle">Grounded Policy Assistant '
            '· Household Support Program</p>',
            unsafe_allow_html=True,
        )

    with header_right:

        theme_label = (
            "☀️ Light"
            if st.session_state.dark_mode
            else "🌙 Dark"
        )

        if st.button(
            theme_label,
            key="theme_button",
            use_container_width=True,
        ):

            st.session_state.dark_mode = (
                not st.session_state.dark_mode
            )

            st.rerun()


st.markdown('<div class="letterhead-rule"></div>', unsafe_allow_html=True)


# ============================================================
# HERO + INTAKE CARD
# (hero copy, search field, and FAQ shortcuts grouped into a
#  single letterhead "intake" card instead of floating loose
#  on the page)
# ============================================================

with st.container(key="hero_card", border=True):

    hero_left, hero_center, hero_right = st.columns(
        [1, 8, 1]
    )

    with hero_center:

        hero_html = (
            '<div class="hero-block">'
            + SEAL_WATERMARK
            + '<h2>🏛️ How can we help you today?</h2>'
            + '<p class="hero-caption">'
            + 'Ask a question about the Household Support Program '
            + 'and receive a clear answer grounded in the official '
            + 'Calder County policy manual.'
            + '</p>'
            + '</div>'
        )

        st.markdown(hero_html, unsafe_allow_html=True)

    st.write("")

    # --------------------------------------------------------
    # QUESTION INPUT
    # --------------------------------------------------------

    question_col, search_col, clear_col = st.columns(
        [11, 1, 1],
        vertical_alignment="bottom",
    )

    with question_col:

        question = st.text_input(
            "Policy question",
            value=st.session_state.question,
            placeholder="Type your policy question...",
            label_visibility="collapsed",
        )

    with search_col:

        search_clicked = st.button(
            "➤",
            key="search",
            type="primary",
            use_container_width=True,
        )

    with clear_col:

        clear_clicked = st.button(
            "✕",
            key="clear",
            use_container_width=True,
        )

    date_col, event_col = st.columns(2, gap="small")

    with date_col:
        determination_date = st.date_input(
            "Determination date",
            value=None,
            min_value=date(2020, 1, 1),
            key="determination_date",
            help="Used for earnings, thresholds, and sanctions.",
        )

    with event_col:
        event_date = st.date_input(
            "Change/event date",
            value=None,
            min_value=date(2020, 1, 1),
            key="event_date",
            help="Used for reporting deadlines and changes of circumstance.",
        )

    # --------------------------------------------------------
    # FAQ CHIPS
    # --------------------------------------------------------

    st.write("")

    st.markdown(
        '<p class="section-label">Frequently Asked Questions</p>',
        unsafe_allow_html=True,
    )

    st.markdown(
        '<p class="section-sub">Start with a common policy question '
        'or enter your own.</p>',
        unsafe_allow_html=True,
    )

    with st.container(key="faq_chips"):

        faq1, faq2, faq3, faq4 = st.columns(4, gap="small")

        def faq_button(
            column,
            text,
            key,
        ):

            with column:

                clicked = st.button(
                    text,
                    key=key,
                    use_container_width=True,
                )

                return clicked

        if faq_button(
            faq1,
            "👤 Who is eligible for the program?",
            "faq_1",
        ):

            st.session_state.question = (
                "Who is eligible for the program?"
            )

            st.session_state.response = None

            st.rerun()

        if faq_button(
            faq2,
            "💰 What are the resource limits?",
            "faq_2",
        ):

            st.session_state.question = (
                "What are the resource limits?"
            )

            st.session_state.response = None

            st.rerun()

        if faq_button(
            faq3,
            "📊 Does income affect eligibility?",
            "faq_3",
        ):

            st.session_state.question = (
                "Does income affect eligibility?"
            )

            st.session_state.response = None

            st.rerun()

        if faq_button(
            faq4,
            "🚗 Can someone owning a car qualify?",
            "faq_4",
        ):

            st.session_state.question = (
                "Can someone owning a car qualify?"
            )

            st.session_state.response = None

            st.rerun()


# ============================================================
# CLEAR
# ============================================================

if clear_clicked:

    st.session_state.question = ""

    st.session_state.response = None

    st.rerun()


# ============================================================
# PROCESS QUESTION
# ============================================================

if search_clicked:

    question = question.strip()

    st.session_state.question = question

    if not question:

        st.warning(
            "Please enter a policy question."
        )

        st.session_state.response = None

    else:

        # ----------------------------------------------------
        # RETRIEVAL
        # ----------------------------------------------------

        with st.spinner(
            "🔎 Searching the policy manual..."
        ):

            try:

                retrieval_response = (
                    search_engine.search(
                        question,
                        determination_date=(
                            determination_date.isoformat()
                            if determination_date
                            else None
                        ),
                        event_date=(
                            event_date.isoformat()
                            if event_date
                            else None
                        ),
                    )
                )

            except Exception as exc:

                st.error(
                    "An error occurred while searching "
                    "the policy manual."
                )

                st.code(str(exc))

                st.session_state.response = None

                st.stop()


        # ----------------------------------------------------
        # GROUNDED ANSWER
        # ----------------------------------------------------

        with st.spinner(
            "🛡️ Validating policy evidence..."
        ):

            try:

                response = generator.generate(
                    question,
                    retrieval_response,
                )

            except Exception as exc:

                st.error(
                    "An error occurred while generating "
                    "the grounded answer."
                )

                st.code(str(exc))

                st.session_state.response = None

                st.stop()


        st.session_state.response = response


# ============================================================
# RESPONSE
# ============================================================

response = st.session_state.response


if response:

    st.write("")

    st.divider()

    st.write("")


    # ========================================================
    # RESPONSE DATA
    # ========================================================

    answerable = response.get(
        "answerable",
        False,
    )

    answer = response.get(
        "answer",
        "No answer available.",
    )

    citations = response.get(
        "citations",
        [],
    )

    sources = response.get(
        "sources",
        [],
    )

    reason = response.get(
        "reason",
        "",
    )

    temporal = {}
    if sources and isinstance(sources[0], dict):
        temporal = sources[0].get("temporal") or {}

    if temporal:
        status = temporal.get("status", "")
        labels = {
            "original_rule": "Original rule",
            "amended_rule": "Amended rule",
            "transitional": "Transitional treatment",
            "date_required": "Date required",
        }
        st.caption(
            f"Policy version: {labels.get(status, status)}"
            f" | {temporal.get('reason', '')}"
        )


    # ========================================================
    # ANSWER HEADER
    # ========================================================

    answer_header_left, answer_header_right = st.columns(
        [8, 3],
        vertical_alignment="center",
    )


    with answer_header_left:

        st.markdown(
            '<p class="section-label">Answer</p>',
            unsafe_allow_html=True,
        )


    with answer_header_right:

        if answerable:

            st.markdown(
                '<div class="status-badge success">✓ Answer found</div>',
                unsafe_allow_html=True,
            )

        else:

            st.markdown(
                '<div class="status-badge warning">⚠ Not grounded</div>',
                unsafe_allow_html=True,
            )


    # ========================================================
    # ANSWER
    # ========================================================

    with st.container(
        key="answer_card",
        border=True,
    ):

        st.markdown(
            answer.replace("$", r"\$")
        )


    # ========================================================
    # SUPPORTING CLAUSE
    # ========================================================

    if citations:

        st.write("")

        st.markdown(
            '<p class="section-label">Supporting clause</p>',
            unsafe_allow_html=True,
        )

        st.markdown(
            '<p class="section-sub">The following policy clause '
            'supports the answer.</p>',
            unsafe_allow_html=True,
        )


        for index, citation in enumerate(
            citations
        ):

            source_text = ""


            if index < len(sources):

                source = sources[index]


                if isinstance(
                    source,
                    dict,
                ):

                    source_text = source.get(
                        "text",
                        "",
                    )

                else:

                    source_text = str(
                        source
                    )


            with st.container(
                key=f"citation_card_{index}",
                border=True,
            ):

                citation_col, text_col = st.columns(
                    [2, 10],
                    vertical_alignment="center",
                )


                with citation_col:

                    st.markdown(
                        f'<div class="citation-tag">{citation}</div>',
                        unsafe_allow_html=True,
                    )


                with text_col:

                    if source_text:

                        st.markdown(
                            source_text.replace("$", r"\$")
                        )

                    else:

                        st.caption(
                            "No policy text available."
                        )


    else:

        st.info(
            "No supporting policy clause was found."
        )


    # ========================================================
    # GENERATION DETAILS
    # ========================================================

    if reason:

        st.write("")

        with st.expander(
            "🔎 Answer Generation Details"
        ):

            st.write(
                reason
            )


# ============================================================
# FOOTER
# ============================================================

st.write("")

st.divider()

st.markdown(
    '<p class="app-footer">🏛️ Calder County · Grounded Policy '
    'Assistant · Hybrid Retrieval + Evidence Validation</p>',
    unsafe_allow_html=True,
)