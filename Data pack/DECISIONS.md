# Engineering Decisions

## Grounded Policy Assistant

This document records important design decisions made while building
the Grounded Policy Assistant.

---

## 1. Hybrid Retrieval

The assistant uses a hybrid retrieval approach combining:

- BM25 lexical retrieval
- Sentence-transformer semantic similarity
- Keyword overlap
- Intent-aware section boosting

The goal is to combine exact terminology matching with semantic
understanding.

---

## 2. Grounded Answers

Answers are generated only from policy clauses retrieved from the
policy manual.

The system does not intentionally generate unsupported policy facts.

Each supported answer includes the relevant policy clause reference.

---

## 3. Refusal Behaviour

If the system does not have sufficient policy support, it returns:

"I don't know."

The assistant also directs the user to the Calder County Department of
Household Services for clarification.

This prevents the system from confidently answering questions that are
not supported by the policy manual.

---

## 4. Policy Scope

Questions outside the Household Support Program policy domain are
rejected before retrieval.

Examples include unrelated questions about:

- weather
- programming
- sports
- movies
- general knowledge

---

## 5. Intent-Aware Retrieval

The retrieval engine identifies important question types such as:

- eligibility
- age
- residence
- income
- resources
- application
- administration
- exclusions

Intent-specific clauses can be prioritized when a controlling policy
clause is known.

---

## 6. Exact Clause References

If a user explicitly references a clause such as §2.1.2, the system
prioritizes that exact clause rather than relying only on semantic
similarity.

---

## 7. Temporal Policy Support

The system includes a temporal policy layer to support policy changes
introduced by amendments.

Amendments are represented separately from the original policy manual.

The temporal layer determines which amendments are effective as of a
requested date.

This allows the system to distinguish between policy versions over
time.

---

## 8. Amendment Ingestion

Amendments are stored as separate source documents and represented in
`data/amendments.json`.

The amendment parser extracts metadata such as:

- amendment identifier
- title
- effective date
- affected clause references
- source text

The system does not invent amendment content.

---

## 9. Source Traceability

Every amendment retains a reference to its original source document.

Example:

`Data pack/Amendment No. 2026-01.md`

This allows the answer system to maintain traceability between an
answer and the underlying policy material.

---

## 10. UI

The assistant uses Streamlit for the demonstration interface.

The UI exposes:

- question input
- grounded answer
- policy support status
- policy evidence
- clause references
- retrieval scores
- retrieval details

The UI is intended to make the system's reasoning and evidence
visible rather than hiding retrieval behaviour.

---

## 11. Current Limitation

The current temporal layer identifies amendments that are effective
for a particular date.

The next implementation step is to apply amendment changes to the
base policy clauses before retrieval.

This will allow historical and current policy questions to use the
correct version of an affected clause.

---

## 12. Design Principle

The primary design principle is:

> Prefer a transparent, grounded answer or an explicit refusal over
> an unsupported confident answer.