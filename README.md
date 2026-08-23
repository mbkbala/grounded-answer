# 🏛️ Grounded Policy Assistant

## Brite Spark 2026 — Hackathon Submission

A **date-aware, evidence-grounded policy assistant** designed to answer policy and claims-related questions using authoritative policy documents while avoiding unsupported or hallucinated answers.

The system retrieves relevant policy evidence, determines whether the question is answerable from the available documents, identifies the policy version applicable to the **date of the claim**, and generates an answer grounded only in the retrieved evidence.

---

## 📌 Problem Statement

Traditional AI assistants may provide plausible answers even when the required information is not available in the provided policy documents.

This creates a major problem in policy and claims processing because:

* An answer may be generated without sufficient evidence.
* The assistant may use an outdated policy version.
* A policy amendment may be incorrectly applied to an earlier claim.
* Conflicting policy documents may not be handled correctly.
* The system may hallucinate information when the question is unanswerable.

The goal of this project is to build a **Grounded Policy Assistant** that answers only when sufficient evidence exists and correctly applies the policy that was effective **on the date relevant to the claim**.

---

# 💡 Solution Overview

Our solution combines:

1. **Document ingestion**
2. **Hybrid information retrieval**
3. **Evidence filtering**
4. **Answerability detection**
5. **Date-aware policy selection**
6. **Grounded answer generation**
7. **Source/evidence presentation**

The assistant does not simply search for similar text and generate an answer.

Instead, it follows a controlled pipeline:

```text
User Question
      │
      ▼
Question Analysis
      │
      ▼
Hybrid Retrieval
      │
      ▼
Relevant Policy Evidence
      │
      ▼
Date / Policy Version Resolution
      │
      ▼
Answerability Check
      │
      ├───────────────┐
      │               │
   Answerable      Not Answerable
      │               │
      ▼               ▼
Grounded Answer   Safe Response
      │
      ▼
Evidence / Sources
```

---

# ✨ Key Features

## 1. Grounded Answers

The assistant generates answers using retrieved policy evidence rather than relying solely on the model's general knowledge.

This reduces unsupported claims and hallucinations.

---

## 2. Answerability Detection

Before generating an answer, the system determines whether the available policy documents contain enough information to answer the question.

If sufficient evidence is unavailable, the assistant does **not** invent an answer.

Instead, it provides a safe response indicating that the available policy evidence is insufficient.

---

## 3. Date-Aware Policy Reasoning

A major requirement of the system is that the answer must be correct for the **date of the claim**, rather than simply being correct according to today's policy.

For example:

```text
Claim Date: February 2026
Policy Amendment: Effective March 1, 2026
```

The March 2026 amendment must **not** be applied to the February 2026 claim.

The system therefore considers:

```text
Claim Date
     +
Policy Effective Date
     +
Policy Version
     ↓
Applicable Policy
```

This prevents newer amendments from being incorrectly applied retroactively.

---

## 4. Policy Version Awareness

Multiple versions of a policy may exist.

The system identifies the policy version that was applicable to the relevant claim date.

This allows the assistant to distinguish between:

* Original policy
* Amended policy
* Updated policy
* Superseded policy

---

## 5. Hybrid Search

The retrieval system combines multiple retrieval signals to improve evidence discovery.

The project uses a hybrid retrieval approach so that both:

* semantic similarity
* keyword / lexical relevance

can contribute to finding the most relevant policy passages.

---

## 6. Evidence-Based Generation

Retrieved evidence is passed to the answer-generation layer as the factual basis for the response.

The generation process is constrained to the available evidence.

This helps prevent the model from introducing unsupported policy rules.

---

## 7. Safe Handling of Unanswerable Questions

If the policy documents do not provide enough information, the assistant should not guess.

Example:

```text
User:
Can I claim an expense that is not mentioned anywhere in the policy?

Assistant:
The available policy documents do not provide sufficient evidence
to determine whether this expense is covered.
```

This behavior is intentional.

---

# 🏗️ System Architecture

```text
                  ┌──────────────────────┐
                  │      User Query      │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │  Query Understanding │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │   Hybrid Retrieval   │
                  │                      │
                  │ Semantic + Keyword   │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Evidence Selection   │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │ Date-Aware Policy    │
                  │ Version Resolution   │
                  └──────────┬───────────┘
                             │
                             ▼
                  ┌──────────────────────┐
                  │    Answerability     │
                  │       Check          │
                  └──────────┬───────────┘
                             │
                    ┌────────┴────────┐
                    │                 │
                    ▼                 ▼
              Answerable        Insufficient
                    │               Evidence
                    ▼                 │
          ┌─────────────────┐         │
          │ Grounded Answer │         │
          └────────┬────────┘         │
                   │                  │
                   └────────┬─────────┘
                            ▼
                  ┌──────────────────────┐
                  │   Streamlit UI       │
                  │ Answer + Evidence    │
                  └──────────────────────┘
```

---

# 📂 Project Structure

```text
grounded-answer/
│
├── app/
│   └── ...
│
├── src/
│   ├── retrieval/
│   │   ├── hybrid_search.py
│   │   └── ...
│   │
│   ├── generation/
│   │   ├── grounded_answer.py
│   │   └── ...
│   │
│   ├── tests/
│   │   └── ...
│   │
│   └── ...
│
├── data/
│   └── ...
│
├── requirements.txt
├── .env.example
├── .gitignore
├── README.md
└── ...
```

> **Note:** Update this section to exactly match the final repository structure before submission.

---

# ⚙️ Requirements

The project requires:

* Python 3.10+
* pip
* Git
* Streamlit
* Required Python dependencies listed in `requirements.txt`
* Access to the configured LLM/API service, if applicable

---

# 🚀 Installation

## 1. Clone the Repository

```bash
git clone <YOUR_REPOSITORY_URL>
cd grounded-answer
```

---

## 2. Create a Virtual Environment

### Windows

```powershell
python -m venv .venv
.venv\Scripts\activate
```

### Linux / macOS

```bash
python3 -m venv .venv
source .venv/bin/activate
```

---

## 3. Install Dependencies

```bash
pip install -r requirements.txt
```

---

# 🔐 Configuration

Create a `.env` file in the project root if the application requires environment variables.

Example:

```text
API_KEY=your_api_key_here
```

Refer to `.env.example` for the required configuration.

### Important

Do **not** commit actual API keys or secrets to GitHub.

The `.env` file should remain local and should be included in `.gitignore`.

---

# ▶️ Running the Application

Activate the virtual environment first.

### Windows

```powershell
.venv\Scripts\activate
```

Then start the Streamlit application:

```powershell
streamlit run <YOUR_STREAMLIT_FILE>.py
```

For example, if the main application is `app.py`:

```powershell
streamlit run app.py
```

The application will open in the browser.

---

# 🧪 Testing

The project includes tests for the retrieval, answer generation, and answerability behavior.

Run the test suite using:

```powershell
python -m pytest
```

If the project uses a custom test runner, use the project's configured test command.

---

# 🧪 Test Scenarios

The system should be tested against the following categories.

### Test 1 — Answerable Question

A question where the policy explicitly contains the required information.

**Expected:**

The assistant provides a grounded answer supported by the policy evidence.

---

### Test 2 — Unanswerable Question

A question asking for information that does not exist in the available policy documents.

**Expected:**

The assistant refuses to invent an answer and clearly states that sufficient evidence is unavailable.

---

### Test 3 — Date-Sensitive Question

A question involving a claim date where multiple policy versions exist.

**Expected:**

The assistant uses the policy version applicable to the claim date.

---

### Test 4 — Pre-Amendment Claim

```text
Claim Date: February 2026
Amendment Effective Date: March 1, 2026
```

**Expected:**

The March 2026 amendment must not be applied to the February claim.

---

### Test 5 — Post-Amendment Claim

```text
Claim Date: March 2026 or later
```

**Expected:**

If the amendment is applicable and the evidence supports the answer, the assistant should use the amended policy.

---

### Test 6 — Missing Evidence

A question where retrieved documents are related to the topic but do not actually establish the requested fact.

**Expected:**

The system should recognize insufficient evidence instead of generating a plausible unsupported answer.

---

# 💬 Sample Questions

The following questions can be used during the hackathon demonstration:

```text
What expenses are covered under the policy?
```

```text
Is this claim eligible under the policy?
```

```text
What is the reimbursement limit?
```

```text
Was this rule applicable to a claim made in February 2026?
```

```text
Does the March 1, 2026 amendment apply to this claim?
```

```text
What does the policy say about an expense that is not covered in the documents?
```

---

# ✅ Expected System Behavior

| Situation                                  | Expected Behavior                        |
| ------------------------------------------ | ---------------------------------------- |
| Evidence clearly supports answer           | Provide grounded answer                  |
| Evidence is insufficient                   | Do not guess                             |
| Question is outside available policy       | State that it cannot be determined       |
| Multiple policy versions exist             | Select version applicable to claim date  |
| Claim predates amendment                   | Do not apply future amendment            |
| Claim falls after amendment effective date | Apply amendment when supported           |
| Retrieved text is only loosely related     | Avoid treating it as sufficient evidence |

---

# 📅 Date-Aware Policy Example

Consider the following policy timeline:

```text
Original Policy
      │
      │
      ▼
Before March 1, 2026
      │
      │
      ├──────── Claim in February 2026
      │         ↓
      │     Original Policy
      │
      ▼
March 1, 2026
      │
      │ Amendment 2026-01
      ▼
After March 1, 2026
      │
      └──────── Claims may use amended policy
```

The key principle is:

> **The applicable policy is determined by the date of the claim, not simply by the latest available policy document.**

---

# 🛡️ Grounding Strategy

The assistant follows a controlled generation strategy:

```text
Retrieve
   ↓
Filter
   ↓
Validate
   ↓
Check Date
   ↓
Check Answerability
   ↓
Generate
```

The system avoids treating retrieval similarity alone as proof.

A retrieved passage must provide meaningful support for the requested claim before it is used as evidence.

---

# 🔎 Answerability Strategy

The system distinguishes between:

### Answerable

The evidence directly supports the requested answer.

```text
Evidence → Supports Claim
```

### Partially Answerable

Some aspects can be established, but the evidence does not support the complete answer.

```text
Evidence → Supports Part of Claim
```

### Unanswerable

The available evidence does not establish the requested information.

```text
Evidence → Does Not Support Claim
```

For unanswerable questions, the assistant should respond conservatively instead of hallucinating.

---

# 🖥️ User Interface

The application provides a Streamlit-based interface where users can:

* Enter policy questions
* View the generated answer
* Inspect supporting evidence
* Understand the reasoning context
* Test date-sensitive policy questions
* Identify when the system cannot answer from the available evidence

### Screenshots

Add screenshots of the final application here:

```text
docs/
├── screenshot-home.png
├── screenshot-answer.png
└── screenshot-date-aware.png
```

Example:

---

# 🧩 Technology Stack

| Component            | Technology                        |
| -------------------- | --------------------------------- |
| User Interface       | Streamlit                         |
| Programming Language | Python                            |
| Retrieval            | Hybrid Search                     |
| Generation           | LLM-based grounded generation     |
| Policy Documents     | Structured policy/document corpus |
| Testing              | Python test framework             |
| Version Control      | Git / GitHub                      |

---

# 🔒 Security Considerations

* API keys are stored outside source code.
* Secrets are not committed to the repository.
* `.env` is excluded using `.gitignore`.
* The assistant does not intentionally fabricate policy information.
* Answers are constrained by retrieved policy evidence.

---

# ⚠️ Limitations

The current system has the following limitations:

1. Answer quality depends on the quality and completeness of the provided policy documents.
2. If the required policy information is absent, the system cannot reliably determine the answer.
3. OCR/document extraction errors may affect retrieval.
4. The system should not be treated as a replacement for official legal or policy review.
5. External policies or documents that are not included in the knowledge base cannot be reliably answered.

---

# 🎯 Hackathon Objective

The primary objective of this project is to demonstrate that an AI policy assistant can be:

* **Grounded**
* **Evidence-aware**
* **Answerability-aware**
* **Date-aware**
* **Resistant to hallucination**
* **Transparent about uncertainty**

Rather than maximizing the number of questions answered, the system prioritizes **correct and defensible answers**.

---

#

---

# 📄 Submission Checklist

Before submitting the repository:

* [ ] README.md updated
* [ ] requirements.txt updated
* [ ] .env.example added
* [ ] .gitignore added
* [ ] No API keys committed
* [ ] Application runs successfully
* [ ] All tests pass
* [ ] Policy documents included
* [ ] Screenshots added
* [ ] Repository pushed to the final branch
* [ ] Final repository URL verified
* [ ] Demo flow tested from a clean environment

---

# 🏁 Quick Start

For the invigilator / evaluator:

```powershell
git clone <YOUR_REPOSITORY_URL>
cd grounded-answer

python -m venv .venv
.venv\Scripts\activate

pip install -r requirements.txt

streamlit run <YOUR_STREAMLIT_FILE>.py
```

Then open the displayed Streamlit URL in a browser.

The application can then be tested using the sample policy questions provided above.

---

## 📌 Core Design Principle

> **If the evidence does not support the answer, the assistant should not guess.**

And for date-sensitive claims:

> **Use the policy that was applicable on the claim date—not simply the latest policy available today.**
