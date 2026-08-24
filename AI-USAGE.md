# AI Usage

AI assistance was used during development to inspect the repository,
identify disconnected temporal plumbing, propose focused code changes,
help write tests, improve the user interface, and prepare project
documentation.

## Tools Used

### ChatGPT

ChatGPT was used for:

- Brainstorming the grounded policy-assistant approach.
- Reviewing the problem statement and expected test scenarios.
- Suggesting architecture and demonstration flow ideas.
- Helping draft explanations and documentation.

### Claude

Claude was used for:

- Reviewing implementation ideas and alternative designs.
- Checking the clarity of project explanations.
- Suggesting improvements to the README and decision record.
- Reviewing the presentation and hackathon demo structure.

### GitHub Copilot

GitHub Copilot was used for:

- Inspecting the repository and tracing the retrieval, reasoning, and
	grounded-answer pipeline.
- Identifying issues in date-aware retrieval and clause normalization.
- Implementing focused fixes for vehicle eligibility, amendment dates,
	reporting deadlines, sanction protection, and conflict detection.
- Improving the Streamlit UI, including dark-mode defaults and date-field
	alignment.
- Running focused runtime checks and diagnostics.
- Preparing `README.md` and `decision.md`.

## Human Review and Responsibility

AI-generated suggestions were reviewed, adapted, and verified by the
project author. The project author remains responsible for the final
implementation, policy interpretation, test results, and submitted code.

All policy facts remain sourced from the files in `data/` and `Data pack/`.
The runtime answerer is deterministic: it retrieves policy text, applies
the configured amendment substitutions for the supplied dates, and cites
the resulting clause. No external AI service or API key is required to
run the application.