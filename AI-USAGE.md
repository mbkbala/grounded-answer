# AI Usage

AI assistance was used during development to inspect the repository,
identify disconnected temporal plumbing, propose focused code changes,
and help write tests and documentation.

All policy facts remain sourced from the files in `data/` and `Data pack/`.
The runtime answerer is deterministic: it retrieves policy text, applies
the configured amendment substitutions for the supplied dates, and cites
the resulting clause. No external AI service or API key is required to
run the application.