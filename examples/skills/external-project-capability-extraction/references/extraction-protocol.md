# Extraction Protocol

Use read-only inspection by default.

Read first:

- `README*`
- `AGENTS.md`, `CLAUDE.md`, or similar agent instructions
- `docs/**`
- `pyproject.toml`, `package.json`, `go.mod`, `Cargo.toml`
- `.github/workflows/**`
- example configs, command references, and example scripts

Classify findings as:

- `documented`: explicitly stated by project docs or code comments.
- `inferred`: visible from manifests, entrypoints, or tests.
- `unsafe`: requires executing untrusted code, accessing secrets, or broad filesystem/network access.

For each useful capability, capture:

- capability name
- trigger phrases
- workflow
- evidence pointers
- required tools or dependencies
- security boundaries
- confidence level

Do not run `npm install`, `pip install`, `pytest`, build scripts, shell scripts, or project binaries unless the user explicitly approves that execution.

