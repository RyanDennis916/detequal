# AI Usage Log

JOSS requires a specific, complete disclosure of AI assistance used in a
project's development. This file records that assistance as it happens. Each
entry names the tool and version, the date, what was generated, and what a
human changed or validated afterward.

This log is maintained continuously, not reconstructed at submission time.

A human author reviewed, edited, and validated all AI-assisted output listed
below, and made the core design decisions for the project.

---

## Format

```
### YYYY-MM-DD — <area: code | docs | tests | paper>
- Tool: <name and version>
- Scope: <what the AI was asked to do>
- Generated: <what it produced>
- Human review: <what was checked, changed, kept, or rejected>
```

---

## Entries

### 2026-09-07 — repository scaffold (code + docs)
- Tool: Claude, Opus 4.8 (Anthropic).
- Scope: Generate the Day-1/Week-1 repository scaffold specified in the project
  proposal — directory layout, packaging metadata, license, contributor and
  changelog files, CI workflow, and skeleton implementations of the multi-run
  harness and instrumentation layer with a first unit test.
- Generated: `LICENSE`, `README.md`, `CONTRIBUTING.md`, `CHANGELOG.md`,
  `pyproject.toml`, `.github/workflows/ci.yml`, `.gitignore`, this log, and
  skeleton modules under `detequal/` plus `tests/test_harness.py`.
- Human review: Complete — Each file reviewed and edited (significantly).
