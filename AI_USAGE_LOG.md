# AI Usage Log

This file records that assistance as it happens. Each
entry names the tool and version, the date, what was generated, and what a
human changed or validated afterward.

This log is maintained continuously.

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

### 2026-09-12 — repository (docs)
- Tool: Claude Sonnet 5 (Anthropic), via Claude Code.
- Scope: Read the full repository and the standing project proposal; author
  `CLAUDE.md` (guidance for future Claude Code sessions in this repo);
  investigate why `pyproject.toml`/`CHANGELOG.md` still showed the `OWNER`
  placeholder locally (found: a "Paths corrected" fix existed on
  `origin/main` but had never been fetched into this local clone); organize
  the untracked working tree (full `detequal/` package beyond the Day-1
  skeleton, `tests/`, `docs/paper/`, `examples/`).
  commits above.
- Human review: contents were reviewed and explicitly
  requested by the project author before being made.
- New code in detequal/ and tests hand written.

### 2026-09-12 — code + tests
- Tool: Claude Sonnet 5 (Anthropic), via Claude Code.
- Scope: Implemented pertinent changes.
  change (per `CLAUDE.md`): wire `stats.is_significant()` and
  `stats.divergence_score()` to the existing `roundoff_floor()` calculation
  instead of forwarding the raw hash-difference signal unconditionally.
- Generated: rewrote `detequal/stats.py` (pairwise, per-dtype relative-error
  grading against full captured tensors, with a majority-of-pairs rule and a
  documented fallback to the prior hash-only behavior when full tensors
  weren't captured); added `tests/test_stats.py` covering round-off-floor
  scaling, sub-floor perturbations, genuine majority divergence, single-run
  outlier suppression, and the no-full-tensor fallback.
- Human review: installed the project's dev environment and ran the full
  suite plus `ruff check`/`ruff format` locally; one design flaw was caught
  and fixed during that verification (comparing each run to the elementwise
  mean let a single outlier drag the mean and falsely flag the *other*,
  consistent runs - switched to pairwise cross-run comparison, which is
  robust to that). All 19 tests plus linting format pass.
