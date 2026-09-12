# Contributing to detequal

Thanks for your interest. detequal is developed in the open, and issues, bug
reports, and pull requests are all welcome.

## Ways to help

- **Report a false positive or false negative.** If detequal flags a layer you
  believe is deterministic, or misses one you know is not, open an issue with a
  minimal reproducing model. These reports directly improve the ground-truth
  validation suite (see `tests/test_ground_truth.py`).
- **Add a repro case.** Minimal, isolated models that exercise a specific
  known-nondeterministic operation are valuable as test fixtures.
- **Improve a cause heuristic.** The causal classification knowledge base
  (`detequal/causes.py`) maps a flagged layer to a probable cause; new
  heuristics with evidence are welcome.
- **Documentation and examples.** Clearer docs and runnable examples lower the
  install-and-run bar for new users.

## Development setup

```bash
git clone https://github.com/RyanDennis916/detequal
cd detequal
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"
pytest
```

Requires Python ≥ 3.10 and PyTorch ≥ 2.7. No GPU is required to develop or run
the test suite: the core is architecture-agnostic, and hardware-specific tests
skip cleanly on machines without the relevant device.

## Before you open a pull request

1. **Run the tests.** `pytest` should pass. If you add behavior, add a test.
2. **Lint.** `ruff check .` and `ruff format .` — CI runs both.
3. **Keep it general-purpose.** detequal is a tool for the whole PyTorch
   community; changes should not narrow it to one architecture or model family.
   The Blackwell/FP4 path is one isolated module (`precision.py`), not the
   center of gravity.
4. **Update the changelog.** Add a line to `CHANGELOG.md` under "Unreleased".
5. **Disclose AI assistance.** If you used an AI tool to help write a
   contribution, note it in the PR and it will be recorded in
   `AI_USAGE_LOG.md`. See that file for the format.

## Testing philosophy

The ground-truth suite is the most important validation artifact in the project.
When you touch the detection or classification logic, check that precision,
recall, and false-positive rate on the ground-truth suite do not regress. A
documented, honest false-positive rate is fine; a silent regression is not.

## Support expectations

This is a solo-maintained research tool. Issues and PRs are read and responded
to, but on a best-effort basis and not to a fixed SLA. Security-relevant reports
are prioritized.

## Code of conduct

Be respectful and constructive. Harassment or abuse is not tolerated in issues,
pull requests, or any project space.
