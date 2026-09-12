# Changelog

All notable changes to detequal are documented here. The format is based on
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and this project aims
to follow [Semantic Versioning](https://semver.org/spec/v2.0.0.html) once it
reaches a first tagged release.

## [Unreleased]

### Added
- Initial public repository scaffold: package layout, packaging metadata,
  MIT license, contributor guide, CI workflow, and AI usage log.
- Multi-run execution harness skeleton (`detequal/harness.py`).
- Instrumentation-layer skeleton (`detequal/instrument.py`).
- First unit test: two seeded runs of a deterministic toy model are
  byte-identical.

### Changed
- `stats.is_significant()` and `stats.divergence_score()` now use
  `roundoff_floor()` when full tensors were captured
  (`capture_full_tensors=True`): divergence is graded against a per-dtype,
  magnitude-scaled relative-error floor via pairwise cross-run comparison,
  and flagged only when a majority of run-pairs exceed it, so a single
  outlier run can no longer trip a false positive. `divergence_score()` is
  now continuous (saturating in `[0, 1)`) instead of binary in this case.
  Both still fall back to the previous conservative hash-only behavior
  (over-flags) when full tensors weren't captured.

## [0.0.1] — 2026-09-07

### Added
- Repository made public. MIT `LICENSE` committed. First commit.

[Unreleased]: https://github.com/RyanDennis916/detequal/compare/v0.0.1...HEAD
[0.0.1]: https://github.com/RyanDennis916/detequal/releases
