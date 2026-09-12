# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## What this is

`detequal` localizes and explains run-to-run nondeterminism in PyTorch models, layer by layer: it runs a model N
times with every RNG fixed identically, hooks every module boundary, statistically separates genuine divergence
from floating-point round-off, and attributes each flagged layer to a probable cause with a suggested fix. One
isolated module additionally characterizes run-to-run behavior of native FP4 (NVFP4/MXFP4) tensor cores on
Blackwell GPUs, degrading to an explicit skip elsewhere.

**Project status: pre-alpha, developed in the open.** The README's "Implementation status" table is the source of
truth for what actually runs versus what is stubbed — check it before assuming a described behavior is implemented.
Stubbed code paths raise `NotImplementedError` or return a documented conservative default rather than silently
returning a wrong answer; preserve that pattern when extending stubs.

This project is being developed against a specific external constraint: it targets a JOSS (Journal of Open Source
Software) submission, which requires the repository to have been public for 6+ months with commits *distributed*
across that period (an automated check; a late or bursty commit dump risks desk rejection), plus demonstrated
research impact and evidence of iterative development. The full plan, rationale, and JOSS compliance mapping live in
`detequal_Master_Proposal.docx.pdf` at the repo root — read it for context on why work is sequenced the way it is
before proposing a restructure of priorities.

## Commands

```bash
pip install -e ".[dev]"        # editable install with dev deps (pytest, ruff)
pytest                         # run the full test suite
pytest tests/test_harness.py   # run one test file
pytest tests/test_harness.py::test_deterministic_model_has_no_divergent_layers  # single test
pytest --cov=detequal --cov-report=term-missing  # with coverage, as CI does
ruff check .                   # lint
ruff format .                  # format (CI runs format --check, not format)
```

No GPU is required for development. Requires Python >= 3.10 and PyTorch >= 2.7. Hardware-specific tests
(`test_cross_arch.py`) use `monkeypatch` on `detequal.precision._cuda_compute_capability` rather than requiring
real CUDA/Blackwell hardware — follow that pattern for new hardware-gated tests rather than adding real device
skips.

CI (`.github/workflows/ci.yml`) runs on Python 3.10/3.11/3.12: install, import check, `ruff check`, `ruff format
--check`, then `pytest --cov`.

## Architecture

Single data flow: the harness produces per-run traces, and each subsequent stage refines a per-module verdict.
Reading `harness.py` → `instrument.py` → `report.py::compare_traces` → `stats.py` → `causes.py` in that order is the
fastest way to understand a change's blast radius, since each stage consumes the previous stage's output type.

- **`harness.py`** — `diagnose()` runs the model `n_runs` times. Before each run, `set_all_seeds()` fixes `torch`,
  CUDA, NumPy, and Python `random` to the same seed. `force_deterministic=True` is a *comparison* mode
  (`torch.use_deterministic_algorithms`), not the default — the tool exists to characterize the common case where
  determinism is off. `isolate_process=True` (fresh-process-per-run, the intended eventual default) is unimplemented
  and raises `NotImplementedError`.
- **`instrument.py`** — `Capture` registers forward/full-backward hooks on every named submodule via `bind_model()`
  + the `capture()` context manager. Snapshots are keyed by a canonical name (`module_path::op_type::callN`) to
  disambiguate repeated calls to the same module, and default to a SHA1 content hash (`tensor_hash`) rather than
  storing full tensors — pass `capture_full_tensors=True` to keep the actual tensor for graded scoring.
  `DispatchCapture` (dispatch-level, sub-module capture via `TorchDispatchMode`) is a deliberate stub.
- **`report.py`** — `compare_traces()` is the join point: for every canonical name seen across runs, it computes
  `raw_divergent` from hash-set cardinality, delegates the real verdict to `stats.py`, and only calls
  `causes.classify()` when a module is flagged. `DiffReport.summary()`/`to_json()` are the two output renderings; a
  new field on `ModuleVerdict` needs both updated.
- **`stats.py`** — the statistical significance layer, intentionally the most incomplete module: `roundoff_floor()`
  implements the intended per-dtype relative-error round-off calculation (adapted from TTrace's method, credited as
  such) but `is_significant()` does not call it yet — it currently just forwards `raw_divergent`, so *any* hash
  difference is flagged (conservative, over-flags). `divergence_score()` is similarly binary (0.0/1.0) rather than
  continuous. Wiring these together is the highest-leverage correctness change in the codebase.
- **`causes.py`** — `CATEGORIES` defines the cause knowledge base (atomic-reduction, heuristic-kernel-selection,
  async-write-race, unseeded-randomness, low-precision-rounding), each with a `detector: Callable | None`. Every
  detector is currently `None`, so `classify()` always falls through to `cause="unattributed"`. A detector is meant
  to *confirm* a hypothesis by re-running the flagged module under a changed condition (e.g. `cudnn.benchmark=False`,
  a forced FP32 pass) — implementing one means giving it a callable that can re-invoke that module, which the
  current `ModuleVerdict`/snapshot data does not yet carry a reference for.
- **`precision.py`** — `probe_support()` (implemented) maps CUDA compute capability to which of
  FP32/FP16/BF16/FP8(E4M3,E5M2)/NVFP4/MXFP4 rungs are usable (FP8 needs Hopper sm_90+, FP4 needs Blackwell
  sm_100+), always failing closed with a `RungSupport.reason` message rather than crashing. `run_ladder()` (the
  per-rung cast-and-diagnose sweep) is an intentional `NotImplementedError` stub.
- **`cli.py`** — `argparse`-based; the `diagnose` subcommand parses args but model/input file loading is
  unimplemented by design, and prints the equivalent Python API call as guidance (exit code 2). Extend this by
  implementing model/batch loading, not by changing the flag surface, which already mirrors the target CLI shape in
  the README.

### Validation philosophy (drives what "correct" means here)

`tests/test_ground_truth.py` is stated (README, CONTRIBUTING, and the master proposal) to be the most important
validation artifact in the project: it's meant to hold 15-20 minimal repro cases built from PyTorch's own maintained
list of known-nondeterministic ops, paired with matched deterministic negative controls, scored as
precision/recall/false-positive rate. It currently holds an empty `GROUND_TRUTH_CASES` registry with an `xfail`
placeholder — this is the gating suite to populate before trusting changes to `stats.py`/`causes.py`. Per
CONTRIBUTING.md, changes to detection or classification logic should not regress precision/recall/FPR on this suite
once it exists.

### AI usage disclosure

This project logs AI assistance in `AI_USAGE_LOG.md` per JOSS policy, updated as it happens (not reconstructed
later). When you (Claude Code) materially generate or edit code, docs, or paper text in this repo, add or extend an
entry there in the existing format (tool/version, date, scope, what was generated, what a human changed/validated)
rather than leaving disclosure to be reconstructed afterward.
