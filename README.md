# detequal

Localize and explain run-to-run nondeterminism in PyTorch models, layer by layer.

Running the same model twice under identical seeds can produce different
outputs. The mechanisms behind this are individually documented: atomic
accumulation in reduction kernels, runtime kernel selection in cuDNN/cuBLAS, and
the non-associativity of floating-point arithmetic under reordered parallel
execution. But the tooling around them reports at the granularity of operation
categories, not the user's model. It can tell you that `scatter_add` is
nondeterministic in general; it cannot tell you that layer 14 of your network is
the reason two runs disagreed.

`detequal` fixes this shortcoming. It executes a model `N` times with every
random-number generator fixed identically, captures a snapshot at each module
boundary, separates genuine cross-run divergence from expected floating-point
round-off, and attributes each flagged layer to a probable cause with a
suggested remedy. The diagnostic core is precision- and architecture-agnostic.
A single isolated module additionally characterizes the run-to-run behavior of
native FP4 (NVFP4/MXFP4) tensor cores, degrading to an explicit skip on hardware
that lacks them.

> **Project status: pre-alpha, developed in the open.** The public repository
> exists from the first commit by design. This README describes both the current
> behavior and the target design. The [Implementation status](#implementation-status)
> section states precisely which is which, and [CHANGELOG.md](CHANGELOG.md)
> tracks what has landed.

## Motivation

Three techniques are individually well established: PyTorch's global determinism
controls (`torch.use_deterministic_algorithms`), dispatch- and module-level
hooking for capturing intermediate tensors, and the statistical separation of
numerical signal from round-off noise. None of the existing tools that use them
answers the operational question a practitioner has when a training run will not
reproduce: which layer, and why? `detequal` composes these techniques into a
single diagnostic that localizes the source and proposes a fix, rather than
enumerating categories of risk in the abstract.

## Installation

```bash
pip install -e ".[dev]"     # from source (current)
pip install detequal        # once published to PyPI
```

Requires Python >= 3.10 and PyTorch >= 2.7. No GPU is required: the core runs on
CPU, and hardware-specific code paths degrade gracefully where the relevant
device is absent.

## Usage

### Python API

```python
import detequal

report = detequal.diagnose(model, batch, n_runs=5)
report.summary()  # human-readable per-layer verdict
report.to_json("report.json")  # machine-readable report
```

`diagnose` accepts:

| Argument | Default | Meaning |
| --- | --- | --- |
| `n_runs` | `5` | Number of independent runs to compare (>= 2). |
| `seed` | `0` | Seed applied to every RNG before each run. |
| `eval_mode` | `True` | Place the model in `eval()` so dropout and other intentional randomness are frozen. |
| `force_deterministic` | `False` | Comparison mode: run with `torch.use_deterministic_algorithms(True)` enabled. Off by default, since the tool exists to characterize the common case where it is not set. |
| `compute_grad` | `False` | Also hook the backward pass. |
| `capture_full_tensors` | `False` | Store full tensors instead of content hashes (higher memory; needed for graded scoring). |
| `isolate_process` | `False` | Run each pass in a fresh process to prevent state leakage (planned; see status). |

### Command line

```bash
detequal diagnose model.py --input batch.pt --runs 5
```

Target output shape:

```
Layer 14 (nn.Linear, backward)   DIVERGENT  score=0.031
  cause: atomic-reduction nondeterminism (cuBLAS backward)
  fix:   torch.use_deterministic_algorithms(True)
Layer 22 (custom scatter op)     DIVERGENT  score=0.402
  cause: unattributed, raw trace saved to report.json
19 / 41 modules checked deterministic within floating-point tolerance.
```

## Architecture

`detequal` is organized as six modules with a single data flow: the harness
produces per-run traces, and each subsequent stage refines a per-module verdict.

**Execution harness** (`harness.py`) runs the model `N` times. Before each run
it fixes the `torch`, CUDA, NumPy, and Python `random` generators to the same
seed, so any observed divergence cannot be attributed to differing initial RNG
state. Fresh-process isolation per run is the intended default to eliminate
state leakage; an in-process mode is offered for speed.

**Instrumentation** (`instrument.py`) registers forward and full-backward hooks
on every named submodule and records a snapshot of each output keyed by a
canonical name (module path, op type, and call index, to disambiguate repeated
calls). Snapshots default to a cheap content hash so an `N`-run sweep stays
affordable; full-tensor capture is opt-in. A secondary dispatch-level capture
(via PyTorch's `TorchDispatchMode`/`DebugMode`) is planned for resolving
multiple kernels concealed within one module.

**Significance test** (`stats.py`) distinguishes real divergence from
floating-point round-off. The intended method, adapted from TTrace's approach
and credited as such, derives a relative-error round-off floor per dtype from
machine epsilon and observed tensor magnitude, rather than a fixed threshold, so
one implementation holds from FP32 down to FP4. A module is flagged only when
divergence exceeds that floor across a majority of runs, never on a single
outlier. Both a binary flag and a continuous score are reported.

**Causal classification** (`causes.py`) maps a flagged module to a probable
cause and remedy. The knowledge base covers atomic-reduction nondeterminism,
heuristic kernel selection (`cudnn.benchmark`), asynchronous write races,
unseeded randomness leaks, and low-precision rounding-path divergence. Several
detectors confirm a hypothesis by re-running the module under a changed
condition (for example `cudnn.benchmark=False`, or a forced FP32 pass); layers
matching no detector are reported honestly as `unattributed` rather than forced
into a guess.

**Precision ladder** (`precision.py`) sweeps a model across FP32, FP16, BF16,
FP8 (E4M3/E5M2), and, on Blackwell (sm_100+) only, NVFP4/MXFP4, producing a
per-precision summary of the fraction of layers flagged and the divergence-score
distribution. Every rung fails closed: on hardware lacking a given precision,
that rung reports `unsupported on this device` and the remaining rungs run
normally.

**Reporting** (`report.py`, `cli.py`) assembles per-module verdicts into a
`DiffReport` with human-readable and JSON renderings.

## The FP4 characterization

One capability the tool exposes exists on no prior hardware: native NVFP4/MXFP4
tensor-core execution, introduced with Blackwell. Because that hardware did not
exist before 2025, the run-to-run determinism of its block-scaled low-precision
path has not been characterized. `detequal` treats this as a capability-gated
study, not the reason to adopt the tool: every other module is
architecture-agnostic. On any non-Blackwell device the FP4 rung skips cleanly
and the rest of the diagnostic is unaffected.

## Validation

Correctness is measured against PyTorch's own maintained list of
known-nondeterministic operations, paired with matched deterministic negative
controls, reported as precision, recall, and false-positive rate. The
known-nondeterministic op list is re-pulled from PyTorch at build time rather
than hard-copied, since it changes across releases. The cross-architecture
behavior (graceful degradation of unsupported precision rungs) is verified with
mocked device capabilities and requires no second GPU. See the `tests/` suite
and [docs/paper/paper.md](docs/paper/paper.md).

## Implementation status

The repository is public from day one, so this table is explicit about what runs
today versus what is designed and stubbed. Stubbed modules raise a clear
`NotImplementedError` or return a documented conservative default rather than
silently returning a wrong answer.

| Component | Status |
| --- | --- |
| Multi-run harness with unified RNG seeding | Implemented |
| Module-level forward/backward hooking + canonical naming | Implemented |
| Content-hash divergence detection and reporting | Implemented |
| Precision-ladder capability probing + graceful skip | Implemented |
| Statistical round-off floor (relative-error, per dtype) | Stubbed; currently flags any hash difference (conservative, over-flags) |
| Causal classification detectors | Scaffolded; knowledge base defined, re-run detectors not yet wired, flagged layers report `unattributed` |
| Precision-ladder per-rung casting and sweep | Stubbed; capability probe works, per-rung diagnosis pending |
| Dispatch-level (sub-module) capture | Planned |
| Process-isolated runs | Planned |
| CLI model/input loading | Planned; use the Python API |

## Contributing

Issues and pull requests are welcome. See [CONTRIBUTING.md](CONTRIBUTING.md) for
development setup and testing expectations. The ground-truth precision/recall
suite is the primary correctness artifact; changes to detection or
classification should not regress it.

## AI assistance

This project uses AI assistance in parts of its development, disclosed in
[AI_USAGE_LOG.md](AI_USAGE_LOG.md) and logged as it occurs, per JOSS policy.

## License

MIT, see [LICENSE](LICENSE).
