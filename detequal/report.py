from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from .harness import HarnessConfig

from . import causes, stats


@dataclass
class ModuleVerdict:
    canonical_name: str
    kind: str
    divergent: bool
    score: float
    cause: str = "not-checked"
    fix: str | None = None
    detail: dict[str, Any] = field(default_factory=dict)


@dataclass
class DiffReport:
    verdicts: list[ModuleVerdict] = field(default_factory=list)
    n_runs: int = 0
    n_modules: int = 0

    @property
    def divergent(self) -> list[ModuleVerdict]:
        return [v for v in self.verdicts if v.divergent]

    def summary(self) -> str:
        lines: list[str] = []
        for v in self.divergent:
            lines.append(f"{v.canonical_name}  DIVERGENT  score={v.score:.3f}")
            if v.cause and v.cause != "not-checked":
                lines.append(f"  cause: {v.cause}")
            if v.fix:
                lines.append(f"  fix:   {v.fix}")
        n_ok = self.n_modules - len(self.divergent)
        lines.append(
            f"{n_ok} / {self.n_modules} modules checked deterministic "
            f"within floating-point tolerance."
        )
        text = "\n".join(lines)
        print(text)
        return text

    def to_json(self, path: str | None = None) -> str:
        import json

        payload = {
            "n_runs": self.n_runs,
            "n_modules": self.n_modules,
            "verdicts": [
                {
                    "canonical_name": v.canonical_name,
                    "kind": v.kind,
                    "divergent": v.divergent,
                    "score": v.score,
                    "cause": v.cause,
                    "fix": v.fix,
                    "detail": v.detail,
                }
                for v in self.verdicts
            ],
        }
        blob = json.dumps(payload, indent=2)
        if path:
            with open(path, "w") as f:
                f.write(blob)
        return blob


def compare_traces(
    traces: dict[int, dict[str, Any]],
    config: HarnessConfig | None = None,
) -> DiffReport:
    run_ids = sorted(traces.keys())
    n_runs = len(run_ids)

    all_names: set[str] = set()
    for rid in run_ids:
        all_names.update(traces[rid].keys())

    verdicts: list[ModuleVerdict] = []
    for name in sorted(all_names):
        snaps = [traces[rid].get(name) for rid in run_ids]
        present = [s for s in snaps if s is not None]
        if not present:
            continue

        hashes = [s.hash for s in present]
        raw_divergent = len(set(hashes)) > 1 or len(present) != n_runs

        score = stats.divergence_score(present, n_runs=n_runs)
        flagged = stats.is_significant(present, n_runs=n_runs, raw_divergent=raw_divergent)

        kind = present[0].kind
        verdict = ModuleVerdict(
            canonical_name=name,
            kind=kind,
            divergent=flagged,
            score=score,
            detail={"distinct_hashes": len(set(hashes)), "runs_present": len(present)},
        )
        if flagged:
            cls = causes.classify(verdict, present)
            verdict.cause = cls.cause
            verdict.fix = cls.fix
        verdicts.append(verdict)

    return DiffReport(verdicts=verdicts, n_runs=n_runs, n_modules=len(verdicts))
