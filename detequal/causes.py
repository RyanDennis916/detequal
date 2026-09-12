from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any


@dataclass
class Classification:
    cause: str
    fix: str | None
    confidence: str = "low"


@dataclass
class CauseCategory:
    key: str
    description: str
    fix: str | None
    detector: Callable[[Any, list[Any]], bool] | None = None


KNOWN_ATOMIC_OPS: set[str] = set()


CATEGORIES: list[CauseCategory] = [
    CauseCategory(
        key="atomic-reduction",
        description=(
            "Flagged op is in the documented set that uses atomicAdd-based "
            "accumulation (scatter/index_add/bincount, certain backward passes)."
        ),
        fix="torch.use_deterministic_algorithms(True) for this op, or restructure "
        "to avoid the reduction.",
    ),
    CauseCategory(
        key="heuristic-kernel-selection",
        description=(
            "Divergence present only with cudnn.benchmark=True; an automatic "
            "re-run with it False confirms."
        ),
        fix="Set cudnn.benchmark=False, or pin the algorithm explicitly.",
    ),
    CauseCategory(
        key="async-write-race",
        description=(
            "Divergence correlates with multi-stream execution; an automatic "
            "re-run forcing cuda.synchronize() clears it."
        ),
        fix="Serialize the region, or accept and document the nondeterminism.",
    ),
    CauseCategory(
        key="unseeded-randomness",
        description=(
            "Present even under use_deterministic_algorithms(True); traced to a "
            "module with an uncovered random call."
        ),
        fix="Seed the specific generator explicitly.",
    ),
    CauseCategory(
        key="low-precision-rounding",
        description=(
            "Appears only at reduced precision; an automatic FP32 re-run of the "
            "same module clears it."
        ),
        fix="Not necessarily fixable; reported as an inherent property of the precision.",
    ),
]

_BY_KEY = {c.key: c for c in CATEGORIES}


def classify(verdict: Any, snapshots: list[Any]) -> Classification:
    for cat in CATEGORIES:
        if cat.detector is not None and cat.detector(verdict, snapshots):
            return Classification(cause=cat.key, fix=cat.fix, confidence="medium")

    return Classification(cause="unattributed", fix=None, confidence="low")
