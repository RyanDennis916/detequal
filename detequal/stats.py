from __future__ import annotations

from typing import Any

DTYPE_EPS: dict[str, float] = {
    "torch.float32": 1.19e-7,
    "torch.float16": 9.77e-4,
    "torch.bfloat16": 7.81e-3,
}


def divergence_score(snapshots: list[Any], *, n_runs: int) -> float:
    hashes = {s.hash for s in snapshots}
    if len(hashes) > 1 or len(snapshots) != n_runs:
        return 1.0
    return 0.0


def is_significant(snapshots: list[Any], *, n_runs: int, raw_divergent: bool) -> bool:
    return raw_divergent


def roundoff_floor(dtype: str, magnitude: float) -> float:
    eps = DTYPE_EPS.get(dtype)
    if eps is None:
        raise NotImplementedError(f"round-off floor for dtype {dtype!r} not implemented yet")
    return eps * abs(magnitude)
