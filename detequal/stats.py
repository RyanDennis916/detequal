from __future__ import annotations

from itertools import combinations
from typing import Any

import torch

DTYPE_EPS: dict[str, float] = {
    "torch.float32": 1.19e-7,
    "torch.float16": 9.77e-4,
    "torch.bfloat16": 7.81e-3,
}


def roundoff_floor(dtype: str, magnitude: float) -> float:
    eps = DTYPE_EPS.get(dtype)
    if eps is None:
        raise NotImplementedError(f"round-off floor for dtype {dtype!r} not implemented yet")
    return eps * abs(magnitude)


def _gradable_tensors(snapshots: list[Any]) -> list[torch.Tensor] | None:
    tensors = [s.tensor for s in snapshots]
    if any(t is None for t in tensors):
        return None
    shape = tensors[0].shape
    if any(t.shape != shape for t in tensors):
        return None
    return [t.float() for t in tensors]


def _pairwise_max_diffs(tensors: list[torch.Tensor]) -> list[float]:
 
    return [(a - b).abs().max().item() for a, b in combinations(tensors, 2)]


def _floor_for(snapshots: list[Any], tensors: list[torch.Tensor]) -> float:
    magnitude = max(t.abs().max().item() for t in tensors)
    return roundoff_floor(snapshots[0].dtype, magnitude)


def divergence_score(snapshots: list[Any], *, n_runs: int) -> float:
    hashes = {s.hash for s in snapshots}
    if len(hashes) <= 1 and len(snapshots) == n_runs:
        return 0.0

    tensors = _gradable_tensors(snapshots)
    if tensors is None:
        # no full-tensor data to grade the divergence against; fall back to
        # the coarse hash-only signal (conservative, over-flags)
        return 1.0

    diffs = _pairwise_max_diffs(tensors)
    max_diff = max(diffs, default=0.0)
    if max_diff == 0.0:
        return 0.0

    floor = _floor_for(snapshots, tensors)
    if floor == 0.0:
        return 1.0
    # saturating ratio: 0 at no deviation, 0.5 exactly at the round-off floor,
    # approaching 1 as the deviation dwarfs it
    return max_diff / (max_diff + floor)


def is_significant(snapshots: list[Any], *, n_runs: int, raw_divergent: bool) -> bool:
    if not raw_divergent:
        return False

    tensors = _gradable_tensors(snapshots)
    if tensors is None:
        # no full-tensor data available to separate real divergence from
        # round-off; keep the conservative hash-based signal
        return raw_divergent

    floor = _floor_for(snapshots, tensors)
    diffs = _pairwise_max_diffs(tensors)
    if not diffs:
        return raw_divergent
    exceeding = sum(1 for d in diffs if d > floor)
    # flagged only when a majority of run-pairs disagree beyond the floor, so
    # a single outlier run (a minority of the pairs) never trips it alone
    return exceeding > len(diffs) / 2
