from __future__ import annotations

import torch

from detequal import stats
from detequal.instrument import Snapshot, tensor_hash


def _snapshot(tensor: torch.Tensor, name: str = "layer::forward::call0") -> Snapshot:
    return Snapshot(
        canonical_name=name,
        kind="forward",
        dtype=str(tensor.dtype),
        shape=tuple(tensor.shape),
        hash=tensor_hash(tensor),
        tensor=tensor,
    )


def test_roundoff_floor_scales_with_epsilon_and_magnitude():
    fp32 = stats.roundoff_floor("torch.float32", 100.0)
    fp16 = stats.roundoff_floor("torch.float16", 100.0)
    assert fp16 > fp32
    assert stats.roundoff_floor("torch.float32", 0.0) == 0.0


def test_tiny_perturbation_within_floor_is_not_significant():
    base = torch.ones(8) * 100.0
    snapshots = [_snapshot(base)] + [_snapshot(base + 1e-6 * torch.randn(8)) for _ in range(4)]
    assert stats.is_significant(snapshots, n_runs=5, raw_divergent=True) is False
    assert stats.divergence_score(snapshots, n_runs=5) < 0.5


def test_large_perturbation_across_majority_is_significant():
    base = torch.ones(8) * 100.0
    # a genuine split into two comparably sized clusters: most cross-run
    # pairs disagree well beyond the round-off floor
    snapshots = [_snapshot(base) for _ in range(2)] + [_snapshot(base + 5.0) for _ in range(3)]
    assert stats.is_significant(snapshots, n_runs=5, raw_divergent=True) is True
    assert stats.divergence_score(snapshots, n_runs=5) > 0.5


def test_single_outlier_among_many_runs_is_not_flagged():
    base = torch.ones(8) * 100.0
    snapshots = [_snapshot(base) for _ in range(4)] + [_snapshot(base + 5.0)]
    assert stats.is_significant(snapshots, n_runs=5, raw_divergent=True) is False


def test_identical_hashes_short_circuit_to_not_divergent():
    base = torch.ones(8)
    snapshots = [_snapshot(base) for _ in range(3)]
    assert stats.is_significant(snapshots, n_runs=3, raw_divergent=False) is False
    assert stats.divergence_score(snapshots, n_runs=3) == 0.0


def test_missing_full_tensors_falls_back_to_raw_divergent():
    snap_a = Snapshot(
        canonical_name="layer::forward::call0",
        kind="forward",
        dtype="torch.float32",
        shape=(8,),
        hash="deadbeef",
        tensor=None,
    )
    snap_b = Snapshot(
        canonical_name="layer::forward::call0",
        kind="forward",
        dtype="torch.float32",
        shape=(8,),
        hash="c0ffee",
        tensor=None,
    )
    snapshots = [snap_a, snap_b]
    assert stats.is_significant(snapshots, n_runs=2, raw_divergent=True) is True
    assert stats.divergence_score(snapshots, n_runs=2) == 1.0
