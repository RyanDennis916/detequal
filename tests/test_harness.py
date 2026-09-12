from __future__ import annotations

import torch
import torch.nn as nn

import detequal


class TinyDeterministic(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(16, 32),
            nn.ReLU(),
            nn.Linear(32, 8),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def _fixed_batch() -> torch.Tensor:
    g = torch.Generator().manual_seed(1234)
    return torch.randn(4, 16, generator=g)


def test_deterministic_model_has_no_divergent_layers():
    model = TinyDeterministic()
    batch = _fixed_batch()

    report = detequal.diagnose(model, batch, n_runs=5, seed=0)

    assert report.n_runs == 5
    assert report.n_modules > 0
    assert report.divergent == [], (
        "a deterministic CPU model should have zero divergent layers, got: "
        f"{[v.canonical_name for v in report.divergent]}"
    )


def test_all_scores_zero_for_deterministic_model():
    model = TinyDeterministic()
    batch = _fixed_batch()
    report = detequal.diagnose(model, batch, n_runs=3, seed=7)
    assert all(v.score == 0.0 for v in report.verdicts)


def test_report_round_trips_to_json():
    model = TinyDeterministic()
    batch = _fixed_batch()
    report = detequal.diagnose(model, batch, n_runs=2)
    blob = report.to_json()
    assert '"n_runs": 2' in blob


def test_n_runs_below_two_rejected():
    model = TinyDeterministic()
    batch = _fixed_batch()
    try:
        detequal.diagnose(model, batch, n_runs=1)
    except ValueError:
        return
    raise AssertionError("n_runs=1 should raise ValueError")


def test_process_isolation_not_yet_supported():
    model = TinyDeterministic()
    batch = _fixed_batch()
    try:
        detequal.diagnose(model, batch, n_runs=2, isolate_process=True)
    except NotImplementedError:
        return
    raise AssertionError("isolate_process=True should raise NotImplementedError for now")
