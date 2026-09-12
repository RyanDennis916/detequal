from __future__ import annotations

import os
import random
from dataclasses import dataclass
from typing import Any

try:
    import numpy as np
except ImportError:
    np = None

import torch

from .instrument import bind_model, capture
from .report import DiffReport, compare_traces


@dataclass
class HarnessConfig:
    n_runs: int = 5
    seed: int = 0
    eval_mode: bool = True
    force_deterministic: bool = False
    compute_grad: bool = False
    isolate_process: bool = False
    capture_full_tensors: bool = False

    def __post_init__(self) -> None:
        if self.n_runs < 2:
            raise ValueError("n_runs must be >= 2 to compare runs against each other")


def set_all_seeds(seed: int) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    if np is not None:
        np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _run_once(
    model: torch.nn.Module,
    batch: Any,
    run_id: int,
    cfg: HarnessConfig,
) -> dict[str, Any]:
    set_all_seeds(cfg.seed)

    if cfg.eval_mode:
        model.eval()

    ctx = torch.enable_grad() if cfg.compute_grad else torch.no_grad()
    bind_model(model)
    with ctx, capture(run_id, full_tensors=cfg.capture_full_tensors) as cap:
        output = model(batch)
        if cfg.compute_grad:
            loss = output.float().sum() if torch.is_tensor(output) else None
            if loss is not None:
                loss.backward()
    return cap.tensors


def diagnose(
    model: torch.nn.Module,
    batch: Any,
    n_runs: int = 5,
    *,
    seed: int = 0,
    eval_mode: bool = True,
    force_deterministic: bool = False,
    compute_grad: bool = False,
    isolate_process: bool = False,
    capture_full_tensors: bool = False,
) -> DiffReport:
    cfg = HarnessConfig(
        n_runs=n_runs,
        seed=seed,
        eval_mode=eval_mode,
        force_deterministic=force_deterministic,
        compute_grad=compute_grad,
        isolate_process=isolate_process,
        capture_full_tensors=capture_full_tensors,
    )

    if cfg.force_deterministic:
        torch.use_deterministic_algorithms(True, warn_only=True)

    if cfg.isolate_process:
        raise NotImplementedError(
            "process-isolated runs are not implemented yet; call with "
            "isolate_process=False for the in-process path"
        )

    traces: dict[int, dict[str, Any]] = {}
    for run_id in range(cfg.n_runs):
        traces[run_id] = _run_once(model, batch, run_id, cfg)

    return compare_traces(traces, config=cfg)
