from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum

import torch


class Precision(str, Enum):
    FP32 = "fp32"
    FP16 = "fp16"
    BF16 = "bf16"
    FP8_E4M3 = "fp8_e4m3"
    FP8_E5M2 = "fp8_e5m2"
    NVFP4 = "nvfp4"
    MXFP4 = "mxfp4"


@dataclass
class RungSupport:
    precision: Precision
    supported: bool
    reason: str


def _cuda_compute_capability() -> tuple[int, int] | None:
    if not torch.cuda.is_available():
        return None
    return torch.cuda.get_device_capability()


def _is_hopper_or_newer(cc: tuple[int, int] | None) -> bool:
    return cc is not None and cc >= (9, 0)


def _is_blackwell(cc: tuple[int, int] | None) -> bool:
    return cc is not None and cc >= (10, 0)


def probe_support() -> list[RungSupport]:
    cc = _cuda_compute_capability()
    hopper = _is_hopper_or_newer(cc)
    blackwell = _is_blackwell(cc)

    def msg(need: str) -> str:
        where = f"device sm_{cc[0]}{cc[1]}" if cc else "CPU (no CUDA device)"
        return f"unsupported on this device: {need} required, found {where}"

    return [
        RungSupport(Precision.FP32, True, ""),
        RungSupport(Precision.FP16, True, ""),
        RungSupport(Precision.BF16, True, ""),
        RungSupport(Precision.FP8_E4M3, hopper, "" if hopper else msg("Hopper or newer")),
        RungSupport(Precision.FP8_E5M2, hopper, "" if hopper else msg("Hopper or newer")),
        RungSupport(Precision.NVFP4, blackwell, "" if blackwell else msg("Blackwell (sm_100+)")),
        RungSupport(Precision.MXFP4, blackwell, "" if blackwell else msg("Blackwell (sm_100+)")),
    ]


@dataclass
class LadderResult:
    device_capability: tuple[int, int] | None
    rungs: list[RungSupport] = field(default_factory=list)
    summaries: dict[str, dict] = field(default_factory=dict)


def run_ladder(model, batch, **diagnose_kwargs) -> LadderResult:
    _ = LadderResult(device_capability=_cuda_compute_capability(), rungs=probe_support())
    raise NotImplementedError(
        "per-rung casting and diagnosis is not implemented yet; "
        "probe_support() gives the supported rungs for this device"
    )
