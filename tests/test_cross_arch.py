from __future__ import annotations

import detequal.precision as precision
from detequal.precision import Precision


def _support_map(rungs):
    return {r.precision: r for r in rungs}


def test_cpu_only_supports_float_rungs_skips_fp8_and_fp4(monkeypatch):
    monkeypatch.setattr(precision, "_cuda_compute_capability", lambda: None)
    rungs = _support_map(precision.probe_support())

    assert rungs[Precision.FP32].supported
    assert rungs[Precision.FP16].supported
    assert rungs[Precision.BF16].supported
    assert not rungs[Precision.FP8_E4M3].supported
    assert not rungs[Precision.NVFP4].supported
    assert "unsupported on this device" in rungs[Precision.NVFP4].reason


def test_hopper_supports_fp8_but_not_fp4(monkeypatch):
    monkeypatch.setattr(precision, "_cuda_compute_capability", lambda: (9, 0))
    rungs = _support_map(precision.probe_support())

    assert rungs[Precision.FP8_E4M3].supported
    assert rungs[Precision.FP8_E5M2].supported
    assert not rungs[Precision.NVFP4].supported
    assert not rungs[Precision.MXFP4].supported


def test_blackwell_supports_full_ladder(monkeypatch):
    monkeypatch.setattr(precision, "_cuda_compute_capability", lambda: (12, 0))
    rungs = _support_map(precision.probe_support())

    assert all(r.supported for r in rungs.values())
    assert rungs[Precision.NVFP4].reason == ""


def test_ladder_run_reports_not_implemented_cleanly(monkeypatch):
    monkeypatch.setattr(precision, "_cuda_compute_capability", lambda: None)
    try:
        precision.run_ladder(model=None, batch=None)
    except NotImplementedError:
        return
    raise AssertionError("run_ladder should raise NotImplementedError for now")
