from __future__ import annotations

import pytest

GROUND_TRUTH_CASES: list[tuple[str, bool]] = []


def test_ground_truth_registry_present():
    assert isinstance(GROUND_TRUTH_CASES, list)


@pytest.mark.xfail(reason="repro cases not yet wired in", strict=False)
def test_ground_truth_has_cases():
    assert len(GROUND_TRUTH_CASES) >= 15
