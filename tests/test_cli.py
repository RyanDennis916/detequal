from __future__ import annotations

import pytest

from detequal import cli


def test_version_flag_exits_zero():
    with pytest.raises(SystemExit) as exc:
        cli.main(["--version"])
    assert exc.value.code == 0


def test_diagnose_requires_input():
    with pytest.raises(SystemExit):
        cli.main(["diagnose", "model.py"])


def test_diagnose_reports_not_implemented():
    rc = cli.main(["diagnose", "model.py", "--input", "batch.pt", "--runs", "3"])
    assert rc == 2
