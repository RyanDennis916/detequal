from __future__ import annotations

__version__ = "0.0.1"

from .harness import diagnose
from .report import DiffReport

__all__ = ["diagnose", "DiffReport", "__version__"]
