from __future__ import annotations

import argparse
import sys

from . import __version__


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="detequal",
        description="Localize and explain run-to-run nondeterminism in PyTorch models.",
    )
    parser.add_argument("--version", action="version", version=f"detequal {__version__}")
    sub = parser.add_subparsers(dest="command", required=True)

    diag = sub.add_parser("diagnose", help="diagnose a model over N runs")
    diag.add_argument("model", help="path to a Python file defining the model")
    diag.add_argument("--input", required=True, help="path to a saved input batch (.pt)")
    diag.add_argument("--runs", type=int, default=5, help="number of runs (default: 5)")
    diag.add_argument("--seed", type=int, default=0, help="fixed seed for every run")
    diag.add_argument("--grad", action="store_true", help="also hook the backward pass")
    diag.add_argument(
        "--deterministic",
        action="store_true",
        help="comparison mode: run with use_deterministic_algorithms(True)",
    )
    diag.add_argument("--json", metavar="PATH", help="write the full report to PATH")
    return parser


def main(argv: list[str] | None = None) -> int:
    args = _parser().parse_args(argv)

    if args.command == "diagnose":
        print(
            "detequal diagnose: model/batch file loading is not implemented yet.\n"
            "Use the Python API in the meantime:\n\n"
            "    import detequal\n"
            f"    report = detequal.diagnose(model, batch, n_runs={args.runs})\n"
            "    report.summary()\n",
            file=sys.stderr,
        )
        return 2

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
