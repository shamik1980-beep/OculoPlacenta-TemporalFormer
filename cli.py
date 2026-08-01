"""Command-line interface for reproducible analyses."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

from .benchmark import run_benchmark
from .constants import DEFAULT_SEED
from .data import sha256_file, verify_bogota_checksum


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="oculoplacenta")
    subparsers = parser.add_subparsers(dest="command", required=True)

    benchmark = subparsers.add_parser("benchmark", help="Run the Bogotá benchmark.")
    benchmark.add_argument("--data", required=True, type=Path)
    benchmark.add_argument("--out", type=Path, default=Path("results"))
    benchmark.add_argument("--seed", type=int, default=DEFAULT_SEED)
    benchmark.add_argument("--splits", type=int, default=5)
    benchmark.add_argument("--repeats", type=int, default=2)
    benchmark.add_argument("--bootstrap", type=int, default=500)
    benchmark.add_argument("--threshold", type=float, default=0.5)

    verify = subparsers.add_parser("verify-data", help="Verify the source dataset checksum.")
    verify.add_argument("data", type=Path)
    verify.add_argument("--allow-mismatch", action="store_true")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    if args.command == "verify-data":
        matches = verify_bogota_checksum(args.data, strict=not args.allow_mismatch)
        print(json.dumps({"path": str(args.data), "sha256": sha256_file(args.data), "matches": matches}, indent=2))
        return

    performance, audit = run_benchmark(
        data_path=args.data,
        output_dir=args.out,
        seed=args.seed,
        n_splits=args.splits,
        n_repeats=args.repeats,
        n_bootstrap=args.bootstrap,
        threshold=args.threshold,
    )
    print(performance.to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
