"""Compare DROT optimality diagnostics across internal residual tolerances."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import torch

from validate_drot_kkt import run_case
from validation_runtime import preflight


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sizes", nargs="+", type=int, default=[128, 1024])
    parser.add_argument("--kinds", nargs="+", default=["random", "gw"])
    parser.add_argument("--rhos", nargs="+", type=float, default=[2.0, 4.0, 10.0])
    parser.add_argument("--epsilons", nargs="+", type=float, default=[1e-6, 1e-7, 1e-8])
    parser.add_argument("--max-iterations", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=20261020)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    preflight()
    records = []
    for solver_eps in args.epsilons:
        for size in args.sizes:
            for kind_index, kind in enumerate(args.kinds):
                for rho_index, rho in enumerate(args.rhos):
                    records.append(
                        run_case(
                            size,
                            kind,
                            rho,
                            solver_eps,
                            args.max_iterations,
                            args.seed + size * 100 + kind_index * 10 + rho_index,
                        )
                    )
    payload = {
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
        "records": records,
    }
    args.output.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    print(f"records={len(records)}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
