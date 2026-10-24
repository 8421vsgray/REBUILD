"""Preflight the installed FASTNEWTON runtime before validation runs."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def ensure_import_paths() -> None:
    """Make the REBUILD project importable without injecting solver source paths."""
    project_root = str(PROJECT_ROOT)
    if project_root not in sys.path:
        sys.path.insert(0, project_root)


def preflight() -> dict[str, object]:
    """Run a small verified projection using the installed FASTNEWTON package."""
    ensure_import_paths()
    try:
        import fastnewton
    except (ImportError, OSError) as error:
        raise RuntimeError(
            "FASTNEWTON is not installed; run `python -m pip install -e ./solver/FASTNEWTON`"
        ) from error

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    candidate = torch.tensor(
        [[1.0, 0.0], [0.0, 1.0]], device=device, dtype=torch.float32
    )
    plan, info = fastnewton.project(
        candidate, relative_tol=1.0e-3, max_iter=200, return_info=True
    )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    status = str(info.get("status", "unknown"))
    if status != "verified":
        raise RuntimeError(f"FASTNEWTON preflight was not verified: {info}")
    marginal = torch.full((2,), 0.5, device=device, dtype=plan.dtype)
    row_error = (plan.sum(dim=1) - marginal).abs().max().item()
    col_error = (plan.sum(dim=0) - marginal).abs().max().item()
    if not torch.isfinite(plan).all().item() or float(plan.min()) < -1.0e-7:
        raise RuntimeError("FASTNEWTON preflight returned an invalid plan")
    if max(row_error, col_error) > 1.0e-3:
        raise RuntimeError(
            f"FASTNEWTON preflight marginal check failed: row={row_error}, col={col_error}"
        )
    return {
        "python": sys.executable,
        "project_root": str(PROJECT_ROOT),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device": str(device),
        "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None,
        "fastnewton_module": str(Path(fastnewton.__file__).resolve()),
        "status": status,
        "iterations": info.get("iterations"),
        "row_residual": row_error,
        "col_residual": col_error,
    }


def main() -> int:
    try:
        print(json.dumps(preflight(), sort_keys=True), flush=True)
    except Exception as error:  # noqa: BLE001 - CLI must return a clear non-zero result
        print(f"REBUILD validation preflight failed: {error}", file=sys.stderr, flush=True)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
