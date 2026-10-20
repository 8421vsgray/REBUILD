"""Shared runtime checks for REBUILD numerical validation scripts."""

from __future__ import annotations

import json
import sys
from pathlib import Path

import torch


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DROT_PYTHON_ROOT = PROJECT_ROOT / "solver" / "Regularized-DROT" / "pytorch"


def ensure_import_paths() -> None:
    """Make the project and compiled third-party extension importable."""
    for path in (PROJECT_ROOT, DROT_PYTHON_ROOT):
        path_string = str(path)
        if path_string not in sys.path:
            sys.path.insert(0, path_string)


def preflight() -> dict[str, object]:
    """Fail before a batch run if the actual numerical backend is unavailable."""
    ensure_import_paths()
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is unavailable")
    try:
        import reg_drot
    except (ImportError, OSError) as error:
        raise RuntimeError(f"reg_drot import failed: {error}") from error

    device = torch.device("cuda")
    cost = torch.tensor([[0.0, 1.0], [1.0, 0.0]], device=device, dtype=torch.float32)
    marginal = torch.full((2,), 0.5, device=device, dtype=torch.float32)
    plan = reg_drot.quadratic_drot(
        cost,
        marginal,
        marginal,
        4.0,
        0.25,
        1000,
        1e-5,
    )
    torch.cuda.synchronize()
    if not isinstance(plan, torch.Tensor) or plan.shape != cost.shape:
        raise RuntimeError("reg_drot preflight returned an invalid plan")
    if not torch.isfinite(plan).all().item():
        raise RuntimeError("reg_drot preflight returned non-finite values")
    return {
        "python": sys.executable,
        "project_root": str(PROJECT_ROOT),
        "drot_python_root": str(DROT_PYTHON_ROOT),
        "torch_version": torch.__version__,
        "torch_cuda": torch.version.cuda,
        "device": torch.cuda.get_device_name(),
        "reg_drot_module": str(Path(reg_drot.__file__).resolve()),
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
