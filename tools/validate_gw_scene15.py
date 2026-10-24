"""Run the standalone GW+FASTNEWTON generator on one real Scene15 batch."""

from __future__ import annotations

from dataclasses import asdict
import json
from pathlib import Path
import time

import torch
import yaml
from torch.utils.data import DataLoader

from dataset_loader import load_dataset
from model import CorreGenBaselineModel
from pseudolabel.gw import generate_bidirectional_gw_targets, projector_from_solver_config


def _compact_diagnostics(diagnostics):
    result = dict(diagnostics)
    result["step_diagnostics"] = [
        asdict(step) for step in diagnostics["step_diagnostics"]
    ]
    return result


def main():
    config_path = Path("config/scene15.yaml")
    with config_path.open(encoding="utf-8") as stream:
        config = yaml.safe_load(stream)

    device = torch.device(config["device"])
    args = type("Scene15Args", (), config)()
    dataset_train, dataset_test = load_dataset(args)
    loader = DataLoader(dataset_test, batch_size=config["batch_size"], shuffle=False)
    batch = next(iter(loader))
    _, data, *_ = batch
    data = [view.to(device, non_blocking=True) for view in data]

    model = CorreGenBaselineModel(
        n_views=config["n_views"],
        layer_dims=config["encoder_dim"],
        temperature=config["temperature"],
        n_classes=config["n_classes"],
        drop_rate=config["drop_rate"],
    ).to(device)
    model.eval()
    with torch.no_grad():
        online = [model.online_encoder[i](data[i]) for i in range(2)]
        predictions = [model.cross_view_decoder[i](online[i]) for i in range(2)]
        target_embeddings = [model.target_encoder[i](data[i]) for i in range(2)]

    projector = projector_from_solver_config(config["solver"])
    torch.cuda.synchronize(device)
    start = time.perf_counter()
    targets, diagnostics = generate_bidirectional_gw_targets(
        predictions, target_embeddings, config, projector
    )
    torch.cuda.synchronize(device)
    elapsed = time.perf_counter() - start

    output = {
        "dataset": config["dataset"],
        "batch_size": int(data[0].shape[0]),
        "device": str(device),
        "elapsed_seconds": elapsed,
        "p12_shape": list(targets["P12"].shape),
        "p21_shape": list(targets["P21"].shape),
        "p12_row_sum_max_error": float(
            (targets["P12"].sum(dim=1) - 1.0).abs().max().item()
        ),
        "p21_row_sum_max_error": float(
            (targets["P21"].sum(dim=1) - 1.0).abs().max().item()
        ),
        "p12": _compact_diagnostics(diagnostics["P12"]),
        "p21": _compact_diagnostics(diagnostics["P21"]),
    }
    print(json.dumps(output, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
