import math
import sys
from pathlib import Path
import torch
from model import CorreGenBaselineModel
from torch.utils.data import DataLoader
from utils import MetricLogger, SmoothedValue ,InfoBuffer
from evaluation import evaluate, update_infoBuffer
from pseudolabel.gw import (
    GWProjectionFailure,
    generate_bidirectional_gw_targets,
    projector_from_solver_config,
)


def train_one_epoch(
    model: CorreGenBaselineModel,
    data_loader_train: DataLoader,
    data_loader_test: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    infolist: InfoBuffer,
    epoch: int,
    scheduler=None,
    args=None,
    projector=None,
):
    metric_logger = MetricLogger(delimiter="  ")
    metric_logger.add_meter("lr", SmoothedValue(window_size=1, fmt="{value:.6f}"))
    header = "Epoch: [{}]".format(epoch)
    if args.print_this_epoch:
        data_loader = enumerate(metric_logger.log_every(data_loader_train, args.print_freq, header))
    else:
        data_loader = enumerate(data_loader_train)

    training_phase = {"rectify": epoch < args.start_rectify_epoch}
    is_stride_gw = getattr(args, "algorithm", "corregen_baseline") == "stride_gw"
    training_state = {"mmt": args.momentum, "rho": args.rho, "dist": None}
    prior_gate_config = getattr(args, "prior_gate", None) or {}
    # The new lock and lockmass fields are explicit controls. The older
    # enabled and offdiag_tol names remain backward-compatible aliases.
    prior_gate_enabled = bool(
        prior_gate_config.get(
            "lock", prior_gate_config.get("enabled", False)
        )
    )
    prior_gate_offdiag_tol = float(
        prior_gate_config.get(
            "lockmass", prior_gate_config.get("offdiag_tol", 1.0e-3)
        )
    )
    prior_gate_on_failure = bool(
        prior_gate_config.get("on_projection_failure", True)
    )
    epoch_prior_locked = False
    epoch_prior_lock_reason = None
    alignment_diagnostics = _start_alignment_diagnostics(args, epoch)
    if is_stride_gw and projector is None:
        projector = projector_from_solver_config(args.solver)
    model.train(True)
    optimizer.zero_grad()
    for data_iter_step, (idx, samples, label1, label2, id1, id2, label_align) in data_loader:
        #droplast
        if len(idx) < math.ceil(args.batch_size / 2):
            continue
        smooth_epoch = epoch + (data_iter_step + 1) / len(data_loader_train)
        if scheduler is not None:
            lr = scheduler(optimizer, smooth_epoch)

        for i in range(args.n_views):
            samples[i] = samples[i].to(device, non_blocking=True)

        with torch.autocast("cuda", enabled=False):
            if is_stride_gw:
                z, p, z_t = model.compute_training_features(
                    samples, momentum=training_state["mmt"], update_target=True
                )
                batch_size = z[0].shape[0]
                identity = torch.eye(
                    batch_size, dtype=z[0].dtype, device=z[0].device
                )
                if training_phase["rectify"]:
                    inter_targets = [identity, identity]
                else:
                    if prior_gate_enabled and epoch_prior_locked:
                        inter_targets = [identity, identity]
                        if args.print_this_epoch:
                            metric_logger.update(epoch_prior_lock=1.0)
                    else:
                        try:
                            with torch.no_grad():
                                generated_targets, gw_diagnostics = generate_bidirectional_gw_targets(
                                    p,
                                    z_t,
                                    {
                                        "init": args.init,
                                        "steps": args.steps,
                                        "step_size": args.step_size,
                                        "transport_gradient": getattr(
                                            args, "transport_gradient", "structural"
                                        ),
                                    },
                                    projector,
                                )
                        except GWProjectionFailure as error:
                            if prior_gate_enabled and prior_gate_on_failure:
                                epoch_prior_locked = True
                                epoch_prior_lock_reason = "projection_failure"
                                inter_targets = [identity, identity]
                                print(
                                    "Epoch {} prior_gate=lock reason={} error={}".format(
                                        epoch,
                                        epoch_prior_lock_reason,
                                        error,
                                    )
                                )
                            else:
                                raise
                        else:
                            p12_offdiag = float(
                                gw_diagnostics["P12"]["non_diagonal_mass"]
                            )
                            p21_offdiag = float(
                                gw_diagnostics["P21"]["non_diagonal_mass"]
                            )
                            for direction in ("P12", "P21"):
                                direction_diagnostics = gw_diagnostics[direction]
                                if direction_diagnostics.get("projection_failed", False):
                                    print(
                                        "Epoch {} GW projection recovered direction={} "
                                        "failed_step={} last_valid_step={} offdiag={:.6g}".format(
                                            epoch,
                                            direction,
                                            direction_diagnostics.get("failed_step"),
                                            direction_diagnostics.get("last_valid_step"),
                                            direction_diagnostics["non_diagonal_mass"],
                                        )
                                    )
                            if prior_gate_enabled:
                                if not epoch_prior_locked and min(
                                    p12_offdiag, p21_offdiag
                                ) <= prior_gate_offdiag_tol:
                                    epoch_prior_locked = True
                                    epoch_prior_lock_reason = "low_diffusion"
                                    print(
                                        "Epoch {} prior_gate=lock reason={} "
                                        "p12_offdiag={:.6g} p21_offdiag={:.6g} "
                                        "tol={:.6g}".format(
                                            epoch,
                                            epoch_prior_lock_reason,
                                            p12_offdiag,
                                            p21_offdiag,
                                            prior_gate_offdiag_tol,
                                        )
                                    )
                            if epoch_prior_locked:
                                inter_targets = [identity, identity]
                            else:
                                inter_targets = [
                                    generated_targets["P12"],
                                    generated_targets["P21"],
                                ]
                            if args.print_this_epoch:
                                metric_logger.update(
                                    p12_offdiag=p12_offdiag,
                                    p21_offdiag=p21_offdiag,
                                    epoch_prior_lock=float(epoch_prior_locked),
                                )
                if alignment_diagnostics is not None:
                    if training_phase["rectify"]:
                        _record_alignment_batch(
                            alignment_diagnostics,
                            idx,
                            id1,
                            id2,
                            None,
                            None,
                            gw_executed=False,
                            prior_locked=False,
                        )
                    elif epoch_prior_locked:
                        _record_alignment_batch(
                            alignment_diagnostics,
                            idx,
                            id1,
                            id2,
                            None,
                            None,
                            gw_executed=False,
                            prior_locked=True,
                        )
                    else:
                        _record_alignment_batch(
                            alignment_diagnostics,
                            idx,
                            id1,
                            id2,
                            inter_targets[0],
                            inter_targets[1],
                            gw_executed=True,
                            prior_locked=False,
                        )
                loss = model.loss_from_training_features(z, p, z_t, inter_targets)
            else:
                dist = None
                if epoch > args.start_guide_epoch:
                    dist = calculate_dist(infolist, samples, idx, args)
                training_state["dist"] = dist
                loss = model(
                    samples,
                    training_state,
                    training_phase,
                    external_targets=None,
                    update_target=True,
                )
        loss_value = loss.item()
        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)
        loss.backward()
        optimizer.step()
        optimizer.zero_grad()

        if args.print_this_epoch:
            metric_logger.update(lr=lr)
            metric_logger.update(loss=loss_value)

    if alignment_diagnostics is not None:
        _save_alignment_diagnostics(alignment_diagnostics, epoch, args)

    if not is_stride_gw and epoch == args.start_guide_epoch:
        update_infoBuffer(infolist, model, data_loader_train, device, args, epoch, 'init')
    elif not is_stride_gw and epoch > args.start_guide_epoch:
        update_infoBuffer(infolist, model, data_loader_train, device, args, epoch, 'guide')

    if args.print_this_epoch:
        eval_result = evaluate(model, data_loader_test, device, args)
    else:
        eval_result = None
    return eval_result


def _start_alignment_diagnostics(args, epoch):
    """Create opt-in per-sample GW diagnostics without changing training."""
    config = getattr(args, "alignment_diagnostics", None) or {}
    if not bool(config.get("enabled", False)):
        return None
    output_subdir = config.get("output_subdir", "alignment_diagnostics")
    output_dir = Path(args.output_dir) / output_subdir
    output_dir.mkdir(parents=True, exist_ok=True)
    n_sample = int(args.n_sample)
    softmax_temperatures = [
        float(value)
        for value in config.get(
            "softmax_temperatures", [0.5, 0.75, 1.0, 1.5, 2.0]
        )
    ]
    return {
        "output_dir": output_dir,
        "epoch": int(epoch),
        "softmax_temperatures": softmax_temperatures,
        "p12_diagonal_quality": torch.full((n_sample,), float("nan")),
        "p21_diagonal_quality": torch.full((n_sample,), float("nan")),
        "p12_batch_diagonal_mean": torch.full((n_sample,), float("nan")),
        "p12_relative_diagonal_quality": torch.full((n_sample,), float("nan")),
        "p12_softmax_relative_quality": {
            str(temperature): torch.full((n_sample,), float("nan"))
            for temperature in softmax_temperatures
        },
        "batch_diagonal_mean": torch.full((n_sample,), float("nan")),
        "relative_diagonal_quality": torch.full((n_sample,), float("nan")),
        "gw_executed": torch.zeros(n_sample, dtype=torch.bool),
        "prior_locked": torch.zeros(n_sample, dtype=torch.bool),
        "observed": torch.zeros(n_sample, dtype=torch.bool),
        "id1": torch.full((n_sample,), -1, dtype=torch.long),
        "id2": torch.full((n_sample,), -1, dtype=torch.long),
        "actual_aligned": torch.full((n_sample,), -1, dtype=torch.int8),
    }




def _record_alignment_batch(
    diagnostics,
    indices,
    id1,
    id2,
    p12,
    p21,
    *,
    gw_executed,
    prior_locked,
):
    """Store only detached batch-level observations indexed by global sample id."""
    global_indices = indices.detach().cpu().long()
    first_ids = id1.detach().cpu().long()
    second_ids = id2.detach().cpu().long()
    diagnostics["observed"][global_indices] = True
    diagnostics["id1"][global_indices] = first_ids
    diagnostics["id2"][global_indices] = second_ids
    diagnostics["actual_aligned"][global_indices] = (
        first_ids == second_ids
    ).to(torch.int8)
    diagnostics["gw_executed"][global_indices] = bool(gw_executed)
    diagnostics["prior_locked"][global_indices] = bool(prior_locked)
    if p12 is not None and p21 is not None:
        p12_quality = p12.detach().diagonal().cpu().float()
        p21_quality = p21.detach().diagonal().cpu().float()
        diagonal_quality = 0.5 * (p12_quality + p21_quality)
        batch_mean = diagonal_quality.mean()
        p12_batch_mean = p12_quality.mean()
        diagnostics["p12_diagonal_quality"][global_indices] = p12_quality
        diagnostics["p21_diagonal_quality"][global_indices] = p21_quality
        diagnostics["p12_batch_diagonal_mean"][global_indices] = p12_batch_mean
        diagnostics["p12_relative_diagonal_quality"][global_indices] = (
            p12_quality / (p12_batch_mean + 1.0e-12)
        )
        for temperature, history in diagnostics[
            "p12_softmax_relative_quality"
        ].items():
            logits = torch.log(p12_quality.clamp_min(1.0e-12)) / float(temperature)
            history[global_indices] = torch.softmax(logits, dim=0) * len(p12_quality)
        diagnostics["batch_diagonal_mean"][global_indices] = batch_mean
        diagnostics["relative_diagonal_quality"][global_indices] = (
            diagonal_quality / (batch_mean + 1.0e-12)
        )


def _save_alignment_diagnostics(diagnostics, epoch, args):
    """Write one immutable epoch record for offline correspondence analysis."""
    record = {
        key: value
        for key, value in diagnostics.items()
        if key not in {"output_dir", "epoch"}
    }
    record.update(
        {
            "epoch": int(epoch),
            "dataset": str(args.dataset),
            "m_ratio": float(args.m_ratio),
            "seed": None if args.seed is None else int(args.seed),
            "batch_size": int(args.batch_size),
            "steps": int(args.steps),
            "step_size": float(args.step_size),
            "solver": dict(args.solver),
            "prior_gate": dict(getattr(args, "prior_gate", None) or {}),
        }
    )
    output_path = diagnostics["output_dir"] / "epoch_{:04d}.pt".format(epoch)
    torch.save(record, output_path)

def calculate_dist(infolist, samples, idx, args):
    emp_dist = infolist.get_Buffer()
    ret_sample_dist = torch.zeros(args.n_views, len(idx), device=samples[0].device)
    for i in range(args.n_views):
        ret_sample_dist[i] = emp_dist[i][idx]
    return ret_sample_dist
