import math
import sys
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
    prior_gate_raw_offdiag_tol = float(
        prior_gate_config.get("raw_offdiag_tol", prior_gate_offdiag_tol)
    )
    prior_gate_on_failure = bool(
        prior_gate_config.get("on_projection_failure", True)
    )
    epoch_prior_locked = False
    epoch_prior_lock_reason = None
    epoch_prior_gate_decided = False
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
                                        "prior_gate_precheck": (
                                            prior_gate_enabled
                                            and not epoch_prior_gate_decided
                                        ),
                                        "raw_offdiag_tol": prior_gate_raw_offdiag_tol,
                                    },
                                    projector,
                                )
                        except GWProjectionFailure as error:
                            if prior_gate_enabled and prior_gate_on_failure:
                                epoch_prior_gate_decided = True
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
                            prior_gate_triggered = bool(
                                gw_diagnostics["P12"].get("prior_gate", False)
                                or gw_diagnostics["P21"].get("prior_gate", False)
                            )
                            if prior_gate_enabled and not epoch_prior_gate_decided:
                                epoch_prior_gate_decided = True
                                if (
                                    prior_gate_triggered
                                    or min(p12_offdiag, p21_offdiag)
                                    <= prior_gate_offdiag_tol
                                ):
                                    epoch_prior_locked = True
                                    epoch_prior_lock_reason = (
                                        "low_raw_diffusion"
                                        if prior_gate_triggered
                                        else "low_diffusion"
                                    )
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

    if not is_stride_gw and epoch == args.start_guide_epoch:
        update_infoBuffer(infolist, model, data_loader_train, device, args, epoch, 'init')
    elif not is_stride_gw and epoch > args.start_guide_epoch:
        update_infoBuffer(infolist, model, data_loader_train, device, args, epoch, 'guide')

    if args.print_this_epoch:
        eval_result = evaluate(model, data_loader_test, device, args)
    else:
        eval_result = None
    return eval_result

def calculate_dist(infolist, samples, idx, args):
    emp_dist = infolist.get_Buffer()
    ret_sample_dist = torch.zeros(args.n_views, len(idx), device=samples[0].device)
    for i in range(args.n_views):
        ret_sample_dist[i] = emp_dist[i][idx]
    return ret_sample_dist
