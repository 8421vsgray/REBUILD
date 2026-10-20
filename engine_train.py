import math
import sys
import torch
from model import CorreGenBaselineModel
from torch.utils.data import DataLoader
from utils import MetricLogger, SmoothedValue ,InfoBuffer
from evaluation import evaluate, update_infoBuffer
from pseudolabel.gw import generate_bidirectional_gw_targets, projector_from_solver_config


@torch.no_grad()
def _make_gw_inputs(model: CorreGenBaselineModel, samples):
    """Build detached online predictions and EMA embeddings for GW labels."""
    was_training = model.training
    model.eval()
    try:
        online = [model.online_encoder[i](samples[i]) for i in range(model.n_views)]
        predictions = [
            model.cross_view_decoder[i](online[i]) for i in range(model.n_views)
        ]
        target_embeddings = [
            model.target_encoder[i](samples[i]) for i in range(model.n_views)
        ]
    finally:
        model.train(was_training)
    return predictions, target_embeddings


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
            dist = None
            if not is_stride_gw and epoch > args.start_guide_epoch:
                dist = calculate_dist(infolist, samples, idx, args)
            training_state["dist"] = dist
            external_targets = None
            if is_stride_gw and not training_phase["rectify"]:
                model._update_target_branch(training_state["mmt"])
                predictions, target_embeddings = _make_gw_inputs(model, samples)
                external_targets, gw_diagnostics = generate_bidirectional_gw_targets(
                    predictions,
                    target_embeddings,
                    {
                        "init": args.init,
                        "steps": args.steps,
                        "step_size": args.step_size,
                    },
                    projector,
                )
                if args.print_this_epoch:
                    metric_logger.update(
                        p12_offdiag=gw_diagnostics["P12"]["non_diagonal_mass"],
                        p21_offdiag=gw_diagnostics["P21"]["non_diagonal_mass"],
                    )
            loss = model(
                samples,
                training_state,
                training_phase,
                external_targets=external_targets,
                update_target=not (is_stride_gw and not training_phase["rectify"]),
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
