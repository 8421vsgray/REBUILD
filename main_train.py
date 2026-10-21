import argparse

import os
from pathlib import Path
import time
import datetime
import warnings
import copy
import uuid

import numpy as np
import torch
import utils
import yaml
from dataset_loader import load_dataset
from engine_train import train_one_epoch
from model import CorreGenBaselineModel

warnings.filterwarnings("ignore")

MAX_FORMAL_BATCH_SIZE = 1024
SUPPORTED_CONFIG_FIELDS = {
    "algorithm",
    "batch_size",
    "beta",
    "blr",
    "config_version",
    "data_norm",
    "data_path",
    "dataset",
    "diagonal_epochs",
    "device",
    "drop_rate",
    "embed_dim",
    "encoder_dim",
    "epochs",
    "eval_batch_size",
    "init",
    "lr",
    "m",
    "m_ratio",
    "min_blr",
    "min_lr",
    "momentum",
    "n_classes",
    "n_views",
    "num_runs",
    "num_workers",
    "output_dir",
    "p",
    "pin_mem",
    "prior_gate",
    "print_freq",
    "reg",
    "rho",
    "save_ckpt",
    "seed",
    "solver",
    "start_epoch",
    "start_guide_epoch",
    "start_rectify_epoch",
    "step_size",
    "steps",
    "temperature",
    "threads",
    "train_time",
    "view_indices",
    "warmup_epochs",
    "weight_decay",
}
SOLVER_CONFIG_FIELDS = {
    "backend",
    "rho",
    "eps",
    "max_iter",
    "relative_tol",
    "absolute_tol",
    "certificate_tol",
    "warm_start",
}
PRIOR_GATE_CONFIG_FIELDS = {
    "enabled",
    "lock",
    "lockmass",
    "offdiag_tol",
    "raw_offdiag_tol",
    "on_projection_failure",
}
STANDARD_CONFIG_REQUIRED_FIELDS = {
    "algorithm",
    "dataset",
    "data_path",
    "data_norm",
    "view_indices",
    "n_views",
    "n_classes",
    "encoder_dim",
    "batch_size",
    "eval_batch_size",
    "epochs",
    "warmup_epochs",
    "diagonal_epochs",
    "blr",
    "min_blr",
    "weight_decay",
    "momentum",
    "temperature",
    "drop_rate",
    "m_ratio",
    "num_runs",
    "device",
    "seed",
    "output_dir",
    "init",
    "steps",
    "step_size",
    "solver",
}


def get_args_parser():
    parser = argparse.ArgumentParser(description="Training")

    # config path
    parser.add_argument("--config_file", type=str, default=None)

    # backbone parameters
    parser.add_argument("--encoder_dim", type=list, nargs="+", default=[])
    parser.add_argument("--embed_dim", type=int, default=0)

    # model parameters
    parser.add_argument("--temperature", type=float, default=0.5)
    parser.add_argument("--start_rectify_epoch", type=int, default=20)
    parser.add_argument("--momentum", type=float, default=0.99)
    parser.add_argument("--drop_rate", type=float, default=0.2)
    parser.add_argument("--n_views", type=int, default=2, help="number of views")
    parser.add_argument("--n_classes", type=int, default=10, help="number of classes")

    # training setting
    parser.add_argument("--batch_size", type=int, default=256, help="batch size per GPU")
    parser.add_argument(
        "--eval_batch_size",
        type=int,
        default=None,
        help="evaluation batch size; defaults to batch_size",
    )
    parser.add_argument("--epochs", type=int, default=200)
    parser.add_argument("--warmup_epochs", type=int, default=20, help="epochs to warmup learning rate")
    parser.add_argument("--blr", type=float, default=5.0e-4)
    parser.add_argument("--min_blr", type=float, default=5.0e-4)
    parser.add_argument("--data_norm",type=str,default="standard",choices=["standard", "min-max", "l2-norm"])
    parser.add_argument("--num_runs", type=int, default=None)
    parser.add_argument("--threads", type=int, default=8, help="number of threads for torch")
    parser.add_argument("--num_workers", default=0, type=int)

    # optimizer parameters
    parser.add_argument("--weight_decay",type=float,default=0,help="Initial value of the weight decay. (default: 0)")
    parser.add_argument("--lr",type=float,default=None,metavar="LR",help="learning rate (absolute lr)")
    parser.add_argument("--min_lr", type=float, default=None, metavar="LR", help="minimum learning rate (absolute lr)")

    #optimal tranport parameters
    parser.add_argument("--beta", type=float, default=0.5, help="the weight for the fusion of identity matrix and the pseudo label")
    parser.add_argument("--reg", type=float, default=0.03, help="regularization term")
    parser.add_argument("--rho", type=float, default=0.2, help="the value for virtual sample mass")

    #GMM settings
    parser.add_argument("--start_guide_epoch",type=int,default=100,help="start clustering guide (GMM) epoch")
    parser.add_argument("--p",type=float,default=0.1,help='penalty for clustering guide')
    parser.add_argument("--m",type=float,default=10)

    # data loader and logger
    parser.add_argument("--device", default="cuda:0", help="device to use for training / testing")
    parser.add_argument("--pin_mem", action="store_true", help="Pin CPU memory in DataLoader for more efficient (sometimes) transfer to GPU.")
    parser.add_argument("--no_pin_mem", action="store_false", dest="pin_mem")
    parser.set_defaults(pin_mem=True)

    parser.add_argument("--print_freq", default=5)
    parser.add_argument("--start_epoch", default=0, type=int, metavar="N", help="start epoch")
    parser.add_argument("--dataset",type=str,default="LandUse21")
    parser.add_argument("--data_path", type=str, default="./", help="path to your folder of dataset")
    parser.add_argument("--output_dir",type=str,default="output_log_test",help="path where to save, empty for no saving")
    parser.add_argument("--m_ratio", type=float, default=0.0)
    parser.add_argument("--seed", default=None, type=int)
    parser.add_argument("--save_ckpt", action="store_true")
    parser.add_argument(
        "--algorithm",
        default="corregen_baseline",
        choices=["corregen_baseline", "stride_gw"],
    )
    parser.add_argument(
        "--validate_config",
        action="store_true",
        help="validate and print a YAML configuration without starting training",
    )
    return parser


def resolve_batch_sizes(args, n_train, n_test):
    """Resolve effective train/evaluation batch sizes without dropping samples."""
    if n_train <= 0 or n_test <= 0:
        raise ValueError("training and evaluation datasets must be non-empty")
    if args.batch_size <= 0:
        raise ValueError("batch_size must be positive")
    if args.batch_size > MAX_FORMAL_BATCH_SIZE:
        raise ValueError(
            "REBUILD formal training batch_size must be <= {} "
            "for the configured solver backend".format(MAX_FORMAL_BATCH_SIZE)
        )

    eval_batch_size = args.eval_batch_size
    if eval_batch_size is None:
        eval_batch_size = args.batch_size
    if eval_batch_size <= 0:
        raise ValueError("eval_batch_size must be positive")
    if eval_batch_size > MAX_FORMAL_BATCH_SIZE:
        raise ValueError(
            "REBUILD formal eval_batch_size must be <= {} "
            "for the configured solver backend".format(MAX_FORMAL_BATCH_SIZE)
        )

    return min(args.batch_size, n_train), min(eval_batch_size, n_test)


def build_data_loaders(dataset_train, dataset_test, args):
    """Build independent random-sampler loaders for training and evaluation."""
    train_batch_size, eval_batch_size = resolve_batch_sizes(
        args, len(dataset_train), len(dataset_test)
    )
    args.batch_size = train_batch_size
    args.eval_batch_size = eval_batch_size

    sampler_train = torch.utils.data.RandomSampler(dataset_train)
    sampler_test = torch.utils.data.RandomSampler(dataset_test)

    data_loader_train = torch.utils.data.DataLoader(
        dataset_train,
        sampler=sampler_train,
        batch_size=train_batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    data_loader_test = torch.utils.data.DataLoader(
        dataset_test,
        sampler=sampler_test,
        batch_size=eval_batch_size,
        num_workers=args.num_workers,
        pin_memory=args.pin_mem,
        drop_last=False,
    )
    return data_loader_train, data_loader_test


def _is_integer(value):
    return isinstance(value, int) and not isinstance(value, bool)


def _validate_standard_config(configs):
    missing = sorted(STANDARD_CONFIG_REQUIRED_FIELDS.difference(configs))
    if missing:
        raise ValueError(
            "Standard REBUILD YAML is missing required field(s): {}".format(
                ", ".join(missing)
            )
        )

    n_views = configs["n_views"]
    if not _is_integer(n_views) or n_views <= 0:
        raise ValueError("n_views must be a positive integer")

    view_indices = configs["view_indices"]
    if (
        not isinstance(view_indices, list)
        or len(view_indices) != n_views
        or any(not _is_integer(index) or index < 0 for index in view_indices)
        or len(set(view_indices)) != len(view_indices)
    ):
        raise ValueError("view_indices must be a unique list of non-negative integers")

    encoder_dim = configs["encoder_dim"]
    if (
        not isinstance(encoder_dim, list)
        or len(encoder_dim) != n_views
        or any(
            not isinstance(view_layers, list)
            or len(view_layers) < 2
            or not _is_integer(view_layers[0])
            or view_layers[0] <= 0
            for view_layers in encoder_dim
        )
    ):
        raise ValueError("encoder_dim must provide one positive input dimension per view")

    if configs["data_norm"] not in {"standard", "min-max", "l2-norm"}:
        raise ValueError("data_norm must be standard, min-max or l2-norm")
    if (
        not _is_integer(configs["batch_size"])
        or not 0 < configs["batch_size"] <= MAX_FORMAL_BATCH_SIZE
    ):
        raise ValueError("formal batch_size must be an integer in [1, 512]")
    if "eval_batch_size" in configs and (
        not _is_integer(configs["eval_batch_size"])
        or not 0 < configs["eval_batch_size"] <= MAX_FORMAL_BATCH_SIZE
    ):
        raise ValueError("formal eval_batch_size must be an integer in [1, 512]")
    if not 0.0 <= float(configs["m_ratio"]) <= 1.0:
        raise ValueError("m_ratio must be in [0, 1]")
    if configs["init"] != "diagonal":
        raise ValueError("the current STRIDE configuration requires init=diagonal")
    if not _is_integer(configs["steps"]) or configs["steps"] <= 0:
        raise ValueError("steps must be a positive integer")
    if float(configs["step_size"]) <= 0:
        raise ValueError("step_size must be positive")

    if "prior_gate" in configs:
        prior_gate = configs["prior_gate"]
        if not isinstance(prior_gate, dict):
            raise ValueError("prior_gate must be a mapping")
        unknown_prior_gate_fields = sorted(
            set(prior_gate).difference(PRIOR_GATE_CONFIG_FIELDS)
        )
        if unknown_prior_gate_fields:
            raise ValueError(
                "Unknown prior_gate field(s): {}".format(
                    ", ".join(unknown_prior_gate_fields)
                )
            )
        if "enabled" in prior_gate and not isinstance(
            prior_gate["enabled"], bool
        ):
            raise ValueError("prior_gate.enabled must be boolean")
        if "lock" in prior_gate and not isinstance(prior_gate["lock"], bool):
            raise ValueError("prior_gate.lock must be boolean")
        if "lockmass" in prior_gate and float(prior_gate["lockmass"]) < 0.0:
            raise ValueError("prior_gate.lockmass must be non-negative")
        for field in ("offdiag_tol", "raw_offdiag_tol"):
            if field in prior_gate and float(prior_gate[field]) < 0.0:
                raise ValueError("prior_gate.{} must be non-negative".format(field))
        if "on_projection_failure" in prior_gate and not isinstance(
            prior_gate["on_projection_failure"], bool
        ):
            raise ValueError("prior_gate.on_projection_failure must be boolean")

    solver = configs["solver"]
    if not isinstance(solver, dict):
        raise ValueError("solver must be a mapping")
    unknown_solver_fields = sorted(set(solver).difference(SOLVER_CONFIG_FIELDS))
    if unknown_solver_fields:
        raise ValueError(
            "Unknown solver field(s): {}".format(", ".join(unknown_solver_fields))
        )
    backend = solver.get("backend")
    if backend == "drot":
        required = {"backend", "rho", "eps", "max_iter"}
        missing_solver_fields = sorted(required.difference(solver))
        if missing_solver_fields:
            raise ValueError(
                "DROT solver config is missing: {}".format(
                    ", ".join(missing_solver_fields)
                )
            )
        if float(solver["rho"]) <= 0:
            raise ValueError("solver.rho must be positive")
        if float(solver["eps"]) <= 0:
            raise ValueError("solver.eps must be positive")
        if not _is_integer(solver["max_iter"]) or solver["max_iter"] <= 0:
            raise ValueError("solver.max_iter must be a positive integer")
    elif backend == "fastnewton":
        if {"rho", "eps"}.intersection(solver):
            raise ValueError(
                "FASTNEWTON solver config must not contain DROT fields rho or eps"
            )
        required = {"backend", "relative_tol", "max_iter"}
        missing_solver_fields = sorted(required.difference(solver))
        if missing_solver_fields:
            raise ValueError(
                "FASTNEWTON solver config is missing: {}".format(
                    ", ".join(missing_solver_fields)
                )
            )
        if float(solver["relative_tol"]) <= 0:
            raise ValueError("solver.relative_tol must be positive")
        for field in ("absolute_tol", "certificate_tol"):
            if field in solver and float(solver[field]) <= 0:
                raise ValueError("solver.{} must be positive".format(field))
        if "warm_start" in solver and not isinstance(solver["warm_start"], bool):
            raise ValueError("solver.warm_start must be boolean")
        if not _is_integer(solver["max_iter"]) or solver["max_iter"] <= 0:
            raise ValueError("solver.max_iter must be a positive integer")
    else:
        raise ValueError("solver.backend must be drot or fastnewton")


def resolve_config_path(config_file):
    """Resolve either a repository-relative path or a config basename."""
    requested = Path(config_file)
    candidates = [requested]
    if not requested.is_absolute():
        candidates.append(Path("config") / requested)
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    raise FileNotFoundError(
        "Configuration file not found; checked: {}".format(
            ", ".join(str(candidate) for candidate in candidates)
        )
    )


def load_config_file(config_path):
    """Load and validate a REBUILD YAML configuration."""
    with open(config_path, encoding="utf-8") as config_stream:
        configs = yaml.safe_load(config_stream) or {}
    if not isinstance(configs, dict):
        raise ValueError("YAML configuration must be a mapping")
    removed_fields = sorted(set(configs).intersection({"c_ratio"}))
    if removed_fields:
        fields = ", ".join(removed_fields)
        raise ValueError(
            "Removed configuration field(s): {}. REBUILD is NC-only; use m_ratio.".format(fields)
        )
    unknown_fields = sorted(set(configs).difference(SUPPORTED_CONFIG_FIELDS))
    if unknown_fields:
        raise ValueError(
            "Unknown configuration field(s): {}".format(", ".join(unknown_fields))
        )
    if "train_time" in configs:
        if "num_runs" in configs:
            raise ValueError("Use only num_runs; train_time and num_runs cannot both be set")
        configs["num_runs"] = configs.pop("train_time")
    standard_markers = {"algorithm", "view_indices", "solver", "steps", "step_size"}
    if standard_markers.intersection(configs):
        _validate_standard_config(configs)
        if configs.get("diagonal_epochs") is not None:
            if (
                "start_rectify_epoch" in configs
                and configs["start_rectify_epoch"] != configs["diagonal_epochs"]
            ):
                raise ValueError(
                    "diagonal_epochs and start_rectify_epoch cannot disagree"
                )
            configs.setdefault("start_rectify_epoch", configs["diagonal_epochs"])
    return configs


def finalize_run_count(args):
    """Apply the stable default after YAML and CLI values have been merged."""
    if args.num_runs is None:
        args.num_runs = 5
    if args.num_runs <= 0:
        raise ValueError("num_runs must be positive")


def build_run_output_dir(experiment_dir, run_id, seed):
    """Return a unique per-run directory and reject accidental reuse."""
    seed_label = "seed_none" if seed is None else "seed_{}".format(seed)
    run_dir = Path(experiment_dir) / run_id / seed_label
    if run_dir.exists():
        raise FileExistsError(
            "Run output already exists; choose a new run_id: {}".format(run_dir)
        )
    return run_dir


def write_config_snapshot(args):
    """Persist the effective configuration without silently overwriting it."""
    snapshot_path = Path(args.output_dir) / "config_effective.yaml"
    snapshot = yaml.safe_dump(vars(args), sort_keys=True)
    if snapshot_path.exists():
        existing = snapshot_path.read_text(encoding="utf-8")
        if existing != snapshot:
            raise FileExistsError(
                "Existing config snapshot differs: {}".format(snapshot_path)
            )
        return
    snapshot_path.write_text(snapshot, encoding="utf-8")


def build_experiment_folder_name(args):
    return "_".join(
        [
            args.dataset,
            "temperature",
            str(args.temperature),
            "mismatch_ratio",
            str(args.m_ratio),
            "epochs",
            str(args.epochs),
        ]
    )


def train_one_time(args, state_logger):
    utils.fix_random_seeds(args.seed)
    device = torch.device(args.device)
    torch.cuda.set_device(device)
    dataset_train, dataset_test = load_dataset(args)
    args.n_sample = len(dataset_train)
    args.n_sample_test = len(dataset_test)


    data_loader_train, data_loader_test = build_data_loaders(
        dataset_train, dataset_test, args
    )

    model = CorreGenBaselineModel(
        n_views=args.n_views,
        layer_dims=args.encoder_dim,
        temperature=args.temperature,
        n_classes=args.n_classes,
        drop_rate=args.drop_rate,
        beta=args.beta,
        reg=args.reg,
        rho=args.rho,
    )

    model = model.to(device)
    projector = None
    if args.algorithm == "stride_gw":
        from pseudolabel.gw import projector_from_solver_config

        projector = projector_from_solver_config(args.solver)
    optimizer = torch.optim.AdamW(model.parameters(),lr=args.lr,weight_decay=args.weight_decay,betas=(0.9, 0.99))
    scheduler = utils.Ratioschedule(args.warmup_epochs,args.epochs,args.lr,args.min_lr,schedule_type='exp',gamma=0.98)
    infolist = utils.InfoBuffer(args)

    if args.train_id == 0:
        print("job dir: {}".format(os.path.dirname(os.path.realpath(__file__))))
        state_logger.write("Train batch size: {}".format(args.batch_size))
        state_logger.write("Eval batch size: {}".format(args.eval_batch_size))
        state_logger.write("Start time: {}".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M")))
        state_logger.write("Train parameters: {}".format(args).replace(", ", ",\n"))
    state_logger.write("\n>> Start training {}-th initial, seed: {},".format(args.train_id, args.seed))

    train_state_dict = dict()
    for epoch in range(args.start_epoch, args.epochs):
        args.print_this_epoch = (epoch + 1) % args.print_freq == 0 or epoch + 1 == args.epochs
        train_state = train_one_epoch(
            model,
            data_loader_train,
            data_loader_test,
            optimizer,
            device,
            infolist,
            epoch,
            scheduler,
            args,
            projector,
        )
        if train_state is not None:
            train_state_dict[epoch] = train_state
        if args.print_this_epoch:
            kmeans_r = train_state['k-means']
            state_logger.write("Epoch {} --------------------------------".format(epoch))
            state_logger.write("K-means  : ACC_k = {:.4f} NMI_k = {:.4f} ARI_k = {:.4f} F_k = {:.4f} CAR = {:.4f}".format(
                    kmeans_r["acc"], kmeans_r["nmi"], kmeans_r["ari"], kmeans_r["f"], kmeans_r["car"])
            )

    if args.save_ckpt:
            torch.save(model, os.path.join(args.output_dir, f"checkpoint_{args.epochs}"))

    return train_state_dict

def main(args):
    start_time = time.time()
    result_avr_def = {'k-means':   {'nmi': [], 'ari': [], 'f': [], 'acc': [], 'car': []}}
    result_dict = dict()

    batch_scale = args.batch_size / 256
    if args.lr is None:  # only base_lr is specified
        args.lr = args.blr * batch_scale
        args.min_lr = args.min_blr * batch_scale
    experiment_dir = Path(args.output_dir)
    run_id = "run_{}_{}".format(
        datetime.datetime.now().strftime("%Y%m%d_%H%M%S"),
        uuid.uuid4().hex[:8],
    )
    run_root = experiment_dir / run_id
    if run_root.exists():
        raise FileExistsError(
            "Run root already exists; refusing to overwrite: {}".format(run_root)
        )
    run_root.mkdir(parents=False)
    summary_logger = utils.FileLogger(str(run_root / "log_summary.txt"))
    summary_logger.write("Experiment run id: {}".format(run_id))
    summary_logger.write("Independent runs: {}".format(args.num_runs))

    for t in range(args.num_runs):
        run_args = copy.deepcopy(args)
        run_args.train_id = t
        run_args.seed = None if args.seed is None else args.seed + t
        run_args.run_id = run_id
        run_args.output_dir = str(
            build_run_output_dir(experiment_dir, run_id, run_args.seed)
        )
        run_args.visual_dir = os.path.join(run_args.output_dir, "visualize")
        Path(run_args.output_dir).mkdir(parents=True, exist_ok=False)
        Path(run_args.visual_dir).mkdir(parents=True, exist_ok=False)
        write_config_snapshot(run_args)
        run_logger = utils.FileLogger(
            os.path.join(run_args.output_dir, "log_train.txt")
        )
        train_state_dict = train_one_time(run_args, run_logger)
        for epoch, train_state in train_state_dict.items():
            result_dict.setdefault(epoch, copy.deepcopy(result_avr_def))
            for method, metrics in train_state.items():
                for k, v in metrics.items():
                    result_dict[epoch][method][k].append(v)

    for epoch, result_avr in result_dict.items():
        for method, metrics in result_avr.items():
            for k, v in metrics.items():
                if k in ['acc', 'nmi', 'ari', 'car', 'f']:
                    x = np.asarray(v) * 100
                    metrics[k] = [x.mean(), x.std()]

    total_time = time.time() - start_time
    total_time_str = str(datetime.timedelta(seconds=int(total_time)))

    for epoch, result_avr in result_dict.items():
        print("Epoch {}:".format(epoch))
        kmeans_r = result_avr['k-means']
        summary_logger.write(f"Epoch {epoch}: title,acc,nmi,ari,car-------------")
        summary_logger.write('Average K-means Result   : ACC = {:.2f}({:.2f}) NMI = {:.2f}({:.2f}) ARI = {:.2f}({:.2f}) CAR = {:.2f}({:.2f})'
                           .format(*kmeans_r['acc'], *kmeans_r['nmi'], *kmeans_r['ari'], *kmeans_r['car']))
    summary_logger.write("Total training time: {}".format(total_time_str))


if __name__ == "__main__":
    args = get_args_parser()
    args = args.parse_args()

    if args.config_file is not None:
        config_path = resolve_config_path(args.config_file)
        configs = load_config_file(config_path)
        args = vars(args)
        args.update(configs)
        args["config_path"] = str(config_path)
        args = argparse.Namespace(**args)
    finalize_run_count(args)

    if args.validate_config:
        print(yaml.safe_dump(vars(args), sort_keys=True))
        raise SystemExit(0)
    os.environ["OMP_NUM_THREADS"] = str(args.threads)
    torch.set_num_threads(args.threads)

    folder_name = build_experiment_folder_name(args)

    args.output_dir = os.path.join(args.output_dir, args.dataset, folder_name)
    args.visual_dir = os.path.join(args.output_dir, "visualize")
    print(f"Output dir: {args.output_dir}")
    args.embed_dim = args.encoder_dim[0][-1]
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(args.visual_dir).mkdir(parents=True, exist_ok=True)
    write_config_snapshot(args)

    main(args)
