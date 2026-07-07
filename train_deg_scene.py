"""Training entry point for the deg_scene point-cloud model.

The DataLoader emits point clouds as [B, N, C]. Before forwarding through
``DegSceneModel`` they are transposed to [B, C, N]. The loop records weighted
loss components and metrics for three-way scene classification, direction
existence, rz, tunnel xy-axis angle error, and tunnel direction-bin metrics.
"""

import argparse
import datetime
import logging
import os
import random
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "models"))

from data_utils.DegSceneDataLoader import DegSceneDataLoader
from models.deg_scene_model import DegSceneModel
from utils.deg_losses import DegLossWeights, compute_deg_loss
from utils.deg_metrics import collect_batch_predictions, summarize_predictions


def parse_args():
    """Parse command-line options for deg_scene training."""

    parser = argparse.ArgumentParser("deg_scene_training")
    parser.add_argument("--label_path", required=True, help="csv/json/npy/pkl pseudo-label file")
    parser.add_argument("--data_root", default="", help="root for relative point cloud paths")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--epoch", type=int, default=100)
    parser.add_argument("--learning_rate", type=float, default=0.0003)
    parser.add_argument("--optimizer", choices=["Adam", "SGD"], default="Adam")
    parser.add_argument("--decay_rate", type=float, default=1e-4)
    parser.add_argument("--num_point", type=int, default=1024)
    parser.add_argument("--input_channel", type=int, default=3)
    parser.add_argument("--num_dir_bins", type=int, default=12)
    parser.add_argument("--backbone", choices=["pointnext"], default="pointnext")
    parser.add_argument("--use_features", action="store_true", help="use channels beyond xyz")
    parser.add_argument("--use_uniform_sample", action="store_true")
    parser.add_argument("--no_normalize_xyz", action="store_true")
    parser.add_argument("--use_cpu", action="store_true")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--log_dir", default=None)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--persistent_workers", action="store_true")
    parser.add_argument("--lambda_cls", type=float, default=0.5)
    parser.add_argument("--lambda_mag", type=float, default=0.2)
    parser.add_argument("--lambda_tun", type=float, default=2.0)
    parser.add_argument("--lambda_rz", type=float, default=0.2)
    parser.add_argument("--lambda_lock", type=float, default=0.1)
    parser.add_argument("--dir_bin_smoothing", type=float, default=0.0)
    parser.add_argument("--dir_neighbor_gamma", type=float, default=0.8)
    return parser.parse_args()


def log_string(logger, text):
    """Write one message to both the logger and stdout."""

    logger.info(text)
    print(text)


def set_random_seed(seed):
    """Seed Python, NumPy, and PyTorch for repeatable split/shuffle behavior."""

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def seed_worker(worker_id):
    """Seed NumPy/random inside DataLoader workers from PyTorch's worker seed."""

    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def to_device(batch, device):
    """Move tensor values in a collated batch to the target device.

    Args:
        batch: Dict returned by PyTorch DataLoader. Tensor fields include
            ``points`` [B, N, C], labels [B], and ``dir_xy_gt`` [B, 2].
        device: Torch device for model execution.

    Returns:
        Batch dict with tensors on ``device`` and metadata left unchanged.
    """

    out = {}
    for key, value in batch.items():
        out[key] = value.to(device) if torch.is_tensor(value) else value
    return out


def format_metrics(prefix, metrics):
    """Format the default deg_scene training metrics for bin direction training."""

    return (
        f"{prefix} acc={metrics['class_accuracy']:.4f} "
        f"macroF1={metrics['macro_f1']:.4f} "
        f"tunnelBinAcc={metrics['tunnel_dir_bin_accuracy']:.4f} "
        f"tunnelBinAdj={metrics['tunnel_dir_bin_adjacent_accuracy']:.4f} "
        f"tunnelBinMAE={metrics['tunnel_dir_bin_mae']:.4f} "
        f"tunnelBinMAEDeg={metrics['tunnel_dir_bin_mae_deg']:.4f}"
    )


def checkpoint_score(metrics):
    """Return a comparable score tuple for best-checkpoint selection."""

    macro_f1 = float(metrics["macro_f1"])
    bin_acc = float(metrics.get("tunnel_dir_bin_accuracy", np.nan))
    bin_mae = float(metrics.get("tunnel_dir_bin_mae", np.nan))
    bin_mae_deg = float(metrics.get("tunnel_dir_bin_mae_deg", np.nan))
    if macro_f1 >= 0.99 and not np.isnan(bin_acc):
        mae_for_score = bin_mae_deg if not np.isnan(bin_mae_deg) else float("inf")
        return (1, bin_acc, -mae_for_score), (
            f"tunnelBinAcc={bin_acc:.4f} tunnelBinMAE={bin_mae:.4f} "
            f"tunnelBinMAEDeg={bin_mae_deg:.4f} macroF1={macro_f1:.4f}"
        )
    return (0, macro_f1, 0.0), f"macroF1={macro_f1:.4f}"


def metrics_for_checkpoint(metrics):
    """Keep checkpoint metrics compact by dropping merged per-sample arrays."""

    compact = {}
    for key, value in metrics.items():
        if key == "merged":
            continue
        if isinstance(value, np.ndarray):
            compact[key] = value.tolist()
        elif isinstance(value, np.generic):
            compact[key] = value.item()
        else:
            compact[key] = value
    return compact


def run_one_epoch(model, loader, optimizer, device, loss_weights):
    """Run one train or evaluation epoch.

    Args:
        model: ``DegSceneModel`` accepting points [B, C, N].
        loader: DataLoader yielding ``points`` [B, N, C] and label tensors.
        optimizer: Optimizer for training, or None for evaluation.
        device: Torch device.
        loss_weights: ``DegLossWeights`` component multipliers.

    Returns:
        ``(avg_losses, metrics)`` where losses are scalar floats and metrics are
        produced by ``utils.deg_metrics.summarize_predictions``.
    """

    training = optimizer is not None
    model.train(training)
    loss_sums = {"loss": 0.0, "L_cls": 0.0, "L_dir": 0.0, "L_mag": 0.0, "L_rz": 0.0, "L_lock": 0.0}
    chunks = []
    count = 0
    for batch in tqdm(loader):
        batch = to_device(batch, device)
        # DataLoader emits [B, N, C]; the point-cloud backbone expects [B, C, N].
        points = batch["points"].float().transpose(2, 1).contiguous()  # [B, C, N]
        if training:
            optimizer.zero_grad()
        outputs = model(points)
        losses = compute_deg_loss(outputs, batch, loss_weights)
        if training:
            losses["loss"].backward()
            optimizer.step()
        bsz = points.shape[0]
        count += bsz
        for key in loss_sums:
            loss_sums[key] += float(losses[key].detach().cpu()) * bsz
        chunks.append(collect_batch_predictions(outputs, batch))
    metrics = summarize_predictions(chunks, num_dir_bins=getattr(model, "num_dir_bins", 12))
    avg_losses = {key: val / max(count, 1) for key, val in loss_sums.items()}
    return avg_losses, metrics


def main(args):
    """Configure data, model, optimizer, scheduler, and run training epochs."""

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    set_random_seed(args.seed)
    device = torch.device("cpu" if args.use_cpu or not torch.cuda.is_available() else "cuda")
    timestr = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M")
    exp_dir = Path("log/deg_scene") / (args.log_dir or timestr)
    checkpoints_dir = exp_dir / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("DegScene")
    logger.setLevel(logging.INFO)
    logger.addHandler(logging.FileHandler(exp_dir / "train.log"))
    log_string(logger, str(args))
    log_string(
        logger,
        "num_point hint: 2048 usually works with batch_size 8 or 16; "
        "4096 usually works with batch_size 4 or 8. Adjust by GPU memory.",
    )

    train_set = DegSceneDataLoader(
        args.label_path,
        args.num_point,
        "train",
        args.data_root,
        args.use_features,
        not args.no_normalize_xyz,
        args.use_uniform_sample,
        seed=args.seed,
        num_dir_bins=args.num_dir_bins,
        deterministic_sample=False,
    )
    val_set = DegSceneDataLoader(
        args.label_path,
        args.num_point,
        "val",
        args.data_root,
        args.use_features,
        not args.no_normalize_xyz,
        args.use_uniform_sample,
        seed=args.seed,
        num_dir_bins=args.num_dir_bins,
        deterministic_sample=True,
    )
    if len(val_set) == 0:
        val_set = DegSceneDataLoader(
            args.label_path,
            args.num_point,
            "test",
            args.data_root,
            args.use_features,
            not args.no_normalize_xyz,
            args.use_uniform_sample,
            seed=args.seed,
            num_dir_bins=args.num_dir_bins,
            deterministic_sample=True,
        )
    generator = torch.Generator()
    generator.manual_seed(args.seed)
    drop_last_train = len(train_set) % args.batch_size == 1 and len(train_set) > args.batch_size
    if drop_last_train:
        log_string(logger, "Dropping the final one-sample training batch to keep BatchNorm stable.")
    train_loader = DataLoader(
        train_set,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        drop_last=drop_last_train,
        worker_init_fn=seed_worker if args.num_workers else None,
        generator=generator,
        persistent_workers=args.persistent_workers and args.num_workers > 0,
    )
    val_loader = (
        DataLoader(
            val_set,
            batch_size=args.batch_size,
            shuffle=False,
            num_workers=args.num_workers,
            worker_init_fn=seed_worker if args.num_workers else None,
            persistent_workers=args.persistent_workers and args.num_workers > 0,
        )
        if len(val_set)
        else None
    )

    model = DegSceneModel(
        input_channel=args.input_channel,
        num_dir_bins=args.num_dir_bins,
        backbone=args.backbone,
    ).to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.decay_rate) if args.optimizer == "Adam" else torch.optim.SGD(model.parameters(), lr=args.learning_rate, momentum=0.9, weight_decay=args.decay_rate)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.7)
    loss_weights = DegLossWeights(
        lambda_cls=args.lambda_cls,
        lambda_mag=args.lambda_mag,
        lambda_tun=args.lambda_tun,
        lambda_rz=args.lambda_rz,
        lambda_lock=args.lambda_lock,
        dir_bin_smoothing=args.dir_bin_smoothing,
        dir_neighbor_gamma=args.dir_neighbor_gamma,
    )

    best_score = None
    for epoch in range(args.epoch):
        log_string(logger, f"Epoch {epoch + 1}/{args.epoch}")
        train_losses, train_metrics = run_one_epoch(model, train_loader, optimizer, device, loss_weights)
        scheduler.step()
        log_string(logger, "Train loss total={loss:.4f} L_cls={L_cls:.4f} L_dir={L_dir:.4f}".format(**train_losses))
        log_string(logger, format_metrics("Train", train_metrics))
        monitor_metrics = train_metrics
        if val_loader is not None:
            with torch.no_grad():
                val_losses, val_metrics = run_one_epoch(model, val_loader, None, device, loss_weights)
            log_string(logger, "Val loss total={loss:.4f} L_cls={L_cls:.4f} L_dir={L_dir:.4f}".format(**val_losses))
            log_string(logger, format_metrics("Val", val_metrics))
            monitor_metrics = val_metrics
        score, score_text = checkpoint_score(monitor_metrics)
        if best_score is None or score >= best_score:
            best_score = score
            best_metrics = metrics_for_checkpoint(monitor_metrics)
            torch.save(
                {
                    "epoch": epoch + 1,
                    "epoch_zero_based": epoch,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_score": best_score,
                    "best_monitor": score_text,
                    "best_metrics": best_metrics,
                    "args": vars(args),
                },
                checkpoints_dir / "best_model.pth",
            )
            log_string(logger, f"Saved best checkpoint: best epoch={epoch + 1}")
            log_string(logger, f"Saved best checkpoint: best monitor={score_text}")
            log_string(
                logger,
                "Saved best checkpoint: "
                f"best tunnelBinAcc={best_metrics.get('tunnel_dir_bin_accuracy', np.nan):.4f} "
                f"best tunnelBinMAE={best_metrics.get('tunnel_dir_bin_mae', np.nan):.4f} "
                f"best tunnelBinMAEDeg={best_metrics.get('tunnel_dir_bin_mae_deg', np.nan):.4f}",
            )
        torch.save(
            {
                "epoch": epoch + 1,
                "epoch_zero_based": epoch,
                "model_state_dict": model.state_dict(),
                "optimizer_state_dict": optimizer.state_dict(),
                "monitor": score_text,
                "metrics": metrics_for_checkpoint(monitor_metrics),
                "args": vars(args),
            },
            checkpoints_dir / "last_model.pth",
        )


if __name__ == "__main__":
    main(parse_args())
