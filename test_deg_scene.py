"""Evaluation entry point for the deg_scene point-cloud model.

The script loads a checkpoint, evaluates a label split, prints accuracy,
macro-F1, confusion matrix, direction/rz accuracy, and tunnel xy-axis angle
error plus direction-bin metrics, then saves a per-sample prediction CSV.
DataLoader points arrive as [B, N, C] and are transposed to [B, C, N] before
model forward.
"""

import argparse
import csv
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(os.path.join(BASE_DIR, "models"))

from data_utils.DegSceneDataLoader import DegSceneDataLoader
from models.deg_scene_model import DegSceneModel, infer_backbone_from_state_dict
from utils.deg_metrics import collect_batch_predictions, summarize_predictions


def parse_args():
    """Parse command-line options for deg_scene evaluation."""

    parser = argparse.ArgumentParser("deg_scene_eval")
    parser.add_argument("--label_path", required=True)
    parser.add_argument("--checkpoint", default=None)
    parser.add_argument("--log_dir", default=None, help="loads log/deg_scene/<log_dir>/checkpoints/best_model.pth by default")
    parser.add_argument("--data_root", default="")
    parser.add_argument("--split", default="test")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--num_point", type=int, default=1024)
    parser.add_argument("--input_channel", type=int, default=3)
    parser.add_argument("--num_dir_bins", type=int, default=12)
    parser.add_argument("--backbone", choices=["pointnext"], default=None)
    parser.add_argument("--use_features", action="store_true")
    parser.add_argument("--use_uniform_sample", action="store_true")
    parser.add_argument("--no_normalize_xyz", action="store_true")
    parser.add_argument("--use_cpu", action="store_true")
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--output_csv", default="deg_scene_predictions.csv")
    parser.add_argument("--summary_json", default="")
    parser.add_argument("--summary_txt", default="")
    parser.add_argument("--summary_csv", default="")
    return parser.parse_args()


def resolve_checkpoint_path(args):
    """Resolve the checkpoint path, defaulting to the run's best model."""

    if args.checkpoint:
        return Path(args.checkpoint)
    if args.log_dir:
        return Path("log/deg_scene") / args.log_dir / "checkpoints" / "best_model.pth"
    raise ValueError("Provide --log_dir or --checkpoint. By default --log_dir loads checkpoints/best_model.pth.")


def print_checkpoint_summary(checkpoint_path: Path, checkpoint: dict) -> None:
    """Print the saved best-checkpoint metadata when available."""

    best_metrics = checkpoint.get("best_metrics", {}) if isinstance(checkpoint, dict) else {}
    print(f"Loaded checkpoint: {checkpoint_path.name}")
    print(f"Checkpoint path: {checkpoint_path}")
    print(f"checkpoint epoch: {checkpoint.get('epoch', 'unknown') if isinstance(checkpoint, dict) else 'unknown'}")
    print(f"best monitor: {checkpoint.get('best_monitor', 'unknown') if isinstance(checkpoint, dict) else 'unknown'}")
    print(f"best tunnelBinAcc: {best_metrics.get('tunnel_dir_bin_accuracy', 'unknown')}")
    print(f"best tunnelBinMAE: {best_metrics.get('tunnel_dir_bin_mae', 'unknown')}")
    print(f"best tunnelBinMAEDeg: {best_metrics.get('tunnel_dir_bin_mae_deg', 'unknown')}")


def to_device(batch, device):
    """Move tensor values in an evaluation batch to the target device.

    Args:
        batch: Dict with ``points`` [B, N, C], labels [B], ``dir_xy_gt`` [B, 2],
            and string metadata.
        device: Torch device.

    Returns:
        Batch dict with tensors on ``device`` and metadata untouched.
    """

    return {key: value.to(device) if torch.is_tensor(value) else value for key, value in batch.items()}


def jsonable(value):
    """Convert NumPy/PyTorch-ish metric values to JSON-serializable objects."""

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [jsonable(val) for val in value]
    return value


def default_summary_path(output_csv: Path, suffix: str) -> Path:
    """Return a summary path beside the prediction CSV."""

    return output_csv.with_name(f"{output_csv.stem}_{suffix}")


def write_summary_files(args, checkpoint_path: Path, checkpoint: dict, dataset_size: int, metrics: dict) -> None:
    """Persist aggregate evaluation metrics beside the prediction CSV."""

    out_csv = Path(args.output_csv)
    best_epoch = checkpoint.get("epoch", "unknown") if isinstance(checkpoint, dict) else "unknown"
    best_monitor = checkpoint.get("best_monitor", "unknown") if isinstance(checkpoint, dict) else "unknown"
    summary = {
        "label_path": str(Path(args.label_path).resolve()),
        "data_root": str(Path(args.data_root).resolve()) if args.data_root else "",
        "split": args.split,
        "dataset_size": int(dataset_size),
        "checkpoint_path": str(checkpoint_path.resolve()),
        "best_epoch": best_epoch,
        "best_monitor": best_monitor,
        "scene_acc": float(metrics["class_accuracy"]),
        "macro_f1": float(metrics["macro_f1"]),
        "confusion_matrix": jsonable(metrics["confusion_matrix"]),
        "dir_exist_accuracy": float(metrics["dir_exist_accuracy"]),
        "rz_accuracy": float(metrics["rz_accuracy"]),
        "tunnel_angle_error_deg": float(metrics["tunnel_angle_error_deg"]),
        "tunnel_dir_top1": float(metrics["tunnel_dir_bin_accuracy"]),
        "tunnel_dir_adjacent": float(metrics["tunnel_dir_bin_adjacent_accuracy"]),
        "tunnel_dir_bin_mae": float(metrics["tunnel_dir_bin_mae"]),
        "tunnel_dir_deg_mae": float(metrics["tunnel_dir_bin_mae_deg"]),
        "prediction_csv": str(out_csv.resolve()),
    }

    summary_json = Path(args.summary_json) if args.summary_json else default_summary_path(out_csv, "summary.json")
    summary_txt = Path(args.summary_txt) if args.summary_txt else default_summary_path(out_csv, "summary.txt")
    summary_csv = Path(args.summary_csv) if args.summary_csv else default_summary_path(out_csv, "summary.csv")
    for path in [summary_json, summary_txt, summary_csv]:
        path.parent.mkdir(parents=True, exist_ok=True)

    with summary_json.open("w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    with summary_txt.open("w", encoding="utf-8") as f:
        for key, value in summary.items():
            f.write(f"{key}: {value}\n")
    csv_row = dict(summary)
    csv_row["confusion_matrix"] = json.dumps(summary["confusion_matrix"])
    with summary_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(csv_row.keys()))
        writer.writeheader()
        writer.writerow(csv_row)
    print(f"Saved evaluation summary JSON to {summary_json}")
    print(f"Saved evaluation summary TXT to {summary_txt}")
    print(f"Saved evaluation summary CSV to {summary_csv}")


def main(args):
    """Run checkpoint evaluation and write the per-sample CSV report."""

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    device = torch.device("cpu" if args.use_cpu or not torch.cuda.is_available() else "cuda")
    dataset = DegSceneDataLoader(
        args.label_path,
        args.num_point,
        args.split,
        args.data_root,
        args.use_features,
        not args.no_normalize_xyz,
        args.use_uniform_sample,
        num_dir_bins=args.num_dir_bins,
        deterministic_sample=True,
    )
    loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False, num_workers=4)
    checkpoint_path = resolve_checkpoint_path(args)
    checkpoint = torch.load(checkpoint_path, map_location=device)
    print_checkpoint_summary(checkpoint_path, checkpoint)
    state = checkpoint.get("model_state_dict", checkpoint)
    ckpt_args = checkpoint.get("args", {}) if isinstance(checkpoint, dict) else {}
    ckpt_backbone = ckpt_args.get("backbone") if isinstance(ckpt_args, dict) else getattr(ckpt_args, "backbone", None)
    backbone = args.backbone or ckpt_backbone or infer_backbone_from_state_dict(state)
    model = DegSceneModel(
        input_channel=args.input_channel,
        num_dir_bins=args.num_dir_bins,
        backbone=backbone,
    ).to(device)
    missing, unexpected = model.load_state_dict(state, strict=False)
    if missing:
        print(f"Checkpoint missing keys: {missing}")
    if unexpected:
        print(f"Checkpoint unexpected keys: {unexpected}")
    model.eval()

    chunks = []
    rows = []
    offset = 0
    with torch.no_grad():
        for batch in tqdm(loader):
            batch_dev = to_device(batch, device)
            # DataLoader emits [B, N, C]; the point-cloud backbone expects [B, C, N].
            points = batch_dev["points"].float().transpose(2, 1).contiguous()  # [B, C, N]
            outputs = model(points)
            chunk = collect_batch_predictions(outputs, batch_dev)
            chunks.append(chunk)
            bsz = points.shape[0]
            sample_ids = batch.get("sample_id", [str(i) for i in range(offset, offset + bsz)])
            for i in range(bsz):
                rows.append({
                    "sample_id": sample_ids[i],
                    "pred_class": int(chunk["pred_class"][i]),
                    "gt_class": int(chunk["gt_class"][i]),
                    "pred_dir_exist": int(chunk["pred_dir_exist"][i]),
                    "gt_dir_exist": int(chunk["gt_dir_exist"][i]),
                    "pred_rz": int(chunk["pred_rz"][i]),
                    "gt_rz": int(chunk["gt_rz"][i]),
                    "pred_dir_x": float(chunk["pred_dir_xy"][i, 0]),
                    "pred_dir_y": float(chunk["pred_dir_xy"][i, 1]),
                    "gt_dir_x": float(chunk["gt_dir_xy"][i, 0]),
                    "gt_dir_y": float(chunk["gt_dir_xy"][i, 1]),
                    "pred_dir_bin": int(chunk["pred_dir_bin"][i]),
                    "gt_dir_bin": int(chunk["gt_dir_bin"][i]),
                    "dir_bin_valid": int(chunk["dir_bin_valid"][i]),
                })
            offset += bsz

    metrics = summarize_predictions(chunks, num_dir_bins=args.num_dir_bins)
    print(f"Class Accuracy: {metrics['class_accuracy']:.6f}")
    print(f"Class Macro-F1: {metrics['macro_f1']:.6f}")
    print("Confusion Matrix (rows=gt, cols=pred):")
    print(metrics["confusion_matrix"])
    print(f"dir_exist Accuracy: {metrics['dir_exist_accuracy']:.6f}")
    print(f"rz Accuracy: {metrics['rz_accuracy']:.6f}")
    print(f"Tunnel dir_xy angle error (deg): {metrics['tunnel_angle_error_deg']:.6f}")
    print(f"Tunnel dir bin Accuracy: {metrics['tunnel_dir_bin_accuracy']:.6f}")
    print(f"Tunnel dir bin Adjacent Accuracy: {metrics['tunnel_dir_bin_adjacent_accuracy']:.6f}")
    print(f"Tunnel dir bin MAE: {metrics['tunnel_dir_bin_mae']:.6f}")
    print(f"Tunnel dir bin MAE (deg): {metrics['tunnel_dir_bin_mae_deg']:.6f}")

    out_path = Path(args.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "sample_id", "pred_class", "gt_class", "pred_dir_exist",
                "gt_dir_exist", "pred_rz", "gt_rz", "pred_dir_x",
                "pred_dir_y", "gt_dir_x", "gt_dir_y", "pred_dir_bin",
                "gt_dir_bin", "dir_bin_valid",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)
    print(f"Saved predictions to {out_path}")
    write_summary_files(args, checkpoint_path, checkpoint, len(dataset), metrics)


if __name__ == "__main__":
    main(parse_args())
