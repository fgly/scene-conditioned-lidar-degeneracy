"""Metrics for deg_scene training and evaluation.

The helpers collect model outputs and batch labels, then report class accuracy,
macro-F1, confusion matrix, direction-existence accuracy, rz accuracy, and
tunnel xy-axis angle error. When direction-bin logits are available, the
helpers also report tunnel direction-bin accuracy and circular bin MAE.
Prediction tensors are accepted with batch shape [B, ...] and converted to
numpy arrays for aggregation.
"""

from typing import Dict, List

import numpy as np
import torch
import torch.nn.functional as F


def macro_f1(pred: np.ndarray, target: np.ndarray, num_classes: int = 3) -> float:
    """Compute unweighted macro-F1 over scene classes.

    Args:
        pred: Predicted class ids with shape [B] or [N].
        target: Ground-truth class ids with shape [B] or [N].
        num_classes: Number of classes in the confusion space.

    Returns:
        Mean F1 across classes as a Python float.
    """

    scores = []
    for cls in range(num_classes):
        tp = np.sum((pred == cls) & (target == cls))
        fp = np.sum((pred == cls) & (target != cls))
        fn = np.sum((pred != cls) & (target == cls))
        denom = 2 * tp + fp + fn
        scores.append(0.0 if denom == 0 else (2 * tp) / denom)
    return float(np.mean(scores))


def confusion_matrix(pred: np.ndarray, target: np.ndarray, num_classes: int = 3) -> np.ndarray:
    """Build a row=ground-truth, column=prediction confusion matrix.

    Args:
        pred: Predicted class ids with shape [B] or [N].
        target: Ground-truth class ids with shape [B] or [N].
        num_classes: Number of scene classes.

    Returns:
        Integer matrix with shape [num_classes, num_classes].
    """

    mat = np.zeros((num_classes, num_classes), dtype=np.int64)
    for gt, pr in zip(target.astype(int), pred.astype(int)):
        if 0 <= gt < num_classes and 0 <= pr < num_classes:
            mat[gt, pr] += 1
    return mat


def angle_error_deg(pred_dir_xy: np.ndarray, gt_dir_xy: np.ndarray, valid: np.ndarray) -> float:
    """Compute tunnel xy-axis angle error in degrees.

    Args:
        pred_dir_xy: Predicted xy unit directions with shape [B, 2].
        gt_dir_xy: Ground-truth xy unit directions with shape [B, 2].
        valid: Boolean or 0/1 mask with shape [B].

    Returns:
        Mean axis angle error in degrees. Returns NaN when no samples are valid.
    """

    mask = valid.astype(bool)
    if not np.any(mask):
        return float("nan")
    pred = pred_dir_xy[mask]
    gt = gt_dir_xy[mask]
    pred = pred / np.clip(np.linalg.norm(pred, axis=1, keepdims=True), 1e-6, None)
    gt = gt / np.clip(np.linalg.norm(gt, axis=1, keepdims=True), 1e-6, None)
    # Tunnel direction is an axis, so +v and -v should have zero angle error.
    cos = np.clip(np.abs(np.sum(pred * gt, axis=1)), -1.0, 1.0)
    return float(np.degrees(np.arccos(cos)).mean())


def circular_bin_mae(pred_bin: np.ndarray, gt_bin: np.ndarray, valid: np.ndarray, num_bins: int = 12) -> float:
    """Compute the mean shortest circular distance between direction bins."""

    mask = valid.astype(bool)
    if not np.any(mask):
        return float("nan")
    diff = np.abs(pred_bin[mask].astype(int) - gt_bin[mask].astype(int))
    diff = np.minimum(diff, int(num_bins) - diff)
    return float(np.mean(diff))


def adjacent_bin_accuracy(pred_bin: np.ndarray, gt_bin: np.ndarray, valid: np.ndarray, num_bins: int = 12) -> float:
    """Compute accuracy allowing the true bin and the two circular neighbors."""

    mask = valid.astype(bool)
    if not np.any(mask):
        return float("nan")
    diff = np.abs(pred_bin[mask].astype(int) - gt_bin[mask].astype(int))
    diff = np.minimum(diff, int(num_bins) - diff)
    return float(np.mean(diff <= 1))


def collect_batch_predictions(outputs: Dict[str, torch.Tensor], batch: Dict[str, torch.Tensor]) -> Dict[str, np.ndarray]:
    """Convert one model batch into numpy prediction/label arrays.

    Args:
        outputs: Model outputs with ``class_logits`` [B, 3],
            ``dir_exist_logit`` [B, 1], ``dir_xy_unit`` [B, 2],
            optional ``dir_bin_logits`` [B, K], and ``rz_logit`` [B, 1].
        batch: Batch labels with ``class_gt`` [B], ``dir_exist_gt`` [B],
            ``dir_xy_gt`` [B, 2], optional ``dir_xy_valid`` [B], and
            ``rz_gt`` [B].

    Returns:
        Dict of numpy arrays keyed by prediction/ground-truth field name.
    """

    pred_class = torch.argmax(outputs["class_logits"], dim=-1)
    pred_dir_exist = (torch.sigmoid(outputs["dir_exist_logit"].view(-1)) >= 0.5).long()
    pred_rz = (torch.sigmoid(outputs["rz_logit"].view(-1)) >= 0.5).long()
    gt_class = batch["class_gt"].view(-1).long()
    gt_dir_exist = batch["dir_exist_gt"].view(-1).long()
    gt_rz = batch["rz_gt"].view(-1).long()
    dir_valid = batch.get("dir_xy_valid", torch.zeros_like(gt_class)).view(-1)
    if "dir_bin_logits" in outputs:
        pred_dir_bin = torch.argmax(outputs["dir_bin_logits"], dim=-1).view(-1)
        dir_bin_valid = batch.get("dir_bin_valid", torch.zeros_like(gt_class)).view(-1)
    else:
        pred_dir_bin = torch.full_like(gt_class, -1)
        dir_bin_valid = torch.zeros_like(gt_class)
    gt_dir_bin = batch.get("dir_bin_gt", torch.zeros_like(gt_class)).view(-1).long()
    return {
        "pred_class": pred_class.detach().cpu().numpy(),
        "gt_class": gt_class.detach().cpu().numpy(),
        "pred_dir_exist": pred_dir_exist.detach().cpu().numpy(),
        "gt_dir_exist": gt_dir_exist.detach().cpu().numpy(),
        "pred_rz": pred_rz.detach().cpu().numpy(),
        "gt_rz": gt_rz.detach().cpu().numpy(),
        "pred_dir_xy": F.normalize(outputs["dir_xy_unit"], p=2, dim=-1, eps=1e-6).detach().cpu().numpy(),
        "gt_dir_xy": batch["dir_xy_gt"].detach().cpu().numpy(),
        "dir_xy_valid": dir_valid.detach().cpu().numpy(),
        "pred_dir_bin": pred_dir_bin.detach().cpu().numpy(),
        "gt_dir_bin": gt_dir_bin.detach().cpu().numpy(),
        "dir_bin_valid": dir_bin_valid.detach().cpu().numpy(),
    }


def summarize_predictions(
    chunks: List[Dict[str, np.ndarray]],
    num_classes: int = 3,
    num_dir_bins: int = 12,
) -> Dict[str, object]:
    """Aggregate collected batch chunks into report metrics.

    Args:
        chunks: List of dictionaries returned by ``collect_batch_predictions``.
        num_classes: Number of scene classes.
        num_dir_bins: Number of axial direction bins over [0, 180).

    Returns:
        Metrics dict containing scalar floats, ``confusion_matrix`` [C, C], and
        the concatenated arrays under ``merged``.
    """

    merged = {key: np.concatenate([chunk[key] for chunk in chunks], axis=0) for key in chunks[0]}
    cls_acc = float(np.mean(merged["pred_class"] == merged["gt_class"]))
    dir_acc = float(np.mean(merged["pred_dir_exist"] == merged["gt_dir_exist"]))
    rz_acc = float(np.mean(merged["pred_rz"] == merged["gt_rz"]))
    tunnel_valid = (merged["gt_class"] == 0) & (merged["dir_xy_valid"] > 0.5)
    tunnel_bin_valid = (merged["gt_class"] == 0) & (merged["dir_bin_valid"] > 0.5)
    if np.any(tunnel_bin_valid):
        bin_acc = float(np.mean(merged["pred_dir_bin"][tunnel_bin_valid] == merged["gt_dir_bin"][tunnel_bin_valid]))
        bin_adj_acc = adjacent_bin_accuracy(
            merged["pred_dir_bin"], merged["gt_dir_bin"], tunnel_bin_valid, num_bins=num_dir_bins
        )
        bin_mae = circular_bin_mae(merged["pred_dir_bin"], merged["gt_dir_bin"], tunnel_bin_valid, num_bins=num_dir_bins)
    else:
        bin_acc = float("nan")
        bin_adj_acc = float("nan")
        bin_mae = float("nan")
    bin_size_deg = 180.0 / float(num_dir_bins)
    return {
        "class_accuracy": cls_acc,
        "macro_f1": macro_f1(merged["pred_class"], merged["gt_class"], num_classes),
        "confusion_matrix": confusion_matrix(merged["pred_class"], merged["gt_class"], num_classes),
        "dir_exist_accuracy": dir_acc,
        "rz_accuracy": rz_acc,
        "tunnel_angle_error_deg": angle_error_deg(merged["pred_dir_xy"], merged["gt_dir_xy"], tunnel_valid),
        "tunnel_dir_bin_accuracy": bin_acc,
        "tunnel_dir_bin_adjacent_accuracy": bin_adj_acc,
        "tunnel_dir_bin_mae": bin_mae,
        "tunnel_dir_bin_mae_deg": bin_mae * bin_size_deg if not np.isnan(bin_mae) else float("nan"),
        "merged": merged,
    }
