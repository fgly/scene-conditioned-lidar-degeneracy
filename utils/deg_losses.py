"""Losses for deg_scene classification and conditional direction heads.

``compute_deg_loss`` keeps the auxiliary direction-existence, rz, and lock
terms available for logging/debugging, but the training objective is the scene
classification loss plus the tunnel direction loss only.
When ``dir_bin_logits`` and bin labels are present, the tunnel direction term
uses circular tolerance over the true and neighboring direction bins.
"""

from dataclasses import dataclass
from typing import Dict, Optional

import torch
import torch.nn.functional as F


@dataclass
class DegLossWeights:
    """Scalar multipliers for each deg_scene loss component."""

    lambda_cls: float = 1.0
    lambda_mag: float = 0.5
    lambda_tun: float = 0.5
    lambda_rz: float = 0.5
    lambda_lock: float = 0.2
    dir_bin_smoothing: float = 0.0
    dir_neighbor_gamma: float = 0.8


def build_circular_dir_bin_targets(
    dir_bin_gt: torch.Tensor,
    num_dir_bins: int,
    smoothing: float,
    dtype: torch.dtype,
) -> torch.Tensor:
    """Build circular label-smoothed targets for axial direction bins."""

    smoothing = min(max(float(smoothing), 0.0), 1.0)
    dir_bin_gt = dir_bin_gt.long().view(-1)
    target = torch.zeros(dir_bin_gt.shape[0], num_dir_bins, device=dir_bin_gt.device, dtype=dtype)
    batch_idx = torch.arange(dir_bin_gt.shape[0], device=dir_bin_gt.device)
    left_bin = (dir_bin_gt - 1) % num_dir_bins
    right_bin = (dir_bin_gt + 1) % num_dir_bins
    target[batch_idx, dir_bin_gt] = 1.0 - smoothing
    target[batch_idx, left_bin] += smoothing * 0.5
    target[batch_idx, right_bin] += smoothing * 0.5
    return target


def circular_dir_bin_cross_entropy(
    dir_bin_logits: torch.Tensor,
    dir_bin_gt: torch.Tensor,
    smoothing: float,
) -> torch.Tensor:
    """Per-sample circular direction-bin loss with optional label smoothing."""

    if smoothing <= 0.0:
        return F.cross_entropy(dir_bin_logits, dir_bin_gt, reduction="none")
    log_prob = F.log_softmax(dir_bin_logits, dim=-1)
    target = build_circular_dir_bin_targets(
        dir_bin_gt,
        num_dir_bins=dir_bin_logits.shape[-1],
        smoothing=smoothing,
        dtype=log_prob.dtype,
    )
    return -(target * log_prob).sum(dim=-1)


def circular_tolerance_dir_loss(
    dir_logits: torch.Tensor,
    dir_bin_gt: torch.Tensor,
    dir_valid: torch.Tensor,
    gamma: float = 0.8,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Circular tolerant direction-bin loss for valid tunnel-like samples."""

    probs = F.softmax(dir_logits, dim=1)
    valid_mask = dir_valid.to(device=dir_logits.device).view(-1) > 0
    if valid_mask.sum() == 0:
        return dir_logits.sum() * 0.0

    num_dir_bins = dir_logits.shape[1]
    idx = dir_bin_gt.to(device=dir_logits.device).long().view(-1)[valid_mask]
    idx = idx.clamp(0, num_dir_bins - 1)
    probs_valid = probs[valid_mask]

    idx_left = (idx - 1) % num_dir_bins
    idx_right = (idx + 1) % num_dir_bins

    p_true = probs_valid.gather(1, idx[:, None]).squeeze(1)
    p_left = probs_valid.gather(1, idx_left[:, None]).squeeze(1)
    p_right = probs_valid.gather(1, idx_right[:, None]).squeeze(1)

    score = p_true + float(gamma) * (p_left + p_right)
    score = torch.clamp(score, min=float(eps), max=1.0)
    return -torch.log(score).mean()


def _as_sample_weight(sample_weight: Optional[torch.Tensor], ref: torch.Tensor) -> torch.Tensor:
    """Return nonnegative sample weights with shape [B] on ``ref`` device."""

    if sample_weight is None:
        return torch.ones(ref.shape[0], device=ref.device, dtype=ref.dtype)
    weight = sample_weight.to(device=ref.device, dtype=ref.dtype).view(-1)
    return weight.clamp_min(0.0)


def _weighted_mean(values: torch.Tensor, weights: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
    """Compute a weighted mean for per-sample tensors of matching shape."""

    weights = weights.to(device=values.device, dtype=values.dtype).view_as(values)
    denom = weights.sum().clamp_min(eps)
    return (values * weights).sum() / denom


def compute_deg_loss(
    outputs: Dict[str, torch.Tensor],
    batch: Dict[str, torch.Tensor],
    weights: Optional[DegLossWeights] = None,
) -> Dict[str, torch.Tensor]:
    """Compute weighted total loss and individual components.

    Args:
        outputs: Model output dict with ``class_logits`` [B, 3],
            ``dir_exist_logit`` [B, 1], ``dir_xy_unit`` [B, 2], and
            ``rz_logit`` [B, 1]. Optional ``dir_bin_logits`` [B, K] enables
            direction-bin classification for ``L_tun``.
        batch: Supervision dict with the fields listed below.
        weights: Optional component multipliers.

    Batch fields and shapes:
        class_gt: [B]
        dir_exist_gt: [B] or [B, 1]
        dir_xy_gt: [B, 2]
        dir_xy_valid: [B] or [B, 1], optional
        dir_bin_gt: [B], optional
        dir_bin_valid: [B] or [B, 1], optional
        rz_gt: [B] or [B, 1]
        sample_weight: [B] or [B, 1], optional

    Returns:
        Dict with scalar tensors ``loss``, ``L_cls``, ``L_dir``, ``L_mag``,
        ``L_tun``, ``L_rz``, and ``L_lock``. ``L_tun`` is kept as an alias for
        the direction loss.
    """

    weights = weights or DegLossWeights()
    class_logits = outputs["class_logits"]  # [B, 3]
    dir_exist_logit = outputs["dir_exist_logit"].view(-1)  # [B]
    pred_dir_xy = F.normalize(outputs["dir_xy_unit"], p=2, dim=-1, eps=1e-6)  # [B, 2]
    rz_logit = outputs["rz_logit"].view(-1)  # [B]

    class_gt = batch["class_gt"].to(class_logits.device).long().view(-1)  # [B]
    dir_exist_gt = batch["dir_exist_gt"].to(class_logits.device, class_logits.dtype).view(-1)
    rz_gt = batch["rz_gt"].to(class_logits.device, class_logits.dtype).view(-1)
    sample_weight = _as_sample_weight(batch.get("sample_weight"), class_logits)

    loss_cls = F.cross_entropy(class_logits, class_gt)

    mag_per_sample = F.binary_cross_entropy_with_logits(
        dir_exist_logit, dir_exist_gt, reduction="none"
    )
    loss_mag = _weighted_mean(mag_per_sample, sample_weight)

    rz_per_sample = F.binary_cross_entropy_with_logits(rz_logit, rz_gt, reduction="none")
    loss_rz = _weighted_mean(rz_per_sample, sample_weight)

    dir_xy_gt = batch.get("dir_xy_gt")
    if dir_xy_gt is None:
        dir_xy_gt = torch.zeros_like(pred_dir_xy)
    dir_xy_gt = F.normalize(dir_xy_gt.to(class_logits.device, class_logits.dtype), p=2, dim=-1, eps=1e-6)
    dir_xy_valid = batch.get("dir_xy_valid")
    if dir_xy_valid is None:
        dir_xy_valid = torch.ones_like(class_gt, dtype=class_logits.dtype, device=class_logits.device)
    else:
        dir_xy_valid = dir_xy_valid.to(class_logits.device, class_logits.dtype).view(-1)
    use_dir_bins = (
        "dir_bin_logits" in outputs
        and "dir_bin_gt" in batch
        and "dir_bin_valid" in batch
    )
    if use_dir_bins:
        dir_bin_logits = outputs["dir_bin_logits"]  # [B, num_dir_bins]
        dir_bin_gt = batch["dir_bin_gt"].to(class_logits.device).long().view(-1)
        dir_bin_gt = dir_bin_gt.clamp(0, dir_bin_logits.shape[-1] - 1)
        dir_bin_valid = batch["dir_bin_valid"].to(class_logits.device, class_logits.dtype).view(-1)
        loss_dir = circular_tolerance_dir_loss(
            dir_bin_logits,
            dir_bin_gt,
            dir_bin_valid,
            gamma=weights.dir_neighbor_gamma,
        )
    else:
        tunnel_mask = (class_gt == 0).to(class_logits.dtype) * dir_xy_valid
        tun_weight = sample_weight * tunnel_mask
        # Fallback for old checkpoints/scripts: +v and -v are equivalent.
        tun_per_sample = 1.0 - torch.abs(torch.sum(pred_dir_xy * dir_xy_gt, dim=-1))
        loss_dir = _weighted_mean(tun_per_sample, tun_weight) if tun_weight.sum() > 0 else class_logits.sum() * 0.0

    prob = torch.softmax(class_logits, dim=-1)
    # p_t/p_o/p_n: tunnel/open/nondeg probabilities; m: direction exists;
    # r: rz/open-like score. L_lock softly aligns conditional heads and class.
    p_t, p_o, p_n = prob[:, 0], prob[:, 1], prob[:, 2]
    m = torch.sigmoid(dir_exist_logit)
    r = torch.sigmoid(rz_logit)
    lock_terms = (
        (m - (p_t + p_o)).pow(2)
        + p_t * r.pow(2)
        + p_o * (1.0 - r).pow(2)
        + p_n * m.pow(2)
    )
    loss_lock = _weighted_mean(lock_terms, sample_weight)

    total = weights.lambda_cls * loss_cls + weights.lambda_tun * loss_dir
    return {
        "loss": total,
        "L_cls": loss_cls,
        "L_dir": loss_dir,
        "L_mag": loss_mag,
        "L_tun": loss_dir,
        "L_rz": loss_rz,
        "L_lock": loss_lock,
    }
