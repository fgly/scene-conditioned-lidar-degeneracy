"""Random-data smoke test for the deg_scene training stack.

This script checks the collated batch fields, [B, N, C] -> [B, C, N] model
input transform, forward output shapes, loss backward, metrics aggregation, and
the empty tunnel-mask branch in ``compute_deg_loss``.
"""

import os
import sys
from typing import Dict

import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "models"))

from models.deg_scene_model import DegSceneModel
from utils.deg_losses import DegLossWeights, compute_deg_loss
from utils.deg_metrics import collect_batch_predictions, summarize_predictions


def make_batch(batch_size: int = 4, num_point: int = 1024, input_channel: int = 3) -> Dict[str, torch.Tensor]:
    """Create one synthetic deg_scene batch.

    Args:
        batch_size: Batch size B.
        num_point: Number of points N.
        input_channel: Point channels C.

    Returns:
        Dict with ``points`` [B, N, C], scalar labels [B], and ``dir_xy_gt``
        [B, 2].
    """

    points = torch.randn(batch_size, num_point, input_channel)
    dir_xy = torch.tensor([[1.0, 0.0], [0.0, 0.0], [0.0, 0.0], [0.0, 1.0]])
    return {
        "points": points,
        "class_gt": torch.tensor([0, 1, 2, 0], dtype=torch.long),
        "dir_exist_gt": torch.tensor([1.0, 1.0, 0.0, 1.0]),
        "dir_xy_gt": dir_xy[:batch_size],
        "dir_xy_valid": torch.tensor([1.0, 0.0, 0.0, 1.0])[:batch_size],
        "dir_bin_gt": torch.tensor([0, 0, 0, 6], dtype=torch.long)[:batch_size],
        "dir_bin_valid": torch.tensor([1.0, 0.0, 0.0, 1.0])[:batch_size],
        "rz_gt": torch.tensor([0.0, 1.0, 0.0, 0.0])[:batch_size],
        "sample_weight": torch.ones(batch_size),
    }


def assert_batch_fields(batch: Dict[str, torch.Tensor]) -> None:
    """Validate smoke-test batch keys and shapes."""

    required = {
        "points",
        "class_gt",
        "dir_exist_gt",
        "dir_xy_gt",
        "dir_xy_valid",
        "dir_bin_gt",
        "dir_bin_valid",
        "rz_gt",
        "sample_weight",
    }
    missing = required.difference(batch)
    if missing:
        raise AssertionError(f"missing batch fields: {sorted(missing)}")
    batch_size = batch["points"].shape[0]
    assert batch["points"].dim() == 3
    assert batch["class_gt"].shape == (batch_size,)
    assert batch["dir_xy_gt"].shape == (batch_size, 2)


def assert_output_shapes(outputs: Dict[str, torch.Tensor], batch_size: int) -> None:
    """Validate ``DegSceneModel`` output tensor shapes."""

    assert outputs["class_logits"].shape == (batch_size, 3)
    assert outputs["dir_exist_logit"].shape == (batch_size, 1)
    assert outputs["dir_xy_unit"].shape == (batch_size, 2)
    assert outputs["dir_bin_logits"].shape == (batch_size, 12)
    assert outputs["rz_logit"].shape == (batch_size, 1)


def run_loss_backward(model: DegSceneModel, batch: Dict[str, torch.Tensor]) -> None:
    """Run forward, loss computation, and backward on one synthetic batch."""

    model.train()
    model.zero_grad(set_to_none=True)
    # DataLoader emits [B, N, C]; the model expects [B, C, N].
    points = batch["points"].float().transpose(2, 1).contiguous()
    outputs = model(points)
    assert_output_shapes(outputs, batch["points"].shape[0])
    losses = compute_deg_loss(outputs, batch, DegLossWeights())
    losses["loss"].backward()
    if not torch.isfinite(losses["loss"]):
        raise AssertionError("loss is not finite")


def run_metrics(model: DegSceneModel, batch: Dict[str, torch.Tensor]) -> None:
    """Run prediction collection and metric aggregation."""

    model.eval()
    with torch.no_grad():
        # [B, N, C] -> [B, C, N] for model forward.
        outputs = model(batch["points"].float().transpose(2, 1).contiguous())
    chunk = collect_batch_predictions(outputs, batch)
    summary = summarize_predictions([chunk])
    for key in [
        "class_accuracy",
        "macro_f1",
        "dir_exist_accuracy",
        "rz_accuracy",
        "tunnel_dir_bin_accuracy",
        "tunnel_dir_bin_mae",
        "tunnel_dir_bin_mae_deg",
    ]:
        if key not in summary:
            raise AssertionError(f"missing metric: {key}")


def run_empty_tunnel_mask_branch(model: DegSceneModel, batch: Dict[str, torch.Tensor]) -> None:
    """Exercise ``L_tun`` when no sample has a valid tunnel xy label."""

    empty_batch = dict(batch)
    empty_batch["class_gt"] = torch.tensor([1, 2, 1, 2], dtype=torch.long)
    empty_batch["dir_xy_valid"] = torch.zeros_like(batch["dir_xy_valid"])
    empty_batch["dir_bin_valid"] = torch.zeros_like(batch["dir_bin_valid"])
    model.train()
    model.zero_grad(set_to_none=True)
    outputs = model(empty_batch["points"].float().transpose(2, 1).contiguous())
    losses = compute_deg_loss(outputs, empty_batch, DegLossWeights())
    losses["loss"].backward()
    if float(losses["L_tun"].detach()) != 0.0:
        raise AssertionError("expected zero tunnel loss for empty tunnel mask")


def main() -> None:
    """Run all deg_scene smoke checks on CPU."""

    torch.manual_seed(7)
    batch = make_batch()
    assert_batch_fields(batch)
    model = DegSceneModel(input_channel=batch["points"].shape[-1], feature_dim=128)
    run_loss_backward(model, batch)
    run_metrics(model, batch)
    run_empty_tunnel_mask_branch(model, batch)
    print("deg_scene smoke test passed")


if __name__ == "__main__":
    main()
