"""Check 15-degree axial direction-bin mapping and training invariants."""

import csv
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(BASE_DIR)

from data_utils.DegSceneDataLoader import DegSceneDataLoader, direction_bin_from_angle_deg
from generate_deg_scene_dataset import balanced_bin_sequence, check_tunnel_bin_balance, tunnel_bin_histogram
from utils.deg_losses import DegLossWeights, build_circular_dir_bin_targets, compute_deg_loss


def check_angle_mapping() -> None:
    """Validate the required angle-to-bin examples."""

    expected = [
        (0.0, 0),
        (14.0, 0),
        (15.0, 1),
        (30.0, 2),
        (179.0, 11),
        (180.0, 0),
        (187.0, 0),
        (195.0, 1),
        (359.0, 11),
    ]
    for angle_deg, want in expected:
        got = direction_bin_from_angle_deg(angle_deg)
        print(f"angle_deg={angle_deg:6.1f} -> bin {got}")
        if got != want:
            raise AssertionError(f"angle {angle_deg} expected bin {want}, got {got}")


def check_non_tunnel_mask() -> None:
    """Ensure open_like and nondeg_or_other do not contribute to L_tun."""

    outputs = {
        "class_logits": torch.zeros(2, 3),
        "dir_exist_logit": torch.zeros(2, 1),
        "dir_xy_unit": torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        "dir_bin_logits": torch.randn(2, 12),
        "rz_logit": torch.zeros(2, 1),
    }
    batch = {
        "class_gt": torch.tensor([1, 2], dtype=torch.long),
        "dir_exist_gt": torch.tensor([1.0, 0.0]),
        "dir_xy_gt": torch.zeros(2, 2),
        "dir_xy_valid": torch.zeros(2),
        "dir_bin_gt": torch.tensor([3, 9], dtype=torch.long),
        "dir_bin_valid": torch.zeros(2),
        "rz_gt": torch.tensor([1.0, 0.0]),
        "sample_weight": torch.ones(2),
    }
    losses = compute_deg_loss(
        outputs,
        batch,
        DegLossWeights(lambda_cls=0.0, lambda_mag=0.0, lambda_tun=1.0, lambda_rz=0.0, lambda_lock=0.0),
    )
    if float(losses["L_tun"].detach()) != 0.0:
        raise AssertionError("expected zero L_tun when no tunnel direction-bin labels are valid")


def write_tiny_dataset(root: Path) -> Path:
    """Create one point cloud and a CSV with train/val/test rows."""

    points_dir = root / "points"
    points_dir.mkdir(parents=True, exist_ok=True)
    points = np.arange(64 * 3, dtype=np.float32).reshape(64, 3)
    np.save(points_dir / "sample.npy", points)

    label_path = root / "labels.csv"
    rows = []
    for split in ["train", "val", "test"]:
        rows.append(
            {
                "sample_id": f"{split}_sample",
                "file_path": "points/sample.npy",
                "scene_type": "tunnel_like",
                "class_gt": 0,
                "dir_x": 1.0,
                "dir_y": 0.0,
                "dir_xy_valid": 1,
                "dir_bin_gt": 0,
                "dir_bin_valid": 1,
                "rz_gt": 0,
                "dir_exist_gt": 1,
                "sample_weight": 1.0,
                "split": split,
            }
        )
    with label_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    return label_path


def check_deterministic_sampling() -> None:
    """Validate deterministic val/test sampling and train random sampling."""

    with tempfile.TemporaryDirectory() as tmp:
        root = Path(tmp)
        label_path = write_tiny_dataset(root)

        val_set = DegSceneDataLoader(
            str(label_path),
            num_point=16,
            split="val",
            root=str(root),
            normalize_xyz=False,
            uniform=False,
            deterministic_sample=True,
        )
        val_a = val_set[0]["points"]
        val_b = val_set[0]["points"]
        if not np.array_equal(val_a, val_b):
            raise AssertionError("deterministic_sample=True should return identical val points")

        test_fps = DegSceneDataLoader(
            str(label_path),
            num_point=16,
            split="test",
            root=str(root),
            normalize_xyz=False,
            uniform=True,
            deterministic_sample=True,
        )
        test_a = test_fps[0]["points"]
        test_b = test_fps[0]["points"]
        if not np.array_equal(test_a, test_b):
            raise AssertionError("deterministic FPS should return identical test points")

        train_set = DegSceneDataLoader(
            str(label_path),
            num_point=16,
            split="train",
            root=str(root),
            normalize_xyz=False,
            uniform=False,
            deterministic_sample=False,
            seed=7,
        )
        train_a = train_set[0]["points"]
        train_b = train_set[0]["points"]
        if np.array_equal(train_a, train_b):
            raise AssertionError("deterministic_sample=False should allow train sampling differences")


def check_circular_smoothing_targets() -> None:
    """Validate circular smoothing weights for boundary bins."""

    targets = build_circular_dir_bin_targets(
        torch.tensor([0, 11]),
        num_dir_bins=12,
        smoothing=0.2,
        dtype=torch.float32,
    )
    if not torch.allclose(targets.sum(dim=1), torch.ones(2)):
        raise AssertionError("smoothed direction-bin targets must sum to 1")
    expected0 = torch.zeros(12)
    expected0[0] = 0.8
    expected0[11] = 0.1
    expected0[1] = 0.1
    expected11 = torch.zeros(12)
    expected11[11] = 0.8
    expected11[10] = 0.1
    expected11[0] = 0.1
    if not torch.allclose(targets[0], expected0) or not torch.allclose(targets[1], expected11):
        raise AssertionError(f"unexpected circular smoothing targets:\n{targets}")


def check_balanced_histograms() -> None:
    """Validate generated tunnel bin histograms are balanced per split."""

    rng = np.random.default_rng(11)
    rows = []
    for split, count in {"train": 25, "val": 14, "test": 9}.items():
        for bin_id in balanced_bin_sequence(count, rng):
            rows.append({"scene_type": "tunnel_like", "split": split, "dir_bin_gt": bin_id})
        check_tunnel_bin_balance(rows, split=split)
        print(f"{split} bin histogram: {tunnel_bin_histogram(rows, split=split).tolist()}")


def main() -> None:
    check_angle_mapping()
    check_non_tunnel_mask()
    check_deterministic_sampling()
    check_circular_smoothing_targets()
    check_balanced_histograms()
    print("dir bin mapping, smoothing, deterministic sampling, and histogram checks passed")


if __name__ == "__main__":
    main()
