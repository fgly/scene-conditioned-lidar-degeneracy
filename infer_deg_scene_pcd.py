import argparse
import math
import os
import sys
from pathlib import Path

import numpy as np
import torch

try:
    import open3d as o3d
except ImportError:
    raise ImportError("Please install open3d first: pip install open3d")

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
sys.path.append(os.path.join(BASE_DIR, "models"))

from models.deg_scene_model import DegSceneModel, infer_backbone_from_state_dict


CLASS_NAMES = {
    0: "tunnel_like",
    1: "open_like",
    2: "nondeg_or_other",
}


def pc_normalize(xyz: np.ndarray) -> np.ndarray:
    centroid = np.mean(xyz, axis=0)
    xyz = xyz - centroid
    radius = np.max(np.sqrt(np.sum(xyz ** 2, axis=1)))
    return xyz / max(float(radius), 1e-6)


def farthest_point_sample(points: np.ndarray, npoint: int) -> np.ndarray:
    """
    Deterministic numpy FPS.
    points: [N, C]
    """
    n, _ = points.shape
    xyz = points[:, :3]

    centroids = np.zeros((npoint,), dtype=np.int64)
    distance = np.ones((n,), dtype=np.float64) * 1e10

    # fixed start: farthest from origin
    farthest = int(np.argmax(np.sum(xyz ** 2, axis=1)))

    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest]
        dist = np.sum((xyz - centroid) ** 2, axis=1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = int(np.argmax(distance))

    return points[centroids]


def sample_points(points: np.ndarray, num_point: int, use_fps: bool = True) -> np.ndarray:
    n = points.shape[0]

    if n >= num_point:
        if use_fps:
            return farthest_point_sample(points, num_point)
        return points[:num_point]

    # deterministic padding when points are fewer than num_point
    extra_num = num_point - n
    extra_idx = np.arange(extra_num) % n
    return np.concatenate([points, points[extra_idx]], axis=0)


def read_pcd_xyz(pcd_path: str) -> np.ndarray:
    pcd = o3d.io.read_point_cloud(pcd_path)
    xyz = np.asarray(pcd.points, dtype=np.float32)

    if xyz.ndim != 2 or xyz.shape[1] < 3 or xyz.shape[0] == 0:
        raise ValueError(f"Invalid or empty PCD file: {pcd_path}")

    return xyz[:, :3].astype(np.float32)


def bin_to_range(bin_id: int, bin_size: int = 15) -> str:
    start = bin_id * bin_size
    end = start + bin_size
    return f"[{start},{end}) or [{start + 180},{end + 180})"


def main():
    parser = argparse.ArgumentParser("infer_deg_scene_pcd")
    parser.add_argument("--pcd", required=True, help="input .pcd file")
    parser.add_argument("--checkpoint", required=True, help="best_model.pth path")
    parser.add_argument("--num_point", type=int, default=2048)
    parser.add_argument("--input_channel", type=int, default=3)
    parser.add_argument("--num_dir_bins", type=int, default=12)
    parser.add_argument("--backbone", choices=["pointnext"], default=None)
    parser.add_argument("--use_cpu", action="store_true")
    parser.add_argument("--no_fps", action="store_true", help="disable FPS and use first num_point points")
    args = parser.parse_args()

    device = torch.device("cpu" if args.use_cpu or not torch.cuda.is_available() else "cuda")

    xyz = read_pcd_xyz(args.pcd)
    points = sample_points(xyz, args.num_point, use_fps=not args.no_fps)
    points[:, :3] = pc_normalize(points[:, :3])

    # [N, C] -> [1, C, N]
    points_tensor = torch.from_numpy(points.astype(np.float32)).unsqueeze(0).transpose(2, 1).contiguous()
    points_tensor = points_tensor.to(device)

    checkpoint = torch.load(args.checkpoint, map_location=device)

    ckpt_args = checkpoint.get("args", {})
    input_channel = int(ckpt_args.get("input_channel", args.input_channel))
    num_dir_bins = int(ckpt_args.get("num_dir_bins", args.num_dir_bins))
    state = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    ckpt_backbone = ckpt_args.get("backbone") if isinstance(ckpt_args, dict) else getattr(ckpt_args, "backbone", None)
    backbone = args.backbone or ckpt_backbone or infer_backbone_from_state_dict(state)

    model = DegSceneModel(
        input_channel=input_channel,
        num_dir_bins=num_dir_bins,
        backbone=backbone,
    ).to(device)

    model.load_state_dict(state, strict=True)
    model.eval()

    with torch.no_grad():
        outputs = model(points_tensor)

        class_prob = torch.softmax(outputs["class_logits"], dim=-1)[0]
        class_id = int(torch.argmax(class_prob).item())
        class_name = CLASS_NAMES.get(class_id, str(class_id))

        dir_exist_prob = float(torch.sigmoid(outputs["dir_exist_logit"])[0, 0].item())
        rz_prob = float(torch.sigmoid(outputs["rz_logit"])[0, 0].item())

        print("\n========== DegScene Inference ==========")
        print(f"PCD file       : {args.pcd}")
        print(f"Checkpoint     : {args.checkpoint}")
        print(f"num_point      : {args.num_point}")
        print(f"device         : {device}")
        print("----------------------------------------")
        print(f"scene_type     : {class_name}  class_id={class_id}")
        print(f"class_prob     : tunnel={class_prob[0]:.4f}, open={class_prob[1]:.4f}, other={class_prob[2]:.4f}")
        print(f"dir_exist_prob : {dir_exist_prob:.4f}")
        print(f"rz_prob        : {rz_prob:.4f}")

        if "dir_bin_logits" in outputs:
            bin_prob = torch.softmax(outputs["dir_bin_logits"], dim=-1)[0]
            pred_bin = int(torch.argmax(bin_prob).item())
            pred_range = bin_to_range(pred_bin, bin_size=180 // num_dir_bins)

            print("----------------------------------------")
            print(f"dir_bin        : {pred_bin}")
            print(f"dir_range      : {pred_range}")
            print(f"dir_bin_prob   : {float(bin_prob[pred_bin].item()):.4f}")

            topk = min(3, num_dir_bins)
            vals, inds = torch.topk(bin_prob, k=topk)
            print("top bins       :")
            for v, idx in zip(vals.cpu().numpy(), inds.cpu().numpy()):
                print(f"  bin {int(idx):2d}, prob={float(v):.4f}, range={bin_to_range(int(idx), 180 // num_dir_bins)}")

        if "dir_xy_unit" in outputs:
            d = outputs["dir_xy_unit"][0].detach().cpu().numpy()
            angle = math.degrees(math.atan2(float(d[1]), float(d[0]))) % 360
            axis_angle = angle % 180
            print("----------------------------------------")
            print(f"dir_xy_unit    : [{d[0]:.4f}, {d[1]:.4f}]")
            print(f"debug angle    : {angle:.2f} deg, axis_angle={axis_angle:.2f} deg")
            print("direction metric: dir_bin is the primary reported direction output.")

        print("========================================\n")


if __name__ == "__main__":
    main()
