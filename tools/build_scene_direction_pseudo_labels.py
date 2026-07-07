"""Build offline deg_scene pseudo labels from single-frame point-cloud lists.

For each listed point cloud, this tool writes CSV fields ``scene_type``,
``dir_x``, ``dir_y``, ``dir_z``, ``dir_xy_valid``, and ``sample_weight``.
The heuristic uses xy spread and vertical spread as an offline bootstrap, then
marks uncertain tunnel directions so training can mask or down-weight them.
"""

import argparse
import csv
import math
from pathlib import Path

import numpy as np

DEFAULT_NUM_DIR_BINS = 12


def direction_bin_from_angle_deg(angle_deg: float, num_dir_bins: int = DEFAULT_NUM_DIR_BINS) -> int:
    """Map a direction angle in degrees to an axial bin over [0, 180)."""

    bin_size = 180.0 / float(num_dir_bins)
    angle = float(angle_deg) % 360.0
    axis_angle = angle % 180.0
    bin_id = int(math.floor(axis_angle / bin_size))
    return min(max(bin_id, 0), int(num_dir_bins) - 1)


def direction_bin_from_xy(dx: float, dy: float, num_dir_bins: int = DEFAULT_NUM_DIR_BINS) -> int:
    """Map an xy direction vector to an axial direction bin."""

    angle_deg = math.degrees(math.atan2(float(dy), float(dx)))
    return direction_bin_from_angle_deg(angle_deg, num_dir_bins)


def direction_bin_range(bin_id: int, num_dir_bins: int = DEFAULT_NUM_DIR_BINS) -> str:
    """Return a readable paired 180-degree-equivalent range for one bin."""

    bin_size = 180.0 / float(num_dir_bins)
    start = int(round(int(bin_id) * bin_size))
    end = int(round((int(bin_id) + 1) * bin_size))
    return f"[{start},{end}) or [{start + 180},{end + 180})"


def load_points(path: str) -> np.ndarray:
    """Load xyz coordinates from a point-cloud file.

    Args:
        path: NPY, NPZ, text/CSV/PTS/XYZ, or BIN point-cloud path.

    Returns:
        XYZ array with shape [N, 3].
    """

    suffix = Path(path).suffix.lower()
    if suffix == ".npy":
        points = np.load(path).astype(np.float32)
    elif suffix == ".npz":
        data = np.load(path)
        key = "points" if "points" in data else data.files[0]
        points = data[key].astype(np.float32)
    elif suffix in {".txt", ".csv", ".pts", ".xyz"}:
        points = np.loadtxt(path, delimiter="," if suffix == ".csv" else None).astype(np.float32)
    elif suffix == ".bin":
        raw = np.fromfile(path, dtype=np.float32)
        if raw.size % 4 == 0:
            points = raw.reshape(-1, 4)
        elif raw.size % 3 == 0:
            points = raw.reshape(-1, 3)
        else:
            raise ValueError(f"Cannot infer channel count for binary point cloud: {path}")
    else:
        raise ValueError(f"Unsupported point cloud file type: {path}")
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"points must be [N, C>=3], got {points.shape}: {path}")
    return points[:, :3]


def classify_cloud(points: np.ndarray, tunnel_anisotropy: float, open_flatness: float, min_dir_norm: float):
    """Assign one conservative deg_scene pseudo label.

    Args:
        points: XYZ point cloud with shape [N, 3].
        tunnel_anisotropy: Minimum xy covariance ratio for tunnel_like.
        open_flatness: Minimum xy/z spread ratio for open_like.
        min_dir_norm: Minimum xy axis norm required for a valid direction.

    Returns:
        Dict with ``scene_type``, direction components, ``dir_xy_valid``, and
        ``sample_weight``.
    """

    xyz = points - np.mean(points, axis=0, keepdims=True)
    xy = xyz[:, :2]
    cov_xy = np.cov(xy.T) + np.eye(2) * 1e-6
    eigvals, eigvecs = np.linalg.eigh(cov_xy)
    order = np.argsort(eigvals)
    small, large = float(eigvals[order[0]]), float(eigvals[order[1]])
    anisotropy = large / max(small, 1e-6)
    z_std = float(np.std(xyz[:, 2]))
    xy_std = float(np.sqrt(np.mean(np.sum(xy * xy, axis=1))))
    flatness = xy_std / max(z_std, 1e-6)

    # Degenerate tunnel direction is the weakly observed xy axis [2].
    direction = eigvecs[:, order[0]]
    dir_norm = float(np.linalg.norm(direction))
    dir_xy_valid = dir_norm > min_dir_norm
    direction = direction / max(dir_norm, 1e-6)

    if anisotropy >= tunnel_anisotropy:
        scene_type = "tunnel_like"
        sample_weight = min(1.0, (anisotropy - tunnel_anisotropy) / tunnel_anisotropy + 0.5)
    elif flatness >= open_flatness:
        scene_type = "open_like"
        sample_weight = min(1.0, (flatness - open_flatness) / open_flatness + 0.5)
        dir_xy_valid = False
    else:
        scene_type = "nondeg_or_other"
        sample_weight = 1.0 - min(0.5, max(anisotropy / tunnel_anisotropy, flatness / open_flatness) * 0.25)
        dir_xy_valid = False
        direction = np.zeros(2, dtype=np.float32)

    if scene_type == "tunnel_like":
        dir_bin_gt = direction_bin_from_xy(float(direction[0]), float(direction[1]))
        dir_bin_valid = 1
        dir_range = direction_bin_range(dir_bin_gt)
    else:
        dir_bin_gt = 0
        dir_bin_valid = 0
        dir_range = ""

    return {
        "scene_type": scene_type,
        "dir_x": float(direction[0]),
        "dir_y": float(direction[1]),
        "dir_z": 0.0,
        "dir_xy_valid": int(dir_xy_valid and scene_type == "tunnel_like"),
        "dir_bin_gt": int(dir_bin_gt),
        "dir_bin_valid": int(dir_bin_valid),
        "dir_range": dir_range,
        "sample_weight": float(np.clip(sample_weight, 0.05, 1.0)),
    }


def read_file_list(path: str):
    """Read a whitespace-delimited point-cloud list.

    Args:
        path: Text file with one point-cloud path per non-empty line.

    Returns:
        List of path strings. Extra columns are ignored.
    """

    with open(path) as f:
        return [line.strip().split()[0] for line in f if line.strip()]


def main():
    """Generate pseudo-label CSV rows for all point clouds in ``--file_list``."""

    parser = argparse.ArgumentParser("build_scene_direction_pseudo_labels")
    parser.add_argument("--file_list", required=True, help="text file with one point cloud path per line")
    parser.add_argument("--output", required=True, help="output csv path")
    parser.add_argument("--root", default="", help="optional root for relative paths in file_list")
    parser.add_argument("--split", default="train", help="split value written to every row")
    parser.add_argument("--tunnel_anisotropy", type=float, default=4.0)
    parser.add_argument("--open_flatness", type=float, default=8.0)
    parser.add_argument("--min_dir_norm", type=float, default=1e-4)
    args = parser.parse_args()

    rows = []
    for item in read_file_list(args.file_list):
        full_path = item if Path(item).is_absolute() else str(Path(args.root) / item)
        points = load_points(full_path)
        label = classify_cloud(points, args.tunnel_anisotropy, args.open_flatness, args.min_dir_norm)
        rows.append({"file_path": item, "split": args.split, **label})

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    with open(out, "w", newline="") as f:
        fieldnames = [
            "file_path", "split", "scene_type", "dir_x", "dir_y", "dir_z",
            "dir_xy_valid", "dir_bin_gt", "dir_bin_valid", "dir_range",
            "sample_weight",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {len(rows)} pseudo labels to {out}")


if __name__ == "__main__":
    main()
