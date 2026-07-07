#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate a synthetic deg_scene point-cloud dataset.

Scene classes:
  0 tunnel_like       : long straight tunnel/corridor; XY degeneracy direction is tunnel axis.
  1 open_like         : dominant ground plane with very sparse clutter; mainly Rz/yaw degeneracy.
  2 nondeg_or_other   : geometrically rich scene with random objects, walls, cylinders, panels, etc.

Output:
  out_dir/
    points/*.npy                  # [N, 3], float32, directly readable by DegSceneDataLoader
    pcd_preview/*.pcd             # optional, for visualization only
    labels/deg_scene_labels.csv   # all samples with split column
    labels/deg_scene_train.csv
    labels/deg_scene_val.csv
    labels/deg_scene_test.csv

Minimal usage:
  python generate_deg_scene_dataset.py --out_dir ./deg_scene_synth --num_each 100 --write_pcd

If ./deg1.pcd or ./nodeg2.pcd exists, the script uses them only to estimate rough scene scale.
"""

from __future__ import annotations

import argparse
import csv
import math
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

SCENE_TYPE_TO_CLASS = {
    "tunnel_like": 0,
    "open_like": 1,
    "nondeg_or_other": 2,
}
DEFAULT_NUM_DIR_BINS = 12


def direction_bin_from_angle_deg(angle_deg: float, num_dir_bins: int = DEFAULT_NUM_DIR_BINS) -> int:
    """Map a direction angle in degrees to an axial bin over [0, 180)."""

    bin_size = 180.0 / float(num_dir_bins)
    angle = float(angle_deg) % 360.0
    axis_angle = angle % 180.0
    bin_id = int(math.floor(axis_angle / bin_size))
    return min(max(bin_id, 0), int(num_dir_bins) - 1)


def direction_bin_range(bin_id: int, num_dir_bins: int = DEFAULT_NUM_DIR_BINS) -> str:
    """Return a readable paired 180-degree-equivalent range for one bin."""

    bin_size = 180.0 / float(num_dir_bins)
    start = int(round(int(bin_id) * bin_size))
    end = int(round((int(bin_id) + 1) * bin_size))
    return f"[{start},{end}) or [{start + 180},{end + 180})"


def ensure_dir(path: str | Path) -> Path:
    p = Path(path)
    p.mkdir(parents=True, exist_ok=True)
    return p


def read_ascii_pcd_xyz(path: str | Path) -> np.ndarray:
    """Read ASCII PCD with x/y/z fields."""
    path = Path(path)
    with path.open("r", encoding="utf-8", errors="ignore") as f:
        lines = f.readlines()

    data_start = None
    fields = None
    for i, line in enumerate(lines):
        s = line.strip()
        if s.startswith("FIELDS"):
            fields = s.split()[1:]
        if s.startswith("DATA"):
            if "ascii" not in s.lower():
                raise ValueError(f"Only ASCII PCD is supported by this script: {path}")
            data_start = i + 1
            break

    if data_start is None:
        raise ValueError(f"Cannot find DATA ascii header in {path}")
    if fields is None:
        raise ValueError(f"Cannot find FIELDS header in {path}")
    if not all(k in fields for k in ["x", "y", "z"]):
        raise ValueError(f"PCD must contain x/y/z fields: {path}")

    x_id, y_id, z_id = fields.index("x"), fields.index("y"), fields.index("z")
    pts = []
    for line in lines[data_start:]:
        if not line.strip():
            continue
        vals = line.split()
        if len(vals) <= max(x_id, y_id, z_id):
            continue
        pts.append([float(vals[x_id]), float(vals[y_id]), float(vals[z_id])])

    points = np.asarray(pts, dtype=np.float32)
    if points.ndim != 2 or points.shape[1] != 3 or points.shape[0] == 0:
        raise ValueError(f"Failed to read valid xyz points from {path}")
    return remove_invalid_and_origin(points)


def write_ascii_pcd_xyz(path: str | Path, points: np.ndarray) -> None:
    """Write [N,3] xyz points to ASCII PCD for CloudCompare/Open3D preview."""
    path = Path(path)
    points = np.asarray(points[:, :3], dtype=np.float32)
    with path.open("w", encoding="utf-8") as f:
        f.write("# .PCD v0.7 - Point Cloud Data file format\n")
        f.write("VERSION 0.7\n")
        f.write("FIELDS x y z\n")
        f.write("SIZE 4 4 4\n")
        f.write("TYPE F F F\n")
        f.write("COUNT 1 1 1\n")
        f.write(f"WIDTH {points.shape[0]}\n")
        f.write("HEIGHT 1\n")
        f.write("VIEWPOINT 0 0 0 1 0 0 0\n")
        f.write(f"POINTS {points.shape[0]}\n")
        f.write("DATA ascii\n")
        np.savetxt(f, points, fmt="%.6f %.6f %.6f")


def remove_invalid_and_origin(points: np.ndarray) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    valid = np.isfinite(points).all(axis=1)
    points = points[valid]
    if len(points) == 0:
        return points
    nonzero = np.linalg.norm(points[:, :3], axis=1) > 1e-6
    if nonzero.sum() > 100:
        points = points[nonzero]
    return points


def robust_extent(points: Optional[np.ndarray], default: Tuple[float, float, float]) -> Tuple[float, float, float]:
    """Estimate robust xyz extents from a reference cloud, with fallback defaults."""
    if points is None or len(points) < 100:
        return default
    pts = remove_invalid_and_origin(points)
    lo = np.percentile(pts[:, :3], 2, axis=0)
    hi = np.percentile(pts[:, :3], 98, axis=0)
    ext = np.maximum(hi - lo, 1e-3)
    return float(ext[0]), float(ext[1]), float(ext[2])


def rotz_xy(xy: np.ndarray, angle_deg: float) -> np.ndarray:
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    R = np.array([[c, -s], [s, c]], dtype=np.float32)
    return xy @ R.T


def rotz(points: np.ndarray, angle_deg: float, center: Optional[Tuple[float, float]] = None) -> np.ndarray:
    out = points.copy()
    if center is None:
        out[:, 0:2] = rotz_xy(out[:, 0:2], angle_deg)
    else:
        c = np.asarray(center, dtype=np.float32)[None, :]
        out[:, 0:2] = rotz_xy(out[:, 0:2] - c, angle_deg) + c
    return out


def resample_points(points: np.ndarray, num_points: int, rng: np.random.Generator) -> np.ndarray:
    points = np.asarray(points, dtype=np.float32)
    points = remove_invalid_and_origin(points)
    if len(points) == 0:
        raise ValueError("Cannot resample an empty point cloud.")
    replace = len(points) < num_points
    idx = rng.choice(len(points), size=num_points, replace=replace)
    return points[idx].astype(np.float32)


def sample_rect_plane(
    n: int,
    fixed_axis: str,
    fixed_value: float,
    range_a: Tuple[float, float],
    range_b: Tuple[float, float],
    rng: np.random.Generator,
    noise: float = 0.01,
) -> np.ndarray:
    if n <= 0:
        return np.empty((0, 3), dtype=np.float32)
    a = rng.uniform(range_a[0], range_a[1], n)
    b = rng.uniform(range_b[0], range_b[1], n)
    pts = np.zeros((n, 3), dtype=np.float32)
    if fixed_axis == "x":
        pts[:, 0] = fixed_value
        pts[:, 1] = a
        pts[:, 2] = b
    elif fixed_axis == "y":
        pts[:, 0] = a
        pts[:, 1] = fixed_value
        pts[:, 2] = b
    elif fixed_axis == "z":
        pts[:, 0] = a
        pts[:, 1] = b
        pts[:, 2] = fixed_value
    else:
        raise ValueError(f"Unknown fixed_axis: {fixed_axis}")
    pts += rng.normal(0.0, noise, pts.shape).astype(np.float32)
    return pts


def sample_oriented_box_surface(
    center: Tuple[float, float, float],
    size: Tuple[float, float, float],
    yaw_deg: float,
    n: int,
    rng: np.random.Generator,
    noise: float = 0.01,
) -> np.ndarray:
    """Sample top and side faces of a box, then rotate it around its center."""
    if n <= 0:
        return np.empty((0, 3), dtype=np.float32)
    cx, cy, cz = center
    sx, sy, sz = size
    counts = rng.multinomial(n, [0.22, 0.195, 0.195, 0.195, 0.195])
    faces = [
        sample_rect_plane(counts[0], "z", sz / 2, (-sx / 2, sx / 2), (-sy / 2, sy / 2), rng, noise),
        sample_rect_plane(counts[1], "x", -sx / 2, (-sy / 2, sy / 2), (-sz / 2, sz / 2), rng, noise),
        sample_rect_plane(counts[2], "x", sx / 2, (-sy / 2, sy / 2), (-sz / 2, sz / 2), rng, noise),
        sample_rect_plane(counts[3], "y", -sy / 2, (-sx / 2, sx / 2), (-sz / 2, sz / 2), rng, noise),
        sample_rect_plane(counts[4], "y", sy / 2, (-sx / 2, sx / 2), (-sz / 2, sz / 2), rng, noise),
    ]
    pts = np.vstack(faces)
    pts = rotz(pts, yaw_deg)
    pts[:, 0] += cx
    pts[:, 1] += cy
    pts[:, 2] += cz
    return pts.astype(np.float32)


def sample_cylinder_surface(
    center: Tuple[float, float, float],
    radius: float,
    height: float,
    n: int,
    rng: np.random.Generator,
    noise: float = 0.01,
) -> np.ndarray:
    """Sample a vertical cylinder/barrel/pillar surface."""
    if n <= 0:
        return np.empty((0, 3), dtype=np.float32)
    cx, cy, cz = center
    n_side = int(n * 0.75)
    n_top = n - n_side

    theta = rng.uniform(-math.pi, math.pi, n_side)
    z = rng.uniform(cz - height / 2, cz + height / 2, n_side)
    side = np.column_stack([
        cx + radius * np.cos(theta),
        cy + radius * np.sin(theta),
        z,
    ])

    theta2 = rng.uniform(-math.pi, math.pi, n_top)
    r2 = radius * np.sqrt(rng.uniform(0.0, 1.0, n_top))
    top = np.column_stack([
        cx + r2 * np.cos(theta2),
        cy + r2 * np.sin(theta2),
        np.full(n_top, cz + height / 2),
    ])
    pts = np.vstack([side, top]).astype(np.float32)
    pts += rng.normal(0.0, noise, pts.shape).astype(np.float32)
    return pts


def sample_sphere_surface(
    center: Tuple[float, float, float],
    radius: float,
    n: int,
    rng: np.random.Generator,
    noise: float = 0.01,
) -> np.ndarray:
    """Sample a partial sphere/dome-like object above the ground."""
    if n <= 0:
        return np.empty((0, 3), dtype=np.float32)
    cx, cy, cz = center
    theta = rng.uniform(-math.pi, math.pi, n)
    u = rng.uniform(0.0, 1.0, n)  # upper half only
    phi = np.arccos(u)
    pts = np.column_stack([
        cx + radius * np.sin(phi) * np.cos(theta),
        cy + radius * np.sin(phi) * np.sin(theta),
        cz + radius * np.cos(phi),
    ]).astype(np.float32)
    pts += rng.normal(0.0, noise, pts.shape).astype(np.float32)
    return pts


def sample_slanted_panel(
    center: Tuple[float, float, float],
    width: float,
    height: float,
    yaw_deg: float,
    pitch_deg: float,
    n: int,
    rng: np.random.Generator,
    noise: float = 0.01,
) -> np.ndarray:
    """Sample a slanted board/ramp, adding non-axis-aligned constraints."""
    if n <= 0:
        return np.empty((0, 3), dtype=np.float32)
    cx, cy, cz = center
    u = rng.uniform(-width / 2, width / 2, n)
    v = rng.uniform(-height / 2, height / 2, n)
    pitch = math.radians(pitch_deg)
    # local panel: x=u, z=v*cos, y=v*sin produces a tilted vertical/ramp-like surface
    local = np.column_stack([u, v * math.sin(pitch), v * math.cos(pitch)]).astype(np.float32)
    local = rotz(local, yaw_deg)
    local[:, 0] += cx
    local[:, 1] += cy
    local[:, 2] += cz
    local += rng.normal(0.0, noise, local.shape).astype(np.float32)
    return local


def make_tunnel_cloud(
    angle_deg: float,
    num_points: int,
    rng: np.random.Generator,
    length: float = 28.0,
    width: float = 4.0,
    height: float = 3.0,
    noise: float = 0.012,
    clutter_ratio: float = 0.008,
) -> np.ndarray:
    """Generate a clean long rectangular tunnel and rotate it in XY."""
    n_clutter = max(0, int(num_points * clutter_ratio))
    n_main = num_points - n_clutter

    n_floor = int(n_main * 0.34)
    n_ceil = int(n_main * 0.14)
    n_left = int(n_main * 0.26)
    n_right = n_main - n_floor - n_ceil - n_left

    x_range = (-length / 2.0, length / 2.0)
    y_range = (-width / 2.0, width / 2.0)
    z_range = (0.0, height)

    pts = np.vstack([
        sample_rect_plane(n_floor, "z", 0.0, x_range, y_range, rng, noise),
        sample_rect_plane(n_ceil, "z", height, x_range, y_range, rng, noise),
        sample_rect_plane(n_left, "y", -width / 2.0, x_range, z_range, rng, noise),
        sample_rect_plane(n_right, "y", width / 2.0, x_range, z_range, rng, noise),
    ])

    # Very sparse internal clutter only; avoid overwhelming the tunnel geometry.
    if n_clutter > 0:
        # Mostly near floor, like tiny roughness/rocks, not full-volume random noise.
        clutter = np.column_stack([
            rng.uniform(-length / 2.0, length / 2.0, n_clutter),
            rng.uniform(-width / 2.0, width / 2.0, n_clutter),
            rng.uniform(0.02, min(0.45, height), n_clutter),
        ]).astype(np.float32)
        pts = np.vstack([pts, clutter])

    pts = rotz(pts, angle_deg)
    return resample_points(pts, num_points, rng)


def make_open_plane_cloud(
    num_points: int,
    rng: np.random.Generator,
    size: float = 26.0,
    noise: float = 0.010,
    clutter_ratio: float = 0.025,
) -> np.ndarray:
    """Generate an open-like scene: dominant ground plane plus very sparse clutter."""
    n_clutter = max(0, int(num_points * clutter_ratio))
    n_plane = num_points - n_clutter

    plane = np.column_stack([
        rng.uniform(-size / 2.0, size / 2.0, n_plane),
        rng.uniform(-size / 2.0, size / 2.0, n_plane),
        rng.normal(0.0, noise, n_plane),
    ]).astype(np.float32)

    # Sparse low clutter; keep it weak so the scene still behaves open/planar.
    if n_clutter > 0:
        clutter = np.column_stack([
            rng.uniform(-size / 2.0, size / 2.0, n_clutter),
            rng.uniform(-size / 2.0, size / 2.0, n_clutter),
            rng.uniform(0.05, 0.75, n_clutter),
        ]).astype(np.float32)
        pts = np.vstack([plane, clutter])
    else:
        pts = plane
    return resample_points(pts, num_points, rng)


def make_nondeg_cloud(
    num_points: int,
    rng: np.random.Generator,
    room_size: float = 13.0,
    height: float = 3.5,
    noise: float = 0.012,
) -> np.ndarray:
    """Generate random, geometrically rich non-degenerate scenes.

    Improvements over the old version:
      - random number of obstacles, not fixed at 4;
      - multiple shape types: boxes, cylinders, domes/spheres, slanted panels;
      - randomly oriented objects and partial walls/corners;
      - scene layout changes every sample.
    """
    half = room_size / 2.0

    n_floor = int(num_points * rng.uniform(0.16, 0.24))
    n_wall_total = int(num_points * rng.uniform(0.25, 0.38))
    n_obj_total = num_points - n_floor - n_wall_total

    parts: List[np.ndarray] = []
    parts.append(sample_rect_plane(n_floor, "z", 0.0, (-half, half), (-half, half), rng, noise))

    # Use 2 to 4 boundary walls, sometimes partial, to avoid one fixed room template.
    wall_candidates = ["x-", "x+", "y-", "y+"]
    rng.shuffle(wall_candidates)
    n_walls = int(rng.integers(2, 5))
    wall_types = wall_candidates[:n_walls]
    wall_counts = rng.multinomial(n_wall_total, np.ones(n_walls) / n_walls)
    for wt, n in zip(wall_types, wall_counts):
        # partial wall extent differs each time
        span = float(rng.uniform(room_size * 0.45, room_size * 0.95))
        shift = float(rng.uniform(-room_size * 0.15, room_size * 0.15))
        if wt == "x-":
            parts.append(sample_rect_plane(int(n), "x", -half, (shift - span / 2, shift + span / 2), (0.0, height), rng, noise))
        elif wt == "x+":
            parts.append(sample_rect_plane(int(n), "x", half, (shift - span / 2, shift + span / 2), (0.0, height), rng, noise))
        elif wt == "y-":
            parts.append(sample_rect_plane(int(n), "y", -half, (shift - span / 2, shift + span / 2), (0.0, height), rng, noise))
        else:
            parts.append(sample_rect_plane(int(n), "y", half, (shift - span / 2, shift + span / 2), (0.0, height), rng, noise))

    # Random object count and mixed shapes.
    n_objects = int(rng.integers(6, 18))
    obj_counts = rng.multinomial(n_obj_total, np.ones(n_objects) / n_objects)
    min_sep = room_size * 0.09
    centers_xy: List[Tuple[float, float]] = []

    def random_xy() -> Tuple[float, float]:
        for _ in range(50):
            x = float(rng.uniform(-half * 0.72, half * 0.72))
            y = float(rng.uniform(-half * 0.72, half * 0.72))
            if all((x - ox) ** 2 + (y - oy) ** 2 > min_sep ** 2 for ox, oy in centers_xy):
                centers_xy.append((x, y))
                return x, y
        x = float(rng.uniform(-half * 0.75, half * 0.75))
        y = float(rng.uniform(-half * 0.75, half * 0.75))
        centers_xy.append((x, y))
        return x, y

    for n in obj_counts:
        n = int(n)
        if n <= 0:
            continue
        x, y = random_xy()
        shape = rng.choice(["box", "box", "cylinder", "cylinder", "sphere", "panel"])
        yaw = float(rng.uniform(-180.0, 180.0))

        if shape == "box":
            sx = float(rng.uniform(0.45, 2.2))
            sy = float(rng.uniform(0.45, 2.0))
            sz = float(rng.uniform(0.5, 2.4))
            parts.append(sample_oriented_box_surface((x, y, sz / 2), (sx, sy, sz), yaw, n, rng, noise))
        elif shape == "cylinder":
            radius = float(rng.uniform(0.25, 0.85))
            h = float(rng.uniform(0.7, 2.6))
            parts.append(sample_cylinder_surface((x, y, h / 2), radius, h, n, rng, noise))
        elif shape == "sphere":
            radius = float(rng.uniform(0.35, 1.1))
            parts.append(sample_sphere_surface((x, y, 0.02), radius, n, rng, noise))
        else:  # slanted panel
            width = float(rng.uniform(0.8, 2.6))
            panel_h = float(rng.uniform(0.8, 2.4))
            pitch = float(rng.uniform(-40.0, 40.0))
            parts.append(sample_slanted_panel((x, y, panel_h / 2), width, panel_h, yaw, pitch, n, rng, noise))

    pts = np.vstack(parts).astype(np.float32)

    # Add a few short interior wall panels / corners for stronger non-degenerate constraints.
    n_extra = int(num_points * rng.uniform(0.03, 0.07))
    if n_extra > 0:
        extra_counts = rng.multinomial(n_extra, [0.5, 0.5])
        for n in extra_counts:
            x, y = random_xy()
            yaw = float(rng.uniform(-180.0, 180.0))
            panel_w = float(rng.uniform(1.0, 3.8))
            panel_h = float(rng.uniform(1.0, height))
            panel = sample_slanted_panel((x, y, panel_h / 2), panel_w, panel_h, yaw, 0.0, int(n), rng, noise)
            pts = np.vstack([pts, panel])

    # Global random yaw so nodeg scenes are not tied to one world orientation.
    pts = rotz(pts, float(rng.uniform(-180.0, 180.0)))
    return resample_points(pts, num_points, rng)


def split_name(i: int, n_total: int, train_ratio: float, val_ratio: float) -> str:
    r = i / max(n_total, 1)
    if r < train_ratio:
        return "train"
    if r < train_ratio + val_ratio:
        return "val"
    return "test"


def balanced_bin_sequence(count: int, rng: np.random.Generator, num_bins: int = DEFAULT_NUM_DIR_BINS) -> List[int]:
    """Return a shuffled bin list whose histogram differs by at most one."""

    if count <= 0:
        return []
    full = np.tile(np.arange(num_bins, dtype=np.int64), count // num_bins)
    remainder = rng.permutation(num_bins)[: count % num_bins]
    bins = np.concatenate([full, remainder]).astype(np.int64)
    rng.shuffle(bins)
    return bins.tolist()


def sample_angle_for_bin(bin_id: int, rng: np.random.Generator, num_bins: int = DEFAULT_NUM_DIR_BINS) -> float:
    """Sample a concrete direction angle inside one axial bin."""

    bin_size = 180.0 / float(num_bins)
    axis_angle = float(rng.uniform(bin_id * bin_size, (bin_id + 1) * bin_size))
    return axis_angle + (180.0 if bool(rng.integers(0, 2)) else 0.0)


def tunnel_bin_histogram(rows: List[Dict[str, object]], split: Optional[str] = None, num_bins: int = DEFAULT_NUM_DIR_BINS) -> np.ndarray:
    """Count tunnel_like rows by direction bin, optionally for one split."""

    hist = np.zeros(num_bins, dtype=np.int64)
    for row in rows:
        if row["scene_type"] != "tunnel_like":
            continue
        if split is not None and row["split"] != split:
            continue
        hist[int(row["dir_bin_gt"])] += 1
    return hist


def check_tunnel_bin_balance(rows: List[Dict[str, object]], split: str, num_bins: int = DEFAULT_NUM_DIR_BINS) -> None:
    """Assert that one split's tunnel bin counts are as balanced as possible."""

    hist = tunnel_bin_histogram(rows, split=split, num_bins=num_bins)
    if hist.sum() == 0:
        return
    if int(hist.max() - hist.min()) > 1:
        raise AssertionError(f"{split} tunnel bin histogram is imbalanced: {hist.tolist()}")


def save_sample(
    sample_id: str,
    points: np.ndarray,
    scene_type: str,
    split: str,
    points_dir: Path,
    pcd_dir: Optional[Path],
    write_pcd: bool,
    dir_x: float = 0.0,
    dir_y: float = 0.0,
    dir_xy_valid: int = 0,
    angle_deg: Optional[float] = None,
    dir_bin_gt: int = 0,
    dir_bin_valid: int = 0,
    dir_range: str = "",
    source: str = "synthetic",
) -> Dict[str, object]:
    npy_path = points_dir / f"{sample_id}.npy"
    np.save(npy_path, points.astype(np.float32))

    if write_pcd and pcd_dir is not None:
        write_ascii_pcd_xyz(pcd_dir / f"{sample_id}.pcd", points)

    if scene_type == "tunnel_like" and angle_deg is not None:
        dir_bin_gt = direction_bin_from_angle_deg(angle_deg)
        dir_bin_valid = 1
        dir_range = direction_bin_range(dir_bin_gt)
    else:
        dir_bin_gt = 0
        dir_bin_valid = 0
        dir_range = ""

    return {
        "sample_id": sample_id,
        "file_path": f"points/{sample_id}.npy",
        "scene_type": scene_type,
        "class_gt": SCENE_TYPE_TO_CLASS[scene_type],
        "dir_x": f"{dir_x:.8f}",
        "dir_y": f"{dir_y:.8f}",
        "dir_xy_valid": int(dir_xy_valid),
        "rz_gt": 1 if scene_type == "open_like" else 0,
        "dir_exist_gt": 1 if scene_type in {"tunnel_like", "open_like"} else 0,
        "sample_weight": 1.0,
        "split": split,
        "angle_deg": "" if angle_deg is None else f"{angle_deg:.3f}",
        "dir_bin_gt": int(dir_bin_gt),
        "dir_bin_valid": int(dir_bin_valid),
        "dir_range": dir_range,
        "source": source,
    }


def write_labels_csv(path: Path, rows: List[Dict[str, object]]) -> None:
    fields = [
        "sample_id", "file_path", "scene_type", "class_gt",
        "dir_x", "dir_y", "dir_xy_valid", "rz_gt", "dir_exist_gt",
        "sample_weight", "split", "angle_deg", "dir_bin_gt",
        "dir_bin_valid", "dir_range", "source",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def generate_dataset(args: argparse.Namespace) -> None:
    rng = np.random.default_rng(args.seed)

    out_dir = ensure_dir(args.out_dir)
    points_dir = ensure_dir(out_dir / "points")
    labels_dir = ensure_dir(out_dir / "labels")
    pcd_dir = ensure_dir(out_dir / "pcd_preview") if args.write_pcd else None

    ref_deg_path = Path(args.ref_deg) if args.ref_deg else Path("deg1.pcd")
    ref_nodeg_path = Path(args.ref_nodeg) if args.ref_nodeg else Path("nodeg2.pcd")

    ref_deg = read_ascii_pcd_xyz(ref_deg_path) if ref_deg_path.exists() else None
    ref_nodeg = read_ascii_pcd_xyz(ref_nodeg_path) if ref_nodeg_path.exists() else None

    # Use reference clouds only to estimate reasonable size, not as direct noisy templates.
    deg_ext = robust_extent(ref_deg, default=(28.0, 4.0, 3.0))
    nodeg_ext = robust_extent(ref_nodeg, default=(13.0, 13.0, 3.5))
    base_length = float(np.clip(deg_ext[0], 18.0, 40.0))
    base_width = float(np.clip(max(deg_ext[1], 3.0), 3.0, 7.0))
    base_height = float(np.clip(max(deg_ext[2], 2.4), 2.4, 5.0))
    base_room = float(np.clip(max(nodeg_ext[0], nodeg_ext[1], 10.0), 9.0, 18.0))
    base_nodeg_h = float(np.clip(max(nodeg_ext[2], 2.8), 2.8, 5.0))

    rows: List[Dict[str, object]] = []

    # 1) Tunnel-like: balance samples across 15-degree axial direction bins.
    tunnel_specs: List[Tuple[str, int]] = []
    for sp in ["train", "val", "test"]:
        split_count = sum(split_name(i, args.num_each, args.train_ratio, args.val_ratio) == sp for i in range(args.num_each))
        tunnel_specs.extend((sp, bin_id) for bin_id in balanced_bin_sequence(split_count, rng))

    for i, (split, bin_id) in enumerate(tunnel_specs):
        angle = sample_angle_for_bin(bin_id, rng)
        dx = math.cos(math.radians(angle))
        dy = math.sin(math.radians(angle))

        length = float(base_length * rng.uniform(0.85, 1.20))
        width = float(base_width * rng.uniform(0.85, 1.20))
        height = float(base_height * rng.uniform(0.85, 1.15))
        clutter_ratio = float(rng.uniform(0.000, 0.015))
        pts = make_tunnel_cloud(
            angle,
            args.num_points,
            rng,
            length=length,
            width=width,
            height=height,
            noise=args.noise,
            clutter_ratio=clutter_ratio,
        )
        sample_id = f"tunnel_rand_{i:04d}"
        rows.append(save_sample(
            sample_id=sample_id,
            points=pts,
            scene_type="tunnel_like",
            split=split,
            points_dir=points_dir,
            pcd_dir=pcd_dir,
            write_pcd=args.write_pcd,
            dir_x=dx,
            dir_y=dy,
            dir_xy_valid=1,
            angle_deg=angle,
            source="synthetic_clean_balanced_tunnel_bins",
        ))
        if rows[-1]["dir_bin_gt"] != bin_id:
            raise AssertionError(f"sampled angle {angle} for bin {bin_id}, got {rows[-1]['dir_bin_gt']}")

    # 2) Open-like: plane + much less clutter than previous version.
    for i in range(args.num_each):
        split = split_name(i, args.num_each, args.train_ratio, args.val_ratio)
        size = float(rng.uniform(20.0, 36.0))
        clutter_ratio = float(rng.uniform(0.000, 0.045))
        pts = make_open_plane_cloud(
            args.num_points,
            rng,
            size=size,
            noise=args.noise,
            clutter_ratio=clutter_ratio,
        )
        sample_id = f"open_rand_{i:04d}"
        rows.append(save_sample(
            sample_id=sample_id,
            points=pts,
            scene_type="open_like",
            split=split,
            points_dir=points_dir,
            pcd_dir=pcd_dir,
            write_pcd=args.write_pcd,
            dir_x=0.0,
            dir_y=0.0,
            dir_xy_valid=0,
            angle_deg=None,
            source="synthetic_clean_plane_sparse_clutter",
        ))

    # 3) Non-degenerate: random object count and mixed geometry, no fixed 4-box layout.
    for i in range(args.num_each):
        split = split_name(i, args.num_each, args.train_ratio, args.val_ratio)
        room_size = float(base_room * rng.uniform(0.80, 1.20))
        height = float(base_nodeg_h * rng.uniform(0.80, 1.15))
        pts = make_nondeg_cloud(
            args.num_points,
            rng,
            room_size=room_size,
            height=height,
            noise=args.noise,
        )
        sample_id = f"nodeg_rand_{i:04d}"
        rows.append(save_sample(
            sample_id=sample_id,
            points=pts,
            scene_type="nondeg_or_other",
            split=split,
            points_dir=points_dir,
            pcd_dir=pcd_dir,
            write_pcd=args.write_pcd,
            dir_x=0.0,
            dir_y=0.0,
            dir_xy_valid=0,
            angle_deg=None,
            source="synthetic_rich_random_nondeg",
        ))

    csv_path = labels_dir / "deg_scene_labels.csv"
    write_labels_csv(csv_path, rows)

    for sp in ["train", "val", "test"]:
        sp_rows = [r for r in rows if r["split"] == sp]
        write_labels_csv(labels_dir / f"deg_scene_{sp}.csv", sp_rows)

    print(f"Done. Output directory: {out_dir}")
    print(f"Total samples: {len(rows)}")
    print(f"Labels: {csv_path}")
    print(f"Points: {points_dir}")
    if args.write_pcd:
        print(f"PCD preview: {pcd_dir}")
    if ref_deg is not None:
        print(f"Used {ref_deg_path} only for rough tunnel scale estimation.")
    if ref_nodeg is not None:
        print(f"Used {ref_nodeg_path} only for rough nondeg scale estimation.")
    print("\nClass counts:")
    for cls in ["tunnel_like", "open_like", "nondeg_or_other"]:
        print(f"  {cls:16s}: {sum(r['scene_type'] == cls for r in rows)}")
    print("\nSplit counts:")
    for sp in ["train", "val", "test"]:
        print(f"  {sp:5s}: {sum(r['split'] == sp for r in rows)}")
    print("\nTunnel bin histograms:")
    for sp in ["train", "val", "test"]:
        check_tunnel_bin_balance(rows, split=sp)
        hist = tunnel_bin_histogram(rows, split=sp)
        print(f"  {sp:5s} bin histogram: {hist.tolist()}")
    print("\nExample tunnel labels:")
    for r in [x for x in rows if x["scene_type"] == "tunnel_like"][:5]:
        print(
            f"  {r['sample_id']}: angle={r['angle_deg']} deg, "
            f"bin={r['dir_bin_gt']}, range={r['dir_range']}, "
            f"dir=({r['dir_x']}, {r['dir_y']})"
        )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate clean random-angle deg_scene point-cloud dataset."
    )
    parser.add_argument("--out_dir", type=str, default="./deg_scene_synth_v2", help="Output dataset directory.")
    parser.add_argument("--num_each", type=int, default=100, help="Samples per class: tunnel/open/nodeg.")
    parser.add_argument("--num_points", type=int, default=20000, help="Points per saved cloud before DataLoader sampling.")
    parser.add_argument("--ref_deg", type=str, default="", help="Optional tunnel reference PCD. Default: auto-use ./deg1.pcd if it exists.")
    parser.add_argument("--ref_nodeg", type=str, default="", help="Optional nondeg reference PCD. Default: auto-use ./nodeg2.pcd if it exists.")
    parser.add_argument("--seed", type=int, default=42, help="Random seed.")
    parser.add_argument("--write_pcd", action="store_true", help="Also write PCD previews for visualization.")

    # Advanced parameters are intentionally hidden to keep normal usage simple.
    parser.add_argument("--noise", type=float, default=0.010, help=argparse.SUPPRESS)
    parser.add_argument("--train_ratio", type=float, default=0.70, help=argparse.SUPPRESS)
    parser.add_argument("--val_ratio", type=float, default=0.15, help=argparse.SUPPRESS)
    return parser.parse_args()


if __name__ == "__main__":
    generate_dataset(parse_args())
