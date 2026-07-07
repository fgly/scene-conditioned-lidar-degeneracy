#!/usr/bin/env python3
# -*- coding: utf-8 -*-
r"""
Extract PointCloud2 frames from a bag file and generate geometric-statistical
deg_scene labels.

Class mapping:
    0: tunnel_like
    1: open_like
    2: nondeg_or_other

Example:
    python -u .\tools\bag_degeneracy_labeler.py --bag .\example.bag --tau_c 25 --sample_hz 1.0
"""

from __future__ import annotations

import argparse
import bisect
import csv
import math
import sys
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np

# Training-label convention.
CLASS_TO_ID = {
    "tunnel_like": 0,
    "open_like": 1,
    "non_degenerate": 2,  # internal name
    "nondeg_or_other": 2,  # CSV/training name
}

POINTFIELD_DTYPE = {
    1: np.int8,     # INT8
    2: np.uint8,    # UINT8
    3: np.int16,    # INT16
    4: np.uint16,   # UINT16
    5: np.int32,    # INT32
    6: np.uint32,   # UINT32
    7: np.float32,  # FLOAT32
    8: np.float64,  # FLOAT64
}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def _get_attr(obj, name: str, default=None):
    """Support both normal objects and namedtuple-like message objects."""
    return getattr(obj, name, default)


def _stamp_to_sec(stamp, fallback: Optional[float] = None) -> Optional[float]:
    """Convert ROS1/ROS2-like stamp object to seconds."""
    if stamp is None:
        return fallback
    sec = getattr(stamp, "sec", None)
    nsec = getattr(stamp, "nanosec", None)
    if sec is None:
        sec = getattr(stamp, "secs", None)
    if nsec is None:
        nsec = getattr(stamp, "nsecs", None)
    if sec is None or nsec is None:
        return fallback
    return float(sec) + float(nsec) * 1e-9


def _msg_stamp_sec(msg, fallback: float) -> float:
    header = _get_attr(msg, "header", None)
    if header is None:
        return fallback
    st = _get_attr(header, "stamp", None)
    return _stamp_to_sec(st, fallback=fallback) or fallback


def quaternion_xyzw_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Return rotation matrix R_parent_child from quaternion [x, y, z, w]."""
    q = np.asarray(q, dtype=np.float64)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.eye(3, dtype=np.float64)
    x, y, z, w = q / n
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array([
        [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz),       2.0 * (xz + wy)],
        [2.0 * (xy + wz),       1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
        [2.0 * (xz - wy),       2.0 * (yz + wx),       1.0 - 2.0 * (xx + yy)],
    ], dtype=np.float64)


def transform_parent_points_to_child(points_parent: np.ndarray,
                                     translation_parent_child: np.ndarray,
                                     quat_parent_child_xyzw: np.ndarray) -> np.ndarray:
    """
    TF convention: T_parent_child maps child-frame coordinates to parent-frame coordinates:
        p_parent = R_parent_child * p_child + t_parent_child
    Therefore, for point cloud already in parent frame:
        p_child = R_parent_child^T * (p_parent - t_parent_child)
    For row-vector numpy points, this is:
        points_child = (points_parent - t) @ R_parent_child
    """
    if points_parent.size == 0:
        return points_parent
    R = quaternion_xyzw_to_rotmat(quat_parent_child_xyzw)
    t = np.asarray(translation_parent_child, dtype=np.float64).reshape(1, 3)
    return ((points_parent.astype(np.float64) - t) @ R).astype(np.float32)


def build_tf_buffer(bag_path: Path, args) -> Tuple[List[float], List[np.ndarray], List[np.ndarray]]:
    """Read /tf and /tf_static transforms for tf_parent -> tf_child from the bag."""
    AnyReader = import_anyreader()
    times: List[float] = []
    trans: List[np.ndarray] = []
    quats: List[np.ndarray] = []

    topics = {args.tf_topic, args.tf_static_topic}
    with AnyReader([bag_path]) as reader:
        tf_connections = [c for c in reader.connections if c.topic in topics]
        if not tf_connections:
            print(f"[WARN] No TF topics found among {sorted(topics)}")
            return times, trans, quats
        for connection, timestamp, rawdata in reader.messages(connections=tf_connections):
            try:
                msg = reader.deserialize(rawdata, connection.msgtype)
            except Exception:
                continue
            transforms = list(_get_attr(msg, "transforms", []))
            for tr in transforms:
                header = _get_attr(tr, "header", None)
                parent = str(_get_attr(header, "frame_id", "")) if header is not None else ""
                child = str(_get_attr(tr, "child_frame_id", ""))
                if parent != args.tf_parent or child != args.tf_child:
                    continue
                tf_stamp = _stamp_to_sec(_get_attr(header, "stamp", None), fallback=int(timestamp) * 1e-9)
                tfm = _get_attr(tr, "transform", None)
                if tfm is None:
                    continue
                tt = _get_attr(tfm, "translation", None)
                rr = _get_attr(tfm, "rotation", None)
                if tt is None or rr is None:
                    continue
                times.append(float(tf_stamp))
                trans.append(np.array([float(tt.x), float(tt.y), float(tt.z)], dtype=np.float64))
                quats.append(np.array([float(rr.x), float(rr.y), float(rr.z), float(rr.w)], dtype=np.float64))

    if not times:
        return times, trans, quats
    order = np.argsort(np.asarray(times, dtype=np.float64))
    times = [times[int(i)] for i in order]
    trans = [trans[int(i)] for i in order]
    quats = [quats[int(i)] for i in order]
    return times, trans, quats


def lookup_tf(times: List[float],
              trans: List[np.ndarray],
              quats: List[np.ndarray],
              stamp: float,
              mode: str,
              max_dt: float) -> Tuple[Optional[np.ndarray], Optional[np.ndarray], float]:
    """Lookup previous or nearest TF. Returns (translation, quaternion, dt_abs)."""
    if not times:
        return None, None, float("inf")
    i = bisect.bisect_left(times, stamp)
    candidates: List[int] = []
    if mode == "previous":
        if i == len(times) or times[i] > stamp:
            i -= 1
        if i >= 0:
            candidates = [i]
    elif mode == "nearest":
        if i < len(times):
            candidates.append(i)
        if i - 1 >= 0:
            candidates.append(i - 1)
    else:
        raise ValueError(f"Unsupported tf_lookup_mode={mode}")
    if not candidates:
        return None, None, float("inf")
    best = min(candidates, key=lambda k: abs(times[k] - stamp))
    dt = abs(times[best] - stamp)
    if math.isfinite(max_dt) and max_dt >= 0 and dt > max_dt:
        return None, None, dt
    return trans[best], quats[best], dt


def _field_dtype(datatype: int, count: int, is_bigendian: bool):
    if datatype not in POINTFIELD_DTYPE:
        raise ValueError(f"Unsupported PointField datatype={datatype}")
    dt = np.dtype(POINTFIELD_DTYPE[datatype])
    dt = dt.newbyteorder(">" if is_bigendian else "<")
    if count and count > 1:
        return (dt, (int(count),))
    return dt


def pointcloud2_to_xyz(msg) -> np.ndarray:
    """Parse x/y/z from a deserialized sensor_msgs/PointCloud2 message."""
    fields = list(_get_attr(msg, "fields", []))
    if not fields:
        return np.empty((0, 3), dtype=np.float32)

    names: List[str] = []
    formats: List[object] = []
    offsets: List[int] = []

    is_bigendian = bool(_get_attr(msg, "is_bigendian", False))
    point_step = int(_get_attr(msg, "point_step", 0))
    width = int(_get_attr(msg, "width", 0))
    height = int(_get_attr(msg, "height", 1))
    count_points = width * height

    for f in fields:
        name = str(_get_attr(f, "name", ""))
        datatype = int(_get_attr(f, "datatype", 0))
        count = int(_get_attr(f, "count", 1))
        offset = int(_get_attr(f, "offset", 0))
        if not name:
            continue
        try:
            fmt = _field_dtype(datatype, count, is_bigendian)
        except ValueError:
            # Keep unsupported fields out of the structured dtype.
            continue
        names.append(name)
        formats.append(fmt)
        offsets.append(offset)

    if not {"x", "y", "z"}.issubset(set(names)):
        available = ", ".join(names)
        raise ValueError(f"PointCloud2 has no x/y/z fields. Available fields: {available}")

    if point_step <= 0:
        raise ValueError(f"Invalid point_step={point_step}")

    dtype = np.dtype({
        "names": names,
        "formats": formats,
        "offsets": offsets,
        "itemsize": point_step,
    })

    data = _get_attr(msg, "data", b"")
    if isinstance(data, np.ndarray):
        raw = data.tobytes()
    else:
        raw = bytes(data)

    if not raw or count_points <= 0:
        return np.empty((0, 3), dtype=np.float32)

    max_points_from_data = len(raw) // point_step
    count_points = min(count_points, max_points_from_data)
    arr = np.frombuffer(raw, dtype=dtype, count=count_points)

    xyz = np.empty((arr.shape[0], 3), dtype=np.float32)
    xyz[:, 0] = arr["x"].astype(np.float32, copy=False)
    xyz[:, 1] = arr["y"].astype(np.float32, copy=False)
    xyz[:, 2] = arr["z"].astype(np.float32, copy=False)
    good = np.isfinite(xyz).all(axis=1)
    return xyz[good]


def point_stats(points: np.ndarray) -> str:
    """Return a compact diagnostic string for a point cloud. Includes robust distance percentiles."""
    if points.size == 0:
        return "n=0"
    good = np.isfinite(points).all(axis=1)
    pts = points[good]
    if pts.size == 0:
        return f"n={points.shape[0]}, finite=0"
    r0 = np.linalg.norm(pts[:, :3], axis=1)
    c_mean = pts.mean(axis=0)
    c_med = np.median(pts, axis=0)
    r_mean = np.linalg.norm(pts[:, :3] - c_mean[None, :], axis=1)
    r_med = np.linalg.norm(pts[:, :3] - c_med[None, :], axis=1)
    qs = np.percentile(r0, [0, 1, 5, 50, 95, 99, 100])
    qs_med = np.percentile(r_med, [0, 1, 5, 50, 95, 99, 100])
    return (
        f"n={points.shape[0]}, finite={pts.shape[0]}, "
        f"x=[{pts[:,0].min():.2f},{pts[:,0].max():.2f}], "
        f"y=[{pts[:,1].min():.2f},{pts[:,1].max():.2f}], "
        f"z=[{pts[:,2].min():.2f},{pts[:,2].max():.2f}], "
        f"r_origin_q=[{qs[0]:.2f},{qs[1]:.2f},{qs[2]:.2f},{qs[3]:.2f},{qs[4]:.2f},{qs[5]:.2f},{qs[6]:.2f}], "
        f"mean=({c_mean[0]:.2f},{c_mean[1]:.2f},{c_mean[2]:.2f}), "
        f"median=({c_med[0]:.2f},{c_med[1]:.2f},{c_med[2]:.2f}), "
        f"r_from_mean=[{r_mean.min():.2f},{r_mean.max():.2f}], "
        f"r_from_median_q=[{qs_med[0]:.2f},{qs_med[1]:.2f},{qs_med[2]:.2f},{qs_med[3]:.2f},{qs_med[4]:.2f},{qs_med[5]:.2f},{qs_med[6]:.2f}]"
    )


def pointcloud2_field_summary(msg) -> str:
    fields = list(_get_attr(msg, "fields", []))
    items = []
    for f in fields:
        items.append(
            f"{_get_attr(f, 'name', '')}:off={_get_attr(f, 'offset', '?')},"
            f"dt={_get_attr(f, 'datatype', '?')},cnt={_get_attr(f, 'count', '?')}"
        )
    return (
        f"height={_get_attr(msg, 'height', '?')}, width={_get_attr(msg, 'width', '?')}, "
        f"point_step={_get_attr(msg, 'point_step', '?')}, row_step={_get_attr(msg, 'row_step', '?')}, "
        f"is_bigendian={_get_attr(msg, 'is_bigendian', '?')}, fields=[" + "; ".join(items) + "]"
    )


def pre_filter_points(points: np.ndarray, radius_max: float, abs_max: float) -> np.ndarray:
    """Remove non-finite points and optional very far / absurd coordinate outliers before range crop."""
    if points.size == 0:
        return points
    mask = np.isfinite(points).all(axis=1)
    if math.isfinite(radius_max) and radius_max > 0:
        r = np.linalg.norm(points[:, :3], axis=1)
        mask &= r <= radius_max
    if math.isfinite(abs_max) and abs_max > 0:
        mask &= (np.abs(points[:, 0]) <= abs_max)
        mask &= (np.abs(points[:, 1]) <= abs_max)
        mask &= (np.abs(points[:, 2]) <= abs_max)
    return points[mask]


def crop_points(points: np.ndarray,
                range_min: float,
                range_max: float,
                z_min: Optional[float],
                z_max: Optional[float],
                crop_mode: str = "origin_range") -> np.ndarray:
    if points.size == 0:
        return points

    mask = np.isfinite(points).all(axis=1)

    if crop_mode == "none":
        pass
    else:
        if crop_mode == "origin_range":
            center = np.zeros(3, dtype=np.float64)
        elif crop_mode == "mean_range":
            valid_pts = points[mask]
            center = valid_pts.mean(axis=0) if valid_pts.shape[0] else np.zeros(3, dtype=np.float64)
        elif crop_mode == "median_range":
            valid_pts = points[mask]
            center = np.median(valid_pts, axis=0) if valid_pts.shape[0] else np.zeros(3, dtype=np.float64)
        else:
            raise ValueError(f"Unsupported crop_mode={crop_mode}")
        r = np.linalg.norm(points[:, :3] - center[None, :], axis=1)
        mask &= (r >= range_min) & (r <= range_max)

    if z_min is not None:
        mask &= points[:, 2] >= z_min
    if z_max is not None:
        mask &= points[:, 2] <= z_max
    return points[mask]


def recenter_points(points: np.ndarray, center_mode: str) -> np.ndarray:
    """Optionally subtract a translation from points. This does not change normals."""
    if points.size == 0 or center_mode == "none":
        return points
    if center_mode == "mean":
        c = points.mean(axis=0)
    elif center_mode == "median":
        c = np.median(points, axis=0)
    else:
        raise ValueError(f"Unsupported center_mode={center_mode}")
    return (points - c[None, :]).astype(np.float32)


def fixed_sample(points: np.ndarray, num_point: int, rng: np.random.Generator) -> np.ndarray:
    if points.shape[0] == 0:
        return np.zeros((num_point, 3), dtype=np.float32)
    n = points.shape[0]
    if n >= num_point:
        idx = rng.choice(n, size=num_point, replace=False)
    else:
        idx = rng.choice(n, size=num_point, replace=True)
    return points[idx].astype(np.float32)


def estimate_normals_open3d(points: np.ndarray,
                            voxel_size: float,
                            normal_radius: float,
                            normal_max_nn: int,
                            max_normal_points: int,
                            rng: np.random.Generator) -> Tuple[np.ndarray, np.ndarray]:
    try:
        import open3d as o3d
    except Exception as exc:
        raise RuntimeError("Open3D import failed. Install with: pip install open3d") from exc

    if points.shape[0] < 30:
        return points, np.empty((0, 3), dtype=np.float32)

    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))

    if voxel_size > 0:
        pcd = pcd.voxel_down_sample(voxel_size=float(voxel_size))

    pts_ds = np.asarray(pcd.points, dtype=np.float32)
    if pts_ds.shape[0] > max_normal_points:
        idx = rng.choice(pts_ds.shape[0], size=max_normal_points, replace=False)
        pts_ds = pts_ds[idx]
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(pts_ds.astype(np.float64))

    if pts_ds.shape[0] < 30:
        return pts_ds, np.empty((0, 3), dtype=np.float32)

    pcd.estimate_normals(
        search_param=o3d.geometry.KDTreeSearchParamHybrid(
            radius=float(normal_radius), max_nn=int(normal_max_nn)
        )
    )
    normals = np.asarray(pcd.normals, dtype=np.float32)
    good = np.isfinite(normals).all(axis=1) & (np.linalg.norm(normals, axis=1) > 1e-6)
    return pts_ds[good], normals[good]


def canonicalize_unoriented_normals(normals: np.ndarray) -> np.ndarray:
    """Treat n and -n as the same unoriented plane-normal axis."""
    n = normals.astype(np.float64).copy()
    norm = np.linalg.norm(n, axis=1, keepdims=True)
    n = n / np.maximum(norm, 1e-12)

    eps = 1e-8
    flip = (n[:, 2] < -eps)
    flip |= (np.abs(n[:, 2]) <= eps) & (n[:, 1] < -eps)
    flip |= (np.abs(n[:, 2]) <= eps) & (np.abs(n[:, 1]) <= eps) & (n[:, 0] < 0)
    n[flip] *= -1.0
    return n.astype(np.float32)


def bin_normals(normals: np.ndarray,
                az_bins: int,
                el_bins: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    n = canonicalize_unoriented_normals(normals)
    z = np.clip(n[:, 2], 0.0, 1.0)
    az = np.arctan2(n[:, 1], n[:, 0])
    el = np.arcsin(z)

    az_idx = np.floor((az + math.pi) / (2.0 * math.pi) * az_bins).astype(np.int64)
    az_idx = np.mod(az_idx, az_bins)
    el_idx = np.floor(el / (0.5 * math.pi) * el_bins).astype(np.int64)
    el_idx = np.clip(el_idx, 0, el_bins - 1)

    counts = np.zeros((el_bins, az_bins), dtype=np.int64)
    np.add.at(counts, (el_idx, az_idx), 1)
    return counts, el_idx, az_idx


def make_region_mask(ei: int,
                     ai: int,
                     counts_shape: Tuple[int, int],
                     merge_bins: int,
                     vertical_start_el_idx: int) -> np.ndarray:
    el_bins, az_bins = counts_shape
    mask = np.zeros((el_bins, az_bins), dtype=bool)

    if ei >= vertical_start_el_idx:
        mask[vertical_start_el_idx:el_bins, :] = True
        return mask

    for de in range(-merge_bins, merge_bins + 1):
        ee = ei + de
        if ee < 0 or ee >= el_bins:
            continue
        for da in range(-merge_bins, merge_bins + 1):
            aa = (ai + da) % az_bins
            mask[ee, aa] = True
    return mask


def find_peak_region(counts: np.ndarray,
                     excluded: np.ndarray,
                     merge_bins: int,
                     vertical_start_el_idx: int) -> Tuple[np.ndarray, int, Tuple[int, int]]:
    best_count = -1
    best_mask = None
    best_cell = (0, 0)

    candidate_cells = np.argwhere((counts > 0) & (~excluded))
    if candidate_cells.size == 0:
        return np.zeros_like(counts, dtype=bool), 0, best_cell

    for ei, ai in candidate_cells:
        region = make_region_mask(int(ei), int(ai), counts.shape, merge_bins, vertical_start_el_idx)
        region = region & (~excluded)
        c = int(counts[region].sum())
        if c > best_count:
            best_count = c
            best_mask = region
            best_cell = (int(ei), int(ai))

    if best_mask is None:
        best_mask = np.zeros_like(counts, dtype=bool)
        best_count = 0
    return best_mask, best_count, best_cell


def mean_direction_for_region(normals: np.ndarray,
                              el_idx: np.ndarray,
                              az_idx: np.ndarray,
                              region_mask: np.ndarray) -> np.ndarray:
    if normals.shape[0] == 0:
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)
    in_region = region_mask[el_idx, az_idx]
    if not np.any(in_region):
        return np.array([0.0, 0.0, 1.0], dtype=np.float32)
    n = canonicalize_unoriented_normals(normals[in_region])
    m = n.mean(axis=0)
    m_norm = np.linalg.norm(m)
    if m_norm < 1e-8:
        return n[0].astype(np.float32)
    return (m / m_norm).astype(np.float32)


def axis_angle_and_bin(d: np.ndarray, num_dir_bins: int) -> Tuple[float, int, np.ndarray]:
    dxy = np.asarray([d[0], d[1]], dtype=np.float64)
    norm_xy = np.linalg.norm(dxy)
    if norm_xy < 1e-8:
        return float("nan"), -1, np.array([0.0, 0.0], dtype=np.float32)
    dxy = dxy / norm_xy
    theta = math.atan2(float(dxy[1]), float(dxy[0]))
    while theta < 0.0:
        theta += math.pi
    while theta >= math.pi:
        theta -= math.pi
    bin_id = int(math.floor(theta / math.pi * num_dir_bins))
    bin_id = max(0, min(num_dir_bins - 1, bin_id))
    return theta, bin_id, np.array([math.cos(theta), math.sin(theta)], dtype=np.float32)


def training_scene_name(scene_type: str) -> str:
    """Map internal class names to the CSV convention used by generate_deg_scene_dataset.py."""
    if scene_type == "non_degenerate":
        return "nondeg_or_other"
    return scene_type


def direction_bin_range(bin_id: int, num_dir_bins: int) -> str:
    """Return readable axial direction-bin range, matching generate_deg_scene_dataset.py."""
    if bin_id < 0:
        return ""
    bin_size = 180.0 / float(num_dir_bins)
    start = int(round(int(bin_id) * bin_size))
    end = int(round((int(bin_id) + 1) * bin_size))
    return f"[{start},{end}) or [{start + 180},{end + 180})"


def split_name_from_index(index: int, total: int, train_ratio: float, val_ratio: float) -> str:
    """Sequential train/val/test split compatible with the synthetic dataset CSV."""
    if total <= 0:
        return "train"
    train_n = int(round(total * train_ratio))
    val_n = int(round(total * val_ratio))
    if index < train_n:
        return "train"
    if index < train_n + val_n:
        return "val"
    return "test"


def angle_deg_from_dir_xy(dir_x: float, dir_y: float) -> float:
    """Return signed vector angle in degrees for CSV readability."""
    return math.degrees(math.atan2(float(dir_y), float(dir_x)))


def make_training_csv_row(label: Dict[str, object],
                          sample_id: str,
                          npy_name: str,
                          split: str,
                          args) -> Dict[str, object]:
    """Convert detailed geometric label to the exact CSV format expected by DegSceneDataLoader."""
    scene_type = training_scene_name(str(label.get("scene_type", "non_degenerate")))
    class_gt = 2 if scene_type == "nondeg_or_other" else int(CLASS_TO_ID[scene_type])

    is_tunnel = scene_type == "tunnel_like"
    is_open = scene_type == "open_like"

    dir_xy_valid = int(label.get("dir_xy_valid", 0)) if is_tunnel else 0
    dir_x = float(label.get("dir_x", 0.0)) if dir_xy_valid else 0.0
    dir_y = float(label.get("dir_y", 0.0)) if dir_xy_valid else 0.0

    dir_bin_gt = int(label.get("dir_bin", -1)) if (is_tunnel and dir_xy_valid) else 0
    dir_bin_valid = 1 if (is_tunnel and dir_xy_valid and dir_bin_gt >= 0) else 0
    angle_deg = angle_deg_from_dir_xy(dir_x, dir_y) if dir_bin_valid else None
    dir_range = direction_bin_range(dir_bin_gt, args.num_dir_bins) if dir_bin_valid else ""

    return {
        "sample_id": sample_id,
        "file_path": str(Path("points") / npy_name).replace("\\", "/"),
        "scene_type": scene_type,
        "class_gt": int(class_gt),
        "dir_x": f"{dir_x:.8f}",
        "dir_y": f"{dir_y:.8f}",
        "dir_xy_valid": int(dir_xy_valid),
        "rz_gt": 1 if is_open else 0,
        "dir_exist_gt": 1 if scene_type in {"tunnel_like", "open_like"} else 0,
        "sample_weight": 1.0,
        "split": split,
        "angle_deg": "" if angle_deg is None else f"{angle_deg:.3f}",
        "dir_bin_gt": int(dir_bin_gt),
        "dir_bin_valid": int(dir_bin_valid),
        "dir_range": dir_range,
        "source": str(args.source),
    }


def make_debug_csv_row(label: Dict[str, object],
                       sample_id: str,
                       npy_name: str,
                       split: str,
                       args,
                       ts: float,
                       topic: str,
                       output_frame_id: str,
                       source_frame_id: str,
                       tf_used: int,
                       tf_dt: float) -> Dict[str, object]:
    """Detailed diagnostic CSV row for threshold tuning and plotting."""
    row = dict(label)
    row.update(make_training_csv_row(label, sample_id, npy_name, split, args))
    row.update({
        "stamp": f"{ts:.6f}",
        "topic": topic,
        "frame_id": output_frame_id,
        "source_frame_id": source_frame_id,
        "tf_used": int(tf_used),
        "tf_dt_sec": "nan" if math.isnan(tf_dt) else f"{tf_dt:.6f}",
    })
    return row



def compute_degeneracy_label(points: np.ndarray,
                             args,
                             rng: np.random.Generator) -> Dict[str, object]:
    pts_n, normals = estimate_normals_open3d(
        points,
        voxel_size=args.voxel_size,
        normal_radius=args.normal_radius,
        normal_max_nn=args.normal_max_nn,
        max_normal_points=args.max_normal_points,
        rng=rng,
    )

    if normals.shape[0] < args.min_normals:
        return {
            "scene_type": "non_degenerate",
            "class_gt": CLASS_TO_ID["non_degenerate"],
            "eta_c": float("nan"),
            "rho21": float("nan"),
            "theta_top1_v_deg": float("nan"),
            "dir_x": 0.0,
            "dir_y": 0.0,
            "dir_z": 0.0,
            "dir_xy_valid": 0,
            "dir_angle_axis_rad": float("nan"),
            "dir_bin": -1,
            "normal_count": int(normals.shape[0]),
            "S_top1": 0,
            "S_top2": 0,
            "n1_x": 0.0,
            "n1_y": 0.0,
            "n1_z": 1.0,
            "n2_x": 0.0,
            "n2_y": 0.0,
            "n2_z": 1.0,
        }

    counts, el_idx, az_idx = bin_normals(normals, args.az_bins, args.el_bins)
    vertical_start = int(math.floor((1.0 - args.alpha_deg / 90.0) * args.el_bins))
    vertical_start = max(0, min(args.el_bins - 1, vertical_start))

    excluded = np.zeros_like(counts, dtype=bool)
    region1, S_top1, cell1 = find_peak_region(counts, excluded, args.merge_bins, vertical_start)
    excluded |= region1
    region2, S_top2, cell2 = find_peak_region(counts, excluded, args.merge_bins, vertical_start)

    n1 = mean_direction_for_region(normals, el_idx, az_idx, region1)
    n2 = mean_direction_for_region(normals, el_idx, az_idx, region2)

    residual = ~(region1 | region2)
    S_res_log = float(np.log1p(counts[residual].astype(np.float64)).sum())
    S_top_log = float(math.log1p(float(S_top1)) + math.log1p(float(S_top2)))
    eta_c = S_res_log / max(S_top_log, 1e-12)
    rho21 = float(S_top2) / max(float(S_top1), 1e-12)
    rho21_log = float(math.log1p(float(S_top2)) / max(math.log1p(float(S_top1)), 1e-12))
    ratio_for_label = rho21_log if args.peak_ratio_mode == "log" else rho21

    cos_v = abs(float(np.dot(n1, np.array([0.0, 0.0, 1.0], dtype=np.float32))))
    cos_v = max(-1.0, min(1.0, cos_v))
    theta_top1_v_deg = math.degrees(math.acos(cos_v))

    if eta_c <= args.tau_c and ratio_for_label >= args.tau_rho:
        scene_type = "tunnel_like"
    elif eta_c <= args.tau_c and ratio_for_label < args.tau_rho and theta_top1_v_deg <= args.alpha_deg:
        scene_type = "open_like"
    else:
        scene_type = "non_degenerate"

    d = np.cross(n1.astype(np.float64), n2.astype(np.float64))
    d_norm = np.linalg.norm(d)
    dir_xy_valid = 0
    dir_x = dir_y = dir_z = 0.0
    dir_angle_axis_rad = float("nan")
    dir_bin = -1

    if scene_type == "tunnel_like" and d_norm >= args.cross_min_norm:
        d = d / d_norm
        theta, b, dxy_unit = axis_angle_and_bin(d, args.num_dir_bins)
        if b >= 0:
            dir_xy_valid = 1
            dir_x = float(dxy_unit[0])
            dir_y = float(dxy_unit[1])
            dir_z = 0.0
            dir_angle_axis_rad = float(theta)
            dir_bin = int(b)

    return {
        "scene_type": scene_type,
        "class_gt": CLASS_TO_ID[scene_type],
        "eta_c": float(eta_c),
        "rho21": float(ratio_for_label),
        "rho21_raw": float(rho21),
        "rho21_log": float(rho21_log),
        "peak_ratio_mode": args.peak_ratio_mode,
        "theta_top1_v_deg": float(theta_top1_v_deg),
        "dir_x": dir_x,
        "dir_y": dir_y,
        "dir_z": dir_z,
        "dir_xy_valid": dir_xy_valid,
        "dir_angle_axis_rad": dir_angle_axis_rad,
        "dir_bin": dir_bin,
        "normal_count": int(normals.shape[0]),
        "S_top1": int(S_top1),
        "S_top2": int(S_top2),
        "n1_x": float(n1[0]),
        "n1_y": float(n1[1]),
        "n1_z": float(n1[2]),
        "n2_x": float(n2[0]),
        "n2_y": float(n2[1]),
        "n2_z": float(n2[2]),
    }


def save_pcd(points: np.ndarray, path: Path) -> None:
    import open3d as o3d
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points.astype(np.float64))
    o3d.io.write_point_cloud(str(path), pcd, write_ascii=False, compressed=True)


def import_anyreader():
    try:
        from rosbags.highlevel import AnyReader
    except Exception as exc:
        print("[ERROR] Cannot import rosbags.", file=sys.stderr)
        print("Install on Windows with: pip install rosbags lz4", file=sys.stderr)
        raise exc
    return AnyReader


def list_bag_topics(bag_path: Path) -> None:
    AnyReader = import_anyreader()
    with AnyReader([bag_path]) as reader:
        print("[INFO] Available topics:")
        for c in reader.connections:
            msgtype = str(c.msgtype)
            count = getattr(c, "msgcount", getattr(c, "message_count", "?"))
            mark = "  <== PointCloud2" if "PointCloud2" in msgtype else ""
            print(f"  {c.topic:50s} {msgtype:35s} count={count}{mark}")


def read_bag_and_export(args) -> int:
    AnyReader = import_anyreader()

    bag_path = Path(args.bag)
    if not bag_path.exists():
        print(f"[ERROR] Bag not found: {bag_path}", file=sys.stderr)
        return 2

    z_min = None if math.isnan(args.z_min) else args.z_min
    z_max = None if math.isnan(args.z_max) else args.z_max

    out_dir = Path(args.out_dir)
    points_dir = out_dir / "points"
    labels_dir = out_dir / "labels"
    pcd_dir = out_dir / "pcd"
    ensure_dir(points_dir)
    ensure_dir(labels_dir)
    if args.save_pcd:
        ensure_dir(pcd_dir)

    label_path = labels_dir / "deg_scene_labels.csv"
    fields = [
        "sample_id", "file_path", "scene_type", "class_gt",
        "dir_x", "dir_y", "dir_xy_valid", "rz_gt", "dir_exist_gt",
        "sample_weight", "split", "angle_deg", "dir_bin_gt",
        "dir_bin_valid", "dir_range", "source",
    ]

    debug_fields = [
        "sample_id", "file_path", "scene_type", "class_gt",
        "dir_x", "dir_y", "dir_z", "dir_xy_valid", "rz_gt", "dir_exist_gt",
        "sample_weight", "split", "angle_deg", "dir_bin_gt", "dir_bin_valid", "dir_range", "source",
        "stamp", "topic", "frame_id", "source_frame_id", "tf_used", "tf_dt_sec",
        "eta_c", "rho21", "rho21_raw", "rho21_log", "peak_ratio_mode", "theta_top1_v_deg",
        "S_top1", "S_top2", "normal_count",
        "n1_x", "n1_y", "n1_z", "n2_x", "n2_y", "n2_z",
        "dir_angle_axis_rad", "dir_bin",
    ]

    rng = np.random.default_rng(args.seed)
    exported = 0
    seen = 0
    class_counter = {k: 0 for k in CLASS_TO_ID.keys()}
    # Approximate total exported samples for assigning sequential train/val/test split.
    # It will be computed after bag_start/bag_end are available.
    expected_total_for_split = 0

    tf_times: List[float] = []
    tf_trans: List[np.ndarray] = []
    tf_quats: List[np.ndarray] = []
    if args.use_tf:
        print(f"[INFO] Loading TF buffer: {args.tf_parent} -> {args.tf_child} from {args.tf_topic}/{args.tf_static_topic}")
        tf_times, tf_trans, tf_quats = build_tf_buffer(bag_path, args)
        print(f"[INFO] Loaded {len(tf_times)} matching TF transforms")
        if tf_times:
            print(f"[INFO] TF time range: {tf_times[0]:.6f} -> {tf_times[-1]:.6f}")
        else:
            print("[ERROR] No matching TF transform found. Check --tf_parent/--tf_child and bag /tf.", file=sys.stderr)
            return 5

    with AnyReader([bag_path]) as reader:
        connections = [c for c in reader.connections if c.topic == args.topic]
        if not connections:
            print(f"[ERROR] Topic not found: {args.topic}", file=sys.stderr)
            print("[INFO] PointCloud2-like topics in this bag:", file=sys.stderr)
            for c in reader.connections:
                if "PointCloud2" in str(c.msgtype):
                    print(f"  {c.topic}  ({c.msgtype})", file=sys.stderr)
            return 3

        pc2_connections = [c for c in connections if "PointCloud2" in str(c.msgtype)]
        if not pc2_connections:
            print(f"[ERROR] Topic exists but is not PointCloud2: {args.topic}", file=sys.stderr)
            for c in connections:
                print(f"  {c.topic}  ({c.msgtype})", file=sys.stderr)
            return 4

        bag_start_ns = int(reader.start_time)
        bag_end_ns = int(reader.end_time)
        bag_start = bag_start_ns * 1e-9
        bag_end = bag_end_ns * 1e-9
        start_abs = bag_start + max(0.0, args.start_sec)
        end_abs = bag_end if args.end_sec < 0 else min(bag_start + args.end_sec, bag_end)

        if args.max_frames is not None and args.max_frames > 0:
            expected_total_for_split = int(args.max_frames)
        elif args.sample_hz and args.sample_hz > 0:
            expected_total_for_split = max(1, int(math.floor((end_abs - start_abs) * args.sample_hz)) + 1)
        else:
            expected_total_for_split = 0

        step = 0.0 if args.sample_hz <= 0 else 1.0 / args.sample_hz
        next_keep = start_abs

        print("[INFO] Backend: rosbags pure-python reader")
        print(f"[INFO] Class mapping: {CLASS_TO_ID}")
        print("[INFO] bag_start={:.3f}, bag_end={:.3f}, duration={:.1f}s".format(
            bag_start, bag_end, bag_end - bag_start
        ))
        print(f"[INFO] topic={args.topic}, output={out_dir}, sample_hz={args.sample_hz}")
        print(f"[INFO] crop_mode={args.crop_mode}, center_mode={args.center_mode}, range=[{args.range_min}, {args.range_max}]")
        print(f"[INFO] pre_filter_radius_max={args.pre_filter_radius_max}, pre_filter_abs_max={args.pre_filter_abs_max}")
        if args.use_tf:
            print(f"[INFO] TF transform enabled: point cloud {args.tf_parent} -> {args.tf_child}, lookup={args.tf_lookup_mode}, max_dt={args.tf_max_dt}s")
        print(f"[INFO] writing labels to: {label_path}")

        training_rows: List[Dict[str, object]] = []
        debug_label_path = labels_dir / "deg_scene_label_stats.csv"
        with open(label_path, "w", newline="", encoding="utf-8") as f, open(debug_label_path, "w", newline="", encoding="utf-8") as f_debug:
            writer = csv.DictWriter(f, fieldnames=fields, extrasaction="ignore")
            writer.writeheader()
            debug_writer = csv.DictWriter(f_debug, fieldnames=debug_fields, extrasaction="ignore")
            debug_writer.writeheader()

            for connection, timestamp, rawdata in reader.messages(connections=pc2_connections):
                ts = int(timestamp) * 1e-9
                if ts < start_abs or ts > end_abs:
                    continue
                seen += 1
                if step > 0.0 and ts + 1e-9 < next_keep:
                    continue
                if step > 0.0:
                    while next_keep <= ts + 1e-9:
                        next_keep += step

                try:
                    msg = reader.deserialize(rawdata, connection.msgtype)
                    raw = pointcloud2_to_xyz(msg)
                except Exception as exc:
                    print(f"[WARN] Failed to parse PointCloud2 at t={ts:.3f}: {exc}")
                    continue

                header = _get_attr(msg, "header", None)
                source_frame_id = ""
                if header is not None:
                    source_frame_id = str(_get_attr(header, "frame_id", ""))
                pc_stamp = _msg_stamp_sec(msg, fallback=ts)

                points_for_crop = raw
                tf_used = 0
                tf_dt = float("nan")
                output_frame_id = source_frame_id
                if args.use_tf:
                    if source_frame_id == args.tf_parent:
                        tt, qq, dt_abs = lookup_tf(tf_times, tf_trans, tf_quats, pc_stamp, args.tf_lookup_mode, args.tf_max_dt)
                        tf_dt = dt_abs
                        if tt is None or qq is None:
                            print(f"[WARN] skip frame t={ts:.3f}: no TF {args.tf_parent}->{args.tf_child} near stamp={pc_stamp:.6f}, dt={dt_abs:.3f}s")
                            continue
                        points_for_crop = transform_parent_points_to_child(raw, tt, qq)
                        tf_used = 1
                        output_frame_id = args.tf_child
                    elif source_frame_id == args.tf_child:
                        points_for_crop = raw
                        output_frame_id = args.tf_child
                        tf_used = 0
                        tf_dt = 0.0
                    else:
                        if args.tf_strict_frame:
                            print(f"[WARN] skip frame t={ts:.3f}: source frame_id={source_frame_id!r}, expected {args.tf_parent!r} or {args.tf_child!r}")
                            continue
                        points_for_crop = raw

                pre = pre_filter_points(points_for_crop, args.pre_filter_radius_max, args.pre_filter_abs_max)
                cropped = crop_points(pre, args.range_min, args.range_max, z_min, z_max, args.crop_mode)

                if args.debug_stats > 0 and seen <= args.debug_stats:
                    print(f"[DEBUG] t={ts:.3f}, pc_stamp={pc_stamp:.6f}, frame_id={source_frame_id}, output_frame={output_frame_id}, tf_used={tf_used}, tf_dt={tf_dt}")
                    if args.debug_fields:
                        print(f"[DEBUG] PointCloud2 fields: {pointcloud2_field_summary(msg)}")
                    print(f"[DEBUG] raw/source: {point_stats(raw)}")
                    if args.use_tf:
                        print(f"[DEBUG] after_tf_or_source: {point_stats(points_for_crop)}")
                    print(f"[DEBUG] pre_filter radius_max={args.pre_filter_radius_max}, abs_max={args.pre_filter_abs_max}: {point_stats(pre)}")
                    print(f"[DEBUG] crop_mode={args.crop_mode}, crop: {point_stats(cropped)}")

                if cropped.shape[0] < args.min_raw_points:
                    print(f"[WARN] skip frame t={ts:.3f}: only {cropped.shape[0]} points after crop | source: {point_stats(raw)} | after_tf: {point_stats(points_for_crop)} | pre: {point_stats(pre)}")
                    continue

                try:
                    proc_points = recenter_points(cropped, args.center_mode)
                    fixed = fixed_sample(proc_points, args.num_point, rng)
                    label = compute_degeneracy_label(proc_points, args, rng)
                except Exception as exc:
                    print(f"[WARN] Failed to label frame t={ts:.3f}: {exc}")
                    continue

                stamp_name = f"{ts:.6f}".replace(".", "_")
                sample_id = f"bag_frame_{exported:06d}_{stamp_name}"
                npy_name = f"{sample_id}.npy"
                npy_path = points_dir / npy_name
                np.save(npy_path, fixed)

                if args.save_pcd:
                    save_pcd(fixed, pcd_dir / (npy_path.stem + ".pcd"))

                split = split_name_from_index(exported, expected_total_for_split, args.train_ratio, args.val_ratio)
                row = make_training_csv_row(label, sample_id, npy_name, split, args)
                debug_row = make_debug_csv_row(
                    label, sample_id, npy_name, split, args,
                    ts=ts,
                    topic=connection.topic,
                    output_frame_id=output_frame_id,
                    source_frame_id=source_frame_id,
                    tf_used=tf_used,
                    tf_dt=tf_dt,
                )
                writer.writerow(row)
                debug_writer.writerow(debug_row)
                training_rows.append(row)
                class_counter[label["scene_type"]] += 1
                exported += 1

                print(
                    "[FRAME] #{:06d} class={}({}) raw={} crop={} normals={} eta={} rho={} rho_raw={} rho_log={} theta_v={} dir_bin={}".format(
                        exported,
                        label["scene_type"],
                        label["class_gt"],
                        raw.shape[0],
                        cropped.shape[0],
                        label["normal_count"],
                        "nan" if math.isnan(label["eta_c"]) else f"{label['eta_c']:.3f}",
                        "nan" if math.isnan(label["rho21"]) else f"{label['rho21']:.3f}",
                        "nan" if math.isnan(label.get("rho21_raw", float("nan"))) else f"{label['rho21_raw']:.3f}",
                        "nan" if math.isnan(label.get("rho21_log", float("nan"))) else f"{label['rho21_log']:.3f}",
                        "nan" if math.isnan(label["theta_top1_v_deg"]) else f"{label['theta_top1_v_deg']:.1f}",
                        label["dir_bin"],
                    ),
                    flush=True,
                )

                if args.max_frames > 0 and exported >= args.max_frames:
                    break

    # Also write split-specific CSV files compatible with generate_deg_scene_dataset.py.
    if exported > 0:
        for sp in ["train", "val", "test"]:
            sp_path = labels_dir / f"deg_scene_{sp}.csv"
            sp_rows = [r for r in training_rows if r.get("split") == sp]
            with open(sp_path, "w", newline="", encoding="utf-8") as f_sp:
                sp_writer = csv.DictWriter(f_sp, fieldnames=fields, extrasaction="ignore")
                sp_writer.writeheader()
                sp_writer.writerows(sp_rows)

    print(f"[DONE] seen topic messages in time window: {seen}")
    print(f"[DONE] exported {exported} frames")
    print(f"[DONE] class counts: {class_counter}")
    print(f"[DONE] labels: {label_path}")
    print(f"[DONE] debug stats: {labels_dir / 'deg_scene_label_stats.csv'}")
    if exported > 0:
        print(f"[DONE] split labels: {labels_dir / 'deg_scene_train.csv'}, {labels_dir / 'deg_scene_val.csv'}, {labels_dir / 'deg_scene_test.csv'}")
    if exported == 0:
        print("[HINT] No frames were exported. Check topic name, time window, range crop, and PointCloud2 fields.")
    return 0


def parse_args():
    p = argparse.ArgumentParser(description="Windows-friendly ROS1 bag PointCloud2 degeneracy labeler.")
    p.add_argument("--bag", required=True, help="Path to ROS1 .bag file")
    p.add_argument("--topic", default="/aligned_points", help="PointCloud2 topic")
    p.add_argument("--out_dir", default="auto", help="Output dataset directory. Use auto to generate by bag date and thresholds.")
    p.add_argument("--sample_hz", type=float, default=4.0, help="Extraction rate. Use 0 to keep every frame.")
    p.add_argument("--start_sec", type=float, default=0.0, help="Start offset from bag start time")
    p.add_argument("--end_sec", type=float, default=-1.0, help="End offset from bag start time; negative means bag end")
    p.add_argument("--max_frames", type=int, default=-1, help="Maximum frames to export")
    p.add_argument("--num_point", type=int, default=8192, help="Saved fixed point number per frame")
    p.add_argument("--seed", type=int, default=2026)
    p.add_argument("--train_ratio", type=float, default=0.70, help="Sequential train split ratio written to CSV.")
    p.add_argument("--val_ratio", type=float, default=0.15, help="Sequential validation split ratio written to CSV.")
    p.add_argument("--source", type=str, default="auto", help="Source string written to CSV. Use auto to generate by bag date and thresholds.")

    p.add_argument("--range_min", type=float, default=0.5)
    p.add_argument("--range_max", type=float, default=80.0)
    p.add_argument("--z_min", type=float, default=float("nan"), help="Optional z lower bound; NaN disables")
    p.add_argument("--z_max", type=float, default=float("nan"), help="Optional z upper bound; NaN disables")
    p.add_argument("--min_raw_points", type=int, default=100, help="Skip frames with too few cropped raw points")
    p.add_argument("--crop_mode", choices=["origin_range", "mean_range", "median_range", "none"],
                   default="origin_range",
                   help="Range-crop mode. Use none or median_range if points are in a global/map frame.")
    p.add_argument("--center_mode", choices=["none", "mean", "median"], default="none",
                   help="Optionally subtract mean/median before saving and labeling. Useful for global-frame clouds.")
    p.add_argument("--pre_filter_radius_max", type=float, default=120.0,
                   help="Optional raw-point radius hard filter before crop. Use e.g. 120 for base_link clouds with 5000m outliers.")
    p.add_argument("--pre_filter_abs_max", type=float, default=120.0,
                   help="Optional per-axis absolute coordinate hard filter before crop. Use e.g. 120 for local base_link clouds.")
    p.add_argument("--debug_stats", type=int, default=0,
                   help="Print raw/cropped coordinate statistics for the first N seen messages.")
    p.add_argument("--debug_fields", action="store_true",
                   help="Print PointCloud2 field layout during debug_stats.")

    p.add_argument("--use_tf", dest="use_tf", action="store_true", default=True,
                   help="Use TF to convert point clouds from --tf_parent frame to --tf_child frame before crop/labeling. Enabled by default.")
    p.add_argument("--no_tf", dest="use_tf", action="store_false",
                   help="Disable TF conversion.")
    p.add_argument("--tf_topic", default="/tf")
    p.add_argument("--tf_static_topic", default="/tf_static")
    p.add_argument("--tf_parent", default="map", help="Parent/source frame in TF, e.g., map for T_map_base_link")
    p.add_argument("--tf_child", default="base_link", help="Child/target frame in TF, e.g., base_link")
    p.add_argument("--tf_lookup_mode", choices=["nearest", "previous"], default="nearest")
    p.add_argument("--tf_max_dt", type=float, default=0.5,
                   help="Maximum allowed time difference between point cloud stamp and TF stamp. Negative disables the check.")
    p.add_argument("--tf_strict_frame", action="store_true",
                   help="When --use_tf is set, skip clouds whose frame_id is neither --tf_parent nor --tf_child.")

    p.add_argument("--voxel_size", type=float, default=0.15)
    p.add_argument("--normal_radius", type=float, default=0.6)
    p.add_argument("--normal_max_nn", type=int, default=30)
    p.add_argument("--max_normal_points", type=int, default=30000)
    p.add_argument("--min_normals", type=int, default=500)

    p.add_argument("--az_bins", type=int, default=18)
    p.add_argument("--el_bins", type=int, default=18)
    p.add_argument("--merge_bins", type=int, default=1)
    p.add_argument("--tau_c", type=float, default=26.0, help="Concentration threshold")
    p.add_argument("--tau_rho", type=float, default=0.80, help="Peak ratio threshold. Its meaning depends on --peak_ratio_mode.")
    p.add_argument("--peak_ratio_mode", choices=["raw", "log"], default="log", help="Use raw S_top2/S_top1 or log1p(S_top2)/log1p(S_top1) for tunnel/open split.")
    p.add_argument("--alpha_deg", type=float, default=20.0, help="Vertical angle threshold for open-like degeneracy")
    p.add_argument("--cross_min_norm", type=float, default=0.08)
    p.add_argument("--num_dir_bins", type=int, default=12, help="Axis bins on [0, pi), 12 means 15 degrees/bin")

    p.add_argument("--save_pcd", dest="save_pcd", action="store_true", default=True, help="Also save .pcd files. Enabled by default.")
    p.add_argument("--no_save_pcd", dest="save_pcd", action="store_false", help="Do not save .pcd files.")
    p.add_argument("--list_topics", action="store_true", help="Only list bag topics and exit")
    return p.parse_args()


def _tag_float_for_name(value: float, scale: int = 1) -> str:
    """Format threshold numbers for directory names."""
    try:
        v = float(value)
    except Exception:
        return str(value).replace(".", "p")
    if scale == 100:
        return f"{int(round(v * 100)):03d}"
    if abs(v - round(v)) < 1e-9:
        return str(int(round(v)))
    return (f"{v:.3f}".rstrip("0").rstrip(".")).replace(".", "p")


def _bag_date_tag(bag_path: Path) -> str:
    """
    Extract a unique date-time tag from bag filename.

    Examples:
        perception_plan_all_2026-05-18-13-04-24_4.bag
            -> 0518_130424_4

        perception_plan_all_2026-05-18-12-42-29_1.bag
            -> 0518_124229_1

    If only date is available:
            -> 0518

    Fallback:
            -> sanitized bag stem
    """
    import re
    name = bag_path.name

    # Full pattern: YYYY-MM-DD-HH-MM-SS_optionalIndex.bag
    m = re.search(
        r"(20\d{2})[-_](\d{2})[-_](\d{2})[-_](\d{2})[-_](\d{2})[-_](\d{2})(?:[_-](\d+))?",
        name,
    )
    if m:
        mmdd = f"{m.group(2)}{m.group(3)}"
        hhmmss = f"{m.group(4)}{m.group(5)}{m.group(6)}"
        idx = m.group(7)
        return f"{mmdd}_{hhmmss}_{idx}" if idx else f"{mmdd}_{hhmmss}"

    # Date-only fallback.
    m = re.search(r"(20\d{2})[-_]?(\d{2})[-_]?(\d{2})", name)
    if m:
        return f"{m.group(2)}{m.group(3)}"

    # Generic safe fallback from bag stem.
    stem = re.sub(r"[^A-Za-z0-9]+", "_", bag_path.stem).strip("_")
    return stem or "bag"


def _auto_name_tokens(args, bag_path: Path):
    date_tag = _bag_date_tag(bag_path)
    tau_tag = _tag_float_for_name(args.tau_c)
    rho_tag = _tag_float_for_name(args.tau_rho, scale=100)
    mode_tag = str(args.peak_ratio_mode).lower()
    return date_tag, tau_tag, mode_tag, rho_tag


def apply_auto_defaults(args):
    """
    Fill automatic output directory and source name.

    Example:
        bag date: 2026-05-18
        tau_c=26, peak_ratio_mode=log, tau_rho=0.80

    out_dir:
        ./0518_deg_file_bin_tau26_log080

    source:
        real_bag_0518_tf_tau26_log080
    """
    bag_path = Path(args.bag)
    date_tag, tau_tag, mode_tag, rho_tag = _auto_name_tokens(args, bag_path)
    auto_suffix = f"tau{tau_tag}_{mode_tag}{rho_tag}"

    if str(args.out_dir).strip().lower() in ["", "auto", "none"]:
        args.out_dir = f"./{date_tag}_deg_file_bin_{auto_suffix}"

    if str(args.source).strip().lower() in ["", "auto", "none"]:
        tf_tag = "tf" if args.use_tf else "notf"
        args.source = f"real_bag_{date_tag}_{tf_tag}_{auto_suffix}"

    return args


def main() -> int:
    args = parse_args()
    bag_path = Path(args.bag)

    if args.list_topics:
        list_bag_topics(bag_path)
        return 0

    args = apply_auto_defaults(args)
    print("[AUTO] simplified defaults enabled")
    print(f"[AUTO] out_dir={args.out_dir}")
    print(f"[AUTO] source={args.source}")
    print(f"[AUTO] tau_c={args.tau_c}, peak_ratio_mode={args.peak_ratio_mode}, tau_rho={args.tau_rho}, alpha_deg={args.alpha_deg}")
    print(f"[AUTO] sample_hz={args.sample_hz}, use_tf={args.use_tf}, tf_max_dt={args.tf_max_dt}, save_pcd={args.save_pcd}")
    return read_bag_and_export(args)


if __name__ == "__main__":
    raise SystemExit(main())
