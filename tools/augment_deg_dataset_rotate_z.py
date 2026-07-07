#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Rotate a deg_scene point-cloud dataset around the z axis.

Tunnel-like samples keep valid axial direction labels in [0, 180). Open-like
and nondeg_or_other samples keep direction labels invalid. If dominant normal
columns are present in the CSV, their xy components are rotated as well.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import csv
import math
import random
import shutil
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd


CLASS_TUNNEL = "tunnel_like"
CLASS_OPEN = "open_like"
CLASS_NONDEG = "nondeg_or_other"


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--input_root", required=True, help="Input dataset root containing points/, pcd/, and CSV.")
    p.add_argument("--csv", default="people_deg_scene_label_stats.csv",
                   help="CSV filename under input_root, or absolute CSV path.")
    p.add_argument("--output_root", default="auto",
                   help="Output dataset root. Use auto for <input_root>_rotaug_xN.")
    p.add_argument("--aug_per_sample", type=int, default=3,
                   help="Number of augmented rotated copies for each original sample.")
    p.add_argument("--rotation_mode", choices=["random", "uniform", "list"], default="random",
                   help="random: random angle in [min,max); uniform: evenly spaced; list: use --rotation_degs.")
    p.add_argument("--rotation_min_deg", type=float, default=0.0)
    p.add_argument("--rotation_max_deg", type=float, default=180.0,
                   help="For axis direction, 0-180 degrees is sufficient.")
    p.add_argument("--rotation_degs", default="",
                   help="Comma-separated angles in degrees, used when --rotation_mode list.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--num_dir_bins", type=int, default=12)
    p.add_argument("--include_original", action="store_true", default=True,
                   help="Copy original samples into output dataset. Default: True.")
    p.add_argument("--no_include_original", dest="include_original", action="store_false",
                   help="Do not copy original samples.")
    p.add_argument("--save_pcd", action="store_true",
                   help="Also save augmented PCD files. Requires open3d.")
    p.add_argument("--copy_original_pcd", action="store_true", default=True,
                   help="Copy original PCD files if they exist. Default: True.")
    p.add_argument("--no_copy_original_pcd", dest="copy_original_pcd", action="store_false")
    p.add_argument("--overwrite", action="store_true",
                   help="Overwrite output_root if it already exists.")
    p.add_argument("--source_suffix", default="rotaug_z",
                   help="Suffix appended to source field.")
    p.add_argument("--write_train_val_test", action="store_true",
                   help="Also write split CSVs according to the split column if present.")
    return p.parse_args()


def resolve_csv(input_root: Path, csv_arg: str) -> Path:
    p = Path(csv_arg)
    if p.is_absolute():
        return p
    return input_root / csv_arg


def auto_output_root(input_root: Path, aug_per_sample: int, rotation_mode: str) -> Path:
    return input_root.parent / f"{input_root.name}_rotaug_{rotation_mode}_x{aug_per_sample}"


def ensure_clean_dir(path: Path, overwrite: bool):
    if path.exists():
        if overwrite:
            shutil.rmtree(path)
        else:
            raise FileExistsError(
                f"Output directory already exists: {path}\n"
                f"Use --overwrite or choose another --output_root."
            )
    path.mkdir(parents=True, exist_ok=True)


def rotz(deg: float) -> np.ndarray:
    a = math.radians(deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s], [s, c]], dtype=np.float64)


def rotate_xy(x: float, y: float, deg: float) -> Tuple[float, float]:
    R = rotz(deg)
    v = R @ np.array([float(x), float(y)], dtype=np.float64)
    return float(v[0]), float(v[1])


def axis_angle_deg_from_xy(x: float, y: float) -> float:
    """Return the unoriented xy-axis angle in [0, 180)."""
    a = math.degrees(math.atan2(float(y), float(x)))
    return a % 180.0


def dir_bin_from_angle(angle_deg: float, num_dir_bins: int) -> int:
    bin_width = 180.0 / float(num_dir_bins)
    b = int(math.floor((angle_deg % 180.0) / bin_width))
    return max(0, min(num_dir_bins - 1, b))


def rotate_points_array(points: np.ndarray, deg: float) -> np.ndarray:
    out = np.array(points, copy=True)
    if out.ndim != 2 or out.shape[1] < 3:
        raise ValueError(f"Point array should be [N,C] with C>=3, got shape {out.shape}")
    R = rotz(deg)
    xy = out[:, :2].astype(np.float64)
    out[:, :2] = xy @ R.T
    return out


def normalize_xy(dx: float, dy: float) -> Tuple[float, float, float]:
    n = math.hypot(float(dx), float(dy))
    if n < 1e-12 or not math.isfinite(n):
        return 0.0, 0.0, 0.0
    return float(dx) / n, float(dy) / n, n


def is_tunnel_row(row: Dict) -> bool:
    scene = str(row.get("scene_type", "")).strip()
    try:
        cls = int(float(row.get("class_gt", -999)))
    except Exception:
        cls = -999
    return scene == CLASS_TUNNEL or cls == 0


def invalidate_direction(row: Dict):
    for k in ["dir_x", "dir_y"]:
        if k in row:
            row[k] = 0.0
    if "dir_z" in row:
        row["dir_z"] = 0.0
    if "dir_xy_valid" in row:
        row["dir_xy_valid"] = 0
    if "dir_exist_gt" in row:
        row["dir_exist_gt"] = 0
    if "angle_deg" in row:
        row["angle_deg"] = -1.0
    if "dir_bin_gt" in row:
        row["dir_bin_gt"] = -1
    if "dir_bin" in row:
        row["dir_bin"] = -1
    if "dir_bin_valid" in row:
        row["dir_bin_valid"] = 0
    if "dir_angle_axis_rad" in row:
        row["dir_angle_axis_rad"] = float("nan")


def rotate_direction_labels(row: Dict, deg: float, num_dir_bins: int):
    if not is_tunnel_row(row):
        invalidate_direction(row)
        return

    valid = int(float(row.get("dir_xy_valid", row.get("dir_bin_valid", 0)))) > 0
    if not valid:
        invalidate_direction(row)
        return

    dx = float(row.get("dir_x", 0.0))
    dy = float(row.get("dir_y", 0.0))
    dx2, dy2 = rotate_xy(dx, dy, deg)
    dx2, dy2, norm = normalize_xy(dx2, dy2)
    if norm < 1e-12:
        invalidate_direction(row)
        return

    angle = axis_angle_deg_from_xy(dx2, dy2)
    b = dir_bin_from_angle(angle, num_dir_bins)

    row["dir_x"] = dx2
    row["dir_y"] = dy2
    if "dir_z" in row:
        row["dir_z"] = 0.0
    row["dir_xy_valid"] = 1
    if "dir_exist_gt" in row:
        row["dir_exist_gt"] = 1
    row["angle_deg"] = angle
    row["dir_bin_gt"] = b
    if "dir_bin" in row:
        row["dir_bin"] = b
    if "dir_bin_valid" in row:
        row["dir_bin_valid"] = 1
    if "dir_angle_axis_rad" in row:
        row["dir_angle_axis_rad"] = math.radians(angle)


def rotate_peak_normals(row: Dict, deg: float):
    """Rotate n1/n2 dominant-normal xy components when present."""
    for prefix in ["n1", "n2"]:
        kx, ky = f"{prefix}_x", f"{prefix}_y"
        if kx in row and ky in row:
            try:
                x, y = float(row[kx]), float(row[ky])
            except Exception:
                continue
            x2, y2 = rotate_xy(x, y, deg)
            row[kx] = x2
            row[ky] = y2


def make_angles(args, sample_idx: int) -> List[float]:
    if args.aug_per_sample <= 0:
        return []

    if args.rotation_mode == "list":
        vals = [float(x.strip()) for x in args.rotation_degs.split(",") if x.strip()]
        if not vals:
            raise ValueError("--rotation_mode list requires --rotation_degs, e.g. 15,45,90")
        if args.aug_per_sample <= len(vals):
            return vals[:args.aug_per_sample]
        reps = int(math.ceil(args.aug_per_sample / len(vals)))
        return (vals * reps)[:args.aug_per_sample]

    if args.rotation_mode == "uniform":
        # Avoid 0 degrees so augmented samples are not exact duplicates.
        step = (args.rotation_max_deg - args.rotation_min_deg) / float(args.aug_per_sample + 1)
        return [args.rotation_min_deg + step * (i + 1) for i in range(args.aug_per_sample)]

    # random
    return [
        random.uniform(args.rotation_min_deg, args.rotation_max_deg)
        for _ in range(args.aug_per_sample)
    ]


def safe_float_str(v, ndigits=6):
    try:
        if math.isnan(float(v)):
            return "nan"
        return f"{float(v):.{ndigits}f}"
    except Exception:
        return v


def save_pcd_open3d(points_xyz: np.ndarray, path: Path):
    try:
        import open3d as o3d
    except Exception as e:
        raise RuntimeError("open3d is required for --save_pcd. Install with: pip install open3d") from e

    path.parent.mkdir(parents=True, exist_ok=True)
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(points_xyz[:, :3].astype(np.float64))
    o3d.io.write_point_cloud(str(path), pcd, write_ascii=False, compressed=False)


def copy_original_sample(row: Dict, input_root: Path, output_root: Path, rows_out: List[Dict]):
    new_row = dict(row)
    src_rel = Path(str(row["file_path"]))
    src_npy = input_root / src_rel
    dst_npy = output_root / src_rel
    dst_npy.parent.mkdir(parents=True, exist_ok=True)
    if src_npy.exists():
        shutil.copy2(src_npy, dst_npy)
    else:
        print(f"[WARN] missing original npy: {src_npy}")

    # copy pcd if it exists
    pcd_rel = Path("pcd") / (Path(str(src_rel)).stem + ".pcd")
    src_pcd = input_root / pcd_rel
    dst_pcd = output_root / pcd_rel
    if src_pcd.exists():
        dst_pcd.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(src_pcd, dst_pcd)

    rows_out.append(new_row)


def augment_one(row: Dict, input_root: Path, output_root: Path, angle_deg: float, aug_idx: int,
                num_dir_bins: int, save_pcd: bool, source_suffix: str) -> Dict | None:
    src_rel = Path(str(row["file_path"]))
    src_npy = input_root / src_rel
    if not src_npy.exists():
        print(f"[WARN] skip missing npy: {src_npy}")
        return None

    pts = np.load(src_npy)
    pts_rot = rotate_points_array(pts, angle_deg)

    base_id = str(row.get("sample_id", src_rel.stem))
    # Use 0.1-degree units in filenames to avoid decimal separators.
    angle_token = f"{int(round(angle_deg * 10)):04d}"
    new_id = f"{base_id}_rz{angle_token}_a{aug_idx:02d}"

    new_rel = Path("points") / f"{new_id}.npy"
    new_npy = output_root / new_rel
    new_npy.parent.mkdir(parents=True, exist_ok=True)
    np.save(new_npy, pts_rot)

    if save_pcd:
        new_pcd = output_root / "pcd" / f"{new_id}.pcd"
        save_pcd_open3d(pts_rot[:, :3], new_pcd)

    new_row = dict(row)
    new_row["sample_id"] = new_id
    new_row["file_path"] = str(new_rel).replace("\\", "/")
    new_row["source"] = f"{row.get('source', 'real')}_{source_suffix}"
    new_row["aug_type"] = "rotate_z"
    new_row["aug_rot_deg"] = float(angle_deg)

    rotate_direction_labels(new_row, angle_deg, num_dir_bins)
    rotate_peak_normals(new_row, angle_deg)

    return new_row


def write_csvs(df_out: pd.DataFrame, output_root: Path, csv_name: str, write_splits: bool):
    out_csv = output_root / csv_name
    df_out.to_csv(out_csv, index=False)

    # Also write the conventional training CSV name.
    if csv_name != "deg_scene_labels.csv":
        df_out.to_csv(output_root / "deg_scene_labels.csv", index=False)

    if write_splits and "split" in df_out.columns:
        for split in ["train", "val", "test"]:
            part = df_out[df_out["split"].astype(str) == split]
            if len(part) > 0:
                part.to_csv(output_root / f"deg_scene_{split}.csv", index=False)

    return out_csv


def main():
    args = parse_args()
    random.seed(args.seed)
    np.random.seed(args.seed)

    input_root = Path(args.input_root)
    csv_path = resolve_csv(input_root, args.csv)
    if not csv_path.exists():
        raise FileNotFoundError(f"CSV not found: {csv_path}")

    output_root = auto_output_root(input_root, args.aug_per_sample, args.rotation_mode) \
        if str(args.output_root).strip().lower() in ["", "auto", "none"] \
        else Path(args.output_root)

    ensure_clean_dir(output_root, args.overwrite)
    (output_root / "points").mkdir(parents=True, exist_ok=True)
    if args.save_pcd or args.copy_original_pcd:
        (output_root / "pcd").mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(csv_path)
    rows_out: List[Dict] = []

    # Ensure augmentation metadata columns exist.
    if "aug_type" not in df.columns:
        df["aug_type"] = "original"
    if "aug_rot_deg" not in df.columns:
        df["aug_rot_deg"] = 0.0

    print(f"[INFO] input_root={input_root}")
    print(f"[INFO] csv={csv_path}")
    print(f"[INFO] output_root={output_root}")
    print(f"[INFO] rows={len(df)}, aug_per_sample={args.aug_per_sample}, mode={args.rotation_mode}")

    for i, row in df.iterrows():
        row_dict = row.to_dict()

        if args.include_original:
            copy_original_sample(row_dict, input_root, output_root, rows_out)

        angles = make_angles(args, int(i))
        for j, deg in enumerate(angles):
            aug_row = augment_one(
                row=row_dict,
                input_root=input_root,
                output_root=output_root,
                angle_deg=float(deg),
                aug_idx=j,
                num_dir_bins=args.num_dir_bins,
                save_pcd=args.save_pcd,
                source_suffix=args.source_suffix,
            )
            if aug_row is not None:
                rows_out.append(aug_row)

        if (i + 1) % 100 == 0:
            print(f"[INFO] processed {i + 1}/{len(df)}", flush=True)

    df_out = pd.DataFrame(rows_out)

    # Preserve original column order and append new metadata columns.
    original_cols = list(df.columns)
    extra_cols = [c for c in df_out.columns if c not in original_cols]
    df_out = df_out[original_cols + extra_cols]

    out_csv_name = csv_path.name
    out_csv = write_csvs(df_out, output_root, out_csv_name, args.write_train_val_test)

    print("[DONE] augmentation finished")
    print(f"[DONE] output csv: {out_csv}")
    print(f"[DONE] compatible csv: {output_root / 'deg_scene_labels.csv'}")
    print(f"[DONE] total rows: {len(df_out)}")
    print("[DONE] scene counts:")
    print(df_out["scene_type"].value_counts(dropna=False).to_string())
    if "dir_bin_gt" in df_out.columns:
        print("[DONE] dir_bin_gt counts for tunnel_like:")
        tunnel = df_out[df_out["scene_type"].astype(str) == CLASS_TUNNEL]
        print(tunnel["dir_bin_gt"].value_counts(dropna=False).sort_index().to_string())


if __name__ == "__main__":
    main()
