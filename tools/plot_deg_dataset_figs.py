#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate diagnostic figures from deg_scene label CSV files.

The script reads the label statistics CSV produced by
``bag_degeneracy_labeler.py`` and writes compact plots under ``fig/`` by
default.
"""

from __future__ import annotations

from pathlib import Path
import argparse
import csv
import math
import re
import shutil
import subprocess
from collections import Counter

import numpy as np
import matplotlib.pyplot as plt


CLASS_NAME_MAP = {
    "non_degenerate": "nondeg_or_other",
    "nondeg_or_other": "nondeg_or_other",
    "tunnel_like": "tunnel_like",
    "open_like": "open_like",
}


def as_float(x, default=float("nan")):
    try:
        return float(x)
    except Exception:
        return default


def unit_or_nan(v):
    n = float(np.linalg.norm(v))
    if not np.isfinite(n) or n < 1e-12:
        return np.array([np.nan, np.nan, np.nan], dtype=float)
    return v / n


def parse_number_token(token: str, scale: int | None = None):
    if token is None:
        return None
    s = str(token).strip().replace("p", ".")
    try:
        v = float(s)
    except Exception:
        return None
    if scale:
        return v / scale
    return v


def parse_params_from_name(name: str):
    """
    Supported:
        0518_deg_file_bin_tau28_log080
        real_bag_0518_tf_tau28_log080
        xxx_tau26_raw045
        xxx_tau26p5_log080
    """
    name = str(name)
    m = re.search(r"tau(?P<tau>\d+(?:p\d+)?)[_-]?(?P<mode>log|raw)(?P<rho>\d+(?:p\d+)?)", name, flags=re.I)
    if not m:
        return {}
    tau_c = parse_number_token(m.group("tau"))
    mode = m.group("mode").lower()
    rho_token = m.group("rho")
    rho = parse_number_token(rho_token, None if "p" in rho_token else 100)
    out = {}
    if tau_c is not None:
        out["tau_c"] = tau_c
    if rho is not None:
        out["tau_rho"] = rho
    out["ratio_mode"] = mode
    return out


def read_first_row(csv_path: Path):
    try:
        with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
            reader = csv.DictReader(f)
            for row in reader:
                return row
    except Exception:
        return {}
    return {}


def choose_stats_csv(dataset_root: Path, explicit_csv: str | None):
    if explicit_csv:
        return Path(explicit_csv)
    candidates = [
        dataset_root / "labels" / "deg_scene_label_stats.csv",
        dataset_root / "labels" / "deg_scene_labels.csv",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError(
        "Cannot find CSV. Expected one of:\n"
        + "\n".join(str(p) for p in candidates)
    )


def auto_fill_plot_params(args, csv_path: Path):
    """
    Parameter priority:
    1. Explicit command-line values
    2. Folder name, e.g. 0518_deg_file_bin_tau28_log080
    3. CSV source column, e.g. real_bag_0518_tf_tau28_log080
    4. CSV explicit columns, if available
    5. Fallback defaults
    """
    first = read_first_row(csv_path)

    folder_params = parse_params_from_name(Path(args.dataset_root).name)
    source_params = parse_params_from_name(str(first.get("source", "")))

    csv_tau_c = as_float(first.get("tau_c_used", first.get("tau_c", "nan")))
    csv_tau_rho = as_float(first.get("tau_rho_used", first.get("tau_rho", "nan")))
    csv_alpha = as_float(first.get("alpha_deg_used", first.get("alpha_deg", "nan")))
    csv_mode = str(first.get("ratio_mode_used", first.get("peak_ratio_mode", first.get("ratio_mode", "")))).strip().lower()

    # Explicit args are None if not specified.
    if args.tau_c is None:
        if "tau_c" in folder_params:
            args.tau_c = folder_params["tau_c"]
            args.param_source_tau_c = "folder"
        elif "tau_c" in source_params:
            args.tau_c = source_params["tau_c"]
            args.param_source_tau_c = "source"
        elif math.isfinite(csv_tau_c):
            args.tau_c = csv_tau_c
            args.param_source_tau_c = "csv"
        else:
            args.tau_c = 25.0
            args.param_source_tau_c = "fallback"
    else:
        args.param_source_tau_c = "cli"

    if args.alpha_deg is None:
        if math.isfinite(csv_alpha):
            args.alpha_deg = csv_alpha
            args.param_source_alpha = "csv"
        else:
            args.alpha_deg = 20.0
            args.param_source_alpha = "fallback"
    else:
        args.param_source_alpha = "cli"

    if args.ratio_mode is None:
        if "ratio_mode" in folder_params:
            args.ratio_mode = folder_params["ratio_mode"]
            args.param_source_ratio_mode = "folder"
        elif "ratio_mode" in source_params:
            args.ratio_mode = source_params["ratio_mode"]
            args.param_source_ratio_mode = "source"
        elif csv_mode in ["raw", "log"]:
            args.ratio_mode = csv_mode
            args.param_source_ratio_mode = "csv"
        else:
            args.ratio_mode = "log"
            args.param_source_ratio_mode = "fallback"
    else:
        args.param_source_ratio_mode = "cli"

    if args.tau_rho_raw is None:
        if args.ratio_mode == "raw":
            if "tau_rho" in folder_params and folder_params.get("ratio_mode") == "raw":
                args.tau_rho_raw = folder_params["tau_rho"]
                args.param_source_tau_rho_raw = "folder"
            elif "tau_rho" in source_params and source_params.get("ratio_mode") == "raw":
                args.tau_rho_raw = source_params["tau_rho"]
                args.param_source_tau_rho_raw = "source"
            elif math.isfinite(csv_tau_rho):
                args.tau_rho_raw = csv_tau_rho
                args.param_source_tau_rho_raw = "csv"
            else:
                args.tau_rho_raw = 0.45
                args.param_source_tau_rho_raw = "fallback"
        else:
            args.tau_rho_raw = 0.45
            args.param_source_tau_rho_raw = "fallback"
    else:
        args.param_source_tau_rho_raw = "cli"

    if args.tau_rho_log is None:
        if args.ratio_mode == "log":
            if "tau_rho" in folder_params and folder_params.get("ratio_mode") == "log":
                args.tau_rho_log = folder_params["tau_rho"]
                args.param_source_tau_rho_log = "folder"
            elif "tau_rho" in source_params and source_params.get("ratio_mode") == "log":
                args.tau_rho_log = source_params["tau_rho"]
                args.param_source_tau_rho_log = "source"
            elif math.isfinite(csv_tau_rho):
                args.tau_rho_log = csv_tau_rho
                args.param_source_tau_rho_log = "csv"
            else:
                args.tau_rho_log = 0.80
                args.param_source_tau_rho_log = "fallback"
        else:
            args.tau_rho_log = 0.80
            args.param_source_tau_rho_log = "fallback"
    else:
        args.param_source_tau_rho_log = "cli"

    return args


def axis_angle_0_180_deg(x, y, xy_norm_min=0.0):
    if not (math.isfinite(x) and math.isfinite(y)):
        return float("nan")
    if math.hypot(x, y) < xy_norm_min:
        return float("nan")
    return math.degrees(math.atan2(y, x)) % 180.0


def vertical_angle_deg(v):
    if not np.all(np.isfinite(v)):
        return float("nan")
    z = abs(float(v[2]))
    z = max(-1.0, min(1.0, z))
    return math.degrees(math.acos(z))


def dir_angle_deg_from_xy(dx, dy, valid):
    if not (math.isfinite(dx) and math.isfinite(dy) and valid > 0.5):
        return float("nan")
    return math.degrees(math.atan2(dy, dx)) % 180.0


def read_rows(csv_path: Path, xy_norm_min):
    with csv_path.open("r", encoding="utf-8-sig", newline="") as f:
        reader = csv.DictReader(f)
        rows_raw = list(reader)
    if not rows_raw:
        raise RuntimeError(f"No rows found in {csv_path}")

    rows = []
    for i, r in enumerate(rows_raw, 1):
        n1 = unit_or_nan(np.array([as_float(r.get("n1_x")), as_float(r.get("n1_y")), as_float(r.get("n1_z"))], dtype=float))
        n2 = unit_or_nan(np.array([as_float(r.get("n2_x")), as_float(r.get("n2_y")), as_float(r.get("n2_z"))], dtype=float))
        s1 = as_float(r.get("S_top1"))
        s2 = as_float(r.get("S_top2"))

        rho_raw = as_float(r.get("rho21_raw", r.get("rho21")))
        if not math.isfinite(rho_raw) and math.isfinite(s1) and math.isfinite(s2) and s1 > 0:
            rho_raw = s2 / max(s1, 1e-12)

        rho_log = as_float(r.get("rho21_log"))
        if not math.isfinite(rho_log) and math.isfinite(s1) and math.isfinite(s2) and s1 > 0:
            rho_log = math.log1p(s2) / max(math.log1p(s1), 1e-12)

        scene = CLASS_NAME_MAP.get(r.get("scene_type", ""), r.get("scene_type", ""))

        dir_x = as_float(r.get("dir_x"))
        dir_y = as_float(r.get("dir_y"))
        dir_xy_valid = as_float(r.get("dir_xy_valid", r.get("dir_bin_valid", 0)))

        rows.append({
            "frame_index": i,
            "sample_id": r.get("sample_id", f"frame_{i:06d}"),
            "scene_type": scene,
            "class_gt": r.get("class_gt", ""),
            "split": r.get("split", ""),
            "source": r.get("source", ""),
            "eta_c": as_float(r.get("eta_c")),
            "rho21_raw": rho_raw,
            "rho21_log": rho_log,
            "theta_top1_v_deg": as_float(r.get("theta_top1_v_deg")),
            "S_top1": s1,
            "S_top2": s2,
            "raw_count": as_float(r.get("raw_count", r.get("raw_points", "nan"))),
            "crop_count": as_float(r.get("crop_count", r.get("crop_points", "nan"))),
            "normal_count": as_float(r.get("normal_count", "nan")),
            "dir_x": dir_x,
            "dir_y": dir_y,
            "dir_xy_valid": dir_xy_valid,
            "dir_angle_deg": dir_angle_deg_from_xy(dir_x, dir_y, dir_xy_valid),
            "dir_bin_gt": as_float(r.get("dir_bin_gt", r.get("dir_bin", "nan"))),
            "n1_x": float(n1[0]), "n1_y": float(n1[1]), "n1_z": float(n1[2]),
            "n2_x": float(n2[0]), "n2_y": float(n2[1]), "n2_z": float(n2[2]),
            "n1_xy_norm": float(np.linalg.norm(n1[:2])) if np.all(np.isfinite(n1[:2])) else float("nan"),
            "n2_xy_norm": float(np.linalg.norm(n2[:2])) if np.all(np.isfinite(n2[:2])) else float("nan"),
            "n1_axis_angle_deg": axis_angle_0_180_deg(float(n1[0]), float(n1[1]), xy_norm_min),
            "n2_axis_angle_deg": axis_angle_0_180_deg(float(n2[0]), float(n2[1]), xy_norm_min),
            "n1_vertical_angle_deg": vertical_angle_deg(n1),
            "n2_vertical_angle_deg": vertical_angle_deg(n2),
        })
    return rows


def arr(rows, key):
    return np.array([r.get(key, float("nan")) for r in rows], dtype=float)


def save_line(x, y, out, xlabel, ylabel, title, hline=None, hlabel=None):
    plt.figure(figsize=(11.5, 4.8))
    plt.plot(x, y, linewidth=1.2)
    if hline is not None:
        plt.axhline(hline, linestyle="--", linewidth=1.0, label=hlabel or f"{hline}")
        plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out, dpi=220)
    plt.close()


def save_two_lines(x, y1, y2, out, ylabel, title, label1="top-1", label2="top-2", hline=None, hlabel=None):
    plt.figure(figsize=(11.5, 4.8))
    plt.plot(x, y1, linewidth=1.1, label=label1)
    plt.plot(x, y2, linewidth=1.1, label=label2)
    if hline is not None:
        plt.axhline(hline, linestyle="--", linewidth=1.0, label=hlabel or f"{hline}")
    plt.xlabel("Frame index")
    plt.ylabel(ylabel)
    plt.title(title)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=220)
    plt.close()


def save_scatter(x, y, out, xlabel, ylabel, title, vline=None, hline=None, vlabel=None, hlabel=None):
    plt.figure(figsize=(7.5, 5.8))
    plt.scatter(x, y, s=14, alpha=0.8)
    if vline is not None:
        plt.axvline(vline, linestyle="--", linewidth=1.0, label=vlabel or f"{vline}")
    if hline is not None:
        plt.axhline(hline, linestyle="--", linewidth=1.0, label=hlabel or f"{hline}")
    if vline is not None or hline is not None:
        plt.legend()
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(out, dpi=220)
    plt.close()


def save_components(x, rows, out):
    plt.figure(figsize=(11.8, 5.5))
    for key in ["n1_x", "n1_y", "n1_z", "n2_x", "n2_y", "n2_z"]:
        plt.plot(x, arr(rows, key), linewidth=1.0, label=key)
    plt.xlabel("Frame index")
    plt.ylabel("Unit-normal component")
    plt.title("Top-1 and top-2 dominant normal vector components")
    plt.legend(ncol=3)
    plt.tight_layout()
    plt.savefig(out, dpi=220)
    plt.close()


def save_xy_projection_scatter(rows, out):
    n1x, n1y = arr(rows, "n1_x"), arr(rows, "n1_y")
    n2x, n2y = arr(rows, "n2_x"), arr(rows, "n2_y")
    s1, s2 = arr(rows, "S_top1"), arr(rows, "S_top2")
    s1 = np.nan_to_num(s1, nan=20.0)
    s2 = np.nan_to_num(s2, nan=20.0)
    s1 = np.clip(s1 / max(np.nanmax(s1), 1.0) * 120.0, 12.0, 120.0)
    s2 = np.clip(s2 / max(np.nanmax(s2), 1.0) * 120.0, 12.0, 120.0)
    plt.figure(figsize=(6.8, 6.5))
    plt.scatter(n1x, n1y, s=s1, alpha=0.65, label="top-1")
    plt.scatter(n2x, n2y, s=s2, alpha=0.65, label="top-2")
    th = np.linspace(0.0, 2.0 * math.pi, 360)
    plt.plot(np.cos(th), np.sin(th), linewidth=1.0)
    plt.axhline(0.0, linewidth=0.8)
    plt.axvline(0.0, linewidth=0.8)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlim(-1.08, 1.08)
    plt.ylim(-1.08, 1.08)
    plt.xlabel("normal x component")
    plt.ylabel("normal y component")
    plt.title("XY projection of top-1/top-2 dominant normal vectors")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=220)
    plt.close()


def save_xy_projection_arrow(rows, out, sample_step):
    x = arr(rows, "frame_index")
    n1x, n1y = arr(rows, "n1_x"), arr(rows, "n1_y")
    n2x, n2y = arr(rows, "n2_x"), arr(rows, "n2_y")
    sel = np.arange(0, len(x), max(1, sample_step), dtype=int)
    plt.figure(figsize=(12.0, 5.0))
    y1 = np.ones_like(sel, dtype=float) * 0.25
    y2 = np.ones_like(sel, dtype=float) * -0.25
    plt.quiver(x[sel], y1, n1x[sel], n1y[sel], angles="xy", scale_units="xy", scale=1.0, width=0.0025, label="top-1 XY projection")
    plt.quiver(x[sel], y2, n2x[sel], n2y[sel], angles="xy", scale_units="xy", scale=1.0, width=0.0025, label="top-2 XY projection")
    plt.ylim(-1.2, 1.2)
    plt.xlabel("Frame index")
    plt.ylabel("Arrow row")
    plt.title("XY projection arrows of top-1 and top-2 dominant normal vectors")
    plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=220)
    plt.close()


def save_dir_projection(rows, out):
    dx, dy = arr(rows, "dir_x"), arr(rows, "dir_y")
    valid = arr(rows, "dir_xy_valid")
    mask = np.isfinite(dx) & np.isfinite(dy) & (valid > 0.5)
    plt.figure(figsize=(6.8, 6.5))
    if np.any(mask):
        plt.scatter(dx[mask], dy[mask], s=18, alpha=0.75, label="deg direction XY")
    th = np.linspace(0.0, 2.0 * math.pi, 360)
    plt.plot(np.cos(th), np.sin(th), linewidth=1.0)
    plt.axhline(0.0, linewidth=0.8)
    plt.axvline(0.0, linewidth=0.8)
    plt.gca().set_aspect("equal", adjustable="box")
    plt.xlim(-1.08, 1.08)
    plt.ylim(-1.08, 1.08)
    plt.xlabel("dir_x")
    plt.ylabel("dir_y")
    plt.title("XY projection of degeneracy direction")
    if np.any(mask):
        plt.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=220)
    plt.close()


def save_3d_scatter(rows, out):
    from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
    n1x, n1y, n1z = arr(rows, "n1_x"), arr(rows, "n1_y"), arr(rows, "n1_z")
    n2x, n2y, n2z = arr(rows, "n2_x"), arr(rows, "n2_y"), arr(rows, "n2_z")
    fig = plt.figure(figsize=(7.0, 6.2))
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(n1x, n1y, n1z, s=12, alpha=0.7, label="top-1")
    ax.scatter(n2x, n2y, n2z, s=12, alpha=0.7, label="top-2")
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    ax.set_zlabel("z")
    ax.set_title("Top-1 and top-2 dominant normal directions on unit sphere")
    ax.legend()
    plt.tight_layout()
    plt.savefig(out, dpi=220)
    plt.close()


def save_bar_counts(rows, out):
    cnt = Counter(r["scene_type"] for r in rows)
    keys = ["tunnel_like", "open_like", "nondeg_or_other"]
    vals = [cnt.get(k, 0) for k in keys]
    plt.figure(figsize=(7.0, 5.0))
    plt.bar(keys, vals)
    plt.ylabel("Count")
    plt.title("Scene label counts")
    plt.tight_layout()
    plt.savefig(out, dpi=220)
    plt.close()


IEEE_DPI = 600
IEEE_LINE_COLOR = "#2F5F88"
IEEE_GRID_COLOR = "#D9DEE7"
IEEE_INK = "#222222"
IEEE_THRESHOLD_LINE_COLOR = "#777777"
IEEE_THRESHOLD_LOW_COLOR = "#F4EEE6"
IEEE_THRESHOLD_HIGH_COLOR = "#E8F1F5"
IEEE_SCENE_COLORS = {
    "nondeg_or_other": "#E8E8E8",
    "tunnel_like": "#D9E8F2",
    "open_like": "#E4EBD8",
}
IEEE_BAR_COLORS = {
    "nondeg_or_other": "#9AA4B2",
    "tunnel_like": "#6F97B8",
    "open_like": "#8EA878",
}
IEEE_SCENE_CODES = {
    "nondeg_or_other": 0,
    "tunnel_like": 1,
    "open_like": 2,
}
IEEE_SCENE_LABELS = {
    "nondeg_or_other": "Non-deg.",
    "tunnel_like": "Tunnel-like",
    "open_like": "Open-like",
}


def ieee_rc_params():
    return {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "mathtext.fontset": "stix",
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "font.size": 8.0,
        "axes.labelsize": 8.0,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 7.5,
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.05,
        "legend.frameon": False,
        "savefig.facecolor": "white",
        "figure.facecolor": "white",
    }


def set_ieee_style():
    plt.rcParams.update(ieee_rc_params())


def save_figure_all_formats(fig, out_base: Path, dpi: int = IEEE_DPI):
    out_base.parent.mkdir(parents=True, exist_ok=True)
    saved = []
    for suffix in [".png", ".pdf", ".svg"]:
        out = out_base.with_suffix(suffix)
        kwargs = {"bbox_inches": "tight", "pad_inches": 0.025, "facecolor": "white"}
        if suffix == ".png":
            kwargs["dpi"] = dpi
        fig.savefig(out, **kwargs)
        saved.append(out)
    plt.close(fig)
    return saved


def export_matlab_csv(records, out_csv: Path, fieldnames):
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            writer.writerow(rec)


def _matlab_str(path_or_name: str) -> str:
    return str(path_or_name).replace("'", "''")


def write_matlab_script_for_timeline(
    out_m: Path,
    csv_name: str,
    output_stem: str,
    ylabel: str,
    line_color: str = IEEE_LINE_COLOR,
    unit_y: bool = True,
    threshold_value: float | None = None,
):
    line_rgb = _hex_to_rgb_triplet(line_color)
    threshold_rgb = _hex_to_rgb_triplet(IEEE_THRESHOLD_LINE_COLOR)
    low_rgb = _hex_to_rgb_triplet(IEEE_THRESHOLD_LOW_COLOR)
    high_rgb = _hex_to_rgb_triplet(IEEE_THRESHOLD_HIGH_COLOR)
    scene_rgb = [_hex_to_rgb_triplet(IEEE_SCENE_COLORS[k]) for k in ["nondeg_or_other", "tunnel_like", "open_like"]]
    out_m.parent.mkdir(parents=True, exist_ok=True)
    out_m.write_text(
        f"""% Auto-generated IEEE-style redraw script.
scriptDir = fileparts(mfilename('fullpath'));
T = readtable(fullfile(scriptDir, '{_matlab_str(csv_name)}'), 'TextType', 'string');
seqs = unique(T.sequence_label, 'stable');
nSeq = numel(seqs);
figH = max(5.2, 3.4 * nSeq + 1.0);
fig = figure('Color', 'w', 'Units', 'centimeters', 'Position', [2 2 8.8 figH]);
tiledlayout(nSeq, 1, 'TileSpacing', 'compact', 'Padding', 'compact');
lineColor = [{line_rgb[0]:.6f} {line_rgb[1]:.6f} {line_rgb[2]:.6f}];
thresholdColor = [{threshold_rgb[0]:.6f} {threshold_rgb[1]:.6f} {threshold_rgb[2]:.6f}];
lowBandColor = [{low_rgb[0]:.6f} {low_rgb[1]:.6f} {low_rgb[2]:.6f}];
highBandColor = [{high_rgb[0]:.6f} {high_rgb[1]:.6f} {high_rgb[2]:.6f}];
sceneColors = [
    {scene_rgb[0][0]:.6f} {scene_rgb[0][1]:.6f} {scene_rgb[0][2]:.6f};
    {scene_rgb[1][0]:.6f} {scene_rgb[1][1]:.6f} {scene_rgb[1][2]:.6f};
    {scene_rgb[2][0]:.6f} {scene_rgb[2][1]:.6f} {scene_rgb[2][2]:.6f}
];
for s = 1:nSeq
    ax = nexttile;
    hold(ax, 'on');
    Ti = T(T.sequence_label == seqs(s), :);
    x = Ti.frame_index;
    y = Ti.value;
    hasThreshold = ismember('threshold', Ti.Properties.VariableNames) && height(Ti) > 0 && isfinite(Ti.threshold(1));
    threshold = NaN;
    if hasThreshold
        threshold = Ti.threshold(1);
    end
    if {_matlab_bool(unit_y)}
        y = min(max(y, 0), 1);
        yBottom = 0.0;
        yTop = 1.02;
        yTicks = [0 0.25 0.5 0.75 1.0];
    else
        finiteY = y(isfinite(y));
        if isempty(finiteY)
            yBottom = 0.0;
            yTop = 1.0;
        else
            yScaleValues = finiteY;
            if hasThreshold
                yScaleValues = [yScaleValues; threshold];
            end
            yLow = min(yScaleValues);
            yHigh = max(yScaleValues);
            yPad = max((yHigh - yLow) * 0.10, max(abs(yHigh), 1.0) * 0.035);
            yBottom = max(0.0, yLow - yPad);
            yTop = yHigh + yPad;
        end
        yTicks = 'auto';
    end
    if hasThreshold
        if {_matlab_bool(unit_y)}
            yTop = max(yTop, threshold * 1.05);
        end
    end
    x0lim = min(x) - 0.5;
    x1lim = max(x) + 0.5;
    if hasThreshold
        thresholdY = min(max(threshold, yBottom), yTop);
        patch(ax, [x0lim x1lim x1lim x0lim], [yBottom yBottom thresholdY thresholdY], lowBandColor, ...
            'EdgeColor', 'none', 'FaceAlpha', 0.55);
        patch(ax, [x0lim x1lim x1lim x0lim], [thresholdY thresholdY yTop yTop], highBandColor, ...
            'EdgeColor', 'none', 'FaceAlpha', 0.55);
    end
    if ismember('scene_code', Ti.Properties.VariableNames) && height(Ti) > 0
        codes = Ti.scene_code;
        runStart = 1;
        for ii = 2:(height(Ti) + 1)
            if ii == height(Ti) + 1 || codes(ii) ~= codes(runStart)
                code = codes(runStart);
                if code >= 0 && code <= 2
                    x0 = x(runStart) - 0.5;
                    x1 = x(ii - 1) + 0.5;
                    patch(ax, [x0 x1 x1 x0], [0 0 yTop yTop], sceneColors(code + 1, :), ...
                        'EdgeColor', 'none', 'FaceAlpha', 0.10);
                end
                runStart = ii;
            end
        end
    end
    plot(ax, x, y, '-', 'Color', lineColor, 'LineWidth', 1.05);
    if hasThreshold
        plot(ax, [x0lim x1lim], [threshold threshold], '--', 'Color', thresholdColor, 'LineWidth', 0.75);
    end
    ylim(ax, [yBottom yTop]);
    if ischar(yTicks)
        yticks(ax, yTicks);
    else
        yticks(ax, yTicks);
    end
    xlim(ax, [x0lim, x1lim]);
    grid(ax, 'on');
    ax.XGrid = 'off';
    ax.YGrid = 'on';
    ax.GridColor = [0.86 0.88 0.91];
    ax.GridAlpha = 0.75;
    ax.LineWidth = 0.8;
    ax.FontName = 'Times New Roman';
    ax.FontSize = 8;
    box(ax, 'on');
    ylabel(ax, '{_matlab_str(ylabel)}', 'Interpreter', 'latex');
    if s == nSeq
        xlabel(ax, 'Frame index');
    else
        ax.XTickLabel = [];
    end
end
outputStem = '{_matlab_str(output_stem)}';
savefig(fig, fullfile(scriptDir, [outputStem '.fig']));
exportgraphics(fig, fullfile(scriptDir, [outputStem '.pdf']), 'ContentType', 'vector');
exportgraphics(fig, fullfile(scriptDir, [outputStem '.png']), 'Resolution', 600);
""",
        encoding="utf-8",
    )


def _matlab_bool(value: bool) -> str:
    return "true" if value else "false"


def write_matlab_script_for_bar(out_m: Path, csv_name: str, output_stem: str):
    bar_rgbs = [_hex_to_rgb_triplet(IEEE_BAR_COLORS[k]) for k in ["nondeg_or_other", "tunnel_like", "open_like"]]
    out_m.parent.mkdir(parents=True, exist_ok=True)
    out_m.write_text(
        f"""% Auto-generated IEEE-style redraw script.
scriptDir = fileparts(mfilename('fullpath'));
T = readtable(fullfile(scriptDir, '{_matlab_str(csv_name)}'), 'TextType', 'string');
cats = ["Non-deg.", "Tunnel-like", "Open-like"];
seqs = unique(T.sequence_label, 'stable');
counts = zeros(numel(seqs), numel(cats));
for s = 1:numel(seqs)
    for c = 1:numel(cats)
        mask = T.sequence_label == seqs(s) & T.category_label == cats(c);
        if any(mask)
            idx = find(mask, 1, 'first');
            counts(s, c) = T.count(idx);
        end
    end
end
fig = figure('Color', 'w', 'Units', 'centimeters', 'Position', [2 2 8.8 4.8]);
ax = axes(fig);
hold(ax, 'on');
colors = [
    {bar_rgbs[0][0]:.6f} {bar_rgbs[0][1]:.6f} {bar_rgbs[0][2]:.6f};
    {bar_rgbs[1][0]:.6f} {bar_rgbs[1][1]:.6f} {bar_rgbs[1][2]:.6f};
    {bar_rgbs[2][0]:.6f} {bar_rgbs[2][1]:.6f} {bar_rgbs[2][2]:.6f}
];
nSeq = max(numel(seqs), 1);
nCat = numel(cats);
groupWidth = min(0.78, nSeq / (nSeq + 1.5));
for s = 1:nSeq
    xs = (1:nCat) - groupWidth / 2 + (2 * s - 1) * groupWidth / (2 * nSeq);
    alphaVal = max(0.42, 1.0 - 0.23 * (s - 1));
    for c = 1:nCat
        bar(ax, xs(c), counts(s, c), groupWidth / nSeq * 0.88, ...
            'FaceColor', colors(c, :), 'EdgeColor', [0.22 0.22 0.22], ...
            'LineWidth', 0.6, 'FaceAlpha', alphaVal);
        text(ax, xs(c), counts(s, c), sprintf('%d', round(counts(s, c))), ...
            'HorizontalAlignment', 'center', 'VerticalAlignment', 'bottom', ...
            'FontName', 'Times New Roman', 'FontSize', 7);
    end
end
set(ax, 'XTick', 1:nCat, 'XTickLabel', cats, 'FontName', 'Times New Roman', 'FontSize', 8, ...
    'LineWidth', 0.8);
ylabel(ax, 'Number of frames');
xlim(ax, [0.45 nCat + 0.55]);
ylim(ax, [0 max(counts(:)) * 1.15 + 1]);
grid(ax, 'on');
ax.XGrid = 'off';
ax.YGrid = 'on';
ax.GridColor = [0.86 0.88 0.91];
ax.GridAlpha = 0.75;
box(ax, 'on');
if nSeq > 1
    lg = legend(ax, seqs, 'Location', 'northeast');
    lg.Box = 'off';
end
outputStem = '{_matlab_str(output_stem)}';
savefig(fig, fullfile(scriptDir, [outputStem '.fig']));
exportgraphics(fig, fullfile(scriptDir, [outputStem '.pdf']), 'ContentType', 'vector');
exportgraphics(fig, fullfile(scriptDir, [outputStem '.png']), 'Resolution', 600);
""",
        encoding="utf-8",
    )


def _hex_to_rgb_triplet(color: str):
    color = color.lstrip("#")
    return tuple(int(color[i:i + 2], 16) / 255.0 for i in (0, 2, 4))


def maybe_run_matlab_scripts(scripts, run_matlab: bool):
    if not scripts:
        return
    matlab = shutil.which("matlab")
    if matlab is None:
        print("[INFO] MATLAB command not found. Run the generated .m scripts in MATLAB to create .fig files.")
        return
    if not run_matlab:
        print("[INFO] MATLAB detected. Pass --run_matlab to execute the generated redraw scripts automatically.")
        return
    for script in scripts:
        script_text = str(Path(script).resolve()).replace("\\", "/").replace("'", "''")
        cmd = [matlab, "-batch", f"run('{script_text}')"]
        print(f"[INFO] Running MATLAB redraw script: {script}")
        subprocess.run(cmd, check=False)


def _split_sequence_groups(rows):
    sources = []
    for r in rows:
        src = str(r.get("source", "")).strip()
        if src and src not in sources:
            sources.append(src)
    if len(sources) <= 1:
        return [("Sequence 1", rows)]
    groups = []
    for i, src in enumerate(sources, 1):
        groups.append((f"Sequence {i}", [r for r in rows if str(r.get("source", "")).strip() == src]))
    return groups


def _scene_code(scene_type: str) -> int:
    return IEEE_SCENE_CODES.get(CLASS_NAME_MAP.get(scene_type, scene_type), -1)


def _draw_scene_spans(ax, x, scene_types):
    if len(x) == 0:
        return
    start = 0
    scenes = [CLASS_NAME_MAP.get(s, s) for s in scene_types]
    for i in range(1, len(scenes) + 1):
        if i == len(scenes) or scenes[i] != scenes[start]:
            color = IEEE_SCENE_COLORS.get(scenes[start])
            if color:
                ax.axvspan(float(x[start]) - 0.5, float(x[i - 1]) + 0.5, color=color, alpha=0.10, lw=0)
            start = i


def save_ieee_timeline(
    rows,
    key: str,
    out_base: Path,
    ylabel: str,
    matlab_dir: Path,
    unit_y: bool = True,
    threshold_value: float | None = None,
):
    groups = _split_sequence_groups(rows)
    threshold = float(threshold_value) if threshold_value is not None and math.isfinite(float(threshold_value)) else float("nan")
    csv_records = []
    for seq_idx, (seq_label, seq_rows) in enumerate(groups, 1):
        for local_idx, row in enumerate(seq_rows, 1):
            raw_value = as_float(row.get(key))
            if unit_y:
                # Unitless ratio panels use a fixed [0, 1] range; outliers are clipped for display only.
                value = float(np.clip(raw_value, 0.0, 1.0)) if math.isfinite(raw_value) else float("nan")
            else:
                # eta_c keeps the project-defined value and scale.
                value = raw_value
            scene = CLASS_NAME_MAP.get(row.get("scene_type", ""), row.get("scene_type", ""))
            csv_records.append({
                "sequence_index": seq_idx,
                "sequence_label": seq_label,
                "frame_index": local_idx,
                "value": value,
                "raw_value": raw_value,
                "threshold": threshold,
                "scene_type": scene,
                "scene_code": _scene_code(scene),
            })

    with plt.rc_context(ieee_rc_params()):
        fig_h = max(2.05, 1.45 * len(groups) + 0.45)
        fig, axes = plt.subplots(len(groups), 1, figsize=(3.5, fig_h), sharex=False)
        if len(groups) == 1:
            axes = [axes]
        for idx, (ax, (seq_label, seq_rows)) in enumerate(zip(axes, groups)):
            x = np.arange(1, len(seq_rows) + 1, dtype=float)
            y_raw = np.array([as_float(r.get(key)) for r in seq_rows], dtype=float)
            y = np.clip(y_raw, 0.0, 1.0) if unit_y else y_raw
            finite_y = y[np.isfinite(y)]
            if unit_y:
                y_bottom = 0.0
                y_top = 1.02
            else:
                if finite_y.size:
                    scale_values = finite_y
                    if math.isfinite(threshold):
                        scale_values = np.concatenate([scale_values, np.array([threshold], dtype=float)])
                    y_low = float(np.nanmin(scale_values))
                    y_high = float(np.nanmax(scale_values))
                    y_pad = max((y_high - y_low) * 0.10, max(abs(y_high), 1.0) * 0.035)
                    y_bottom = max(0.0, y_low - y_pad)
                    y_top = y_high + y_pad
                else:
                    y_bottom = 0.0
                    y_top = 1.0
            if math.isfinite(threshold):
                if unit_y:
                    y_top = max(y_top, threshold * 1.05)
                threshold_y = min(max(threshold, y_bottom), y_top)
                ax.axhspan(y_bottom, threshold_y, color=IEEE_THRESHOLD_LOW_COLOR, alpha=0.62, lw=0, zorder=0)
                ax.axhspan(threshold_y, y_top, color=IEEE_THRESHOLD_HIGH_COLOR, alpha=0.62, lw=0, zorder=0)
            _draw_scene_spans(ax, x, [r.get("scene_type", "") for r in seq_rows])
            ax.plot(x, y, color=IEEE_LINE_COLOR, lw=1.05, zorder=3)
            if math.isfinite(threshold):
                ax.axhline(threshold, color=IEEE_THRESHOLD_LINE_COLOR, lw=0.75, ls=(0, (3, 2)), zorder=2)
            ax.set_ylim(y_bottom, y_top)
            if unit_y:
                ax.set_yticks([0.0, 0.25, 0.5, 0.75, 1.0])
            ax.set_xlim(float(x[0]) - 0.5 if len(x) else 0.5, float(x[-1]) + 0.5 if len(x) else 1.5)
            ax.set_ylabel(ylabel)
            ax.grid(axis="y", color=IEEE_GRID_COLOR, linewidth=0.45, alpha=0.85)
            ax.tick_params(direction="out", length=2.2, width=0.7, colors=IEEE_INK)
            for spine in ax.spines.values():
                spine.set_linewidth(0.8)
                spine.set_color(IEEE_INK)
            if idx == len(groups) - 1:
                ax.set_xlabel("Frame index")
            else:
                ax.set_xticklabels([])
        fig.align_ylabels()
        saved = save_figure_all_formats(fig, out_base)

    csv_path = matlab_dir / f"{out_base.name}.csv"
    export_matlab_csv(
        csv_records,
        csv_path,
        ["sequence_index", "sequence_label", "frame_index", "value", "raw_value", "threshold", "scene_type", "scene_code"],
    )
    script_path = matlab_dir / f"draw_{out_base.name}.m"
    write_matlab_script_for_timeline(
        script_path,
        csv_path.name,
        out_base.name,
        ylabel,
        unit_y=unit_y,
        threshold_value=threshold_value,
    )
    return saved, [csv_path, script_path]


def save_ieee_scene_label_count_bar(rows, out_base: Path, matlab_dir: Path):
    groups = _split_sequence_groups(rows)
    categories = ["nondeg_or_other", "tunnel_like", "open_like"]
    csv_records = []
    counts_by_group = []
    for seq_idx, (seq_label, seq_rows) in enumerate(groups, 1):
        cnt = Counter(CLASS_NAME_MAP.get(r["scene_type"], r["scene_type"]) for r in seq_rows)
        vals = [cnt.get(k, 0) for k in categories]
        counts_by_group.append(vals)
        for key, value in zip(categories, vals):
            csv_records.append({
                "sequence_index": seq_idx,
                "sequence_label": seq_label,
                "category_key": key,
                "category_label": IEEE_SCENE_LABELS[key],
                "count": int(value),
            })

    counts = np.asarray(counts_by_group, dtype=float)
    with plt.rc_context(ieee_rc_params()):
        fig, ax = plt.subplots(figsize=(3.5, 1.95))
        x = np.arange(len(categories), dtype=float)
        n_seq = max(len(groups), 1)
        group_width = min(0.78, n_seq / (n_seq + 1.5))
        for seq_idx, (seq_label, _) in enumerate(groups):
            offsets = x - group_width / 2.0 + (2 * seq_idx + 1) * group_width / (2.0 * n_seq)
            alpha = max(0.42, 1.0 - 0.23 * seq_idx)
            bars = ax.bar(
                offsets,
                counts[seq_idx],
                width=group_width / n_seq * 0.88,
                color=[IEEE_BAR_COLORS[k] for k in categories],
                edgecolor=IEEE_INK,
                linewidth=0.45,
                alpha=alpha,
                label=seq_label,
            )
            for bar, value in zip(bars, counts[seq_idx]):
                ax.text(
                    bar.get_x() + bar.get_width() / 2.0,
                    bar.get_height(),
                    f"{int(value)}",
                    ha="center",
                    va="bottom",
                    fontsize=7.2,
                    color=IEEE_INK,
                )
        ax.set_xticks(x)
        ax.set_xticklabels([IEEE_SCENE_LABELS[k] for k in categories])
        ax.set_ylabel("Number of frames")
        ax.set_ylim(0.0, max(float(np.max(counts)) * 1.16, 1.0))
        ax.grid(axis="y", color=IEEE_GRID_COLOR, linewidth=0.45, alpha=0.85)
        ax.tick_params(direction="out", length=2.2, width=0.7, colors=IEEE_INK)
        for spine in ax.spines.values():
            spine.set_linewidth(0.8)
            spine.set_color(IEEE_INK)
        if len(groups) > 1:
            ax.legend(loc="upper right")
        saved = save_figure_all_formats(fig, out_base)

    csv_path = matlab_dir / f"{out_base.name}.csv"
    export_matlab_csv(
        csv_records,
        csv_path,
        ["sequence_index", "sequence_label", "category_key", "category_label", "count"],
    )
    script_path = matlab_dir / f"draw_{out_base.name}.m"
    write_matlab_script_for_bar(script_path, csv_path.name, out_base.name)
    return saved, [csv_path, script_path]


def save_ieee_label_generation_figures(rows, out_dir: Path, args, run_matlab: bool = False):
    ieee_dir = out_dir / "ieee_label_generation"
    matlab_dir = ieee_dir / "matlab"
    matlab_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    matlab_assets = []

    saved, assets = save_ieee_timeline(
        rows,
        "eta_c",
        ieee_dir / "01_eta_c_timeline_ieee",
        r"Normal concentration factor, $\eta_c$",
        matlab_dir,
        unit_y=False,
        threshold_value=args.tau_c,
    )
    outputs.extend(saved)
    matlab_assets.extend(assets)

    saved, assets = save_ieee_timeline(
        rows,
        "rho21_log",
        ieee_dir / "02_rho21_timeline_ieee",
        r"Relative peak ratio, $\rho_{21}$",
        matlab_dir,
        unit_y=True,
        threshold_value=args.tau_rho_log,
    )
    outputs.extend(saved)
    matlab_assets.extend(assets)

    saved, assets = save_ieee_scene_label_count_bar(
        rows,
        ieee_dir / "14_scene_label_count_bar_ieee",
        matlab_dir,
    )
    outputs.extend(saved)
    matlab_assets.extend(assets)

    maybe_run_matlab_scripts([p for p in matlab_assets if p.suffix.lower() == ".m"], run_matlab)
    return outputs + matlab_assets


def save_dir_bin_hist(rows, out):
    bins = arr(rows, "dir_bin_gt")
    bins = bins[np.isfinite(bins)]
    if len(bins) == 0:
        bins = np.array([-1.0])
    plt.figure(figsize=(8.0, 5.0))
    uniq, counts = np.unique(bins.astype(int), return_counts=True)
    plt.bar(uniq.astype(str), counts)
    plt.xlabel("dir_bin_gt")
    plt.ylabel("Count")
    plt.title("Direction-bin distribution")
    plt.tight_layout()
    plt.savefig(out, dpi=220)
    plt.close()


def write_summary(rows, out_txt, args, csv_path):
    def stat(key):
        v = arr(rows, key)
        if not np.any(np.isfinite(v)):
            return "nan/nan/nan"
        return f"{np.nanmin(v):.4f}/{np.nanmedian(v):.4f}/{np.nanmax(v):.4f}"
    cnt = Counter(r["scene_type"] for r in rows)
    with out_txt.open("w", encoding="utf-8") as f:
        f.write(f"csv_path = {csv_path}\n")
        f.write(f"rows = {len(rows)}\n")
        f.write(f"tau_c = {args.tau_c} ({args.param_source_tau_c})\n")
        f.write(f"ratio_mode = {args.ratio_mode} ({args.param_source_ratio_mode})\n")
        f.write(f"tau_rho_raw = {args.tau_rho_raw} ({args.param_source_tau_rho_raw})\n")
        f.write(f"tau_rho_log = {args.tau_rho_log} ({args.param_source_tau_rho_log})\n")
        f.write(f"alpha_deg = {args.alpha_deg} ({args.param_source_alpha})\n")
        f.write(f"xy_norm_min = {args.xy_norm_min}\n")
        f.write("\nCounts:\n")
        f.write(f"tunnel_like = {cnt.get('tunnel_like', 0)}\n")
        f.write(f"open_like = {cnt.get('open_like', 0)}\n")
        f.write(f"nondeg_or_other = {cnt.get('nondeg_or_other', 0)}\n")
        f.write("\nStatistics (min/median/max):\n")
        for key in ["eta_c", "rho21_raw", "rho21_log", "theta_top1_v_deg",
                    "n1_vertical_angle_deg", "n2_vertical_angle_deg",
                    "n1_xy_norm", "n2_xy_norm", "dir_angle_deg",
                    "raw_count", "crop_count", "normal_count"]:
            f.write(f"{key} = {stat(key)}\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset_root", default=".", help="Dataset root directory.")
    ap.add_argument("--csv", default="", help="Optional explicit stats CSV path.")
    ap.add_argument("--out_dir", default="", help="Optional figure output directory. Default: <dataset_root>/fig")
    ap.add_argument("--tau_c", type=float, default=None, help="Override tau_c. If omitted, parse from folder/source/CSV.")
    ap.add_argument("--tau_rho_raw", type=float, default=None, help="Override raw ratio threshold.")
    ap.add_argument("--tau_rho_log", type=float, default=None, help="Override log ratio threshold.")
    ap.add_argument("--ratio_mode", choices=["raw", "log"], default=None, help="Override ratio mode.")
    ap.add_argument("--alpha_deg", type=float, default=None, help="Override vertical angle threshold.")
    ap.add_argument("--xy_norm_min", type=float, default=0.08)
    ap.add_argument("--sample_step", type=int, default=5)
    ap.add_argument("--run_matlab", action="store_true", help="Run generated MATLAB redraw scripts if matlab is available.")
    args = ap.parse_args()

    dataset_root = Path(args.dataset_root)
    csv_path = choose_stats_csv(dataset_root, args.csv if args.csv else None)
    args = auto_fill_plot_params(args, csv_path)

    out_dir = Path(args.out_dir) if args.out_dir else (dataset_root / "fig")
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = read_rows(csv_path, xy_norm_min=args.xy_norm_min)
    x = arr(rows, "frame_index")

    save_line(x, arr(rows, "eta_c"), out_dir / "01_eta_c_timeline.png",
              "Frame index", "eta_c", "Concentration factor eta_c",
              args.tau_c, f"tau_c={args.tau_c}")
    save_line(x, arr(rows, "rho21_raw"), out_dir / "02_rho21_raw_timeline.png",
              "Frame index", "rho21_raw", "Raw peak ratio rho21_raw = S_top2 / S_top1",
              args.tau_rho_raw, f"tau_rho_raw={args.tau_rho_raw}")
    save_line(x, arr(rows, "rho21_log"), out_dir / "03_rho21_log_timeline.png",
              "Frame index", "rho21_log", "Log peak ratio rho21_log = log1p(S_top2) / log1p(S_top1)",
              args.tau_rho_log, f"tau_rho_log={args.tau_rho_log}")
    save_line(x, arr(rows, "theta_top1_v_deg"), out_dir / "04_theta_top1_v_timeline.png",
              "Frame index", "theta_top1_v_deg", "Vertical angle of top-1 dominant normal peak",
              args.alpha_deg, f"alpha_deg={args.alpha_deg}")
    save_scatter(arr(rows, "eta_c"), arr(rows, "rho21_raw"), out_dir / "05_eta_vs_rho21_raw_scatter.png",
                 "eta_c", "rho21_raw", "eta_c vs raw peak ratio",
                 args.tau_c, args.tau_rho_raw, f"tau_c={args.tau_c}", f"tau_rho_raw={args.tau_rho_raw}")
    save_scatter(arr(rows, "eta_c"), arr(rows, "rho21_log"), out_dir / "06_eta_vs_rho21_log_scatter.png",
                 "eta_c", "rho21_log", "eta_c vs log peak ratio",
                 args.tau_c, args.tau_rho_log, f"tau_c={args.tau_c}", f"tau_rho_log={args.tau_rho_log}")
    save_components(x, rows, out_dir / "07_top1_top2_direction_components_timeline.png")
    save_two_lines(x, arr(rows, "n1_vertical_angle_deg"), arr(rows, "n2_vertical_angle_deg"),
                   out_dir / "08_top1_top2_vertical_angle_timeline.png",
                   "angle to z-axis (deg)", "Top-1/top-2 vertical angle")
    save_two_lines(x, arr(rows, "n1_xy_norm"), arr(rows, "n2_xy_norm"),
                   out_dir / "09_top1_top2_xy_norm_timeline.png",
                   "sqrt(nx^2+ny^2)", "Top-1/top-2 XY projection norm",
                   "top-1", "top-2", args.xy_norm_min, f"xy_norm_min={args.xy_norm_min}")
    save_xy_projection_scatter(rows, out_dir / "10_xy_projection_normal_scatter.png")
    save_xy_projection_arrow(rows, out_dir / "11_xy_projection_normal_arrow.png", args.sample_step)
    save_dir_projection(rows, out_dir / "12_xy_projection_deg_direction_scatter.png")
    save_3d_scatter(rows, out_dir / "13_top1_top2_unit_sphere_scatter.png")
    save_bar_counts(rows, out_dir / "14_scene_label_count_bar.png")
    save_dir_bin_hist(rows, out_dir / "15_dir_bin_hist.png")
    save_two_lines(x, arr(rows, "raw_count"), arr(rows, "crop_count"),
                   out_dir / "16_raw_crop_count_timeline.png",
                   "point count", "Raw and cropped point counts", "raw_count", "crop_count")
    save_line(x, arr(rows, "normal_count"), out_dir / "17_normal_count_timeline.png",
              "Frame index", "normal_count", "Normal count timeline")
    save_two_lines(x, arr(rows, "n1_axis_angle_deg"), arr(rows, "n2_axis_angle_deg"),
                   out_dir / "18_top1_top2_axis_angle_timeline.png",
                   "axis angle in XY plane (deg, 0-180)", "Top-1/top-2 axis angle in XY plane")
    save_line(x, arr(rows, "dir_angle_deg"), out_dir / "19_deg_direction_angle_timeline.png",
              "Frame index", "deg direction angle (deg, 0-180)", "Degeneracy direction angle in XY plane")

    ieee_outputs = save_ieee_label_generation_figures(rows, out_dir, args, run_matlab=args.run_matlab)
    write_summary(rows, out_dir / "summary_figures.txt", args, csv_path)

    print(f"[DONE] csv used: {csv_path}")
    print(
        "[DONE] auto params: "
        f"tau_c={args.tau_c}({args.param_source_tau_c}), "
        f"ratio_mode={args.ratio_mode}({args.param_source_ratio_mode}), "
        f"tau_rho_raw={args.tau_rho_raw}({args.param_source_tau_rho_raw}), "
        f"tau_rho_log={args.tau_rho_log}({args.param_source_tau_rho_log}), "
        f"alpha_deg={args.alpha_deg}({args.param_source_alpha})"
    )
    print(f"[DONE] figures saved to: {out_dir}")
    print(f"[DONE] IEEE label-generation outputs saved to: {out_dir / 'ieee_label_generation'}")
    print(f"[DONE] IEEE output file count: {len(ieee_outputs)}")


if __name__ == "__main__":
    main()
