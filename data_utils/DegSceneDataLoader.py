"""Load single-frame point clouds and pseudo labels for the deg_scene task.

The dataset reads one point-cloud frame per sample and returns the fields used
by ``utils.deg_losses.compute_deg_loss``:
``points`` [N, C], ``class_gt`` [], ``dir_exist_gt`` [], ``dir_xy_gt`` [2],
``dir_xy_valid`` [], ``rz_gt`` [], ``sample_weight`` [], ``dir_bin_gt`` [],
and ``dir_bin_valid`` [].

Label files may be CSV, JSON/JSONL, NPY, or pickle. Each row must contain a
point-cloud path (``file_path``/``path``/``points``/``point_path``) and either
``class_gt`` or ``scene_type``. Optional fields are ``dir_x``, ``dir_y``,
``dir_z``, ``dir_xy_valid``, ``dir_bin_gt``, ``dir_bin_valid``,
``angle_deg``, ``sample_weight``, and ``split``.
"""

import csv
import json
import math
import os
import pickle
import random
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from torch.utils.data import Dataset

SCENE_TYPE_TO_CLASS = {
    "tunnel_like": 0,
    "tunnel-like": 0,
    "tunnel": 0,
    "open_like": 1,
    "open-like": 1,
    "open": 1,
    "nondeg_or_other": 2,
    "non_degenerate": 2,
    "non-degenerate": 2,
    "nondegenerate": 2,
    "non_deg": 2,
    "non-deg": 2,
    "nondeg": 2,
    "nodeg": 2,
    "other": 2,
}
CLASS_TO_SCENE_TYPE = {0: "tunnel_like", 1: "open_like", 2: "nondeg_or_other"}
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


def pc_normalize(pc: np.ndarray) -> np.ndarray:
    """Center and scale xyz coordinates to a unit-radius cloud."""

    centroid = np.mean(pc, axis=0)
    pc = pc - centroid
    radius = np.max(np.sqrt(np.sum(pc**2, axis=1)))
    return pc / max(float(radius), 1e-6)


def farthest_point_sample(
    points: np.ndarray,
    npoint: int,
    start_idx: Optional[int] = None,
) -> np.ndarray:
    """Subsample a point cloud with numpy farthest-point sampling.

    ``start_idx`` pins the first centroid for deterministic validation/test
    sampling. When omitted, the first centroid is random for training.
    """

    num_points, _ = points.shape
    xyz = points[:, :3]
    centroids = np.zeros((npoint,), dtype=np.int32)
    distance = np.ones((num_points,), dtype=np.float64) * 1e10
    farthest = np.random.randint(0, num_points) if start_idx is None else int(start_idx) % num_points
    for i in range(npoint):
        centroids[i] = farthest
        centroid = xyz[farthest, :]
        dist = np.sum((xyz - centroid) ** 2, axis=-1)
        closer = dist < distance
        distance[closer] = dist[closer]
        farthest = int(np.argmax(distance, axis=-1))
    return points[centroids]


def _load_rows(label_path: str) -> List[Dict[str, Any]]:
    """Load label rows from a supported table-like file."""

    suffix = Path(label_path).suffix.lower()
    if suffix == ".csv":
        with open(label_path, newline="") as f:
            return list(csv.DictReader(f))
    if suffix in {".json", ".jsonl"}:
        with open(label_path) as f:
            if suffix == ".jsonl":
                return [json.loads(line) for line in f if line.strip()]
            data = json.load(f)
        if isinstance(data, dict):
            data = data.get("samples", data.get("data", data))
        if isinstance(data, dict):
            return [
                dict({"sample_id": k}, **v) if isinstance(v, dict) else {"sample_id": k, "file_path": v}
                for k, v in data.items()
            ]
        return list(data)
    if suffix == ".npy":
        data = np.load(label_path, allow_pickle=True)
        if isinstance(data, np.ndarray) and data.dtype.names:
            return [
                {name: row[name].item() if hasattr(row[name], "item") else row[name] for name in data.dtype.names}
                for row in data
            ]
        return [dict(x) for x in data.tolist()]
    if suffix in {".pkl", ".pickle"}:
        with open(label_path, "rb") as f:
            data = pickle.load(f)
        if isinstance(data, dict):
            data = data.get("samples", data.get("data", data))
        if isinstance(data, dict):
            return [
                dict({"sample_id": k}, **v) if isinstance(v, dict) else {"sample_id": k, "file_path": v}
                for k, v in data.items()
            ]
        return [dict(x) for x in data]
    raise ValueError(f"Unsupported label file type: {label_path}")


def _normalize_scene_type(value: Any) -> str:
    """Normalize scene-type aliases without changing the three-class mapping."""

    text = str(value).strip().lower().replace(" ", "_")
    if text in SCENE_TYPE_TO_CLASS:
        return text
    text = text.replace("-", "_")
    aliases = {
        "tunnel": "tunnel_like",
        "tunnel_like": "tunnel_like",
        "open": "open_like",
        "open_like": "open_like",
        "nodeg": "nondeg_or_other",
        "nondeg": "nondeg_or_other",
        "nondegenerate": "nondeg_or_other",
        "non_degenerate": "nondeg_or_other",
        "non_deg": "nondeg_or_other",
        "nondeg_or_other": "nondeg_or_other",
        "other": "nondeg_or_other",
    }
    return aliases.get(text, text)


def _split_rows(rows: List[Dict[str, Any]], split: str, seed: int) -> List[Dict[str, Any]]:
    """Filter explicit splits, or create a deterministic 70/15/15 split."""

    if split in {"all", ""}:
        return rows

    requested = str(split).strip().lower()
    explicit_values = [
        str(row.get("split", "")).strip().lower()
        for row in rows
        if str(row.get("split", "")).strip()
    ]
    if explicit_values:
        return [row for row in rows if str(row.get("split", "")).strip().lower() == requested]

    indexed = list(enumerate(rows))
    rng = random.Random(seed)
    rng.shuffle(indexed)
    n = len(indexed)
    train_end = int(round(n * 0.70))
    val_end = train_end + int(round(n * 0.15))
    if n >= 3:
        train_end = min(max(train_end, 1), n - 2)
        val_end = min(max(val_end, train_end + 1), n - 1)
    split_ranges = {
        "train": indexed[:train_end],
        "val": indexed[train_end:val_end],
        "test": indexed[val_end:],
    }
    selected = split_ranges.get(requested, [])
    selected.sort(key=lambda item: item[0])
    return [row for _, row in selected]


def _to_float(row: Dict[str, Any], key: str, default: float) -> float:
    """Read a numeric scalar from a label row with an empty-value default."""

    value = row.get(key, default)
    if value is None or value == "":
        return float(default)
    return float(value)


def _has_value(row: Dict[str, Any], key: str) -> bool:
    """Return True when a row field exists and is not empty/NaN."""

    if key not in row:
        return False
    value = row[key]
    if value is None:
        return False
    if isinstance(value, str):
        return value != ""
    try:
        return not bool(np.isnan(value))
    except (TypeError, ValueError):
        return True


def _load_points(path: str) -> np.ndarray:
    """Load a point-cloud frame from disk."""

    suffix = Path(path).suffix.lower()
    if suffix == ".npy":
        points = np.load(path).astype(np.float32)
    elif suffix == ".npz":
        data = np.load(path)
        key = "points" if "points" in data else data.files[0]
        points = data[key].astype(np.float32)
    elif suffix in {".txt", ".csv", ".pts", ".xyz"}:
        delimiter = "," if suffix == ".csv" else None
        points = np.loadtxt(path, delimiter=delimiter).astype(np.float32)
    elif suffix == ".bin":
        raw = np.fromfile(path, dtype=np.float32)
        if raw.size % 4 == 0:
            points = raw.reshape(-1, 4)
        elif raw.size % 3 == 0:
            points = raw.reshape(-1, 3)
        else:
            raise ValueError(f"Cannot infer channel count for binary point cloud: {path}")
    elif suffix in {".pkl", ".pickle"}:
        with open(path, "rb") as f:
            data = pickle.load(f)
        points = data.get("points", data) if isinstance(data, dict) else data
        points = np.asarray(points, dtype=np.float32)
    else:
        raise ValueError(f"Unsupported point cloud file type: {path}")
    if points.ndim != 2 or points.shape[1] < 3:
        raise ValueError(f"points must have shape [N, C>=3], got {points.shape} from {path}")
    if points.shape[0] == 0:
        raise ValueError(f"point cloud is empty: {path}")
    return points


class DegSceneDataLoader(Dataset):
    """PyTorch dataset for deg_scene single-frame classification samples.

    ``__getitem__`` returns numpy arrays/scalars that the default PyTorch
    collate function stacks into batch tensors. The training/evaluation scripts
    receive ``points`` as [B, N, C] and transpose it to [B, C, N] before calling
    ``DegSceneModel``.
    """

    def __init__(
        self,
        label_path: str,
        num_point: int = 1024,
        split: str = "train",
        root: str = "",
        use_features: bool = False,
        normalize_xyz: bool = True,
        uniform: bool = False,
        dir_xy_eps: float = 1e-4,
        seed: int = 0,
        num_dir_bins: int = DEFAULT_NUM_DIR_BINS,
        deterministic_sample: bool = False,
    ):
        """Create a dataset over the requested split.

        Args:
            label_path: Pseudo-label table path.
            num_point: Number of rows in each returned point cloud [N, C].
            split: Split name to filter on, or ``all``/``""`` for no filter.
            root: Base directory for relative point-cloud paths.
            use_features: Keep channels beyond xyz when True.
            normalize_xyz: Apply unit-radius xyz normalization when True.
            uniform: Use farthest-point sampling when enough points exist.
            dir_xy_eps: Minimum xy direction norm considered valid.
            seed: Seed for train-time random sampling.
            num_dir_bins: Number of axial direction bins over [0, 180).
            deterministic_sample: Use fixed sampling for repeated val/test reads.
        """

        self.label_path = label_path
        self.num_point = num_point
        self.split = split
        label_parent = Path(label_path).parent
        if root:
            self.root = root
        elif label_parent.name.lower() == "labels":
            self.root = str(label_parent.parent)
        else:
            self.root = str(label_parent)
        self.use_features = use_features
        self.normalize_xyz = normalize_xyz
        self.uniform = uniform
        self.dir_xy_eps = dir_xy_eps
        self.seed = seed
        self.num_dir_bins = int(num_dir_bins)
        self.deterministic_sample = deterministic_sample

        np.random.seed(seed)
        random.seed(seed)

        rows = _load_rows(label_path)
        self.rows = _split_rows(rows, split, seed)
        print(f"The size of {split} deg scene data is {len(self.rows)}")

    def __len__(self):
        """Return the number of label rows in the selected split."""

        return len(self.rows)

    def _resolve_path(self, row: Dict[str, Any]) -> str:
        """Resolve a label row to an absolute or root-relative point-cloud path."""

        path = row.get("file_path") or row.get("path") or row.get("points") or row.get("point_path")
        if path is None:
            raise KeyError("label row must contain file_path/path/points/point_path")
        path = str(path)
        return path if os.path.isabs(path) else os.path.join(self.root, path)

    def _sample_points(self, points: np.ndarray) -> np.ndarray:
        """Return a sampled point cloud, deterministic when requested."""

        num_points = points.shape[0]
        if self.uniform and num_points >= self.num_point:
            start_idx = None
            if self.deterministic_sample:
                start_idx = int(np.argmax(np.sum(points[:, :3] ** 2, axis=1)))
            return farthest_point_sample(points, self.num_point, start_idx=start_idx)

        if num_points >= self.num_point:
            if self.deterministic_sample:
                return points[: self.num_point]
            choice = np.random.choice(num_points, self.num_point, replace=False)
            return points[choice]

        deficit = self.num_point - num_points
        if self.deterministic_sample:
            choice = np.arange(deficit, dtype=np.int64) % num_points
        else:
            choice = np.random.choice(num_points, deficit, replace=True)
        return np.concatenate([points, points[choice]], axis=0)

    def _labels_from_row(self, row: Dict[str, Any]) -> Dict[str, Any]:
        """Convert one label row into deg_scene supervision tensors."""

        if "class_gt" in row and row["class_gt"] not in {None, ""}:
            class_gt = int(float(row["class_gt"]))
        else:
            scene_type = _normalize_scene_type(row.get("scene_type", "nondeg_or_other"))
            class_gt = SCENE_TYPE_TO_CLASS.get(scene_type)
            if class_gt is None:
                raise ValueError(f"Unknown scene_type: {scene_type}")

        dir_exist_gt = _to_float(row, "dir_exist_gt", 1.0 if class_gt in {0, 1} else 0.0)
        rz_gt = _to_float(row, "rz_gt", 1.0 if class_gt == 1 else 0.0)
        sample_weight = _to_float(row, "sample_weight", 1.0)

        dx = _to_float(row, "dir_x", _to_float(row, "gt_dir_x", 0.0))
        dy = _to_float(row, "dir_y", _to_float(row, "gt_dir_y", 0.0))
        norm = float(np.sqrt(dx * dx + dy * dy))
        if norm > self.dir_xy_eps:
            dir_xy = np.array([dx / norm, dy / norm], dtype=np.float32)
            auto_valid = 1.0
        else:
            dir_xy = np.zeros(2, dtype=np.float32)
            auto_valid = 0.0

        if _has_value(row, "dir_xy_valid"):
            dir_xy_valid = float(row["dir_xy_valid"])
        else:
            dir_xy_valid = auto_valid if class_gt == 0 else 0.0
        if class_gt != 0:
            dir_xy_valid = 0.0

        dir_bin_gt = 0
        has_dir_bin = False
        if _has_value(row, "dir_bin_gt"):
            dir_bin_gt = int(float(row["dir_bin_gt"]))
            has_dir_bin = True
        elif _has_value(row, "angle_deg"):
            dir_bin_gt = direction_bin_from_angle_deg(float(row["angle_deg"]), self.num_dir_bins)
            has_dir_bin = True
        elif (
            _has_value(row, "dir_x")
            or _has_value(row, "dir_y")
            or _has_value(row, "gt_dir_x")
            or _has_value(row, "gt_dir_y")
        ) and norm > self.dir_xy_eps:
            dir_bin_gt = direction_bin_from_xy(dx, dy, self.num_dir_bins)
            has_dir_bin = True

        dir_bin_gt = min(max(int(dir_bin_gt), 0), self.num_dir_bins - 1)
        if class_gt == 0:
            dir_bin_valid = 1.0 if has_dir_bin else 0.0
            if _has_value(row, "dir_bin_valid"):
                dir_bin_valid = float(row["dir_bin_valid"]) if has_dir_bin else 0.0
        else:
            dir_bin_valid = 0.0
            dir_bin_gt = 0

        return {
            "class_gt": np.int64(class_gt),
            "dir_exist_gt": np.float32(dir_exist_gt),
            "dir_xy_gt": dir_xy,
            "dir_xy_valid": np.float32(dir_xy_valid),
            "rz_gt": np.float32(rz_gt),
            "sample_weight": np.float32(sample_weight),
            "dir_bin_gt": np.int64(dir_bin_gt),
            "dir_bin_valid": np.float32(dir_bin_valid),
        }

    def __getitem__(self, index: int):
        """Return one sampled and labeled point-cloud item."""

        row = self.rows[index]
        point_path = self._resolve_path(row)
        points = self._sample_points(_load_points(point_path))

        if self.normalize_xyz:
            points[:, 0:3] = pc_normalize(points[:, 0:3])

        if not self.use_features:
            points = points[:, 0:3]

        item = self._labels_from_row(row)
        item["points"] = points.astype(np.float32)
        item["sample_id"] = str(row.get("sample_id", Path(point_path).stem))
        item["file_path"] = point_path
        return item
