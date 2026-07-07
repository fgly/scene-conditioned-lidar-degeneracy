"""PointNeXt backbone plus deg_scene task heads.

This model maps one single-frame point cloud to degeneracy scene predictions:
input ``points`` is [B, C, N], and outputs are ``class_logits`` [B, 3],
``dir_exist_logit`` [B, 1], ``dir_xy_unit`` [B, 2],
``dir_bin_logits`` [B, 12], and ``rz_logit`` [B, 1].
The release build uses PointNeXt as the only backbone.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from pointnext_backbone import PointNeXtBackbone
except ImportError:  # Allows importing as models.deg_scene_model.
    from models.pointnext_backbone import PointNeXtBackbone


def _normalize_backbone_name(backbone):
    """Normalize the release backbone name."""

    name = str(backbone).strip().lower().replace("-", "").replace("_", "")
    if name == "pointnext":
        return "pointnext"
    raise ValueError(f"Unsupported backbone '{backbone}'. This release supports only 'pointnext'.")


def infer_backbone_from_state_dict(state_dict, default="pointnext"):
    """Return the release backbone for checkpoints that do not store args."""

    return _normalize_backbone_name(default)


class DegSceneModel(nn.Module):
    """Degeneracy type classifier with conditional direction heads.

    Outputs:
        class_logits: [B, 3] in order tunnel_like/open_like/nondeg_or_other.
        dir_exist_logit: [B, 1]. Apply sigmoid for dir_exist score.
        dir_xy_unit: [B, 2]. L2-normalized xy tunnel direction.
        dir_bin_logits: [B, num_dir_bins]. Tunnel axis bin logits.
        rz_logit: [B, 1]. Apply sigmoid for rz score.
    """

    def __init__(
        self,
        input_channel=3,
        feature_dim=1024,
        dropout=0.4,
        num_dir_bins=12,
        backbone="pointnext",
    ):
        super().__init__()
        self.num_dir_bins = int(num_dir_bins)
        self.backbone_name = _normalize_backbone_name(backbone)
        self.backbone = PointNeXtBackbone(input_channel=input_channel, feature_dim=feature_dim)
        self.shared = nn.Sequential(
            nn.Linear(feature_dim, 512),
            nn.BatchNorm1d(512),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(512, 256),
            nn.BatchNorm1d(256),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
        )
        self.class_head = nn.Linear(256, 3)
        self.dir_exist_head = nn.Linear(256, 1)
        self.dir_xy_head = nn.Linear(256, 2)
        self.dir_bin_head = nn.Sequential(
            nn.Linear(256, 128),
            nn.BatchNorm1d(128),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout * 0.5),
            nn.Linear(128, self.num_dir_bins),
        )
        self.rz_head = nn.Linear(256, 1)

    def forward(self, points):
        """Predict scene class and conditional direction heads.

        Args:
            points: Input point cloud tensor [B, C, N].

        Returns:
            Dict with ``class_logits`` [B, 3], ``dir_exist_logit`` [B, 1],
            ``dir_xy_unit`` [B, 2], ``dir_bin_logits`` [B, num_dir_bins],
            and ``rz_logit`` [B, 1].
        """

        global_feature = self.backbone(points)  # [B, H]
        feature = self.shared(global_feature)  # [B, 256]
        class_logits = self.class_head(feature)  # [B, 3]
        dir_exist_logit = self.dir_exist_head(feature)  # [B, 1]
        raw_dir_xy = self.dir_xy_head(feature)  # [B, 2]
        # raw_dir_xy -> L2 normalize -> dir_xy_unit, preserving [B, 2].
        dir_xy_unit = F.normalize(raw_dir_xy, p=2, dim=-1, eps=1e-6)  # [B, 2]
        dir_bin_logits = self.dir_bin_head(feature)  # [B, num_dir_bins]
        rz_logit = self.rz_head(feature)  # [B, 1]
        return {
            "class_logits": class_logits,
            "dir_exist_logit": dir_exist_logit,
            "dir_xy_unit": dir_xy_unit,
            "dir_bin_logits": dir_bin_logits,
            "rz_logit": rz_logit,
        }


# Keep the dynamic import style used by the original training scripts.
def get_model(input_channel=3, **kwargs):
    """Factory compatible with the original dynamic model import pattern.

    Args:
        input_channel: Number of point channels C in input [B, C, N].
        **kwargs: Forwarded to ``DegSceneModel``.

    Returns:
        A ``DegSceneModel`` instance.
    """

    return DegSceneModel(input_channel=input_channel, **kwargs)
