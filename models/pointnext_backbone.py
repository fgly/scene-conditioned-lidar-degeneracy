"""PointNeXt-style global feature backbone for deg_scene.

Input tensors are [B, C, N] with xyz in channels 0:3, and the output is one
global feature vector [B, feature_dim].
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

try:
    from pointcloud_ops import farthest_point_sample, index_points, query_ball_point
except ImportError:  # Allows importing as models.pointnext_backbone.
    from models.pointcloud_ops import farthest_point_sample, index_points, query_ball_point


class ConvBNReLU1d(nn.Module):
    """1x1 Conv1d with BatchNorm and ReLU."""

    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv1d(in_channel, out_channel, 1, bias=False),
            nn.BatchNorm1d(out_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class ConvBNReLU2d(nn.Module):
    """1x1 Conv2d with BatchNorm and ReLU."""

    def __init__(self, in_channel, out_channel):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(in_channel, out_channel, 1, bias=False),
            nn.BatchNorm2d(out_channel),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        return self.net(x)


class PointNeXtSetAbstraction(nn.Module):
    """PointNeXt-style local aggregation with optional downsampling."""

    def __init__(self, npoint, radius, nsample, in_channel, out_channel, group_all=False):
        super().__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.group_all = group_all
        hidden_channel = max(out_channel // 2, 64)
        self.local_mlp = nn.Sequential(
            ConvBNReLU2d(in_channel + 3, hidden_channel),
            ConvBNReLU2d(hidden_channel, out_channel),
        )

    def forward(self, xyz, features):
        """Aggregate local neighborhoods.

        Args:
            xyz: Point coordinates [B, 3, N].
            features: Per-point features [B, C, N].

        Returns:
            new_xyz: Sampled coordinates [B, 3, S].
            new_features: Aggregated features [B, out_channel, S].
        """

        xyz_t = xyz.permute(0, 2, 1).contiguous()
        features_t = features.permute(0, 2, 1).contiguous()
        batch_size, num_points, _ = xyz_t.shape

        if self.group_all:
            new_xyz = torch.zeros(batch_size, 1, 3, device=xyz.device, dtype=xyz.dtype)
            grouped_xyz_norm = xyz_t.view(batch_size, 1, num_points, 3)
            grouped_features = features_t.view(batch_size, 1, num_points, -1)
        else:
            fps_idx = farthest_point_sample(xyz_t, self.npoint)
            new_xyz = index_points(xyz_t, fps_idx)
            group_idx = query_ball_point(self.radius, self.nsample, xyz_t, new_xyz)
            grouped_xyz = index_points(xyz_t, group_idx)
            grouped_xyz_norm = grouped_xyz - new_xyz.view(batch_size, self.npoint, 1, 3)
            grouped_features = index_points(features_t, group_idx)

        grouped = torch.cat([grouped_xyz_norm, grouped_features], dim=-1)
        grouped = grouped.permute(0, 3, 2, 1).contiguous()
        new_features = torch.max(self.local_mlp(grouped), dim=2)[0]
        return new_xyz.permute(0, 2, 1).contiguous(), new_features


class PointNeXtBlock(nn.Module):
    """Residual local aggregation block used after each downsampling stage."""

    def __init__(self, channel, radius, nsample, expansion=2):
        super().__init__()
        hidden_channel = int(channel * expansion)
        self.radius = radius
        self.nsample = nsample
        self.local_mlp = nn.Sequential(
            ConvBNReLU2d(channel + 3, hidden_channel),
            nn.Conv2d(hidden_channel, channel, 1, bias=False),
            nn.BatchNorm2d(channel),
        )
        self.point_mlp = nn.Sequential(
            ConvBNReLU1d(channel, hidden_channel),
            nn.Conv1d(hidden_channel, channel, 1, bias=False),
            nn.BatchNorm1d(channel),
        )

    def forward(self, xyz, features):
        xyz_t = xyz.permute(0, 2, 1).contiguous()
        features_t = features.permute(0, 2, 1).contiguous()
        batch_size, num_points, _ = xyz_t.shape
        group_idx = query_ball_point(self.radius, self.nsample, xyz_t, xyz_t)
        grouped_xyz = index_points(xyz_t, group_idx)
        grouped_xyz_norm = grouped_xyz - xyz_t.view(batch_size, num_points, 1, 3)
        grouped_features = index_points(features_t, group_idx)
        grouped = torch.cat([grouped_xyz_norm, grouped_features], dim=-1)
        grouped = grouped.permute(0, 3, 2, 1).contiguous()
        local_feature = torch.max(self.local_mlp(grouped), dim=2)[0]
        return F.relu(features + self.point_mlp(local_feature), inplace=True)


class PointNeXtBackbone(nn.Module):
    """Hierarchical PointNeXt-style backbone returning a global feature vector."""

    def __init__(self, input_channel=3, feature_dim=1024):
        super().__init__()
        if input_channel < 3:
            raise ValueError("input_channel must be at least 3 for xyz coordinates")
        self.input_channel = input_channel
        self.feature_dim = feature_dim

        self.stem = nn.Sequential(
            ConvBNReLU1d(input_channel, 32),
            ConvBNReLU1d(32, 64),
        )
        self.sa1 = PointNeXtSetAbstraction(512, 0.2, 32, 64, 128)
        self.block1 = PointNeXtBlock(128, 0.2, 16)
        self.sa2 = PointNeXtSetAbstraction(128, 0.4, 32, 128, 256)
        self.block2 = PointNeXtBlock(256, 0.4, 16)
        self.sa3 = PointNeXtSetAbstraction(32, 0.8, 32, 256, 512)
        self.block3 = PointNeXtBlock(512, 0.8, 16)
        self.global_sa = PointNeXtSetAbstraction(None, None, None, 512, feature_dim, group_all=True)

    def forward(self, points):
        """Encode a point cloud into one global feature vector.

        Args:
            points: Input point cloud tensor [B, C, N].

        Returns:
            Global feature tensor [B, feature_dim].
        """

        if points.dim() != 3:
            raise ValueError(f"points must be [B, C, N], got {tuple(points.shape)}")
        batch_size, channels, _ = points.shape
        if channels != self.input_channel:
            raise ValueError(f"expected {self.input_channel} channels, got {channels}")

        xyz = points[:, :3, :]
        features = self.stem(points)
        xyz, features = self.sa1(xyz, features)
        features = self.block1(xyz, features)
        xyz, features = self.sa2(xyz, features)
        features = self.block2(xyz, features)
        xyz, features = self.sa3(xyz, features)
        features = self.block3(xyz, features)
        _, global_feature = self.global_sa(xyz, features)
        return global_feature.view(batch_size, -1)
