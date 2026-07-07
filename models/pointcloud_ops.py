"""Point-cloud sampling and neighborhood query helpers for PointNeXt.

The functions in this module use [B, N, C] tensors internally. They are kept
small and framework-local so the release model does not expose legacy backbone
modules in its public API.
"""

import torch


def square_distance(src, dst):
    """Calculate pairwise squared Euclidean distance.

    Args:
        src: Source points [B, N, C].
        dst: Target points [B, M, C].

    Returns:
        Pairwise distances [B, N, M].
    """

    batch_size, num_src, _ = src.shape
    _, num_dst, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src**2, dim=-1).view(batch_size, num_src, 1)
    dist += torch.sum(dst**2, dim=-1).view(batch_size, 1, num_dst)
    return dist


def index_points(points, idx):
    """Gather points by batched index tensor."""

    device = points.device
    batch_size = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(batch_size, dtype=torch.long, device=device).view(view_shape).repeat(repeat_shape)
    return points[batch_indices, idx, :]


def farthest_point_sample(xyz, npoint):
    """Farthest point sampling over xyz coordinates.

    Args:
        xyz: Point cloud coordinates [B, N, 3].
        npoint: Number of centroids to sample.

    Returns:
        Sampled centroid indices [B, npoint].
    """

    device = xyz.device
    batch_size, num_points, _ = xyz.shape
    centroids = torch.zeros(batch_size, npoint, dtype=torch.long, device=device)
    distance = torch.full((batch_size, num_points), 1e10, device=device)
    farthest = torch.randint(0, num_points, (batch_size,), dtype=torch.long, device=device)
    batch_indices = torch.arange(batch_size, dtype=torch.long, device=device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(batch_size, 1, 3)
        dist = torch.sum((xyz - centroid) ** 2, dim=-1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, dim=-1)[1]
    return centroids


def query_ball_point(radius, nsample, xyz, new_xyz):
    """Group neighboring points within a radius around query centers."""

    device = xyz.device
    batch_size, num_points, _ = xyz.shape
    _, num_centers, _ = new_xyz.shape
    group_idx = torch.arange(num_points, dtype=torch.long, device=device).view(1, 1, num_points)
    group_idx = group_idx.repeat(batch_size, num_centers, 1)
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius**2] = num_points
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(batch_size, num_centers, 1).repeat(1, 1, nsample)
    mask = group_idx == num_points
    group_idx[mask] = group_first[mask]
    return group_idx
