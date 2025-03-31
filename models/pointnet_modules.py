import torch
import torch.nn as nn
import torch.nn.functional as F


class PointNetSetAbstraction(nn.Module):
    """
    PointNet Set Abstraction Module for processing point clouds.

    Attributes:
        npoint (int): Number of points to sample
        radius (float): Radius to use for ball query
        nsample (int): Number of samples in each local region
        mlp (list): List of feature dimensions for MLP
        group_all (bool): Whether to group all points into one cluster
    """

    def __init__(self, npoint, radius, nsample, in_channel, mlp, group_all=False, use_xyz=True):
        super(PointNetSetAbstraction, self).__init__()
        self.npoint = npoint
        self.radius = radius
        self.nsample = nsample
        self.mlp_convs = nn.ModuleList()
        self.mlp_bns = nn.ModuleList()
        self.group_all = group_all
        self.use_xyz = use_xyz

        if use_xyz:
            in_channel += 3

        last_channel = in_channel
        for out_channel in mlp:
            self.mlp_convs.append(nn.Conv2d(last_channel, out_channel, 1))
            self.mlp_bns.append(nn.BatchNorm2d(out_channel))
            last_channel = out_channel

    def forward(self, xyz, points):
        """
        Input:
            xyz: input points position data, [B, N, C] where C is channels (e.g., 2 for complex data)
            points: input points data, [B, N, D]
        Return:
            new_xyz: sampled points position data, [B, S, C]
            new_points_concat: sample points feature data, [B, S, D']
        """
        device = xyz.device
        B, N, C = xyz.shape

        # For our texture data, we treat coordinates and features the same way
        # Use first channel as "coordinates" for sampling
        if points is None:
            points = xyz  # Use the input itself as features

        # Sample points
        if self.group_all:
            # Use all points in a single group
            new_xyz = torch.zeros(B, 1, C).to(device)
            grouped_points = points.view(B, 1, N, -1).repeat(1, 1, 1, 1)
        else:
            # Simple random sampling for texture data (no need for farthest point sampling)
            if self.npoint is not None:
                idx = torch.randperm(N)[:self.npoint]
                new_xyz = xyz[:, idx, :]
                new_points = points[:, idx, :]
            else:
                new_xyz = xyz
                new_points = points

            # Simple neighborhood aggregation (use K-nearest neighbors)
            idx = knn_point(self.nsample, points, new_points)
            grouped_points = index_points(points, idx)

        # Process points through MLP
        grouped_points = grouped_points.permute(0, 3, 2, 1)  # [B, C, nsample, npoint]

        for i, conv in enumerate(self.mlp_convs):
            grouped_points = F.relu(self.mlp_bns[i](conv(grouped_points)))

        # Max pooling
        new_points = torch.max(grouped_points, 2)[0]  # [B, C', npoint]
        new_points = new_points.permute(0, 2, 1)  # [B, npoint, C']

        return new_xyz, new_points


# Helper functions for PointNetSetAbstraction
def farthest_point_sampling(xyz, npoint):
    """
    Input:
        xyz: pointcloud data, [B, N, 3]
        npoint: number of samples
    Return:
        centroids: sampled pointcloud index, [B, npoint]
    """
    device = xyz.device
    B, N, C = xyz.shape
    centroids = torch.zeros(B, npoint, dtype=torch.long).to(device)
    distance = torch.ones(B, N).to(device) * 1e10
    farthest = torch.randint(0, N, (B,), dtype=torch.long).to(device)
    batch_indices = torch.arange(B, dtype=torch.long).to(device)
    for i in range(npoint):
        centroids[:, i] = farthest
        centroid = xyz[batch_indices, farthest, :].view(B, 1, C)
        dist = torch.sum((xyz - centroid) ** 2, -1)
        mask = dist < distance
        distance[mask] = dist[mask]
        farthest = torch.max(distance, -1)[1]
    return centroids


def index_points(points, idx):
    """
    Input:
        points: input points data, [B, N, C]
        idx: sample index data, [B, S]
    Return:
        new_points:, indexed points data, [B, S, C]
    """
    device = points.device
    B = points.shape[0]
    view_shape = list(idx.shape)
    view_shape[1:] = [1] * (len(view_shape) - 1)
    repeat_shape = list(idx.shape)
    repeat_shape[0] = 1
    batch_indices = torch.arange(B, dtype=torch.long).to(device).view(view_shape).repeat(repeat_shape)
    new_points = points[batch_indices, idx, :]
    return new_points


def query_ball_point(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample]
    group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx


def square_distance(src, dst):
    """
    Calculate squared distance between each two points.
    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    dist += torch.sum(src ** 2, -1).view(B, N, 1)
    dist += torch.sum(dst ** 2, -1).view(B, 1, M)
    return dist


def knn_point(nsample, xyz, new_xyz):
    """
    Input:
        nsample: max sample number in local region
        xyz: all points, [B, N, C]
        new_xyz: query points, [B, S, C]
    Return:
        group_idx: grouped points index, [B, S, nsample]
    """
    sqrdists = square_distance(new_xyz, xyz)
    _, group_idx = torch.topk(sqrdists, nsample, dim=-1, largest=False, sorted=False)
    return group_idx