import torch
import torch.nn as nn
import torch.nn.functional as F
from models.pointnet_modules import PointNetSetAbstraction


class TexturePointNet(nn.Module):
    """
    PointNet++ model adapted for texture analysis.

    Attributes:
        num_points (int): Number of points in the input
        in_channels (int): Number of input channels (1 or 2 for complex data)
        output_dim (int): Dimensionality of the output (number of r-values)
        dropout_rate (float): Dropout rate for regularization
    """

    def __init__(self, num_points, in_channels=2, output_dim=3, dropout_rate=0.4):
        super(TexturePointNet, self).__init__()

        self.num_points = num_points
        self.in_channels = in_channels
        self.output_dim = output_dim

        # Set abstraction layers
        self.sa1 = PointNetSetAbstraction(
            npoint=128,
            radius=0.2,
            nsample=32,
            in_channel=in_channels,
            mlp=[in_channels, 32, 32, 64],
            use_xyz=False
        )

        self.sa2 = PointNetSetAbstraction(
            npoint=64,
            radius=0.4,
            nsample=32,
            in_channel=64,
            mlp=[64, 64, 64, 128],
            use_xyz=False
        )

        self.sa3 = PointNetSetAbstraction(
            npoint=None,  # Global pooling
            radius=None,
            nsample=None,
            in_channel=128,
            mlp=[128, 128, 256, 512],
            group_all=True,
            use_xyz=False
        )

        # MLP for regression output
        self.fc1 = nn.Linear(512, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(dropout_rate)

        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.drop2 = nn.Dropout(dropout_rate)

        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        """
        Forward pass of the TexturePointNet.

        Args:
            x (Tensor): Input tensor of shape [B, N, C]
                where B is batch size, N is number of points, and C is channels

        Returns:
            Tensor: Output tensor of shape [B, output_dim]
        """
        # Set abstraction layers
        l1_xyz, l1_points = self.sa1(x, None)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)

        # Flatten points for MLP
        x = l3_points.view(-1, 512)

        # MLP layers
        x = F.relu(self.bn1(self.fc1(x)))
        x = self.drop1(x)

        x = F.relu(self.bn2(self.fc2(x)))
        x = self.drop2(x)

        # Output layer (no activation for regression)
        x = self.fc3(x)

        return x