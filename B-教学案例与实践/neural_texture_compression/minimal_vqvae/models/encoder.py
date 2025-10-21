import torch
import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):
    def __init__(self, in_channels=3, hidden_channels=128, embed_dim=64, downsample=8):
        super().__init__()
        assert downsample in (8, 16), "downsample must be 8 or 16"
        layers = []
        c = hidden_channels
        layers.append(nn.Conv2d(in_channels, c, kernel_size=3, stride=1, padding=1))
        layers.append(nn.ReLU(inplace=True))
        # Downsample by powers of two to reach the factor
        ds = downsample
        steps = 0
        while ds > 1:
            layers.append(nn.Conv2d(c, c, kernel_size=4, stride=2, padding=1))
            layers.append(nn.ReLU(inplace=True))
            ds //= 2
            steps += 1
        layers.append(nn.Conv2d(c, embed_dim, kernel_size=3, stride=1, padding=1))
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        z_e = self.net(x)
        return z_e
