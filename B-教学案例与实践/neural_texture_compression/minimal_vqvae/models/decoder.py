import torch
import torch.nn as nn
import torch.nn.functional as F


class Decoder(nn.Module):
    def __init__(self, out_channels=3, hidden_channels=128, embed_dim=64, downsample=8):
        super().__init__()
        assert downsample in (8, 16), "downsample must be 8 or 16"
        layers = []
        c = hidden_channels
        layers.append(nn.Conv2d(embed_dim, c, kernel_size=3, stride=1, padding=1))
        layers.append(nn.ReLU(inplace=True))
        ds = downsample
        while ds > 1:
            layers.append(nn.Upsample(scale_factor=2, mode='nearest'))
            layers.append(nn.Conv2d(c, c, kernel_size=3, stride=1, padding=1))
            layers.append(nn.ReLU(inplace=True))
            ds //= 2
        layers.append(nn.Conv2d(c, out_channels, kernel_size=3, stride=1, padding=1))
        # Use Sigmoid to map to [0,1]
        layers.append(nn.Sigmoid())
        self.net = nn.Sequential(*layers)

    def forward(self, z_q):
        x_hat = self.net(z_q)
        return x_hat
