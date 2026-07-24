"""JSCC transceiver for 32x32 RGB satellite imagery over the complex fading channel."""

import torch
import torch.nn as nn

from channel import transmit

M = 128  # complex channel symbols (=> 256 real dims); ~12x bandwidth compression


class SatEncoder(nn.Module):
    def __init__(self, m=M):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.GELU(),    # 32->16
            nn.Conv2d(32, 64, 3, 2, 1), nn.GELU(),   # 16->8
            nn.Conv2d(64, 64, 3, 1, 1), nn.GELU(),
        )
        self.proj = nn.Linear(64 * 8 * 8, 2 * m)

    def forward(self, x):
        return self.proj(self.body(x).flatten(1))    # (B, 2m)


class SatDecoder(nn.Module):
    def __init__(self, m=M):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(4 * m, 64 * 8 * 8), nn.GELU())
        self.up = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, 2, 1), nn.GELU(),   # 8->16
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.GELU(),   # 16->32
            nn.Conv2d(32, 3, 3, 1, 1), nn.Sigmoid(),
        )

    def forward(self, feats):                         # feats: (B, m, 4)
        h = self.proj(feats.flatten(1)).view(-1, 64, 8, 8)
        return self.up(h)


class ContestedJSCC(nn.Module):
    def __init__(self, m=M):
        super().__init__()
        self.m = m
        self.enc = SatEncoder(m)
        self.dec = SatDecoder(m)

    def forward(self, x, **chan):
        feats, sinr = transmit(self.enc(x), **chan)
        return self.dec(feats), sinr


def psnr(x, xhat, eps=1e-8):
    mse = torch.mean((x - xhat) ** 2, dim=[1, 2, 3])
    return 10.0 * torch.log10(1.0 / (mse + eps))


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
