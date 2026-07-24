"""
Latent Radio — a neural transceiver (Deep JSCC).

The sender's Encoder maps a 28x28 image to k real channel symbols. Those symbols
are power-normalized, pushed through a noisy AWGN channel, and the receiver's
Decoder reconstructs the image. Encoder + Decoder are trained end-to-end: they are
the *shared model* both endpoints agree on. Nothing about "meaning" is hand-coded
— the network discovers a joint source-channel code that packs the image into a
noise-robust latent and unpacks it on the far side.

Why this can beat classical digital comms: a standard system first compresses to
bits (source coding) then protects those bits (channel coding), and Shannon's
separation theorem says that's optimal *only* in the infinite-blocklength limit.
At the short blocklengths and low SNRs of real images, a jointly-learned analog
code degrades gracefully instead of falling off the digital "cliff".
"""

import torch
import torch.nn as nn

from channel import power_normalize, awgn


class Encoder(nn.Module):
    def __init__(self, k=128):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1), nn.GELU(),   # 28->14
            nn.Conv2d(32, 64, 3, stride=2, padding=1), nn.GELU(),  # 14->7
            nn.Conv2d(64, 64, 3, stride=1, padding=1), nn.GELU(),
        )
        self.proj = nn.Linear(64 * 7 * 7, k)

    def forward(self, x):
        h = self.body(x).flatten(1)
        z = self.proj(h)
        return power_normalize(z)  # (B, k), unit average power per symbol


class Decoder(nn.Module):
    def __init__(self, k=128):
        super().__init__()
        self.proj = nn.Linear(k, 64 * 7 * 7)
        self.body = nn.Sequential(
            nn.GELU(),
            nn.ConvTranspose2d(64, 64, 3, stride=1, padding=1), nn.GELU(),
            nn.ConvTranspose2d(64, 32, 4, stride=2, padding=1), nn.GELU(),  # 7->14
            nn.ConvTranspose2d(32, 1, 4, stride=2, padding=1),             # 14->28
            nn.Sigmoid(),
        )

    def forward(self, y):
        h = self.proj(y).view(-1, 64, 7, 7)
        return self.body(h)


class LatentRadio(nn.Module):
    def __init__(self, k=128):
        super().__init__()
        self.k = k
        self.enc = Encoder(k)
        self.dec = Decoder(k)

    def forward(self, x, snr_db):
        z = self.enc(x)
        y = awgn(z, snr_db)
        return self.dec(y)

    def transmit(self, x, snr_db):
        """Same as forward but returns the channel symbols too (for inspection)."""
        z = self.enc(x)
        y = awgn(z, snr_db)
        return self.dec(y), z, y


def psnr(x, xhat, eps=1e-8):
    """Peak SNR in dB for images in [0,1]. Higher is better."""
    mse = torch.mean((x - xhat) ** 2, dim=[1, 2, 3])
    return 10.0 * torch.log10(1.0 / (mse + eps))


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
