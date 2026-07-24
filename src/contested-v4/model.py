"""
SNR-adaptive JSCC transceiver.

The v2 model used one fixed code for all channels, so it saturated and lost to
rate-adaptive digital in good channels. Here the encoder and decoder are
*conditioned on the SNR*: a small network turns the operating SNR into an
embedding that is fused at both bottlenecks, letting the model allocate its
representation differently for good vs. bad channels — encode finer detail when
the channel can carry it, fall back to robust coarse structure when it can't.
This is the deep-JSCC analogue of adaptive modulation and coding.
"""

import torch
import torch.nn as nn

from channel import transmit

M = 128


class SNREmbed(nn.Module):
    def __init__(self, dim=64):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(1, 64), nn.GELU(), nn.Linear(64, dim), nn.GELU())

    def forward(self, snr_norm):          # (B,1) -> (B,dim)
        return self.net(snr_norm)


class SatEncoder(nn.Module):
    def __init__(self, m=M, cdim=64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1), nn.GELU(),
            nn.Conv2d(32, 64, 3, 2, 1), nn.GELU(),
            nn.Conv2d(64, 64, 3, 1, 1), nn.GELU(),
        )
        self.proj = nn.Linear(64 * 8 * 8 + cdim, 2 * m)

    def forward(self, x, cond):
        h = self.body(x).flatten(1)
        return self.proj(torch.cat([h, cond], dim=1))


class SatDecoder(nn.Module):
    def __init__(self, m=M, cdim=64):
        super().__init__()
        self.proj = nn.Sequential(nn.Linear(4 * m + cdim, 64 * 8 * 8), nn.GELU())
        self.up = nn.Sequential(
            nn.ConvTranspose2d(64, 64, 4, 2, 1), nn.GELU(),
            nn.ConvTranspose2d(64, 32, 4, 2, 1), nn.GELU(),
            nn.Conv2d(32, 3, 3, 1, 1), nn.Sigmoid(),
        )

    def forward(self, feats, cond):
        h = self.proj(torch.cat([feats.flatten(1), cond], dim=1)).view(-1, 64, 8, 8)
        return self.up(h)


class AdaptiveJSCC(nn.Module):
    def __init__(self, m=M):
        super().__init__()
        self.m = m
        self.emb = SNREmbed()
        self.enc = SatEncoder(m)
        self.dec = SatDecoder(m)

    @staticmethod
    def _norm(snr_db, B, device):
        return torch.full((B, 1), (snr_db + 10.0) / 30.0, device=device)

    def encode(self, x, snr_db):
        cond = self.emb(self._norm(snr_db, x.shape[0], x.device))
        return self.enc(x, cond), cond

    def decode(self, feats, cond):
        return self.dec(feats, cond)

    def forward(self, x, snr_db, **chan):
        z, cond = self.encode(x, snr_db)
        feats, sinr = transmit(z, snr_db=snr_db, **chan)
        return self.dec(feats, cond), sinr


def psnr(x, xhat, eps=1e-8):
    mse = torch.mean((x - xhat) ** 2, dim=[1, 2, 3])
    return 10.0 * torch.log10(1.0 / (mse + eps))


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
