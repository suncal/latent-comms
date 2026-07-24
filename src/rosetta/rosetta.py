"""
Rosetta — a semantic interoperability layer for independently-trained transceivers.

The open problem (the #1 named barrier to deploying semantic communication): two
endpoints built by different vendors, or trained in separate runs, learn private
latent codes and cannot understand each other — communication collapses to noise
even on a perfect channel. Standards bodies have no answer yet; formal models of
this "knowledge mismatch" are, per the literature, largely undeveloped.

Rosetta's answer: don't retrain the base models (you can't — they're deployed /
proprietary). Instead standardize a shared REFERENCE SPACE and give each model two
tiny adapters:
    to_ref_i   : model_i's private symbols  ->  the shared reference signal (transmitted)
    from_ref_j : the received reference      ->  model_j's private symbols
Every base encoder/decoder stays frozen. Train the adapters jointly so that for
EVERY sender i and receiver j, dec_j(from_ref_j(channel(to_ref_i(enc_i(x))))) ~ x.
Then any transmitter interoperates with any receiver through the standard — the
O(N^2) vendor-pair problem collapses to O(N) adapters against one reference.
"""

import torch
import torch.nn as nn

from channel import power_normalize, awgn


class Adapter(nn.Module):
    """A lightweight residual map between a private symbol space and the reference."""

    def __init__(self, k=128, hid=256):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(k, hid), nn.GELU(),
            nn.Linear(hid, hid), nn.GELU(),
            nn.Linear(hid, k),
        )

    def forward(self, z):
        return z + self.net(z)          # residual: identity is a sensible init prior


class Rosetta(nn.Module):
    def __init__(self, base_models, k=128):
        """base_models: list of frozen LatentRadio models (one per 'vendor')."""
        super().__init__()
        self.k = k
        self.n = len(base_models)
        self.bases = base_models
        for m in self.bases:
            for p in m.parameters():
                p.requires_grad_(False)
            m.eval()
        self.to_ref = nn.ModuleList([Adapter(k) for _ in range(self.n)])
        self.from_ref = nn.ModuleList([Adapter(k) for _ in range(self.n)])

    def transmit(self, x, i, snr_db):
        """Sender i -> reference signal over the channel -> received reference."""
        z = self.bases[i].enc(x)                       # private symbols
        r = power_normalize(self.to_ref[i](z))         # standardized, unit-power signal
        return awgn(r, snr_db)

    def receive(self, r_rx, j):
        """Receiver j reconstructs from the received reference signal."""
        z_j = self.from_ref[j](r_rx)
        return self.bases[j].dec(z_j)

    def cross(self, x, i, j, snr_db):
        return self.receive(self.transmit(x, i, snr_db), j)
