"""
Sigil — a visual message code that fades instead of breaking.

An Encoder turns a short bit-string (the message) into a small grayscale glyph.
A Decoder reads the glyph back into bits. They are trained together end-to-end
*through a simulated damage channel* — noise, blur, block erasure (occlusion),
brightness/contrast shifts — so the pair learns a joint source-channel code:
the message is spread redundantly across the whole glyph in a way the decoder
can still recover after heavy corruption.

This is the Latent Radio principle (jointly-learned coding that degrades
gracefully) applied to the real-world "channel" a printed or photographed code
goes through. A QR code has fixed algebraic error-correction with a hard limit —
past it, nothing decodes (a cliff). A Sigil's learned code has no such wall; its
accuracy slides down smoothly, so a partly-damaged Sigil still delivers most of
the message.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F

GLYPH = 24        # 24x24 grayscale glyph
NBITS = 32        # message length in bits (~ a short code / a few characters)


class Encoder(nn.Module):
    def __init__(self, nbits=NBITS, g=GLYPH):
        super().__init__()
        self.g = g
        self.fc = nn.Sequential(nn.Linear(nbits, 256), nn.GELU(), nn.Linear(256, 8 * 6 * 6), nn.GELU())
        self.up = nn.Sequential(
            nn.ConvTranspose2d(8, 32, 4, 2, 1), nn.GELU(),    # 6 -> 12
            nn.ConvTranspose2d(32, 32, 4, 2, 1), nn.GELU(),   # 12 -> 24
            nn.Conv2d(32, 1, 3, 1, 1), nn.Sigmoid(),
        )

    def forward(self, bits):
        h = self.fc(bits).view(-1, 8, 6, 6)
        return self.up(h)                       # (B,1,24,24) in [0,1]


class Decoder(nn.Module):
    def __init__(self, nbits=NBITS, g=GLYPH):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 32, 3, 2, 1), nn.GELU(),    # 24 -> 12
            nn.Conv2d(32, 64, 3, 2, 1), nn.GELU(),   # 12 -> 6
            nn.Conv2d(64, 64, 3, 1, 1), nn.GELU(),
        )
        self.fc = nn.Linear(64 * 6 * 6, nbits)

    def forward(self, glyph):
        return self.fc(self.body(glyph).flatten(1))  # bit logits


def damage(glyph, severity, kinds=("noise", "erase", "blur", "bright"), gen=None):
    """Apply a random mix of real-world corruptions at a given severity in [0,1].
    Differentiable (so it can sit inside end-to-end training)."""
    B = glyph.shape[0]
    x = glyph
    if "blur" in kinds and severity > 0.15:
        # mild low-pass: blend in an average-pooled version
        k = F.avg_pool2d(x, 3, 1, 1)
        a = 0.6 * severity
        x = (1 - a) * x + a * k
    if "bright" in kinds:
        gain = 1.0 + (torch.rand(B, 1, 1, 1, device=x.device) - 0.5) * 1.2 * severity
        bias = (torch.rand(B, 1, 1, 1, device=x.device) - 0.5) * 0.6 * severity
        x = x * gain + bias
    if "erase" in kinds:
        # blocky occlusion: a coarse 6x6 Bernoulli mask upsampled to glyph size,
        # zeroing ~severity fraction of blocks (like a covered corner / sticker).
        low = (torch.rand(B, 1, 6, 6, device=x.device) > (0.6 * severity)).float()
        mask = F.interpolate(low, size=(GLYPH, GLYPH), mode="nearest")
        x = x * mask
    if "noise" in kinds:
        x = x + torch.randn_like(x) * (0.7 * severity)
    return x.clamp(0, 1)


def bit_metrics(logits, bits):
    pred = (logits > 0).float()
    bit_acc = (pred == bits).float().mean().item()
    msg_acc = (pred == bits).all(dim=1).float().mean().item()  # whole message exactly right
    return bit_acc, msg_acc
