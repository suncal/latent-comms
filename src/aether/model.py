"""
Aether — two agents that invent a compressed, noise-robust language.

Setup (a referential game):
  * The SENDER sees a target image and emits a short message: d real numbers,
    power-normalized to unit average power (so it's a fair "channel signal").
  * The message crosses a noisy AWGN channel (reused from the Latent Radio work).
  * The RECEIVER sees K candidate images, turns the noisy message into a "query",
    embeds each candidate, and picks the candidate whose embedding best matches.

Nothing about the message is designed. The only training signal is "did the
receiver pick the right candidate?" (cross-entropy). To succeed the two agents
must agree on a code — an emergent language — and, because we inject channel
noise during training, that language comes out noise-robust. The message
dimension d is the language's "bandwidth"; sweeping it shows how few numbers the
agents actually need to cooperate.

Capability angle: today's LLM agents cooperate by exchanging verbose natural
language. These agents exchange d≈a-handful of numbers that survive a channel
which would corrupt the equivalent digital message — a form of efficient,
robust machine-to-machine communication current chat models don't do natively.
"""

import torch
import torch.nn as nn

from channel import power_normalize, awgn


class ImageEncoder(nn.Module):
    def __init__(self, out=64):
        super().__init__()
        self.body = nn.Sequential(
            nn.Conv2d(1, 16, 3, 2, 1), nn.GELU(),   # 28->14
            nn.Conv2d(16, 32, 3, 2, 1), nn.GELU(),  # 14->7
            nn.Conv2d(32, 32, 3, 1, 1), nn.GELU(),
        )
        self.proj = nn.Linear(32 * 7 * 7, out)

    def forward(self, x):
        return self.proj(self.body(x).flatten(1))


class Sender(nn.Module):
    def __init__(self, d, feat=64):
        super().__init__()
        self.enc = ImageEncoder(feat)
        self.head = nn.Sequential(nn.GELU(), nn.Linear(feat, d))

    def forward(self, target_img):
        z = self.head(self.enc(target_img))
        return power_normalize(z)               # (B, d) unit-power message


class Receiver(nn.Module):
    def __init__(self, d, emb=64):
        super().__init__()
        self.enc = ImageEncoder(emb)
        self.query = nn.Sequential(nn.Linear(d, emb), nn.GELU(), nn.Linear(emb, emb))

    def forward(self, msg, candidates):
        B, K = candidates.shape[:2]
        q = self.query(msg)                                       # (B, emb)
        cand = self.enc(candidates.reshape(B * K, 1, 28, 28)).reshape(B, K, -1)  # (B,K,emb)
        scores = torch.einsum("be,bke->bk", q, cand)              # (B, K) match scores
        return scores


class Aether(nn.Module):
    def __init__(self, d=16, feat=64):
        super().__init__()
        self.d = d
        self.sender = Sender(d, feat)
        self.receiver = Receiver(d, feat)

    def forward(self, target_img, candidates, snr_db):
        msg = self.sender(target_img)
        msg = awgn(msg, snr_db)
        return self.receiver(msg, candidates)


def count_params(m):
    return sum(p.numel() for p in m.parameters() if p.requires_grad)
