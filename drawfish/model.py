"""The Drawfish network.

Residual tower with squeeze-and-excitation, four heads:
  wdl   - win/draw/loss logits for the side to move (Stockfish-WDL distilled)
  adv   - tanh-squashed advantage (tanh(cp/600)); unlike WDL it does not saturate when
          one side is a piece up vs. a queen up, which tells the drawer *how much* to
          give back when it is winning
  pbest - "what a strong player would play" policy (models the opponent)
  pdraw - "what keeps the game drawn" policy (proposes the drawer's own moves)
Policies use attention-style from/to logits: logit(from,to) = <q_from, k_to>/sqrt(d).
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from .encoding import N_FEAT, N_PLANES, NUM_MOVES


def expand_features(feat: torch.Tensor) -> torch.Tensor:
    """(B,14) int64 words -> (B,18,8,8) float planes, on whatever device feat is on."""
    b = feat.shape[0]
    shifts = torch.arange(64, device=feat.device, dtype=torch.int64)
    bits = ((feat[:, :13, None] >> shifts) & 1).to(torch.float32).view(b, 13, 8, 8)
    castle = ((feat[:, 13:14] >> torch.arange(4, device=feat.device)) & 1).to(torch.float32)
    castle = castle[:, :, None, None].expand(b, 4, 8, 8)
    ones = torch.ones(b, 1, 8, 8, device=feat.device)
    return torch.cat([bits, castle, ones], 1)


class SEResBlock(nn.Module):
    def __init__(self, c, se_ratio=4):
        super().__init__()
        self.c1 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.b1 = nn.BatchNorm2d(c)
        self.c2 = nn.Conv2d(c, c, 3, padding=1, bias=False)
        self.b2 = nn.BatchNorm2d(c)
        self.se1 = nn.Linear(c, c // se_ratio)
        self.se2 = nn.Linear(c // se_ratio, 2 * c)

    def forward(self, x):
        y = F.relu(self.b1(self.c1(x)), inplace=True)
        y = self.b2(self.c2(y))
        s = F.relu(self.se1(y.mean((2, 3))), inplace=True)
        w, b = self.se2(s).chunk(2, 1)
        y = y * torch.sigmoid(w)[:, :, None, None] + b[:, :, None, None]
        return F.relu(x + y, inplace=True)


def _underpromo_index():
    frm, to, tofile, piece = [], [], [], []
    for p in range(3):
        for f in range(8):
            for d in range(3):
                tf = min(7, max(0, f + d - 1))  # off-board combos are never legal; clamp
                frm.append(48 + f)
                to.append(56 + tf)
                tofile.append(tf)
                piece.append(p)
    return [torch.tensor(v) for v in (frm, to, tofile, piece)]


class PolicyHead(nn.Module):
    def __init__(self, c, d_model=128, d_key=64):
        super().__init__()
        self.embed = nn.Linear(c, d_model)
        self.q = nn.Linear(d_model, d_key)
        self.k = nn.Linear(d_model, d_key)
        self.promo = nn.Linear(d_key, 3)
        self.scale = 1.0 / math.sqrt(d_key)
        for name, t in zip(("up_from", "up_to", "up_tofile", "up_piece"), _underpromo_index()):
            self.register_buffer(name, t, persistent=False)

    def forward(self, x):
        b = x.shape[0]
        tok = F.relu(self.embed(x.flatten(2).transpose(1, 2)))  # (B,64,d_model)
        q, k = self.q(tok), self.k(tok)
        ft = torch.bmm(q, k.transpose(1, 2)) * self.scale  # (B,64 from,64 to)
        pb = self.promo(k[:, 56:64])  # (B,8,3): bias per (to-file, piece)
        up = ft[:, self.up_from, self.up_to] + pb[:, self.up_tofile, self.up_piece]
        return torch.cat([ft.reshape(b, 4096), up], 1)


class DrawNet(nn.Module):
    def __init__(self, blocks=10, channels=128):
        super().__init__()
        self.config = {"blocks": blocks, "channels": channels}
        c = channels
        self.inp = nn.Sequential(nn.Conv2d(N_PLANES, c, 3, padding=1, bias=False),
                                 nn.BatchNorm2d(c), nn.ReLU(inplace=True))
        self.tower = nn.Sequential(*[SEResBlock(c) for _ in range(blocks)])
        self.pbest = PolicyHead(c)
        self.pdraw = PolicyHead(c)
        self.vconv = nn.Sequential(nn.Conv2d(c, 32, 1, bias=False), nn.BatchNorm2d(32),
                                   nn.ReLU(inplace=True))
        self.vfc = nn.Linear(32 * 64, 128)
        self.wdl = nn.Linear(128, 3)
        self.adv = nn.Linear(128, 1)

    def forward(self, feat):
        x = self.tower(self.inp(expand_features(feat)))
        v = F.relu(self.vfc(self.vconv(x).flatten(1)), inplace=True)
        return {"wdl": self.wdl(v), "adv": torch.tanh(self.adv(v)).squeeze(1),
                "pbest": self.pbest(x), "pdraw": self.pdraw(x)}


def save_model(net: DrawNet, path, extra=None):
    torch.save({"config": net.config, "state": net.state_dict(), **(extra or {})}, path)


def load_model(path, device="cpu") -> DrawNet:
    ck = torch.load(path, map_location=device, weights_only=False)
    net = DrawNet(**ck["config"])
    net.load_state_dict(ck["state"])
    return net.to(device).eval()


assert N_FEAT == 14 and NUM_MOVES == 4168
