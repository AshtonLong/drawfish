"""Train the Drawfish network on Stockfish-labelled shards (see gen_data.py).

The whole dataset lives on the GPU; batches, legal-move masks and policy targets are
assembled there, so the CPU (busy with Stockfish) is never the bottleneck.

Targets per position:
  wdl   <- Stockfish WDL of the position (soft cross-entropy)
  adv   <- tanh(cp / 600) (MSE)
  pbest <- 0.8 * one-hot(Stockfish best move) + 0.2 * softmax(cp_move / 50 cp)
  pdraw <- softmax(U_move / tau),  U = D_move + gamma * (1 - |tanh(cp_move / 600)|)
           i.e. prefer the move that keeps the draw most likely; when every move is
           hopelessly won/lost (D ~ 0) prefer the one that keeps the score closest to equal.

Usage: python scripts/train.py --steps 24000 [--resume]
"""
import argparse
import copy
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from drawfish.encoding import MIRROR_PERM, NUM_MOVES  # noqa: E402
from drawfish.model import DrawNet, save_model  # noqa: E402
from drawfish.paths import CKPT_DIR, DATA_DIR  # noqa: E402

DRAW_GAMMA = 0.2
DRAW_TAU = 0.03
BEST_SOFT = 0.2
BEST_T = 50.0
VAL_EVERY_GAME = 50  # game_id % 50 == 0 -> validation (2%)

dev = torch.device("cuda")
M55 = 0x5555555555555555
M33 = 0x3333333333333333
M0F = 0x0F0F0F0F0F0F0F0F


def mirror_bb(x):
    """Mirror int64 bitboards a<->h (reverse the bits of every byte)."""
    x = ((x >> 1) & M55) | ((x & M55) << 1)
    x = ((x >> 2) & M33) | ((x & M33) << 2)
    return ((x >> 4) & M0F) | ((x & M0F) << 4)


class GPUData:
    """All shards concatenated on the GPU, split by game into train / val.
    A master copy is kept in system RAM so growing the dataset never needs two GPU copies."""

    FIELDS = ("feat", "wdl", "cp", "best", "nmv", "mv_idx", "mv_d", "mv_cp", "game")
    ON_GPU = ("feat", "wdl", "cp", "best", "nmv", "mv_idx", "mv_d", "mv_cp")

    def __init__(self, dirs):
        self.dirs = dirs
        self.loaded = set()
        self.parts = {k: [] for k in self.FIELDS}
        self.n = 0

    def refresh(self):
        new = []
        for d in self.dirs:
            for f in sorted(Path(d).glob("*.npz")):
                if f.name.endswith(".tmp.npz") or str(f) in self.loaded:
                    continue
                new.append(f)
        if not new:
            return 0
        for f in new:
            z = np.load(f)
            self.parts["feat"].append(torch.from_numpy(z["feat"].view(np.int64)))
            self.parts["wdl"].append(torch.from_numpy(z["wdl"].astype(np.int16)))
            self.parts["cp"].append(torch.from_numpy(z["cp"]))
            self.parts["best"].append(torch.from_numpy(z["best"].astype(np.int16)))
            self.parts["nmv"].append(torch.from_numpy(z["nmv"]))
            self.parts["mv_idx"].append(torch.from_numpy(z["mv_idx"].astype(np.int16)))
            self.parts["mv_d"].append(torch.from_numpy(np.ascontiguousarray(z["mv_wdl"][:, 1])))
            self.parts["mv_cp"].append(torch.from_numpy(z["mv_cp"]))
            self.parts["game"].append(torch.from_numpy(z["game"]))
            self.loaded.add(str(f))
        for k in self.FIELDS:  # consolidate the RAM copy
            self.parts[k] = [torch.cat(self.parts[k])]
        for k in self.ON_GPU:  # drop the old GPU copy before uploading the new one
            if hasattr(self, k):
                delattr(self, k)
        for k in ("off", "train_idx", "val_idx"):
            if hasattr(self, k):
                delattr(self, k)
        torch.cuda.empty_cache()
        for k in self.ON_GPU:
            setattr(self, k, self.parts[k][0].to(dev))
        self.n = self.feat.shape[0]
        self.off = torch.zeros(self.n + 1, dtype=torch.int64, device=dev)
        self.off[1:] = torch.cumsum(self.nmv.long(), 0)
        is_val = (self.parts["game"][0] % VAL_EVERY_GAME) == 0
        self.train_idx = torch.nonzero(~is_val).squeeze(1).to(dev)
        self.val_idx = torch.nonzero(is_val).squeeze(1).to(dev)
        return len(new)

    def feats(self, idx):
        return self.feat[idx]

    def batch(self, idx, augment):
        b = idx.shape[0]
        n = self.nmv[idx].long()
        starts = self.off[idx]
        row = torch.repeat_interleave(torch.arange(b, device=dev), n)
        first = torch.cumsum(n, 0) - n
        flat = starts[row] + torch.arange(row.shape[0], device=dev) - first[row]
        mv = self.mv_idx[flat].long()
        best = self.best[idx].long()
        feat = self.feat[idx].clone()
        if augment:
            flip = (feat[:, 13] == 0) & (torch.rand(b, device=dev) < 0.5)
            if flip.any():
                feat[flip, :13] = mirror_bb(feat[flip, :13])
                mv = torch.where(flip[row], PERM[mv], mv)
                best = torch.where(flip, PERM[best], best)
        md = self.mv_d[flat].float() / 250.0
        mcp = self.mv_cp[flat].float()
        legal = torch.zeros(b, NUM_MOVES, dtype=torch.bool, device=dev)
        legal[row, mv] = True
        u = md + DRAW_GAMMA * (1 - torch.tanh(mcp / 600).abs())
        tgt_draw = torch.full((b, NUM_MOVES), -1e9, device=dev)
        tgt_draw[row, mv] = u / DRAW_TAU
        tgt_draw = torch.softmax(tgt_draw, 1)
        soft = torch.full((b, NUM_MOVES), -1e9, device=dev)
        soft[row, mv] = mcp.clamp(-3000, 3000) / BEST_T
        tgt_best = BEST_SOFT * torch.softmax(soft, 1)
        tgt_best[torch.arange(b, device=dev), best] += 1 - BEST_SOFT
        return {
            "feat": feat, "legal": legal,
            "wdl": self.wdl[idx].float() / 1000.0,
            "adv": torch.tanh(self.cp[idx].float() / 600.0),
            "best": best, "tgt_best": tgt_best, "tgt_draw": tgt_draw,
        }


PERM = torch.from_numpy(MIRROR_PERM).to(dev)


def losses(out, bt):
    neg = torch.finfo(torch.float32).min
    lb = F.log_softmax(out["pbest"].masked_fill(~bt["legal"], neg), 1)
    ld = F.log_softmax(out["pdraw"].masked_fill(~bt["legal"], neg), 1)
    l_best = -(bt["tgt_best"] * lb.masked_fill(~bt["legal"], 0)).sum(1).mean()
    l_draw = -(bt["tgt_draw"] * ld.masked_fill(~bt["legal"], 0)).sum(1).mean()
    l_wdl = -(bt["wdl"] * F.log_softmax(out["wdl"], 1)).sum(1).mean()
    l_adv = F.mse_loss(out["adv"], bt["adv"])
    return {"wdl": l_wdl, "adv": l_adv, "best": l_best, "draw": l_draw}, lb, ld


W = {"wdl": 1.0, "adv": 1.0, "best": 0.7, "draw": 1.0}


@torch.no_grad()
def recalibrate_bn(model, data, batches=40, bs=1024):
    """Recompute BatchNorm statistics for (EMA-averaged) weights; averaged running stats
    do not match averaged weights."""
    bns = [m for m in model.modules() if isinstance(m, torch.nn.BatchNorm2d)]
    momenta = [m.momentum for m in bns]
    for m in bns:
        m.reset_running_stats()
        m.momentum = None  # cumulative average
    model.train()
    for _ in range(batches):
        idx = data.train_idx[torch.randint(0, data.train_idx.numel(), (bs,), device=dev)]
        model(data.feats(idx))
    for m, mom in zip(bns, momenta):
        m.momentum = mom
    model.eval()


@torch.no_grad()
def validate(net, data, max_n=60000, bs=2048):
    net.eval()
    idx_all = data.val_idx[:max_n]
    agg = {"wdl": 0., "adv": 0., "best": 0., "draw": 0., "acc_best": 0., "acc_draw": 0., "d_mae": 0.}
    cnt = 0
    for i in range(0, idx_all.shape[0], bs):
        idx = idx_all[i:i + bs]
        bt = data.batch(idx, augment=False)
        out = net(bt["feat"])
        ls, lb, ld = losses(out, bt)
        k = idx.shape[0]
        for name, v in ls.items():
            agg[name] += float(v) * k
        agg["acc_best"] += float((lb.argmax(1) == bt["best"]).float().sum())
        agg["acc_draw"] += float((ld.argmax(1) == bt["tgt_draw"].argmax(1)).float().sum())
        agg["d_mae"] += float((torch.softmax(out["wdl"], 1)[:, 1] - bt["wdl"][:, 1]).abs().sum())
        cnt += k
    net.train()
    return {k: v / max(cnt, 1) for k, v in agg.items()}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--steps", type=int, default=24000)
    ap.add_argument("--batch", type=int, default=1024)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--wd", type=float, default=0.01)
    ap.add_argument("--blocks", type=int, default=10)
    ap.add_argument("--channels", type=int, default=128)
    ap.add_argument("--data", nargs="*", default=[str(DATA_DIR / "sf")])
    ap.add_argument("--run", default="main")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--init", default=None, help="start from these weights (fine-tune)")
    ap.add_argument("--refresh-every", type=int, default=1500)
    ap.add_argument("--warmup", type=int, default=600)
    args = ap.parse_args()
    torch.backends.cudnn.benchmark = True
    out_dir = CKPT_DIR / args.run
    out_dir.mkdir(parents=True, exist_ok=True)

    data = GPUData(args.data)
    data.refresh()
    print(f"loaded {data.n:,} positions (train {data.train_idx.numel():,}, val {data.val_idx.numel():,})",
          flush=True)

    net = DrawNet(args.blocks, args.channels).to(dev)
    if args.init:
        net.load_state_dict(torch.load(args.init, map_location=dev, weights_only=False)["state"])
    decay = [p for n_, p in net.named_parameters() if p.ndim > 1]
    no_decay = [p for n_, p in net.named_parameters() if p.ndim <= 1]
    opt = torch.optim.AdamW([{"params": decay, "weight_decay": args.wd},
                             {"params": no_decay, "weight_decay": 0.0}], lr=args.lr)
    ema = torch.optim.swa_utils.AveragedModel(
        net, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.999), use_buffers=True)
    step = 0
    if args.resume and (out_dir / "latest.pt").exists():
        ck = torch.load(out_dir / "latest.pt", map_location=dev, weights_only=False)
        net.load_state_dict(ck["net"])
        opt.load_state_dict(ck["opt"])
        ema.load_state_dict(ck["ema"])
        step = ck["step"]
        print(f"resumed at step {step}", flush=True)

    def lr_at(s):
        if s < args.warmup:
            return args.lr * (s + 1) / args.warmup
        p = min(1.0, (s - args.warmup) / max(1, args.steps - args.warmup))
        return 1e-5 + (args.lr - 1e-5) * 0.5 * (1 + math.cos(math.pi * p))

    log = open(out_dir / "train_log.tsv", "a")
    run = {k: 0.0 for k in W}
    t0, seen = time.time(), 0
    net.train()
    while step < args.steps:
        if step % args.refresh_every == 0 and step > 0:
            if data.refresh():
                print(f"  data refreshed: {data.n:,} positions", flush=True)
        for g in opt.param_groups:
            g["lr"] = lr_at(step)
        idx = data.train_idx[torch.randint(0, data.train_idx.numel(), (args.batch,), device=dev)]
        bt = data.batch(idx, augment=True)
        out = net(bt["feat"])
        ls, _, _ = losses(out, bt)
        loss = sum(W[k] * v for k, v in ls.items())
        ls = {k: v.detach() for k, v in ls.items()}
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(net.parameters(), 5.0)
        opt.step()
        ema.update_parameters(net)
        step += 1
        seen += args.batch
        for k, v in ls.items():
            run[k] = 0.98 * run[k] + 0.02 * v.item() if step > 1 else v.item()
        if step % 200 == 0:
            el = time.time() - t0
            print(f"step {step:6d} lr {lr_at(step):.2e} | " +
                  " ".join(f"{k} {v:.4f}" for k, v in run.items()) +
                  f" | {seen / el:.0f} pos/s, epoch {seen / max(1, data.train_idx.numel()):.2f}",
                  flush=True)
        if step % 2000 == 0 or step == args.steps:
            ema_net = copy.deepcopy(ema.module)
            recalibrate_bn(ema_net, data)
            vm = validate(ema_net, data)
            msg = " ".join(f"{k} {v:.4f}" for k, v in vm.items())
            print(f"  [val ema @ {step}] {msg}", flush=True)
            log.write(f"{step}\t{time.time() - t0:.0f}\t{data.n}\t" +
                      "\t".join(f"{k}={v:.5f}" for k, v in vm.items()) + "\n")
            log.flush()
            save_model(ema_net, out_dir / f"net_{step:06d}.pt", {"step": step, "val": vm})
            save_model(ema_net, out_dir / "net_latest.pt", {"step": step, "val": vm})
            torch.save({"net": net.state_dict(), "opt": opt.state_dict(),
                        "ema": ema.state_dict(), "step": step}, out_dir / "latest.pt")
    print("training finished", flush=True)


if __name__ == "__main__":
    main()
