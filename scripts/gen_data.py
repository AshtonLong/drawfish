"""Generate Stockfish-labelled training positions.

Each worker plays games between two randomly chosen player *profiles* and labels every
position it visits with two Stockfish searches:

  A) a node-limited search  -> value target (Stockfish WDL), advantage target (cp)
                               and the "best move" policy target;
  B) a shallow MultiPV search over ALL legal moves -> W/D/L and cp for every move,
                               which becomes the draw-seeking policy target.

Profiles: strong (mostly Stockfish's best move), club / weak (softmax over move scores,
weak ones also blunder at random) and drawer (greedily maximizes draw probability).
Mixing them yields balanced, lopsided and "someone is steering toward a draw" positions,
which is exactly the distribution Drawfish meets when it plays.

Positions are sent to Stockfish without history and with the move counters reset, so the
labels never contain fifty-move-rule effects (move counts never draw under our rules).

Usage:  python scripts/gen_data.py --workers 11 --hours 2
Stop early by creating the file STOP inside the output directory.
"""
import argparse
import math
import multiprocessing as mp
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import chess  # noqa: E402

from drawfish.encoding import board_features, clean_fen, move_index  # noqa: E402
from drawfish.paths import DATA_DIR, stockfish_path  # noqa: E402
from drawfish.rules import Game  # noqa: E402
from drawfish.sfclient import Stockfish  # noqa: E402

NODES_A = 6000
DEPTH_B = 4
SHARD_POSITIONS = 25000
LOPSIDED_CP = 2500
LOPSIDED_PLIES = 30
MAX_PLIES = 350


def pick_profile(rng):
    r = rng.random()
    if r < 0.25:
        return {"name": "strong", "p_best": 0.9, "T": 30.0, "p_rand": 0.0}
    if r < 0.55:
        return {"name": "club", "p_best": 0.4, "T": rng.uniform(50, 120), "p_rand": 0.0}
    if r < 0.70:
        return {"name": "weak", "p_best": 0.1, "T": rng.uniform(120, 350), "p_rand": 0.03}
    return {"name": "drawer", "gamma": 0.2, "tau": 0.02}


def softmax_sample(rng, scores, temp):
    m = max(scores)
    w = [math.exp((s - m) / temp) for s in scores]
    return rng.choices(range(len(scores)), weights=w)[0]


def choose_move(rng, prof, legal, best_uci, mv_wdl, mv_cp):
    if prof["name"] == "drawer":
        u = [d[1] / 1000.0 + prof["gamma"] * (1.0 - abs(math.tanh(c / 600.0)))
             for d, c in zip(mv_wdl, mv_cp)]
        return legal[softmax_sample(rng, u, prof["tau"])]
    if rng.random() < prof["p_rand"]:
        return rng.choice(legal)
    if rng.random() < prof["p_best"]:
        return chess.Move.from_uci(best_uci)
    cps = [max(-3000, min(3000, c)) for c in mv_cp]
    return legal[softmax_sample(rng, cps, prof["T"])]


class ShardWriter:
    def __init__(self, out_dir, wid):
        self.out_dir, self.wid, self.k = out_dir, wid, 0
        while (out_dir / f"w{wid:02d}_{self.k:04d}.npz").exists():
            self.k += 1
        self.reset()

    def reset(self):
        self.feat, self.wdl, self.cp, self.best = [], [], [], []
        self.nmv, self.mv_idx, self.mv_wdl, self.mv_cp = [], [], [], []
        self.game, self.ply, self.result = [], [], []

    def __len__(self):
        return len(self.feat)

    def add_game(self, recs, result_white):
        for r in recs:
            self.feat.append(r["feat"])
            self.wdl.append(r["wdl"])
            self.cp.append(r["cp"])
            self.best.append(r["best"])
            self.nmv.append(len(r["mv_idx"]))
            self.mv_idx.extend(r["mv_idx"])
            self.mv_wdl.extend(r["mv_wdl"])
            self.mv_cp.extend(r["mv_cp"])
            self.game.append(r["game"])
            self.ply.append(r["ply"])
            if result_white is None:
                self.result.append(-2)
            else:
                self.result.append(result_white if r["white"] else -result_white)

    def flush(self):
        if not self.feat:
            return
        name = self.out_dir / f"w{self.wid:02d}_{self.k:04d}"
        tmp = str(name) + ".tmp.npz"
        mv_wdl = np.round(np.asarray(self.mv_wdl, dtype=np.float32) / 4.0).astype(np.uint8)
        np.savez(tmp,
                 feat=np.asarray(self.feat, dtype=np.uint64),
                 wdl=np.asarray(self.wdl, dtype=np.uint16),
                 cp=np.clip(np.asarray(self.cp), -32000, 32000).astype(np.int16),
                 best=np.asarray(self.best, dtype=np.uint16),
                 nmv=np.asarray(self.nmv, dtype=np.uint8),
                 mv_idx=np.asarray(self.mv_idx, dtype=np.uint16),
                 mv_wdl=mv_wdl.reshape(-1, 3),
                 mv_cp=np.clip(np.asarray(self.mv_cp), -32000, 32000).astype(np.int16),
                 game=np.asarray(self.game, dtype=np.int64),
                 ply=np.asarray(self.ply, dtype=np.uint16),
                 result=np.asarray(self.result, dtype=np.int8))
        os.replace(tmp, str(name) + ".npz")
        self.k += 1
        self.reset()


def play_game(rng, sf, game_id):
    g = Game()
    if rng.random() < 0.8:
        for _ in range(rng.randint(1, 8)):
            legal = list(g.board.legal_moves)
            if not legal:
                break
            g.push(rng.choice(legal))
    if g.outcome():
        return [], None
    profiles = {chess.WHITE: pick_profile(rng), chess.BLACK: pick_profile(rng)}
    recs, lopsided, plies = [], 0, 0
    while True:
        out = g.outcome()
        if out:
            res = {"1-0": 1, "0-1": -1}.get(out[0], 0)
            return recs, res
        b = g.board
        fen = clean_fen(b)
        best_uci, la = sf.search(fen, nodes=NODES_A)
        _, lb = sf.search(fen, depth=DEPTH_B, multipv=256)
        legal = list(b.legal_moves)
        by_move = {ln.move: ln for ln in lb.values()}
        mv_wdl, mv_cp = [], []
        for m in legal:
            ln = by_move.get(m.uci())
            if ln is None:  # should not happen; treat as a terrible move
                mv_wdl.append((0, 0, 1000))
                mv_cp.append(-20000)
            else:
                mv_wdl.append(ln.wdl)
                mv_cp.append(ln.cp)
        top = la.get(1)
        if top is None:
            return recs, None
        turn = b.turn
        recs.append({
            "feat": board_features(b), "wdl": top.wdl, "cp": top.cp,
            "best": move_index(chess.Move.from_uci(best_uci), turn),
            "mv_idx": [move_index(m, turn) for m in legal],
            "mv_wdl": mv_wdl, "mv_cp": mv_cp,
            "game": game_id, "ply": min(plies, 65535), "white": turn == chess.WHITE,
        })
        lopsided = lopsided + 1 if abs(top.cp) >= LOPSIDED_CP else 0
        plies += 1
        if lopsided >= LOPSIDED_PLIES or plies >= MAX_PLIES:
            return recs, None
        g.push(choose_move(rng, profiles[turn], legal, best_uci, mv_wdl, mv_cp))


def worker(wid, out_dir, deadline, counter):
    rng = random.Random(int(time.time() * 1000) ^ (wid * 7919))
    sf = Stockfish(stockfish_path(), threads=1, hash_mb=64, low_priority=True)
    writer = ShardWriter(out_dir, wid)
    stop_file = out_dir / "STOP"
    gid = 0
    try:
        while time.time() < deadline and not stop_file.exists():
            try:
                if gid % 50 == 0:
                    sf.new_game()
                recs, res = play_game(rng, sf, wid * 10_000_000 + writer.k * 100_000 + gid)
            except Exception as e:  # Stockfish died or hung: drop this game, restart it
                print(f"worker {wid}: {e!r}; restarting Stockfish", flush=True)
                sf.close()
                sf = Stockfish(stockfish_path(), threads=1, hash_mb=64, low_priority=True)
                gid += 1
                continue
            gid += 1
            writer.add_game(recs, res)
            with counter.get_lock():
                counter.value += len(recs)
            if len(writer) >= SHARD_POSITIONS:
                writer.flush()
    finally:
        writer.flush()
        sf.close()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--workers", type=int, default=max(1, os.cpu_count() - 1))
    ap.add_argument("--hours", type=float, default=2.0)
    ap.add_argument("--out", type=Path, default=DATA_DIR / "sf")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "STOP").unlink(missing_ok=True)
    deadline = time.time() + args.hours * 3600
    counter = mp.Value("q", 0)
    procs = [mp.Process(target=worker, args=(w, args.out, deadline, counter), daemon=True)
             for w in range(args.workers)]
    for p in procs:
        p.start()
    t0 = time.time()
    while any(p.is_alive() for p in procs):
        time.sleep(30)
        n = counter.value
        el = time.time() - t0
        print(f"[{el / 60:6.1f} min] {n:,} positions  ({n / max(el, 1):.0f}/s)", flush=True)
    for p in procs:
        p.join()
    print("done", counter.value)


if __name__ == "__main__":
    main()
