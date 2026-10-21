"""Benchmark: draw regret on held-out positions.

For positions from validation games (never trained on), each "player" chooses a move and a
deep Stockfish MultiPV search over ALL legal moves judges the draw probability D of each
choice.  regret = max_move D - D(chosen move).  Lower is better; 0 = always found the most
drawish move.  Positions are bucketed by Stockfish's eval for the side to move:
  balanced (|cp| < 100), worse (-500..-100: must defend), better (100..500: must not press),
  winning (500..3000: must give something back). Completely lost positions are skipped:
  there every move has D ~ 0, so they cannot tell players apart.

Players: drawfish (MCTS, draw mode), drawfish-policy (draw head only, no search),
drawfish-win (same network, ordinary engine), stockfish-best (Stockfish's own move),
random.

  python scripts/bench_positions.py --n 300 --nodes 800
  python scripts/bench_positions.py --configs "c1.5:cpuct=1.5;g0.4:gamma=0.4"   # extra search settings
"""
import argparse
import json
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import chess  # noqa: E402

from drawfish.encoding import board_features  # noqa: E402
from drawfish.paths import DATA_DIR, DEFAULT_MODEL, HOME, stockfish_path  # noqa: E402
from drawfish.rules import Game  # noqa: E402
from drawfish.sfclient import Stockfish  # noqa: E402

BUCKETS = ("balanced", "worse", "better", "winning")
PIECES = [chess.PAWN, chess.KNIGHT, chess.BISHOP, chess.ROOK, chess.QUEEN, chess.KING]


def features_to_board(f) -> chess.Board:
    """Rebuild a board (White to move, i.e. in the side-to-move frame) from features."""
    b = chess.Board(None)
    for c, color in ((0, chess.WHITE), (6, chess.BLACK)):
        for k, pt in enumerate(PIECES):
            bb = int(f[c + k])
            for sq in chess.scan_forward(bb):
                b.set_piece_at(sq, chess.Piece(pt, color))
    b.turn = chess.WHITE
    cr = int(f[13])
    fen_c = ("K" if cr & 1 else "") + ("Q" if cr & 2 else "") + ("k" if cr & 4 else "") + ("q" if cr & 8 else "")
    b.set_castling_fen(fen_c or "-")
    ep = int(f[12])
    b.ep_square = (ep.bit_length() - 1) if ep else None
    return b


def load_positions(n, seed):
    rng = random.Random(seed)
    files = sorted((DATA_DIR / "sf").glob("*.npz"))
    buckets = {b: [] for b in BUCKETS}
    seen = set()
    for f in files:
        z = np.load(f)
        feat, cps, nmv = z["feat"], z["cp"], z["nmv"]
        val = np.nonzero(z["game"] % 50 == 0)[0]
        for i in val:
            cp = int(cps[i])
            if nmv[i] < 2:
                continue
            key = ("balanced" if abs(cp) < 100 else "worse" if -500 < cp <= -100 else
                   "better" if 100 <= cp < 500 else "winning" if 500 <= cp < 3000 else None)
            if key and feat[i].tobytes() not in seen:
                seen.add(feat[i].tobytes())
                buckets[key].append(feat[i].copy())
    out = []
    for key, items in buckets.items():
        rng.shuffle(items)
        for f in items[:n // len(BUCKETS)]:
            b = features_to_board(f)
            if b.is_valid() and board_features(b) == [int(x) for x in f]:
                out.append((key, b))
    return out


def judge(sf, board, depth):
    """D (0..1) of every legal move from a deep MultiPV search, plus Stockfish's best move."""
    best, lines = sf.search(board.fen(), depth=depth, multipv=256)
    d = {ln.move: ln.wdl[1] / 1000.0 for ln in lines.values()}
    return d, best


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--nodes", type=int, default=800)
    ap.add_argument("--depth", type=int, default=10)
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--configs", default="", help='extra Drawfish settings, e.g. "c2:cpuct=2.0;t1:policy_temp=1.0"')
    ap.add_argument("--no-baselines", action="store_true")
    args = ap.parse_args()
    from drawfish.engine import DrawfishEngine
    from drawfish.search import SearchParams

    pos_file = HOME / "bench" / f"positions_n{args.n}_seed{args.seed}.json"
    if pos_file.exists():  # frozen set, so results stay comparable as more data is generated
        positions = [(k, chess.Board(fen)) for k, fen in json.loads(pos_file.read_text())]
    else:
        positions = load_positions(args.n, args.seed)
        pos_file.parent.mkdir(parents=True, exist_ok=True)
        pos_file.write_text(json.dumps([(k, b.fen()) for k, b in positions]))
    print(f"{len(positions)} positions:", {k: sum(1 for p in positions if p[0] == k) for k in BUCKETS})
    sf = Stockfish(stockfish_path(), threads=4, hash_mb=256)
    draw_eng = DrawfishEngine(args.model)
    win_eng = DrawfishEngine(args.model, params=SearchParams(mode="win"), evaluator=draw_eng.ev)
    extra = {}
    for spec in filter(None, args.configs.split(";")):
        name, _, kv = spec.partition(":")
        params = SearchParams()
        for item in filter(None, kv.split(",")):
            k, v = item.split("=")
            setattr(params, k.strip(), type(getattr(params, k.strip()))(v))
        extra[name.strip()] = DrawfishEngine(args.model, params=params, evaluator=draw_eng.ev)
    rng = random.Random(args.seed)
    players = ["drawfish"] + list(extra)
    if not args.no_baselines:
        players += ["drawfish-policy", "drawfish-win", "stockfish-best", "random"]
    reg = {p: {b: [] for b in BUCKETS} for p in players}
    cache_file = HOME / "bench" / f"judge_depth{args.depth}.json"
    cache = json.loads(cache_file.read_text()) if cache_file.exists() else {}
    for k, (bucket, b) in enumerate(positions):
        fen = b.fen()
        if fen not in cache:
            cache[fen] = judge(sf, b, args.depth)
        dmap, sf_best = cache[fen]
        if not dmap:
            continue
        dmax = max(dmap.values())
        g = Game(b.fen())
        draw_eng.new_game()
        win_eng.new_game()
        choice = {"drawfish": draw_eng.think(g, nodes=args.nodes)[0]}
        for name, eng in extra.items():
            eng.new_game()
            choice[name] = eng.think(g, nodes=args.nodes)[0]
        if not args.no_baselines:
            draw_eng.new_game()
            choice.update({
                "drawfish-policy": draw_eng.think(g, nodes=1)[0],
                "drawfish-win": win_eng.think(g, nodes=args.nodes)[0],
                "stockfish-best": chess.Move.from_uci(sf_best),
                "random": rng.choice(list(b.legal_moves)),
            })
        for p, m in choice.items():
            reg[p][bucket].append(dmax - dmap.get(m.uci(), 0.0))
        if (k + 1) % 25 == 0:
            print(f"  {k + 1}/{len(positions)}", flush=True)
    print(f"\nmean draw regret (lower is better), judge = Stockfish depth {args.depth}")
    print(f"{'player':16s} " + " ".join(f"{b:>9s}" for b in BUCKETS) + f" {'all':>9s}   {'found best (all)':>16s}")
    for p in players:
        allr = sum(reg[p].values(), [])
        cells = " ".join(f"{np.mean(reg[p][b]):9.3f}" for b in BUCKETS)
        hit = np.mean([r <= 0.02 for r in allr])
        print(f"{p:16s} {cells} {np.mean(allr):9.3f}   {100 * hit:15.1f}%")
    sf.close()


if __name__ == "__main__":
    main()
