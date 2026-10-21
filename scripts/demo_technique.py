"""Show Drawfish's drawing technique from positions where a normal engine would just win
(or lose). The opponent is the same network running as an ordinary, win-seeking engine.

  python scripts/demo_technique.py [--nodes 3000]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from drawfish.engine import DrawfishEngine  # noqa: E402
from drawfish.paths import DEFAULT_MODEL  # noqa: E402
from drawfish.rules import Game  # noqa: E402
from drawfish.search import SearchParams  # noqa: E402

POSITIONS = {
    "Drawfish has K+R vs K (must get rid of the rook)": "8/8/8/4k3/8/8/8/R3K3 w - - 0 1",
    "Drawfish has K+Q+P vs K": "4k3/8/8/8/8/8/4P3/3QK3 w - - 0 1",
    "Drawfish is a full rook up": "r5k1/5ppp/8/8/8/8/5PPP/R2R2K1 w - - 0 1",
    "Drawfish is a queen up in the middlegame": "r1b2rk1/pp3ppp/2n1p3/3p4/3P4/2PB1N2/PP3PPP/R1BQ1RK1 w - - 0 1",
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--nodes", type=int, default=3000)
    ap.add_argument("--max-plies", type=int, default=80)
    args = ap.parse_args()
    drawer = DrawfishEngine(args.model)
    opponent = DrawfishEngine(args.model, params=SearchParams(mode="win"), evaluator=drawer.ev)
    for name, fen in POSITIONS.items():
        g = Game(fen)
        drawfish_color = g.board.turn
        drawer.new_game()
        opponent.new_game()
        sans, result = [], "unfinished"
        for _ in range(args.max_plies):
            out = g.outcome()
            if out:
                result = f"{out[0]} ({out[1]})"
                break
            eng = drawer if g.board.turn == drawfish_color else opponent
            mv, _ = eng.think(g, nodes=args.nodes)
            sans.append(g.board.san(mv))
            g.push(mv)
        print(f"{name}\n  {fen}\n  {' '.join(sans)}\n  -> {result}\n", flush=True)


if __name__ == "__main__":
    main()
