"""Benchmark: how often does Drawfish draw against opponents that try to beat it?

Plays Drawfish (fixed nodes per move) against Stockfish at several strength levels from
16 balanced openings, each opening with both colours. Games end only by checkmate,
stalemate, insufficient material or threefold repetition: there is no move-count draw.
A game that reaches the --max-plies safety cap is reported as "unfinished" and counted
as NOT drawn.

  python scripts/bench_match.py --nodes 800 --games-per-opp 32 --workers 4
  python scripts/bench_match.py --mode win      # same network as an ordinary engine (baseline)
"""
import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import chess  # noqa: E402
import chess.pgn  # noqa: E402

from drawfish.paths import DEFAULT_MODEL, HOME, stockfish_path  # noqa: E402
from drawfish.rules import DRAW, Game  # noqa: E402

OPENINGS = [
    "e4 e5 Nf3 Nc6 Bb5 a6", "e4 e5 Nf3 Nc6 Bc4 Bc5", "e4 c5 Nf3 d6 d4 cxd4 Nxd4 Nf6",
    "e4 c5 Nc3 Nc6 g3", "e4 e6 d4 d5 Nc3 Nf6", "e4 c6 d4 d5 e5 Bf5", "d4 d5 c4 e6 Nc3 Nf6",
    "d4 d5 c4 c6 Nf3 Nf6", "d4 Nf6 c4 g6 Nc3 Bg7 e4 d6", "d4 Nf6 c4 e6 Nc3 Bb4",
    "c4 e5 Nc3 Nf6 g3", "Nf3 d5 g3 Nf6 Bg2", "e4 d5 exd5 Qxd5 Nc3 Qa5", "e4 e5 Nf3 Nf6 Nxe5 d6",
    "d4 f5 g3 Nf6 Bg2", "e4 g6 d4 Bg7 Nc3 d6",
]

OPPONENTS = {
    "sf1320": {"elo": 1320}, "sf1600": {"elo": 1600}, "sf2000": {"elo": 2000},
    "sf2400": {"elo": 2400}, "sf2800": {"elo": 2800}, "sfmax": {"elo": None},
}


def sf_position_args(game: Game):
    """FEN + moves for Stockfish: the last <=20 plies with the halfmove clock reset, so its
    internal fifty-move rule never triggers while repetitions remain visible."""
    b = game.board.copy()
    moves = []
    while b.move_stack and len(moves) < 20:
        moves.append(b.pop().uci())
    parts = b.fen().split(" ")
    parts[4] = "0"
    return " ".join(parts), list(reversed(moves))


def play_one(engine, sf, opp, opening, df_white, nodes, movetime_ms, max_plies):
    g = Game()
    for san in opening.split():
        g.push(g.board.parse_san(san))
    engine.new_game()
    sf.new_game()
    df_color = chess.WHITE if df_white else chess.BLACK
    d_sum, d_n = 0.0, 0
    while True:
        out = g.outcome()
        if out:
            res, reason = out
            break
        if len(g.board.move_stack) >= max_plies:
            res, reason = "*", "unfinished (ply cap)"
            break
        if g.board.turn == df_color:
            mv, st = engine.think(g, nodes=nodes)
            if st and st.get("wdl"):
                d_sum += st["wdl"][1]
                d_n += 1
        else:
            fen, moves = sf_position_args(g)
            best, _ = sf.search(fen, moves=moves, movetime=movetime_ms)
            mv = chess.Move.from_uci(best)
        g.push(mv)
    if res == DRAW:
        score = "D"
    elif res == "*":
        score = "U"
    else:
        df_won = (res == "1-0") == df_white
        score = "W" if df_won else "L"
    pg = chess.pgn.Game.from_board(g.board)
    pg.headers["White"] = "Drawfish" if df_white else opp
    pg.headers["Black"] = opp if df_white else "Drawfish"
    pg.headers["Result"] = res
    pg.headers["Termination"] = reason
    return {"opp": opp, "opening": opening, "df_white": df_white, "score": score,
            "reason": reason, "plies": len(g.board.move_stack),
            "avg_draw_belief": d_sum / max(d_n, 1), "pgn": str(pg)}


def worker(jobs, results, args):
    from drawfish.engine import DrawfishEngine
    from drawfish.search import SearchParams
    from drawfish.sfclient import Stockfish
    params = SearchParams(mode=args.mode)
    for item in filter(None, args.params.split(",")):
        k, v = item.split("=")
        setattr(params, k.strip(), type(getattr(params, k.strip()))(v))
    engine = DrawfishEngine(args.model, params=params)
    sfs = {}
    while True:
        job = jobs.get()
        if job is None:
            break
        opp, opening, df_white = job
        if opp not in sfs:
            elo = OPPONENTS[opp]["elo"]
            opts = {"UCI_LimitStrength": "true", "UCI_Elo": elo} if elo else {}
            sfs[opp] = Stockfish(stockfish_path(), threads=1, hash_mb=32, options=opts)
        r = play_one(engine, sfs[opp], opp, opening, df_white, args.nodes, args.sf_movetime,
                     args.max_plies)
        results.put(r)
    for sf in sfs.values():
        sf.close()


def summarize(rows, opps):
    lines = [f"{'opponent':10s} {'games':>5s} {'draw%':>6s} {'DF win':>6s} {'DF loss':>7s} "
             f"{'unfin':>5s} {'avg plies':>9s}  draw reasons"]
    for opp in opps:
        rs = [r for r in rows if r["opp"] == opp]
        if not rs:
            continue
        n = len(rs)
        c = {k: sum(r["score"] == k for r in rs) for k in "WDLU"}
        reasons = {}
        for r in rs:
            if r["score"] == "D":
                reasons[r["reason"]] = reasons.get(r["reason"], 0) + 1
        rs_txt = ", ".join(f"{k} {v}" for k, v in sorted(reasons.items(), key=lambda x: -x[1]))
        lines.append(f"{opp:10s} {n:5d} {100 * c['D'] / n:5.1f}% {c['W']:6d} {c['L']:7d} {c['U']:5d} "
                     f"{sum(r['plies'] for r in rs) / n:9.1f}  {rs_txt}")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--mode", default="draw", choices=["draw", "win"])
    ap.add_argument("--opponents", default="sf1320,sf1600,sf2000,sf2400,sf2800,sfmax")
    ap.add_argument("--games-per-opp", type=int, default=32)
    ap.add_argument("--nodes", type=int, default=800)
    ap.add_argument("--sf-movetime", type=int, default=100)
    ap.add_argument("--max-plies", type=int, default=600)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--params", default="", help='search overrides, e.g. "win_value=0.95,cpuct=1.5"')
    ap.add_argument("--tag", default=None)
    args = ap.parse_args()
    opps = args.opponents.split(",")
    jobs, results = mp.Queue(), mp.Queue()
    n_jobs = 0
    for opp in opps:
        for k in range(args.games_per_opp):
            jobs.put((opp, OPENINGS[(k // 2) % len(OPENINGS)], k % 2 == 0))
            n_jobs += 1
    for _ in range(args.workers):
        jobs.put(None)
    procs = [mp.Process(target=worker, args=(jobs, results, args)) for _ in range(args.workers)]
    for p in procs:
        p.start()
    rows, t0 = [], time.time()
    tag = args.tag or f"{args.mode}_n{args.nodes}_{time.strftime('%Y%m%d_%H%M%S')}"
    out_dir = HOME / "bench" / tag
    out_dir.mkdir(parents=True, exist_ok=True)
    while len(rows) < n_jobs:
        r = results.get()
        rows.append(r)
        print(f"[{len(rows)}/{n_jobs} {(time.time() - t0) / 60:5.1f}m] {r['opp']:7s} "
              f"{'W' if r['df_white'] else 'B'} {r['score']} {r['plies']:4d} plies  {r['reason']}",
              flush=True)
    for p in procs:
        p.join()
    summary = summarize(rows, opps)
    print("\n" + summary)
    (out_dir / "summary.txt").write_text(
        f"model={args.model} mode={args.mode} nodes={args.nodes} sf_movetime={args.sf_movetime}ms "
        f"params={args.params or 'defaults'}\n" + summary + "\n")
    (out_dir / "games.json").write_text(json.dumps([{k: v for k, v in r.items() if k != "pgn"} for r in rows], indent=1))
    (out_dir / "games.pgn").write_text("\n\n".join(r["pgn"] for r in rows))
    print(f"\nsaved to {out_dir}")


if __name__ == "__main__":
    main()
