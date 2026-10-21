"""Check the Rust search core (drawfish_core) against the Python reference.

1. Rules/encoding: on thousands of positions from random games plus hand-picked edge cases
   (castling, en passant incl. pinned pawns, under-promotions, insufficient material), the
   Rust side must produce exactly python-chess's legal moves, check and insufficient-material
   flags, legal en-passant square, network features and policy indices.
2. Search: with a deterministic fake network, RustMCTS and search.MCTS must build identical
   trees (same visit counts, value sums, priors and PV) over whole games with tree reuse,
   repetitions in history, and both modes. Priors are taken raw (no softmax) on both sides so
   the comparison does not depend on how numpy and Rust round exp().
3. Softmax: Rust priors equal numpy's float32 softmax to within float rounding.

    python scripts/test_core.py [--games 300] [--search-games 12]
"""
import argparse
import random
import sys
import zlib
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import chess  # noqa: E402
import numpy as np  # noqa: E402

from drawfish import fastsearch  # noqa: E402
from drawfish.encoding import NUM_MOVES, board_features, move_index  # noqa: E402
from drawfish.rules import Game  # noqa: E402
from drawfish.search import MCTS, SearchParams  # noqa: E402

EDGE_FENS = [
    chess.STARTING_FEN,
    "r3k2r/8/8/8/8/8/8/R3K2R w KQkq - 0 1",
    "r3k2r/8/8/8/8/8/8/R3K2R b KQkq - 0 1",
    "r3k2r/8/8/8/8/8/8/R3K2R w Kq - 0 1",
    "r3k2r/8/8/8/8/8/8/R3K2R b Qk - 0 1",
    "r3k2r/8/8/8/8/5q2/8/R3K2R w KQkq - 0 1",          # castling through check
    "rnbqkbnr/ppp1p1pp/8/3pPp2/8/8/PPPP1PPP/RNBQKBNR w KQkq f6 0 3",  # ep legal
    "8/8/8/K2pP2r/8/8/8/7k w - d6 0 1",                 # ep illegal: horizontal pin
    "8/8/8/8/k2pP2Q/8/8/7K b - e3 0 1",                 # ep illegal for black (pin)
    "4k3/8/8/2KpP3/8/8/8/8 w - d6 0 1",
    "8/2P5/8/8/8/8/5p2/K6k w - - 0 1",                  # promotions
    "8/2P5/8/8/8/8/5p2/K6k b - - 0 1",
    "1n2k3/2P5/8/8/8/8/8/4K3 w - - 0 1",                # capture-promotions
    "4k3/8/8/8/8/8/8/4K3 w - - 0 1",                    # K v K
    "4k3/8/8/8/8/8/8/3NK3 w - - 0 1",                   # K+N v K
    "4k3/8/8/8/8/8/8/2NNK3 w - - 0 1",                  # K+NN v K (sufficient)
    "4k3/8/8/8/8/8/8/2B1K3 w - - 0 1",                  # K+B v K
    "4k3/8/8/8/8/8/8/2BBK3 w - - 0 1",                  # bishops on both colors
    "4kb2/8/8/8/8/8/8/2B1K3 w - - 0 1",                 # opposite-colored bishops
    "4k1b1/8/8/8/8/8/8/2B1K3 w - - 0 1",                # same-colored bishops
    "4k3/8/8/8/8/8/8/1n1NK3 w - - 0 1",                 # knight each side
    "4k3/8/8/8/8/8/8/3NKq2 w - - 0 1",                  # knight v queen
    "7k/5Q2/6K1/8/8/8/8/8 b - - 0 1",                   # stalemate
    "6k1/5ppp/8/8/8/8/8/R5K1 b - - 0 1",
    "R5k1/5ppp/8/8/8/8/8/6K1 b - - 0 1",                # checkmate
    "r1bqkb1r/pppp1Qpp/2n2n2/4p3/2B1P3/8/PPPP1PPP/RNB1K1NR b KQkq - 0 4",
]


# ---------------------------------------------------------------------- 1. rules/encoding
def check_position(board: chess.Board):
    fen = board.fen()
    ref = chess.Board(fen)
    got = fastsearch.probe(fen)
    assert got is not None, f"Rust rejected {fen}"
    legal = list(ref.legal_moves)
    assert set(got["moves"]) == set(legal) and len(got["moves"]) == len(legal), \
        f"legal moves differ in {fen}: {sorted(m.uci() for m in got['moves'])} vs {sorted(m.uci() for m in legal)}"
    for m, i in zip(got["moves"], got["idx"]):
        assert i == move_index(m, ref.turn), f"move_index {m} in {fen}"
    assert got["white"] == (ref.turn == chess.WHITE)
    assert got["check"] == ref.is_check(), f"is_check {fen}"
    assert got["insufficient"] == ref.is_insufficient_material(), f"insufficient material {fen}"
    ep = ref.ep_square if ref.has_legal_en_passant() else None
    assert got["ep"] == ep, f"ep {fen}: {got['ep']} vs {ep}"
    want = np.array(board_features(ref), dtype=np.uint64)
    assert np.array_equal(got["feats"], want), f"features differ in {fen}"


def test_rules(n_games, rng):
    count = 0
    for fen in EDGE_FENS:
        check_position(chess.Board(fen))
        count += 1
    for _ in range(n_games):
        b = chess.Board()
        for _ in range(rng.randrange(20, 400)):
            check_position(b)
            count += 1
            moves = list(b.legal_moves)
            if not moves:
                break
            # bias towards captures and promotions so endgames are reached often
            spicy = [m for m in moves if b.is_capture(m) or m.promotion]
            b.push(rng.choice(spicy if spicy and rng.random() < 0.5 else moves))
    print(f"rules/encoding: {count} positions identical to python-chess")


# ---------------------------------------------------------------------- 2. search parity
class FakeEvaluator:
    """Deterministic pseudo-network: outputs depend only on the 14 feature words (and sel)."""

    def _row(self, f, s):
        rng = np.random.default_rng(zlib.crc32(f.tobytes()))
        wdl = rng.random(3, dtype=np.float32) + np.float32(0.05)
        wdl /= wdl.sum()
        adv = np.float32(rng.random() * 2 - 1)
        best = rng.random(NUM_MOVES, dtype=np.float32)
        draw = rng.random(NUM_MOVES, dtype=np.float32)
        return wdl, adv, np.float32(s) * draw + (np.float32(1) - np.float32(s)) * best

    def launch(self, feats, sel):
        rows = [self._row(feats[j], sel[j]) for j in range(len(feats))]
        return (np.stack([r[0] for r in rows]), np.array([r[1] for r in rows], dtype=np.float32),
                np.stack([r[2] for r in rows]))

    def wait(self, handle):
        return handle


def raw_python_mcts(ev, params):
    m = MCTS(ev, params)
    m._priors = lambda node, logits: logits[[move_index(mv, node.board.turn) for mv in node.moves]]
    return m


def raw_rust_mcts(ev, params):
    m = fastsearch.RustMCTS(ev, params)
    m.raw_priors = True
    return m


def compare_stats(a, b, where):
    assert (a is None) == (b is None), f"{where}: one search returned None"
    if a is None:
        return
    for k in ("best", "wdl", "adv", "nodes", "tree", "pv"):
        assert a[k] == b[k], f"{where}: {k} differs: {a[k]} vs {b[k]}"
    assert len(a["moves"]) == len(b["moves"]), f"{where}: move list length"
    for x, y in zip(a["moves"], b["moves"]):
        assert x == y, f"{where}: root move stats differ: {x} vs {y}"


def play_parity_game(start_fen, prefix, mode, nodes_seq, rng, params_kw):
    ev = FakeEvaluator()
    params = SearchParams(mode=mode, batch=rng.choice([1, 8, 32]), **params_kw)
    py, rs = raw_python_mcts(ev, params), raw_rust_mcts(ev, params)
    game = Game(start_fen)
    for uci in prefix:
        game.push(chess.Move.from_uci(uci))
    searches = 0
    for ply, nodes in enumerate(nodes_seq):
        if game.outcome():
            break
        a = py.search(game, nodes=nodes)
        b = rs.search(game, nodes=nodes)
        compare_stats(a, b, f"{start_fen} +{len(game.board.move_stack)} plies ({mode}, {nodes} nodes)")
        searches += 1
        # mostly follow the search (exercises tree reuse), sometimes an off-tree move
        moves = list(game.board.legal_moves)
        mv = a["best"] if rng.random() < 0.8 else rng.choice(moves)
        game.push(mv)
        if rng.random() < 0.05:
            py.reset()
            rs.reset()
    return searches


def test_search(n_games, rng):
    shuffle = ["g1f3", "g8f6", "f3g1", "f6g8", "g1f3", "g8f6", "f3g1", "f6g8"]  # start pos x3
    cases = [
        (chess.STARTING_FEN, []),
        (chess.STARTING_FEN, shuffle),
        (chess.STARTING_FEN, shuffle[:4]),
        ("r1bqkbnr/pppp1ppp/2n5/4p3/4P3/5N2/PPPP1PPP/RNBQKB1R w KQkq - 2 3", []),
        ("8/8/4k3/8/8/4K3/4P3/8 w - - 0 1", []),
        ("8/5k2/8/8/8/8/1Q6/K7 w - - 0 1", []),
        ("8/5k2/8/8/8/8/1Q6/K7 w - - 0 1", ["b2b3", "f7f6", "b3b2", "f6f7"]),  # repeated once
        ("6k1/5ppp/8/8/8/8/5PPP/3R2K1 w - - 0 1", []),
        ("r3k2r/pp3ppp/8/8/8/8/PP3PPP/R3K2R w KQkq - 0 1", []),
    ]
    total = 0
    for g in range(n_games):
        fen, prefix = cases[g % len(cases)]
        mode = "draw" if g % 3 else "win"
        kw = {}
        if g % 4 == 3:
            kw = {"draw_prior": 0.5, "cpuct": 1.7, "fpu_red": 0.35, "win_value": 0.8, "gamma": 0.3}
        nodes_seq = [rng.choice([1, 2, 50, 300, 800, 2000]) for _ in range(rng.randrange(6, 16))]
        total += play_parity_game(fen, prefix, mode, nodes_seq, rng, kw)
    print(f"search: {total} searches over {n_games} games -> identical trees in Rust and Python")


def test_cache_eviction(rng):
    """With a tiny cache the two searches may evict different entries, but they must stay
    legal and sane (no crash, visit counts add up)."""
    ev = FakeEvaluator()
    params = SearchParams(cache_size=50)
    rs = raw_rust_mcts(ev, params)
    st = rs.search(Game(), nodes=3000)
    assert st["nodes"] >= 3000 and sum(m["n"] for m in st["moves"]) <= st["tree"]
    print("tiny cache: ok")


# ---------------------------------------------------------------------- 3. softmax
def test_softmax(rng):
    """Production priors (softmax in Rust) against numpy, for every root move searched."""
    ev = FakeEvaluator()
    checked = 0
    for temp in (1.0, 1.7):
        params = SearchParams(policy_temp=temp)
        for fen in EDGE_FENS:
            g = Game(fen)
            if g.outcome():
                continue
            st = fastsearch.RustMCTS(ev, params).search(g, nodes=400)
            lg = ev.launch(np.array([board_features(g.board)], dtype=np.uint64), np.zeros(1, np.float32))[2][0]
            moves = list(g.board.legal_moves)
            x = lg[[move_index(m, g.board.turn) for m in moves]] / params.policy_temp
            e = np.exp(x - x.max())
            want = dict(zip(moves, (e / e.sum()).tolist()))
            for m in st["moves"]:
                assert abs(m["p"] - want[m["move"]]) <= 2e-6 * want[m["move"]] + 1e-9,                     f"prior {m['move']} in {fen}: {m['p']} vs {want[m['move']]}"
                checked += 1
    print(f"softmax priors: {checked} priors match numpy")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--games", type=int, default=300)
    ap.add_argument("--search-games", type=int, default=24)
    ap.add_argument("--seed", type=int, default=1)
    args = ap.parse_args()
    if not fastsearch.available():
        sys.exit(f"drawfish_core not available: {fastsearch.load_error()}")
    rng = random.Random(args.seed)
    test_rules(args.games, rng)
    test_softmax(rng)
    test_search(args.search_games, rng)
    test_cache_eviction(rng)
    print("ALL OK")


if __name__ == "__main__":
    main()
