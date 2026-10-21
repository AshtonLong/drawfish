"""High-level Drawfish engine used by the GUI, the UCI front-end and the benchmarks."""
import chess

from .evaluator import Evaluator
from .paths import DEFAULT_MODEL
from .rules import DRAW, Game
from .fastsearch import make_search
from .search import SearchParams


class DrawfishEngine:
    def __init__(self, model_path=DEFAULT_MODEL, device=None, params: SearchParams | None = None,
                 evaluator: Evaluator | None = None):
        self.ev = evaluator or Evaluator(model_path, device=device)
        self.params = params or SearchParams()
        self.mcts = make_search(self.ev, self.params)

    def new_game(self):
        self.mcts.reset()

    def instant_draw(self, game: Game):
        """A legal move that ends the game in a draw right now, if one exists."""
        if self.params.mode != "draw":
            return None
        for m in list(game.board.legal_moves):
            game.push(m)
            out = game.outcome()
            game.pop()
            if out and out[0] == DRAW:
                return m
        return None

    def think(self, game: Game, nodes=None, movetime=None, stop_event=None, info_cb=None):
        """Pick a move for the side to move in `game`. Returns (move, stats)."""
        m = self.instant_draw(game)
        if m is not None:
            return m, {"best": m, "wdl": (0.0, 1.0, 0.0), "adv": 0.0, "nodes": 0, "time": 0.0,
                       "nps": 0.0, "pv": [m], "moves": [], "note": "draw by rule"}
        if movetime is None and nodes is None:
            movetime = 1.0
        st = self.mcts.search(game, nodes=nodes, movetime=movetime, stop_event=stop_event,
                              info_cb=info_cb)
        if st is None or st["best"] is None:
            return next(iter(game.board.legal_moves), None), st
        best = st["best"]
        if self.params.mode == "draw" and _gives_mate(game.board, best):
            # Never win on purpose: prefer the most-searched move that does not mate.
            alts = [m["move"] for m in st["moves"] if not _gives_mate(game.board, m["move"])]
            alts += [m for m in game.board.legal_moves if not _gives_mate(game.board, m)]
            if alts:
                best = alts[0]
                st["best"] = best
        return best, st


def _gives_mate(board: chess.Board, move: chess.Move) -> bool:
    board.push(move)
    mate = board.is_checkmate()
    board.pop()
    return mate


def budget_from_clock(board: chess.Board, wtime=None, btime=None, winc=0, binc=0, movestogo=None):
    """Seconds to spend on this move given UCI clock parameters."""
    rem = wtime if board.turn == chess.WHITE else btime
    inc = winc if board.turn == chess.WHITE else binc
    if rem is None:
        return None
    rem, inc = rem / 1000.0, (inc or 0) / 1000.0
    mtg = movestogo or 30
    t = rem / mtg + 0.8 * inc
    return max(0.02, min(t, rem * 0.5 - 0.05))
