"""Game-termination rules Drawfish plays under.

A game is drawn ONLY by stalemate, insufficient mating material, or threefold
repetition. Move counts never draw a game: no fifty-move rule, no seventy-five-move
rule, and no game-length cap counts as a draw.
"""
from collections import Counter

import chess

DRAW = "1/2-1/2"


def rep_key(board: chess.Board):
    return board._transposition_key()


class Game:
    """A board plus repetition bookkeeping."""

    def __init__(self, fen: str | None = None):
        self.board = chess.Board(fen) if fen else chess.Board()
        self.start_fen = self.board.fen()
        self.reps = Counter({rep_key(self.board): 1})

    def copy(self) -> "Game":
        g = Game.__new__(Game)
        g.board = self.board.copy()
        g.start_fen = self.start_fen
        g.reps = Counter(self.reps)
        return g

    def push(self, move: chess.Move):
        self.board.push(move)
        self.reps[rep_key(self.board)] += 1

    def pop(self) -> chess.Move:
        k = rep_key(self.board)
        self.reps[k] -= 1
        if self.reps[k] <= 0:
            del self.reps[k]
        return self.board.pop()

    def outcome(self):
        """Return (result, reason) if the game is over, else None."""
        b = self.board
        if not any(b.generate_legal_moves()):
            if b.is_check():
                return ("0-1" if b.turn == chess.WHITE else "1-0"), "checkmate"
            return DRAW, "stalemate"
        if b.is_insufficient_material():
            return DRAW, "insufficient material"
        if self.reps[rep_key(b)] >= 3:
            return DRAW, "threefold repetition"
        return None
