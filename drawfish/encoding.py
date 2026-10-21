"""Board and move encoding shared by data generation, training and search.

A position is stored as 14 uint64 words, always from the side-to-move's point of view
(the board is flipped vertically when Black is to move):
    0-5   own   P N B R Q K bitboards
    6-11  enemy P N B R Q K bitboards
    12    en-passant target square bitboard (only if an ep capture is actually legal)
    13    castling bits: 1 own O-O, 2 own O-O-O, 4 enemy O-O, 8 enemy O-O-O
The GPU expands these words into 18 8x8 input planes (see model.expand_features).

There is deliberately no half-move clock and no history: by the rules Drawfish plays
under, no move count ever makes a game a draw, and repetitions are handled exactly by
the search instead of being guessed by the network.

Moves are indexed in the same flipped frame: from*64 + to for ordinary moves and queen
promotions (4096 entries), plus 72 under-promotion slots (3 pieces x 8 files x 3 dirs).
"""
import chess
import numpy as np

N_FEAT = 14
N_PLANES = 18
NUM_MOVES = 4096 + 72

_flipv = chess.flip_vertical
_UNDERPROMO = {chess.KNIGHT: 0, chess.BISHOP: 1, chess.ROOK: 2}


def board_features(board: chess.Board) -> list:
    us = board.turn
    oc_us = board.occupied_co[us]
    oc_them = board.occupied_co[not us]
    p, n, b, r, q, k = (board.pawns, board.knights, board.bishops,
                        board.rooks, board.queens, board.kings)
    f = [p & oc_us, n & oc_us, b & oc_us, r & oc_us, q & oc_us, k & oc_us,
         p & oc_them, n & oc_them, b & oc_them, r & oc_them, q & oc_them, k & oc_them]
    ep = board.ep_square
    f.append(chess.BB_SQUARES[ep] if ep is not None and board.has_legal_en_passant() else 0)
    cr = board.clean_castling_rights()
    if us == chess.WHITE:
        c = ((cr & chess.BB_H1 != 0) | (cr & chess.BB_A1 != 0) << 1 |
             (cr & chess.BB_H8 != 0) << 2 | (cr & chess.BB_A8 != 0) << 3)
    else:
        c = ((cr & chess.BB_H8 != 0) | (cr & chess.BB_A8 != 0) << 1 |
             (cr & chess.BB_H1 != 0) << 2 | (cr & chess.BB_A1 != 0) << 3)
        f = [_flipv(x) for x in f]
    f.append(int(c))
    return f


def features_array(boards) -> np.ndarray:
    out = np.empty((len(boards), N_FEAT), dtype=np.uint64)
    for i, b in enumerate(boards):
        out[i] = board_features(b)
    return out


def move_index(move: chess.Move, turn: bool) -> int:
    fr, to = move.from_square, move.to_square
    if not turn:  # black to move: flip ranks
        fr ^= 56
        to ^= 56
    promo = move.promotion
    if promo and promo != chess.QUEEN:
        return 4096 + _UNDERPROMO[promo] * 24 + (fr & 7) * 3 + ((to & 7) - (fr & 7) + 1)
    return fr * 64 + to


def _mirror_perm() -> np.ndarray:
    """Policy-index permutation for mirroring the board a<->h (an involution)."""
    perm = np.arange(NUM_MOVES)
    for fr in range(64):
        for to in range(64):
            perm[fr * 64 + to] = (fr ^ 7) * 64 + (to ^ 7)
    for piece in range(3):
        for file in range(8):
            for d in range(3):
                perm[4096 + piece * 24 + file * 3 + d] = 4096 + piece * 24 + (7 - file) * 3 + (2 - d)
    return perm


MIRROR_PERM = _mirror_perm()


def clean_fen(board: chess.Board) -> str:
    """FEN with the move counters reset, so Stockfish never sees a fifty-move clock."""
    parts = board.fen().split(" ")
    parts[4], parts[5] = "0", "1"
    return " ".join(parts)
