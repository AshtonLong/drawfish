//! Board/move encoding and draw rules, bit-for-bit compatible with drawfish/encoding.py and
//! the python-chess functions the Python search relies on.

use shakmaty::{Chess, Color, EnPassantMode, Move, Position, Role};

pub const N_FEAT: usize = 14;
pub const NUM_MOVES: usize = 4096 + 72;

const BB_A1: u64 = 1;
const BB_H1: u64 = 1 << 7;
const BB_A8: u64 = 1 << 56;
const BB_H8: u64 = 1 << 63;
const BB_DARK: u64 = 0xaa55_aa55_aa55_aa55;
const BB_LIGHT: u64 = 0x55aa_55aa_55aa_55aa;

/// Repetition key: python-chess `_transposition_key()` (pieces, colors, turn, castling
/// rights, en-passant square only if an en-passant capture is legal).
#[derive(Clone, Copy, PartialEq, Eq, Hash, Debug)]
pub struct Key {
    bb: [u64; 8],
    castling: u64,
    white: bool,
    ep: u8, // 64 = none
}

impl Key {
    pub fn ep(&self) -> Option<u32> {
        (self.ep < 64).then_some(self.ep as u32)
    }
}

pub fn key(pos: &Chess) -> Key {
    let b = pos.board();
    Key {
        bb: [b.pawns().0, b.knights().0, b.bishops().0, b.rooks().0, b.queens().0, b.kings().0,
             b.white().0, b.black().0],
        castling: pos.castles().castling_rights().0,
        white: pos.turn() == Color::White,
        ep: pos.ep_square(EnPassantMode::Legal).map_or(64, |s| s.to_u32() as u8),
    }
}

/// encoding.board_features: 14 words from the side to move's point of view.
pub fn features(pos: &Chess, k: &Key) -> [u64; N_FEAT] {
    let b = pos.board();
    let us = pos.turn();
    let ou = b.by_color(us).0;
    let ot = b.by_color(!us).0;
    let roles = [b.pawns().0, b.knights().0, b.bishops().0, b.rooks().0, b.queens().0, b.kings().0];
    let mut f = [0u64; N_FEAT];
    for i in 0..6 {
        f[i] = roles[i] & ou;
        f[6 + i] = roles[i] & ot;
    }
    f[12] = k.ep().map_or(0, |s| 1u64 << s);
    let cr = k.castling;
    let bit = |m: u64| (cr & m != 0) as u64;
    if us == Color::White {
        f[13] = bit(BB_H1) | bit(BB_A1) << 1 | bit(BB_H8) << 2 | bit(BB_A8) << 3;
    } else {
        f[13] = bit(BB_H8) | bit(BB_A8) << 1 | bit(BB_H1) << 2 | bit(BB_A1) << 3;
        for x in f.iter_mut().take(13) {
            *x = x.swap_bytes(); // vertical flip
        }
    }
    f
}

/// (from, to, promotion piece type) as python-chess sees the move: castling is the king's
/// two-square step, promotion uses python-chess piece numbers (N=2 .. Q=5, 0 = none).
pub fn uci_parts(m: &Move) -> (u32, u32, u32) {
    match *m {
        Move::Normal { from, to, promotion, .. } => {
            (from.to_u32(), to.to_u32(), promotion.map_or(0, |r| r as u32))
        }
        Move::EnPassant { from, to } => (from.to_u32(), to.to_u32(), 0),
        Move::Castle { king, rook } => {
            let k = king.to_u32();
            (k, (k & !7) | if rook > king { 6 } else { 2 }, 0)
        }
        Move::Put { to, .. } => (to.to_u32(), to.to_u32(), 0),
    }
}

/// Packed move for the Python side: from | to << 6 | promotion << 12.
pub fn move_code(m: &Move) -> u32 {
    let (f, t, p) = uci_parts(m);
    f | t << 6 | p << 12
}

/// encoding.move_index
pub fn move_index(m: &Move, white: bool) -> usize {
    let (mut fr, mut to, promo) = uci_parts(m);
    if !white {
        fr ^= 56;
        to ^= 56;
    }
    if promo != 0 && promo != Role::Queen as u32 {
        let up = promo - Role::Knight as u32; // N=0, B=1, R=2
        return (4096 + up * 24 + (fr & 7) * 3 + ((to & 7) + 1 - (fr & 7))) as usize;
    }
    (fr * 64 + to) as usize
}

fn has_insufficient(pos: &Chess, color: Color) -> bool {
    let b = pos.board();
    let ours = b.by_color(color).0;
    let theirs = b.by_color(!color).0;
    let (pawns, knights, bishops) = (b.pawns().0, b.knights().0, b.bishops().0);
    if ours & (pawns | b.rooks().0 | b.queens().0) != 0 {
        return false;
    }
    if ours & knights != 0 {
        return ours.count_ones() <= 2 && theirs & !b.kings().0 & !b.queens().0 == 0;
    }
    if ours & bishops != 0 {
        let same_color = bishops & BB_DARK == 0 || bishops & BB_LIGHT == 0;
        return same_color && pawns == 0 && knights == 0;
    }
    true
}

/// python-chess `is_insufficient_material()`.
pub fn insufficient_material(pos: &Chess) -> bool {
    has_insufficient(pos, Color::White) && has_insufficient(pos, Color::Black)
}
