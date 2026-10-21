//! drawfish_core: the Drawfish tree search in Rust, exposed through a small C ABI that
//! drawfish/fastsearch.py loads with ctypes. Every entry point catches panics and reports
//! failure through its return value, so a bug makes Python fall back to the pure-Python
//! search instead of killing the process.

mod enc;
mod search;

use std::ffi::{c_char, CStr};
use std::panic::{catch_unwind, AssertUnwindSafe};
use std::slice;

use shakmaty::fen::Fen;
use shakmaty::{CastlingMode, Chess, Color, Position};

use enc::{N_FEAT, NUM_MOVES};
use search::{Params, Prepared, Search};

/// Bumped whenever the ABI changes; fastsearch.py refuses a DLL with another version.
pub const ABI_VERSION: u32 = 1;

fn guard<T>(fail: T, f: impl FnOnce() -> T) -> T {
    catch_unwind(AssertUnwindSafe(f)).unwrap_or(fail)
}

unsafe fn cstr<'a>(p: *const c_char) -> Option<&'a str> {
    if p.is_null() {
        return None;
    }
    CStr::from_ptr(p).to_str().ok()
}

/// params: [draw_mode, cpuct, fpu_red, win_value, gamma, draw_prior, policy_temp, cache_size,
/// raw_priors]
unsafe fn read_params(p: *const f64) -> Params {
    let v = slice::from_raw_parts(p, 9);
    Params {
        draw_mode: v[0] != 0.0,
        cpuct: v[1],
        fpu_red: v[2],
        win_value: v[3],
        gamma: v[4],
        draw_prior: v[5] as f32,
        policy_temp: v[6] as f32,
        cache_size: v[7].max(1.0) as usize,
        raw_priors: v[8] != 0.0,
    }
}

#[no_mangle]
pub extern "C" fn df_abi_version() -> u32 {
    ABI_VERSION
}

#[no_mangle]
pub unsafe extern "C" fn df_new(params: *const f64) -> *mut Search {
    guard(std::ptr::null_mut(), || Box::into_raw(Box::new(Search::new(read_params(params)))))
}

#[no_mangle]
pub unsafe extern "C" fn df_free(s: *mut Search) {
    if !s.is_null() {
        let _ = guard((), || drop(Box::from_raw(s)));
    }
}

#[no_mangle]
pub unsafe extern "C" fn df_reset(s: *mut Search) -> i32 {
    guard(-1, || {
        (*s).reset();
        0
    })
}

/// Set the root. Returns 0 = root is terminal, 1 = root needs an evaluation
/// (call df_root_batch), 2 = ready, -1 = error (message copied into err).
#[no_mangle]
pub unsafe extern "C" fn df_prepare(s: *mut Search, params: *const f64, start_fen: *const c_char,
                                    moves: *const c_char, white: i32, err: *mut u8, err_len: usize) -> i32 {
    guard(-1, || {
        let s = &mut *s;
        s.set_params(read_params(params));
        let (Some(fen), Some(mv)) = (cstr(start_fen), cstr(moves)) else {
            return -1;
        };
        let color = if white != 0 { Color::White } else { Color::Black };
        match s.prepare(fen, mv, color) {
            Ok(Prepared::Terminal) => 0,
            Ok(Prepared::NeedsEval) => 1,
            Ok(Prepared::Ready) => 2,
            Err(e) => {
                if !err.is_null() && err_len > 0 {
                    let b = e.as_bytes();
                    let n = b.len().min(err_len - 1);
                    std::ptr::copy_nonoverlapping(b.as_ptr(), err, n);
                    *err.add(n) = 0;
                }
                -1
            }
        }
    })
}

/// Select up to `want` leaves; writes features (want x 14 u64) and policy blend weights.
/// Returns the number of leaves (0 = nothing to evaluate) or -1; batch id in *batch.
#[no_mangle]
pub unsafe extern "C" fn df_select(s: *mut Search, want: i32, feats: *mut u64, sel: *mut f32,
                                   batch: *mut u64) -> i32 {
    guard(-1, || {
        let want = want.max(0) as usize;
        let f = slice::from_raw_parts_mut(feats, want * N_FEAT);
        let sl = slice::from_raw_parts_mut(sel, want);
        let (n, id) = (*s).select_batch(want, f, sl);
        *batch = id;
        n as i32
    })
}

#[no_mangle]
pub unsafe extern "C" fn df_root_batch(s: *mut Search, feats: *mut u64, sel: *mut f32, batch: *mut u64) -> i32 {
    guard(-1, || {
        let f = slice::from_raw_parts_mut(feats, N_FEAT);
        let sl = slice::from_raw_parts_mut(sel, 1);
        *batch = (*s).root_batch(f, sl);
        1
    })
}

/// Back up network outputs for a batch: wdl (n x 3), adv (n), logits (n x 4168), float32.
#[no_mangle]
pub unsafe extern "C" fn df_finish(s: *mut Search, batch: u64, n: i32, wdl: *const f32, adv: *const f32,
                                   logits: *const f32) -> i32 {
    guard(-1, || {
        let n = n.max(0) as usize;
        let r = (*s).finish_batch(batch, slice::from_raw_parts(wdl, 3 * n), slice::from_raw_parts(adv, n),
                                  slice::from_raw_parts(logits, NUM_MOVES * n));
        match r {
            Ok(k) => k as i32,
            Err(_) => -1,
        }
    })
}

/// out: [root visits, legal moves, expanded moves, most visits, second-most visits, state,
/// tree nodes]
#[no_mangle]
pub unsafe extern "C" fn df_summary(s: *mut Search, out: *mut i64) -> i32 {
    guard(-1, || {
        let s = &*s;
        let m = s.summary();
        let o = slice::from_raw_parts_mut(out, 7);
        o.copy_from_slice(&[m.root_n as i64, m.nmoves as i64, m.nexp as i64, m.top1 as i64,
                            m.top2 as i64, m.state as i64, s.tree_size() as i64]);
        0
    })
}

/// Root move statistics: up to k moves -> codes[k], vals[k x 6] (n, w, d, l, adv, p);
/// principal variation -> pv[max_pv]. Returns n_moves | n_pv << 16.
#[no_mangle]
pub unsafe extern "C" fn df_root_stats(s: *mut Search, k: i32, codes: *mut u32, vals: *mut f64, max_pv: i32,
                                       pv: *mut u32) -> i32 {
    guard(-1, || {
        let (moves, line) = (*s).root_stats(k.max(0) as usize, max_pv.max(0) as usize);
        let c = slice::from_raw_parts_mut(codes, moves.len());
        let v = slice::from_raw_parts_mut(vals, moves.len() * 6);
        for (j, m) in moves.iter().enumerate() {
            c[j] = m.code;
            v[j * 6..j * 6 + 6].copy_from_slice(&[m.n as f64, m.w, m.d, m.l, m.adv, m.p]);
        }
        slice::from_raw_parts_mut(pv, line.len()).copy_from_slice(&line);
        (moves.len() | line.len() << 16) as i32
    })
}

/// Test probe for one FEN: features (14 u64), legal moves (codes + policy indices, in
/// generation order) and info [turn is white, in check, insufficient material, legal ep
/// square or -1]. Returns the number of legal moves, or -1 if the FEN is rejected.
#[no_mangle]
pub unsafe extern "C" fn df_probe(fen: *const c_char, feats: *mut u64, codes: *mut u32, idx: *mut u32,
                                  max: i32, info: *mut i32) -> i32 {
    guard(-1, || {
        let Some(f) = cstr(fen) else { return -1 };
        let Ok(fen) = Fen::from_ascii(f.trim().as_bytes()) else { return -1 };
        let Ok(pos) = fen.into_position::<Chess>(CastlingMode::Standard) else { return -1 };
        let k = enc::key(&pos);
        slice::from_raw_parts_mut(feats, N_FEAT).copy_from_slice(&enc::features(&pos, &k));
        let white = pos.turn() == Color::White;
        let moves = pos.legal_moves();
        let n = moves.len().min(max.max(0) as usize);
        let c = slice::from_raw_parts_mut(codes, n);
        let ix = slice::from_raw_parts_mut(idx, n);
        for (j, m) in moves.iter().take(n).enumerate() {
            c[j] = enc::move_code(m);
            ix[j] = enc::move_index(m, white) as u32;
        }
        slice::from_raw_parts_mut(info, 4).copy_from_slice(&[
            white as i32, pos.is_check() as i32, enc::insufficient_material(&pos) as i32,
            k.ep().map_or(-1, |e| e as i32),
        ]);
        moves.len() as i32
    })
}
