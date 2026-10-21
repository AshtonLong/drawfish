"""Rust-accelerated search: the tree search of drawfish/search.py running in the compiled
drawfish_core library (source in core/), driven from here with the same PyTorch evaluator.

The Python search spends most of its time generating moves and walking the tree, which
left the GPU idle half the time; in Rust that work is nearly free, so the search runs as
fast as the network can evaluate positions. Results are the same: core/ is a line-by-line
port, checked against search.py by scripts/test_core.py.

make_search() returns the Rust search when the library loads and the pure-Python one
otherwise (or when DRAWFISH_SEARCH=python). RustMCTS itself falls back to the Python search
for any position the Rust side rejects (e.g. a FEN that python-chess accepts but the Rust
chess library considers impossible).
"""
import ctypes
import os
import time
from pathlib import Path

import chess
import numpy as np

from .encoding import N_FEAT
from .search import MCTS, SearchParams

ABI_VERSION = 1
LIB_PATH = Path(os.environ.get("DRAWFISH_CORE") or Path(__file__).resolve().parent / "drawfish_core.dll")
_lib = None
_load_error = None


def _load():
    global _lib, _load_error
    if _lib is not None or _load_error is not None:
        return _lib
    try:
        lib = ctypes.CDLL(str(LIB_PATH))
        if lib.df_abi_version() != ABI_VERSION:
            raise OSError(f"{LIB_PATH.name} has ABI {lib.df_abi_version()}, expected {ABI_VERSION}; "
                          "rebuild it with core/build.bat")
        vp, i32, f64p = ctypes.c_void_p, ctypes.c_int32, ctypes.POINTER(ctypes.c_double)
        u64p = ctypes.POINTER(ctypes.c_uint64)
        sigs = {
            "df_new": (vp, [f64p]),
            "df_free": (None, [vp]),
            "df_reset": (i32, [vp]),
            "df_prepare": (i32, [vp, f64p, ctypes.c_char_p, ctypes.c_char_p, i32, ctypes.c_char_p,
                                 ctypes.c_size_t]),
            "df_select": (i32, [vp, i32, vp, vp, u64p]),
            "df_root_batch": (i32, [vp, vp, vp, u64p]),
            "df_finish": (i32, [vp, ctypes.c_uint64, i32, vp, vp, vp]),
            "df_summary": (i32, [vp, vp]),
            "df_root_stats": (i32, [vp, i32, vp, vp, i32, vp]),
            "df_probe": (i32, [ctypes.c_char_p, vp, vp, vp, i32, vp]),
        }
        for name, (res, args) in sigs.items():
            fn = getattr(lib, name)
            fn.restype, fn.argtypes = res, args
        _lib = lib
    except (OSError, AttributeError) as e:
        _load_error = e
    return _lib


def available() -> bool:
    return _load() is not None


def load_error():
    _load()
    return _load_error


def make_search(evaluator, params: SearchParams | None = None):
    """The search used by the engine: Rust if available, else the pure-Python MCTS."""
    if os.environ.get("DRAWFISH_SEARCH", "").lower() != "python" and available():
        return RustMCTS(evaluator, params)
    return MCTS(evaluator, params)


def _decode(code: int) -> chess.Move:
    promo = (code >> 12) & 7
    return chess.Move(code & 63, (code >> 6) & 63, promo or None)


class RustMCTS:
    """Drop-in replacement for search.MCTS (same search(), stats() and reset())."""
    raw_priors = False  # test hook, see core/src/search.rs

    def __init__(self, evaluator, params: SearchParams | None = None):
        lib = _load()
        if lib is None:
            raise RuntimeError(f"drawfish_core unavailable: {_load_error}")
        self.lib = lib
        self.ev = evaluator
        self.p = params or SearchParams()
        self._ptr = lib.df_new(self._params())
        if not self._ptr:
            raise RuntimeError("drawfish_core: df_new failed")
        self._py = None  # pure-Python fallback, created on first use
        self._bid = ctypes.c_uint64()
        self._sum = np.zeros(7, dtype=np.int64)
        self._bufs = [(np.zeros((max(self.p.batch, 1), N_FEAT), dtype=np.uint64),
                       np.zeros(max(self.p.batch, 1), dtype=np.float32)) for _ in range(2)]
        self._turn = 0

    def __del__(self):
        ptr, self._ptr = getattr(self, "_ptr", None), None
        if ptr:
            self.lib.df_free(ptr)

    def _params(self):
        p = self.p
        vals = (1.0 if p.mode == "draw" else 0.0, p.cpuct, p.fpu_red, p.win_value, p.gamma,
                p.draw_prior, p.policy_temp, float(p.cache_size), 1.0 if self.raw_priors else 0.0)
        return (ctypes.c_double * 9)(*vals)

    def _fallback(self):
        if self._py is None:
            self._py = MCTS(self.ev, self.p)
        return self._py

    def reset(self):
        self.lib.df_reset(self._ptr)
        if self._py is not None:
            self._py.reset()

    # ------------------------------------------------------------------ core calls
    def _summary(self):
        if self.lib.df_summary(self._ptr, self._sum.ctypes.data) != 0:
            raise RuntimeError("drawfish_core: df_summary failed")
        return self._sum.tolist()

    def _select(self, want):
        self._turn ^= 1
        feats, sel = self._bufs[self._turn]
        if want > len(sel):
            feats = np.zeros((want, N_FEAT), dtype=np.uint64)
            sel = np.zeros(want, dtype=np.float32)
            self._bufs[self._turn] = (feats, sel)
        n = self.lib.df_select(self._ptr, want, feats.ctypes.data, sel.ctypes.data,
                               ctypes.byref(self._bid))
        if n < 0:
            raise RuntimeError("drawfish_core: df_select failed")
        return n, self._bid.value, feats, sel

    def _finish(self, bid, n, handle):
        wdl, adv, logits = self.ev.wait(handle)
        wdl = np.ascontiguousarray(wdl, dtype=np.float32)
        adv = np.ascontiguousarray(adv, dtype=np.float32)
        logits = np.ascontiguousarray(logits, dtype=np.float32)
        if self.lib.df_finish(self._ptr, bid, n, wdl.ctypes.data, adv.ctypes.data, logits.ctypes.data) != n:
            raise RuntimeError("drawfish_core: df_finish failed")

    def _prepare_root(self, game):
        err = ctypes.create_string_buffer(256)
        moves = " ".join(m.uci() for m in game.board.move_stack).encode()
        r = self.lib.df_prepare(self._ptr, self._params(), game.start_fen.encode(), moves,
                                1 if game.board.turn == chess.WHITE else 0, err, len(err))
        if r == 1:  # root needs an evaluation
            feats, sel = self._bufs[0]
            self.lib.df_root_batch(self._ptr, feats.ctypes.data, sel.ctypes.data, ctypes.byref(self._bid))
            self._finish(self._bid.value, 1, self.ev.launch(feats[:1].copy(), sel[:1].copy()))
        return r

    # ------------------------------------------------------------------ main loop
    def search(self, game, nodes=None, movetime=None, stop_event=None, info_cb=None,
               info_every=0.5):
        """Search the position of `game` (Drawfish = side to move). Returns a stats dict."""
        t0 = time.perf_counter()
        r = self._prepare_root(game)
        if r < 0:
            return self._fallback().search(game, nodes, movetime, stop_event, info_cb, info_every)
        if r == 0:
            return None
        s = self._summary()
        if s[1] == 1 and (nodes is None or nodes > 1):
            nodes = min(nodes or 10**9, 64)
        start_n = s[0]
        deadline = t0 + movetime if movetime else None
        next_info = t0 + info_every
        evals = 0
        inflight = None  # (batch id, n, handle): a batch the GPU is working on
        while True:
            s = self._summary()
            done = s[0] - start_n
            pending = inflight[1] if inflight else 0
            now = time.perf_counter()
            stop = ((nodes is not None and done + pending >= nodes)
                    or (deadline is not None and now >= deadline)
                    or (stop_event is not None and stop_event.is_set())
                    or self._can_stop_early(s, nodes, done, deadline, now, t0))
            batch = None
            if not stop:
                want = self.p.batch
                if nodes is not None:
                    want = max(1, min(want, nodes - done - pending))
                n, bid, feats, sel = self._select(want)
                if n:
                    batch = (bid, n, self.ev.launch(feats[:n], sel[:n]))
            if inflight is not None:
                self._finish(*inflight)
                evals += inflight[1]
            inflight = batch
            if stop and inflight is None:
                break
            if info_cb is not None and time.perf_counter() >= next_info:
                next_info = time.perf_counter() + info_every
                info_cb(self.stats(time.perf_counter() - t0, start_n))
        st = self.stats(time.perf_counter() - t0, start_n)
        st["evals"] = evals
        st["searched"] = self._summary()[0] - start_n
        return st

    @staticmethod
    def _can_stop_early(s, nodes, done, deadline, now, t0):
        """Stop when the most-visited root move can no longer be overtaken."""
        if done < 64 or s[1] < 2 or s[2] < 1:
            return False
        if deadline is not None:
            remaining = done / max(now - t0, 1e-3) * (deadline - now)
        elif nodes is not None:
            remaining = nodes - done
        else:
            return False
        return s[3] - s[4] > remaining

    def stats(self, elapsed=0.0, start_n=0):
        k, max_pv = 8, 20
        codes = np.zeros(k, dtype=np.uint32)
        vals = np.zeros((k, 6), dtype=np.float64)
        pvbuf = np.zeros(max_pv, dtype=np.uint32)
        r = self.lib.df_root_stats(self._ptr, k, codes.ctypes.data, vals.ctypes.data, max_pv,
                                   pvbuf.ctypes.data)
        if r < 0:
            raise RuntimeError("drawfish_core: df_root_stats failed")
        nm, npv = r & 0xFFFF, r >> 16
        moves = [{"move": _decode(int(codes[j])), "n": int(vals[j, 0]), "w": vals[j, 1], "d": vals[j, 2],
                  "l": vals[j, 3], "adv": vals[j, 4], "p": vals[j, 5]} for j in range(nm)]
        best = moves[0] if moves else None
        tree = self._summary()[0]
        return {"best": best["move"] if best else None,
                "wdl": (best["w"], best["d"], best["l"]) if best else None,
                "adv": best["adv"] if best else 0.0, "nodes": tree - start_n, "tree": tree,
                "time": elapsed, "nps": (tree - start_n) / elapsed if elapsed > 0 else 0.0,
                "pv": [_decode(int(c)) for c in pvbuf[:npv]], "moves": moves}


def probe(fen: str, max_moves=256):
    """Test helper: the Rust side's view of a position (see df_probe)."""
    lib = _load()
    feats = np.zeros(N_FEAT, dtype=np.uint64)
    codes = np.zeros(max_moves, dtype=np.uint32)
    idx = np.zeros(max_moves, dtype=np.uint32)
    info = np.zeros(4, dtype=np.int32)
    n = lib.df_probe(fen.encode(), feats.ctypes.data, codes.ctypes.data, idx.ctypes.data, max_moves,
                     info.ctypes.data)
    if n < 0:
        return None
    return {"feats": feats, "moves": [_decode(int(c)) for c in codes[:n]], "idx": idx[:n].tolist(),
            "white": bool(info[0]), "check": bool(info[1]), "insufficient": bool(info[2]),
            "ep": None if info[3] < 0 else int(info[3])}
