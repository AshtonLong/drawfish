"""Draw-seeking Monte-Carlo tree search (PUCT) with vector-valued, non-zero-sum backups.

Chess is zero-sum, but Drawfish's goal is not: it wants a DRAW, while its opponent is
assumed to want to WIN. Every node therefore accumulates an outcome vector from its
side-to-move's perspective and each side selects moves with its own utility (a 2-player
"max^n" search):

  Opponent to move:  u = W + D / 2
      a realistic opponent maximizing its own expected score.
  Drawfish to move:  u = D + kappa * Wc + gamma * (1 - max(0, -adv))
      Wc is Drawfish's winning chance in positions that are NOT yet decided. Against an
      opponent that maximizes its score, being ahead is the best place to draw from: the
      opponent, now losing, wants the draw too, so the advantage can be cashed in by
      repetition or simplification. Such chances are therefore worth almost as much as a
      draw (kappa = 0.97: just enough that keeping a big advantage beats handing it back
      for Stockfish-style "equal = 90% drawn"), while an actual checkmate delivered by
      Drawfish is a failure (worth 0). Benchmarks showed why this matters: maximizing Stockfish's draw
      probability alone made Drawfish hug dead equality, where its own occasional
      blunders lost far more games than an ordinary engine on the same network.
      The small one-sided balance term only pushes back when Drawfish is behind.

Move priors come from the best-move policy head at every node by default (draw_prior=0):
the draw-policy head, distilled from shallow Stockfish searches, proposed tactically weaker
moves and lowered the draw rate in matches (81% -> 100% vs Stockfish-1600 when switched).

The vector is (W, D, L, adv, Wc, Lc): Wc/Lc equal W/L for network evaluations and are 0
for checkmates, so "could win" and "has won" are told apart.

Draws are scored exactly by the rules: stalemate, insufficient material, and repetition
(a position repeated inside the search tree counts as a draw, as does a true third
occurrence against game history). Move counts never draw (no fifty-move rule).

In mode "win" both sides use expected score and the best-move policy: an ordinary
engine built on the same network, used as a benchmark baseline.
"""
import math
import time
from collections import OrderedDict
from dataclasses import dataclass

import numpy as np

from .encoding import board_features, move_index
from .rules import rep_key

PENDING, EXPANDED, TERMINAL, NEW = 0, 1, 2, 3
MATED = (0.0, 0.0, 1.0, -1.0, 0.0, 0.0)
DRAWN = (0.0, 1.0, 0.0, 0.0, 0.0, 0.0)


@dataclass
class SearchParams:
    mode: str = "draw"          # "draw" (Drawfish) or "win" (ordinary-engine baseline)
    cpuct: float = 1.2
    fpu_red: float = 0.2
    win_value: float = 0.97     # kappa: worth of an undecided winning chance to Drawfish
    gamma: float = 0.1          # one-sided balance term (only when Drawfish is behind)
    draw_prior: float = 0.0     # prior at Drawfish's nodes: 1 = draw head, 0 = best-move head
    policy_temp: float = 1.0
    batch: int = 32
    cache_size: int = 400_000


class Node:
    """Search node. Children are sorted by prior; per-child statistics are plain lists
    (W/D/L/adv sums from this node's side-to-move perspective) and Q holds each visited
    child's utility for the player choosing here. Unvisited children all share the same
    first-play-urgency value, so the next one worth trying is always index `nexp`."""
    __slots__ = ("board", "key", "parent", "moves", "P", "cP", "N", "VL", "SW", "SD", "SL",
                 "SA", "Q", "children", "nexp", "ntot", "pvis", "state", "tv", "drawer",
                 "n", "vw", "vd", "vl", "va", "SWc", "SLc", "vwc", "vlc", "u0")

    def __init__(self, board, parent):
        self.board = board
        self.parent = parent
        self.key = rep_key(board)
        self.n = 0
        self.vw = self.vd = self.vl = self.va = self.vwc = self.vlc = 0.0
        self.children = None


class MCTS:
    def __init__(self, evaluator, params: SearchParams | None = None):
        self.ev = evaluator
        self.p = params or SearchParams()
        self.root = None
        self.root_stack = []
        self.start_fen = None
        self.engine_color = None
        self.hist = {}
        self.cache = OrderedDict()

    # ------------------------------------------------------------------ tree basics
    def _make_node(self, board, parent):
        node = Node(board, parent)
        node.drawer = self.p.mode == "draw" and board.turn == self.engine_color
        moves = list(board.generate_legal_moves())
        if not moves:
            node.state, node.tv = TERMINAL, (MATED if board.is_check() else DRAWN)
        elif board.is_insufficient_material() or self._repeats(node):
            node.state, node.tv = TERMINAL, DRAWN
        else:
            node.state, node.moves = PENDING, moves
        return node

    def _repeats(self, node):
        if node.parent is None:
            return False  # the root is the current game position, already counted in hist
        k = node.key
        p = node.parent
        while p is not None and p is not self.root:
            if p.key == k:
                return True  # repeated inside the search tree -> treated as a draw
            p = p.parent
        return self.hist.get(k, 0) >= 2  # third occurrence counting game history

    def _expand(self, node, wdl, adv, priors):
        order = np.argsort(-priors, kind="stable")
        p = priors[order]
        moves = node.moves
        node.moves = [moves[i] for i in order.tolist()]
        n = len(p)
        node.P = p.tolist()
        node.cP = (p * self.p.cpuct).tolist()
        node.N = [0] * n
        node.VL = [0] * n
        node.SW = [0.0] * n
        node.SD = [0.0] * n
        node.SL = [0.0] * n
        node.SA = [0.0] * n
        node.SWc = [0.0] * n
        node.SLc = [0.0] * n
        node.Q = [0.0] * n
        node.children = [None] * n
        node.nexp = 0
        node.ntot = 0
        node.pvis = 0.0
        node.state = EXPANDED
        w, d, l, a = float(wdl[0]), float(wdl[1]), float(wdl[2]), float(adv)
        if node.drawer:
            node.u0 = d + self.p.win_value * w + self.p.gamma * (1.0 - max(0.0, -a))
        else:
            node.u0 = w + 0.5 * d
        return (w, d, l, a, w, l)

    def _priors(self, node, logits):
        turn = node.board.turn
        lg = logits[[move_index(m, turn) for m in node.moves]] / self.p.policy_temp
        e = np.exp(lg - lg.max())
        return e / e.sum()

    def _pick(self, node):
        N, VL, Q, cP = node.N, node.VL, node.Q, node.cP
        sq = math.sqrt(node.ntot) if node.ntot > 0 else 1.0
        best, bs = -1, -1e18
        for i in range(node.nexp):
            n = N[i]
            ne = n + VL[i]
            s = (Q[i] * n / ne if n else 0.0) + cP[i] * sq / (1 + ne)
            if s > bs:
                bs, best = s, i
        k = node.nexp
        if k < len(cP):
            # First-play urgency from the node's own network evaluation (not the running
            # average): with a non-zero-sum utility a high-prior move can be terrible for the
            # player choosing (e.g. a checkmate Drawfish does not want to give); a running
            # average dragged down by it would hide every alternative.
            fpu = node.u0 - self.p.fpu_red * math.sqrt(node.pvis)
            if fpu + cP[k] * sq > bs:
                best = k
        return best

    def _backprop(self, path, leaf, v):
        w, d, l, a, wc, lc = v
        leaf.n += 1
        leaf.vw += w
        leaf.vd += d
        leaf.vl += l
        leaf.va += a
        leaf.vwc += wc
        leaf.vlc += lc
        g, kappa = self.p.gamma, self.p.win_value
        for node, i in reversed(path):
            # flip to the perspective of the player choosing at node
            w, l, a, wc, lc = l, w, -a, lc, wc
            n = node.N[i] + 1
            node.N[i] = n
            node.VL[i] -= 1
            sw = node.SW[i] + w
            sd = node.SD[i] + d
            sa = node.SA[i] + a
            swc = node.SWc[i] + wc
            node.SW[i] = sw
            node.SD[i] = sd
            node.SL[i] += l
            node.SA[i] = sa
            node.SWc[i] = swc
            node.SLc[i] += lc
            if n == 1:
                node.pvis += node.P[i]
            if node.drawer:
                node.Q[i] = (sd + kappa * swc + g * (n - max(0.0, -sa))) / n
            else:
                node.Q[i] = (sw + 0.5 * sd) / n
            node.n += 1
            node.vw += w
            node.vd += d
            node.vl += l
            node.va += a
            node.vwc += wc
            node.vlc += lc

    def _select(self):
        node, path = self.root, []
        while True:
            if node.state == TERMINAL:
                return path, node, TERMINAL
            if node.state == PENDING:
                return path, node, PENDING  # collision with a leaf already in flight
            i = self._pick(node)
            path.append((node, i))
            node.VL[i] += 1
            node.ntot += 1
            if i == node.nexp:
                node.nexp += 1
            child = node.children[i]
            if child is None:
                b = node.board.copy(stack=False)
                b.push(node.moves[i])
                child = self._make_node(b, node)
                node.children[i] = child
                return path, child, (TERMINAL if child.state == TERMINAL else NEW)
            node = child

    def _undo_virtual(self, path):
        for node, i in path:
            node.VL[i] -= 1
            node.ntot -= 1

    # ------------------------------------------------------------------ evaluation
    def _cache_get(self, node):
        key = (node.key, node.drawer)
        hit = self.cache.get(key)
        if hit is not None:
            self.cache.move_to_end(key)
        return hit

    def _cache_put(self, node, entry):
        self.cache[(node.key, node.drawer)] = entry
        if len(self.cache) > self.p.cache_size:
            self.cache.popitem(last=False)

    def _launch(self, leaves):
        feats = np.empty((len(leaves), 14), dtype=np.uint64)
        sel = np.empty(len(leaves), dtype=np.float32)
        dp = self.p.draw_prior
        for j, (_, node) in enumerate(leaves):
            feats[j] = board_features(node.board)
            sel[j] = dp if node.drawer else 0.0
        return self.ev.launch(feats, sel)

    def _finish(self, leaves, handle):
        wdl, adv, logits = self.ev.wait(handle)
        for j, (path, node) in enumerate(leaves):
            entry = (wdl[j], float(adv[j]), self._priors(node, logits[j]))
            self._cache_put(node, entry)
            self._backprop(path, node, self._expand(node, *entry))

    def _evaluate(self, leaves):
        self._finish(leaves, self._launch(leaves))

    # ------------------------------------------------------------------ root handling
    def _prepare_root(self, game, engine_color):
        if engine_color != self.engine_color:
            self.root = None
            self.cache.clear()
        self.engine_color = engine_color
        self.hist = dict(game.reps)
        stack = game.board.move_stack
        node = None
        old = self.root_stack
        if (self.root is not None and game.start_fen == self.start_fen
                and len(stack) >= len(old) and stack[:len(old)] == old):
            node = self.root
            for m in stack[len(old):]:
                if node.state != EXPANDED or m not in node.moves:
                    node = None
                    break
                node = node.children[node.moves.index(m)]
                if node is None:
                    break
        self.root = None
        if node is not None and node.state != TERMINAL:
            node.parent = None
            self.root = node
        else:
            self.root = self._make_node(game.board.copy(stack=False), None)
        self.root_stack = list(stack)
        self.start_fen = game.start_fen
        if self.root.state == PENDING:
            self._evaluate([([], self.root)])

    def reset(self):
        self.root = None
        self.root_stack = []
        self.cache.clear()

    # ------------------------------------------------------------------ main loop
    def search(self, game, nodes=None, movetime=None, stop_event=None, info_cb=None,
               info_every=0.5):
        """Search the position of `game` (Drawfish = side to move). Returns a stats dict."""
        t0 = time.perf_counter()
        self._prepare_root(game, game.board.turn)
        root = self.root
        if root.state == TERMINAL:
            return None
        if len(root.moves) == 1 and (nodes is None or nodes > 1):
            nodes = min(nodes or 10**9, 64)
        start_n = root.n
        deadline = t0 + movetime if movetime else None
        next_info = t0 + info_every
        evals = 0
        inflight = None  # (leaves, handle): a batch the GPU is working on
        while True:
            done = root.n - start_n
            pending = len(inflight[0]) if inflight else 0
            stop = ((nodes is not None and done + pending >= nodes)
                    or (deadline is not None and time.perf_counter() >= deadline)
                    or (stop_event is not None and stop_event.is_set())
                    or self._can_stop_early(nodes, done, deadline, time.perf_counter(), t0))
            leaves = []
            if not stop:
                want = self.p.batch
                if nodes is not None:
                    want = max(1, min(want, nodes - done - pending))
                for _ in range(want):
                    path, leaf, kind = self._select()
                    if kind == TERMINAL:
                        self._backprop(path, leaf, leaf.tv)
                        continue
                    if kind == PENDING:
                        self._undo_virtual(path)
                        break
                    hit = self._cache_get(leaf)
                    if hit is not None:
                        self._backprop(path, leaf, self._expand(leaf, *hit))
                    else:
                        leaves.append((path, leaf))
            handle = self._launch(leaves) if leaves else None
            if inflight is not None:
                self._finish(*inflight)
                evals += len(inflight[0])
            inflight = (leaves, handle) if leaves else None
            if stop and inflight is None:
                break
            if info_cb is not None and time.perf_counter() >= next_info:
                next_info = time.perf_counter() + info_every
                info_cb(self.stats(time.perf_counter() - t0, start_n))
        st = self.stats(time.perf_counter() - t0, start_n)
        st["evals"] = evals
        st["searched"] = root.n - start_n
        return st

    def _can_stop_early(self, nodes, done, deadline, now, t0):
        """Stop when the most-visited root move can no longer be overtaken."""
        root = self.root
        if done < 64 or len(root.moves) < 2 or root.nexp < 1:
            return False
        if deadline is not None:
            remaining = done / max(now - t0, 1e-3) * (deadline - now)
        elif nodes is not None:
            remaining = nodes - done
        else:
            return False
        a, b = sorted(root.N[:root.nexp + 1])[-2:] if root.nexp + 1 >= 2 else (0, root.N[0])
        return b - a > remaining

    def stats(self, elapsed=0.0, start_n=0):
        root = self.root
        order = sorted(range(root.nexp), key=lambda i: -root.N[i])
        moves = []
        for i in order[:8]:
            n = root.N[i]
            if n == 0:
                break
            moves.append({"move": root.moves[i], "n": n, "w": root.SW[i] / n, "d": root.SD[i] / n,
                          "l": root.SL[i] / n, "adv": root.SA[i] / n, "p": root.P[i]})
        best = moves[0] if moves else None
        pv, node = [], root
        while node is not None and node.state == EXPANDED and node.nexp > 0 and len(pv) < 20:
            i = max(range(node.nexp), key=node.N.__getitem__)
            if node.N[i] == 0:
                break
            pv.append(node.moves[i])
            node = node.children[i]
        return {"best": best["move"] if best else None,
                "wdl": (best["w"], best["d"], best["l"]) if best else None,
                "adv": best["adv"] if best else 0.0, "nodes": root.n - start_n, "tree": root.n,
                "time": elapsed, "nps": (root.n - start_n) / elapsed if elapsed > 0 else 0.0,
                "pv": pv, "moves": moves}
