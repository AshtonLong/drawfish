//! Draw-seeking PUCT search: a line-by-line port of drawfish/search.py (see its docstring for
//! the method). Arithmetic is kept in the same order and precision as the Python version, so
//! both build identical trees from identical network outputs.
//!
//! The network itself stays in Python (PyTorch + CUDA graphs). The Python driver calls
//! `select` to collect a batch of leaves (their features are written straight into numpy
//! buffers), evaluates them, and hands the outputs back through `finish`.

use std::collections::HashMap;
use std::num::NonZeroUsize;

use lru::LruCache;
use shakmaty::fen::Fen;
use shakmaty::uci::UciMove;
use shakmaty::{CastlingMode, Chess, Color, Move, Position, PositionError};

use crate::enc::{self, Key, N_FEAT, NUM_MOVES};

const NONE: u32 = u32::MAX;
const PENDING: u8 = 0;
const EXPANDED: u8 = 1;
const TERMINAL: u8 = 2;
const MATED: [f64; 6] = [0.0, 0.0, 1.0, -1.0, 0.0, 0.0];
const DRAWN: [f64; 6] = [0.0, 1.0, 0.0, 0.0, 0.0, 0.0];

#[derive(Clone, Copy, Debug)]
pub struct Params {
    pub draw_mode: bool,
    pub cpuct: f64,
    pub fpu_red: f64,
    pub win_value: f64,
    pub gamma: f64,
    pub draw_prior: f32,
    pub policy_temp: f32,
    pub cache_size: usize,
    /// Test hook: use the "logits" buffer as unnormalized priors (no softmax), so tests can
    /// compare trees with the Python search without depending on exp() rounding.
    pub raw_priors: bool,
}

/// Statistics of one visited child, from the perspective of the player choosing at the parent.
#[derive(Clone, Copy)]
struct Edge {
    n: u32,
    vl: i32,
    sw: f64,
    sd: f64,
    sl: f64,
    sa: f64,
    swc: f64,
    slc: f64,
    q: f64,
    child: u32,
}

impl Edge {
    const NEW: Edge = Edge { n: 0, vl: 0, sw: 0.0, sd: 0.0, sl: 0.0, sa: 0.0, swc: 0.0,
                             slc: 0.0, q: 0.0, child: NONE };
}

/// Moves are sorted by prior once expanded. Unvisited moves all share the same first-play
/// urgency, so only the first `nexp` moves have statistics (`edges.len() == nexp`).
struct Node {
    pos: Chess,
    key: Key,
    parent: u32,
    state: u8,
    mated: bool,
    drawer: bool,
    moves: Vec<Move>,
    p: Vec<f32>,
    edges: Vec<Edge>,
    ntot: u32,
    pvis: f64,
    u0: f64,
    n: u32,
    v: [f64; 6], // value sums: W, D, L, adv, Wc, Lc
}

#[derive(Clone)]
struct Eval {
    wdl: [f32; 3],
    adv: f32,
    priors: Vec<f32>, // in legal-move generation order
}

type Path = Vec<(u32, u32)>;

pub struct Summary {
    pub root_n: u32,
    pub nmoves: u32,
    pub nexp: u32,
    pub top1: u32,
    pub top2: u32,
    pub state: u8,
}

pub struct MoveStat {
    pub code: u32,
    pub n: u32,
    pub w: f64,
    pub d: f64,
    pub l: f64,
    pub adv: f64,
    pub p: f64,
}

pub enum Prepared {
    Terminal,
    NeedsEval,
    Ready,
}

pub struct Search {
    pub p: Params,
    nodes: Vec<Node>,
    root: u32,
    root_moves: Vec<String>,
    start_fen: String,
    engine_color: Option<Color>,
    hist: HashMap<Key, u32>,
    cache: LruCache<(Key, bool), Eval>,
    batches: HashMap<u64, Vec<(Path, u32)>>,
    next_batch: u64,
}

fn parse_position(fen: &str) -> Result<Chess, String> {
    let fen = Fen::from_ascii(fen.trim().as_bytes()).map_err(|e| e.to_string())?;
    fen.into_position::<Chess>(CastlingMode::Standard)
        .or_else(PositionError::ignore_invalid_castling_rights)
        .or_else(PositionError::ignore_invalid_ep_square)
        .or_else(PositionError::ignore_too_much_material)
        .or_else(PositionError::ignore_impossible_check)
        .map_err(|e| e.to_string())
}

impl Search {
    pub fn new(p: Params) -> Search {
        Search {
            p,
            nodes: Vec::new(),
            root: NONE,
            root_moves: Vec::new(),
            start_fen: String::new(),
            engine_color: None,
            hist: HashMap::new(),
            cache: LruCache::new(NonZeroUsize::new(p.cache_size.max(1)).unwrap()),
            batches: HashMap::new(),
            next_batch: 0,
        }
    }

    pub fn set_params(&mut self, p: Params) {
        self.p = p;
        if let Some(cap) = NonZeroUsize::new(p.cache_size.max(1)) {
            if cap != self.cache.cap() {
                self.cache.resize(cap);
            }
        }
    }

    pub fn reset(&mut self) {
        self.nodes.clear();
        self.root = NONE;
        self.root_moves.clear();
        self.batches.clear();
        self.cache.clear();
    }

    // ------------------------------------------------------------------ tree basics
    fn make_node(&mut self, pos: Chess, parent: u32) -> u32 {
        let key = enc::key(&pos);
        let drawer = self.p.draw_mode && Some(pos.turn()) == self.engine_color;
        let moves: Vec<Move> = pos.legal_moves().into_iter().collect();
        let mut node = Node {
            pos, key, parent, state: PENDING, mated: false, drawer, moves: Vec::new(),
            p: Vec::new(), edges: Vec::new(), ntot: 0, pvis: 0.0, u0: 0.0, n: 0, v: [0.0; 6],
        };
        if moves.is_empty() {
            node.state = TERMINAL;
            node.mated = node.pos.is_check();
        } else if enc::insufficient_material(&node.pos) || self.repeats(&node) {
            node.state = TERMINAL;
        } else {
            node.moves = moves;
        }
        self.nodes.push(node);
        (self.nodes.len() - 1) as u32
    }

    fn repeats(&self, node: &Node) -> bool {
        if node.parent == NONE {
            return false; // the root is the current game position, already counted in hist
        }
        let mut p = node.parent;
        while p != NONE && p != self.root {
            let pn = &self.nodes[p as usize];
            if pn.key == node.key {
                return true; // repeated inside the search tree -> treated as a draw
            }
            p = pn.parent;
        }
        self.hist.get(&node.key).copied().unwrap_or(0) >= 2
    }

    fn expand(&mut self, idx: u32, ev: &Eval) -> [f64; 6] {
        let p = self.p;
        let node = &mut self.nodes[idx as usize];
        // np.argsort(-priors, kind="stable"): descending prior, ties keep generation order
        let mut order: Vec<usize> = (0..ev.priors.len()).collect();
        order.sort_by(|&a, &b| ev.priors[b].partial_cmp(&ev.priors[a]).unwrap_or(std::cmp::Ordering::Equal));
        node.moves = order.iter().map(|&i| node.moves[i]).collect();
        node.p = order.iter().map(|&i| ev.priors[i]).collect();
        node.edges = Vec::new();
        node.ntot = 0;
        node.pvis = 0.0;
        node.state = EXPANDED;
        let (w, d, l, a) = (ev.wdl[0] as f64, ev.wdl[1] as f64, ev.wdl[2] as f64, ev.adv as f64);
        node.u0 = if node.drawer {
            d + p.win_value * w + p.gamma * (1.0 - f64::max(0.0, -a))
        } else {
            w + 0.5 * d
        };
        [w, d, l, a, w, l]
    }

    fn priors(&self, idx: u32, logits: &[f32]) -> Vec<f32> {
        let node = &self.nodes[idx as usize];
        let white = node.pos.turn() == Color::White;
        let lg: Vec<f32> = node.moves.iter().map(|m| logits[enc::move_index(m, white)]).collect();
        if self.p.raw_priors {
            return lg;
        }
        let t = self.p.policy_temp;
        let lg: Vec<f32> = lg.iter().map(|x| x / t).collect();
        let mx = lg.iter().copied().fold(f32::NEG_INFINITY, f32::max);
        let e: Vec<f32> = lg.iter().map(|x| (x - mx).exp()).collect();
        let s = numpy_sum(&e);
        e.iter().map(|x| x / s).collect()
    }

    #[inline]
    fn cp(&self, node: &Node, i: usize) -> f64 {
        (node.p[i] * self.p.cpuct as f32) as f64 // numpy: float32 priors * cpuct -> float32
    }

    fn pick(&self, idx: u32) -> usize {
        let node = &self.nodes[idx as usize];
        let sq = if node.ntot > 0 { (node.ntot as f64).sqrt() } else { 1.0 };
        let mut best = usize::MAX;
        let mut bs = -1e18;
        for (i, e) in node.edges.iter().enumerate() {
            let n = e.n as f64;
            let ne = (e.n as i64 + e.vl as i64) as f64;
            let s = (if e.n != 0 { e.q * n / ne } else { 0.0 }) + self.cp(node, i) * sq / (1.0 + ne);
            if s > bs {
                bs = s;
                best = i;
            }
        }
        let k = node.edges.len();
        if k < node.moves.len() {
            // first-play urgency from the node's own evaluation (see search.py)
            let fpu = node.u0 - self.p.fpu_red * node.pvis.sqrt();
            if fpu + self.cp(node, k) * sq > bs {
                best = k;
            }
        }
        best
    }

    fn backprop(&mut self, path: &[(u32, u32)], leaf: u32, v: [f64; 6]) {
        let [mut w, d, mut l, mut a, mut wc, mut lc] = v;
        {
            let lf = &mut self.nodes[leaf as usize];
            lf.n += 1;
            for (s, x) in lf.v.iter_mut().zip(v) {
                *s += x;
            }
        }
        let (g, kappa) = (self.p.gamma, self.p.win_value);
        for &(ni, i) in path.iter().rev() {
            // flip to the perspective of the player choosing at node
            (w, l, a, wc, lc) = (l, w, -a, lc, wc);
            let node = &mut self.nodes[ni as usize];
            let pi = node.p[i as usize] as f64;
            let drawer = node.drawer;
            let e = &mut node.edges[i as usize];
            let n = e.n + 1;
            e.n = n;
            e.vl -= 1;
            let sw = e.sw + w;
            let sd = e.sd + d;
            let sa = e.sa + a;
            let swc = e.swc + wc;
            e.sw = sw;
            e.sd = sd;
            e.sl += l;
            e.sa = sa;
            e.swc = swc;
            e.slc += lc;
            let nf = n as f64;
            e.q = if drawer {
                (sd + kappa * swc + g * (nf - f64::max(0.0, -sa))) / nf
            } else {
                (sw + 0.5 * sd) / nf
            };
            if n == 1 {
                node.pvis += pi;
            }
            node.n += 1;
            for (s, x) in node.v.iter_mut().zip([w, d, l, a, wc, lc]) {
                *s += x;
            }
        }
    }

    /// Descend to a leaf. Returns (path, leaf, kind) with kind PENDING (collision with a leaf
    /// already in flight), TERMINAL, or EXPANDED meaning "new leaf, needs an evaluation".
    fn select(&mut self) -> (Path, u32, u8) {
        let mut idx = self.root;
        let mut path = Vec::new();
        loop {
            let node = &self.nodes[idx as usize];
            if node.state == TERMINAL || node.state == PENDING {
                return (path, idx, node.state);
            }
            let i = self.pick(idx);
            path.push((idx, i as u32));
            let node = &mut self.nodes[idx as usize];
            if i == node.edges.len() {
                node.edges.push(Edge::NEW);
            }
            node.edges[i].vl += 1;
            node.ntot += 1;
            let child = node.edges[i].child;
            if child == NONE {
                let mut pos = node.pos.clone();
                pos.play_unchecked(node.moves[i]);
                let c = self.make_node(pos, idx);
                self.nodes[idx as usize].edges[i].child = c;
                let kind = if self.nodes[c as usize].state == TERMINAL { TERMINAL } else { EXPANDED };
                return (path, c, kind);
            }
            idx = child;
        }
    }

    fn undo_virtual(&mut self, path: &[(u32, u32)]) {
        for &(ni, i) in path {
            let node = &mut self.nodes[ni as usize];
            node.edges[i as usize].vl -= 1;
            node.ntot -= 1;
        }
    }

    fn terminal_value(&self, idx: u32) -> [f64; 6] {
        if self.nodes[idx as usize].mated { MATED } else { DRAWN }
    }

    fn write_leaf(&self, idx: u32, feats: &mut [u64], sel: &mut f32) {
        let node = &self.nodes[idx as usize];
        feats.copy_from_slice(&enc::features(&node.pos, &node.key));
        *sel = if node.drawer { self.p.draw_prior } else { 0.0 };
    }

    // ------------------------------------------------------------------ batches
    fn new_batch(&mut self, leaves: Vec<(Path, u32)>) -> u64 {
        let id = self.next_batch;
        self.next_batch += 1;
        self.batches.insert(id, leaves);
        id
    }

    /// Collect up to `want` leaves needing a network evaluation; terminal leaves and cache
    /// hits are backed up immediately. Returns (number of leaves written, batch id).
    pub fn select_batch(&mut self, want: usize, feats: &mut [u64], sel: &mut [f32]) -> (usize, u64) {
        let mut leaves = Vec::new();
        for _ in 0..want {
            let (path, leaf, kind) = self.select();
            if kind == TERMINAL {
                let v = self.terminal_value(leaf);
                self.backprop(&path, leaf, v);
                continue;
            }
            if kind == PENDING {
                self.undo_virtual(&path);
                break;
            }
            let ck = (self.nodes[leaf as usize].key, self.nodes[leaf as usize].drawer);
            if let Some(hit) = self.cache.get(&ck).cloned() {
                let v = self.expand(leaf, &hit);
                self.backprop(&path, leaf, v);
            } else {
                let j = leaves.len();
                self.write_leaf(leaf, &mut feats[j * N_FEAT..(j + 1) * N_FEAT], &mut sel[j]);
                leaves.push((path, leaf));
            }
        }
        let n = leaves.len();
        if n == 0 {
            return (0, u64::MAX);
        }
        (n, self.new_batch(leaves))
    }

    /// The root as a one-leaf batch (it is evaluated without consulting the cache).
    pub fn root_batch(&mut self, feats: &mut [u64], sel: &mut [f32]) -> u64 {
        let root = self.root;
        self.write_leaf(root, &mut feats[..N_FEAT], &mut sel[0]);
        self.new_batch(vec![(Vec::new(), root)])
    }

    pub fn finish_batch(&mut self, id: u64, wdl: &[f32], adv: &[f32], logits: &[f32]) -> Result<usize, String> {
        let leaves = self.batches.remove(&id).ok_or("unknown batch")?;
        if wdl.len() < 3 * leaves.len() || adv.len() < leaves.len() || logits.len() < NUM_MOVES * leaves.len() {
            return Err("output buffers too small".into());
        }
        for (j, (path, leaf)) in leaves.iter().enumerate() {
            let ev = Eval {
                wdl: [wdl[3 * j], wdl[3 * j + 1], wdl[3 * j + 2]],
                adv: adv[j],
                priors: self.priors(*leaf, &logits[j * NUM_MOVES..(j + 1) * NUM_MOVES]),
            };
            let node = &self.nodes[*leaf as usize];
            self.cache.put((node.key, node.drawer), ev.clone());
            let v = self.expand(*leaf, &ev);
            self.backprop(path, *leaf, v);
        }
        Ok(leaves.len())
    }

    // ------------------------------------------------------------------ root handling
    /// Set the root to the game position (start FEN + moves), reusing the subtree from the
    /// previous search when the game continued from it.
    pub fn prepare(&mut self, start_fen: &str, moves: &str, engine_color: Color) -> Result<Prepared, String> {
        if Some(engine_color) != self.engine_color {
            self.root = NONE;
            self.cache.clear();
        }
        self.engine_color = Some(engine_color);
        self.batches.clear();

        let mut pos = parse_position(start_fen)?;
        let mut hist: HashMap<Key, u32> = HashMap::new();
        *hist.entry(enc::key(&pos)).or_insert(0) += 1;
        let ucis: Vec<String> = moves.split_whitespace().map(str::to_owned).collect();
        let mut played = Vec::with_capacity(ucis.len());
        for u in &ucis {
            let m = UciMove::from_ascii(u.as_bytes())
                .map_err(|e| format!("{u}: {e}"))?
                .to_move(&pos)
                .map_err(|e| format!("{u}: {e}"))?;
            pos.play_unchecked(m);
            played.push(m);
            *hist.entry(enc::key(&pos)).or_insert(0) += 1;
        }
        self.hist = hist;

        let old = self.root_moves.len();
        let mut node = NONE;
        if self.root != NONE && start_fen == self.start_fen && ucis.len() >= old && ucis[..old] == self.root_moves[..] {
            node = self.root;
            for m in &played[old..] {
                let nd = &self.nodes[node as usize];
                let i = match nd.moves.iter().position(|x| x == m) {
                    Some(i) if nd.state == EXPANDED => i,
                    _ => {
                        node = NONE;
                        break;
                    }
                };
                node = nd.edges.get(i).map_or(NONE, |e| e.child);
                if node == NONE {
                    break;
                }
            }
        }
        self.root = NONE;
        if node != NONE && self.nodes[node as usize].state != TERMINAL {
            self.root = self.compact(node);
        } else {
            self.nodes.clear();
            self.root = self.make_node(pos, NONE);
        }
        self.root_moves = ucis;
        self.start_fen = start_fen.to_owned();
        Ok(match self.nodes[self.root as usize].state {
            TERMINAL => Prepared::Terminal,
            PENDING => Prepared::NeedsEval,
            _ => Prepared::Ready,
        })
    }

    /// Keep only the subtree under `root` (renumbered, root first); returns the new root.
    fn compact(&mut self, root: u32) -> u32 {
        let old = std::mem::take(&mut self.nodes);
        let mut map = vec![NONE; old.len()];
        let mut order = vec![root];
        map[root as usize] = 0;
        let mut i = 0;
        while i < order.len() {
            for e in &old[order[i] as usize].edges {
                if e.child != NONE {
                    map[e.child as usize] = order.len() as u32;
                    order.push(e.child);
                }
            }
            i += 1;
        }
        let mut slots: Vec<Option<Node>> = old.into_iter().map(Some).collect();
        let mut nodes = Vec::with_capacity(order.len());
        for &o in &order {
            let mut n = slots[o as usize].take().unwrap();
            n.parent = if o == root { NONE } else { map[n.parent as usize] };
            for e in n.edges.iter_mut() {
                if e.child != NONE {
                    e.child = map[e.child as usize];
                }
            }
            nodes.push(n);
        }
        self.nodes = nodes;
        0
    }

    // ------------------------------------------------------------------ reporting
    pub fn summary(&self) -> Summary {
        let r = &self.nodes[self.root as usize];
        // sorted(N[:nexp + 1])[-2:]; unvisited moves have N = 0
        let mut ns: Vec<u32> = r.edges.iter().map(|e| e.n).collect();
        if ns.len() < r.moves.len() {
            ns.push(0);
        }
        ns.sort_unstable();
        let top1 = ns.last().copied().unwrap_or(0);
        let top2 = if ns.len() >= 2 { ns[ns.len() - 2] } else { 0 };
        Summary { root_n: r.n, nmoves: r.moves.len() as u32, nexp: r.edges.len() as u32, top1,
                  top2, state: r.state }
    }

    /// Up to `k` most-visited root moves (stats from the root player's perspective), and the
    /// principal variation (at most `max_pv` moves).
    pub fn root_stats(&self, k: usize, max_pv: usize) -> (Vec<MoveStat>, Vec<u32>) {
        let r = &self.nodes[self.root as usize];
        let mut order: Vec<usize> = (0..r.edges.len()).collect();
        order.sort_by_key(|&i| std::cmp::Reverse(r.edges[i].n)); // stable, like sorted()
        let mut moves = Vec::new();
        for &i in order.iter().take(k) {
            let e = &r.edges[i];
            if e.n == 0 {
                break;
            }
            let n = e.n as f64;
            moves.push(MoveStat { code: enc::move_code(&r.moves[i]), n: e.n, w: e.sw / n,
                                  d: e.sd / n, l: e.sl / n, adv: e.sa / n, p: r.p[i] as f64 });
        }
        let mut pv = Vec::new();
        let mut idx = self.root;
        while idx != NONE && pv.len() < max_pv {
            let node = &self.nodes[idx as usize];
            if node.state != EXPANDED || node.edges.is_empty() {
                break;
            }
            let mut best = 0;
            for (i, e) in node.edges.iter().enumerate() {
                if e.n > node.edges[best].n {
                    best = i;
                }
            }
            if node.edges[best].n == 0 {
                break;
            }
            pv.push(enc::move_code(&node.moves[best]));
            idx = node.edges[best].child;
        }
        (moves, pv)
    }

    pub fn tree_size(&self) -> usize {
        self.nodes.len()
    }
}

/// numpy's float32 add-reduce (first element, then pairwise summation of the rest), so the
/// softmax normalizer matches drawfish/search.py as closely as possible.
fn numpy_sum(a: &[f32]) -> f32 {
    fn pairwise(a: &[f32]) -> f32 {
        let n = a.len();
        if n < 8 {
            let mut res = 0.0f32;
            for &x in a {
                res += x;
            }
            res
        } else if n <= 128 {
            let mut r = [0.0f32; 8];
            r.copy_from_slice(&a[..8]);
            let mut i = 8;
            while i < n - n % 8 {
                for j in 0..8 {
                    r[j] += a[i + j];
                }
                i += 8;
            }
            let mut res = ((r[0] + r[1]) + (r[2] + r[3])) + ((r[4] + r[5]) + (r[6] + r[7]));
            while i < n {
                res += a[i];
                i += 1;
            }
            res
        } else {
            let mut n2 = n / 2;
            n2 -= n2 % 8;
            pairwise(&a[..n2]) + pairwise(&a[n2..])
        }
    }
    match a.split_first() {
        None => 0.0,
        Some((first, rest)) => *first + pairwise(rest),
    }
}
