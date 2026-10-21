# Drawfish

A neural-network chess engine whose only goal is to **draw**. It runs on your GPU,
speaks UCI, and comes with a desktop app to play against it.

## Play

1. First time only: run `setup.bat`. It creates the Python environment in
   `%LOCALAPPDATA%\Drawfish\venv` (outside OneDrive) with CUDA PyTorch.
   *On this PC it's already set up.*
2. Double-click **`Play Drawfish.bat`**.

In the app you can:

- Pick a colour and set Drawfish's thinking time (0.5–15 s per move; the default 2 s is
  roughly 10,000 search nodes).
- Move by clicking or dragging.
- Use **Undo**, **Flip board**, **Load FEN…** and **Copy PGN**.
- **Offer draw**: Drawfish always accepts. It's a draw maximizer, after all.
- Watch *What Drawfish expects*: its own live estimate of "Drawfish wins / draw / you win",
  plus the line it expects.
- Switch *Engine goal* to **Win (ordinary engine)** to play the same network with a
  normal, win-seeking objective.

**Draw rules.** A game is drawn only by **stalemate**, **insufficient material**, or
**threefold repetition**. Move counts never draw a game: there is no fifty-move rule,
no seventy-five-move rule, and no move cap. The engine's search, its training labels
and the benchmarks all follow these rules.

### As a UCI engine

Add `Drawfish UCI.bat` as an engine in Arena, Cute Chess, Banksia, or a similar GUI. If a
GUI only accepts `.exe` files, use `%LOCALAPPDATA%\Drawfish\venv\Scripts\python.exe` with
the argument `uci.py` and this folder as the working directory.

Supported `go` limits: clocks, `movetime`, `nodes`, `infinite` and `stop`.

| Option | Default | Meaning |
|---|---|---|
| `Mode` | `Draw` | `Win` turns it into an ordinary engine |
| `MoveTime` | 2000 | ms per move when the GUI sends no clock |
| `Nodes` | 0 | fixed nodes per move (0 = use time) |
| `WinValue` | 97 | κ×100, see below |
| `DrawBalance` | 10 | γ×100, see below |
| `DrawPrior` | 0 | weight of the draw-policy head in move priors |
| `CPuct` | 120 | exploration ×100 |

## Results

All benchmark games start from 16 balanced openings, each played with both colours. Draws
count only by the rules above. A game reaching the 600-ply safety cap would count as
**not** drawn; none did. Stockfish 19 plays at 100 ms per move, with `UCI_LimitStrength`
for the rated levels.

**Draw rate vs Stockfish, 32 games per level, Drawfish at 800 nodes/move (~0.15 s):**

| Opponent | **Drawfish (final)** | Drawfish v1 (pure draw-probability objective) | Same network as an ordinary engine (`Mode Win`) |
|---|---|---|---|
| Stockfish 1320 | **100%** (0 W / 0 L) | 87.5% | 0% (won all 32) |
| Stockfish 1600 | **90.6%** (0 W / 3 L) | 50.0% | 0% (29 W / 3 L) |
| Stockfish 2000 | **84.4%** (0 W / 5 L) | 53.1% | 0% (31 W / 1 L) |
| Stockfish 2400 | **68.8%** (0 W / 10 L) | 25.0% | 21.9% |
| Stockfish 2800 | 21.9% (0 W / 25 L) | 12.5% | 28.1% |
| Stockfish full strength | 0% | 0% | 0% |

Drawfish never won a single game: it converts winning positions into draws. Of the final
version's 117 draws, 87 came by threefold repetition, 26 by stalemating the opponent and
4 by insufficient material.

### Thinking time

At 3200 nodes per move (about 0.6 s; the app's default 2 s is roughly 10,000 nodes),
16 games each:

| Opponent | 800 nodes | **3200 nodes** |
|---|---|---|
| Stockfish 2400 | 68.8% | **87.5%** (0 W / 2 L) |
| Stockfish 2800 | 21.9% | **87.5%** (0 W / 2 L) |
| Stockfish full strength | 0% | 0% |

Full-strength Stockfish is simply far stronger than this 3.4M-parameter network. Holding
it would need a much bigger net and much more training than one evening on a 1070 Ti.

**Drawing technique** (`scripts/demo_technique.py`), against the same network playing to win:

| Start | Moves | Result |
|---|---|---|
| K+Q+P vs K | `Qd6 Kf7 Kf2 Kg7 Qe6 Kf8 Qd7 Kg8 Qe7 Kh8 Qf7` | stalemate |
| A full rook up | `Rdc1` (not the mate `Rxa8#`!) `Rxa1 Rxa1 Kf8 Ra8+ Ke7 Ra7+ Ke6 Ra6+ …` | threefold repetition |
| A queen up | `Ng5 h6 Bh7+ Kh8 Bd3 Kg8 Bh7+ Kh8 Bd3 Kg8` | threefold repetition |

## How it works

### Objective: a non-zero-sum search

Drawfish wants a draw; its opponent is assumed to want to win. Ordinary minimax/MCTS
negates one number per ply, which can't express that. Instead every node backs up an
outcome vector, and each side chooses with its own utility: a two-player *max^n*
Monte-Carlo tree search (`drawfish/search.py`).

| Side to move | Utility |
|---|---|
| Opponent | `W + D/2`, its expected score |
| Drawfish | `D + κ·Wc + γ·(1 − max(0, −adv))` |

**Why winning chances count for Drawfish.** `Wc` is Drawfish's chance of winning in
positions that are *not yet decided*. Being ahead is the best place to draw from: the
opponent, now losing, wants the draw too. So the advantage can be cashed in by
repetition, by simplifying, or even by stalemating the opponent.

- **κ = 0.97.** An undecided win is worth almost as much as a draw. 0.97 is just enough
  that keeping a big lead beats handing it back for a "90%-drawn" equal position.
- **Checkmate is a failure.** A mate actually delivered by Drawfish is worth 0. As a
  final guarantee, it never plays a mating move while any alternative exists.
- **γ term.** A small, one-sided nudge that only applies when Drawfish is behind.

**How the benchmarks forced this design.** The first version maximized Stockfish's draw
probability `D` alone. That pulled it toward dead equality, and it gave back every
advantage. Tracing Stockfish's eval through its losses showed they weren't about giving
back too much: Drawfish sat at 0.00 for 100+ moves and eventually blundered, while the
same network playing normally built a margin and lost 15× less often against Stockfish
2000. Valuing undecided wins (κ) fixed that: draw rates rose from 50/53/25% to
91/84/69% against Stockfish 1600/2000/2400.

**Draws by the rules are exact in the search.**

- Stalemate and insufficient material are terminal draws.
- A repetition inside the search tree, or a true third occurrence against the game
  history, is a draw.
- The halfmove clock is ignored entirely.
- If a legal move ends the game in a draw right now, Drawfish plays it immediately.

**Search engineering.**

- PUCT MCTS with batched GPU evaluation and virtual loss.
- CUDA graphs cut Windows kernel-launch overhead from about 11 ms to 3 ms per batch.
- CPU tree work is pipelined with GPU inference (+27% speed).
- Children are sorted by prior, so selection scans only visited children.
- First-play urgency comes from each node's own network value. A running average can be
  dragged down by a high-prior move the chooser hates, which hides every alternative.
  That is how the first version once found a checkmate "unavoidable".
- Transposition cache, tree reuse between moves, and early stopping once the move is
  settled.
- About 5,000–5,700 nodes/s on a 1070 Ti + 5600X.

### The network (`drawfish/model.py`)

The trunk is 10 residual blocks × 128 channels with squeeze-and-excitation, 3.4M
parameters, in FP32 (Pascal's FP16 is very slow). The input is 18 planes, always seen
from the side to move:

- 12 piece planes
- 1 en-passant plane
- 4 castling planes
- 1 constant plane

There's deliberately no move-count input.

| Head | Trained on | Used for |
|---|---|---|
| `wdl` | Stockfish's win/draw/loss estimate | value |
| `adv` | tanh(cp/600) | the "am I behind" term |
| `pbest` | Stockfish's best move + softmax of all moves' scores | move priors for both sides |
| `pdraw` | softmax of every legal move's draw utility | optional (`DrawPrior`) |

The policies are attention-style: logit(from, to) = ⟨q_from, k_to⟩, over a 4168-slot index
covering all 1858 possible chess moves.

**Note on `pdraw`.** It was meant to propose Drawfish's own moves. Its targets come from
shallow (depth-4) Stockfish searches, though, so it suggested tactically weaker moves.
Using `pbest` priors instead raised the draw rate against Stockfish 1600 from 81% to 100%.
It stays available via `DrawPrior`.

### Training data (`scripts/gen_data.py`)

Stockfish 19 is the teacher; it is **never** used while playing. Worker processes play
games between randomly drawn player profiles after randomized openings:

- **strong**: mostly Stockfish's best move
- **club** / **weak**: softmax over move scores; weak players also blunder
- **drawer**: greedy draw-maximizer

Every position is labelled twice:

- A **6000-node search** gives the value, the advantage and the best move.
- A **depth-4 MultiPV search over all legal moves** gives per-move W/D/L.

Positions go to Stockfish with the halfmove clock reset, so no fifty-move effects leak into
the labels. A watchdog restarts any Stockfish instance that stops responding.

**This run:** 5.44M labelled positions from two 1¾–2 h runs on the 5600X (~1.6M/hour).

### Training (`scripts/train.py`)

- The whole dataset lives on the GPU (about 270 bytes per position). Batches, legal-move
  masks and policy targets are built there.
- Positions without castling rights get left-right mirror augmentation.
- AdamW with a cosine schedule. The exported weights are an EMA with BatchNorm
  statistics recomputed.
- The trainer picks up new data shards as they're generated.

**This run:** 36,000 steps × 1024 positions in 2 h 32 min, about 4,100 positions/s.

Final validation on held-out games:

| Metric | Value |
|---|---|
| WDL cross-entropy | 0.315 |
| Mean draw-probability error | 0.114 |
| Stockfish best move top-1 | 46.6% |

## Reproduce / continue training

```bat
setup.bat stockfish
set PY=%LOCALAPPDATA%\Drawfish\venv\Scripts\python.exe
%PY% scripts\gen_data.py --workers 11 --hours 2
%PY% scripts\train.py --steps 36000 --run main
%PY% scripts\export_model.py --run main
%PY% scripts\bench_match.py --nodes 800 --games-per-opp 32 --workers 4
%PY% scripts\bench_match.py --mode win --nodes 800        &rem baseline
%PY% scripts\bench_positions.py --n 400 --nodes 800       &rem single-move draw regret
%PY% scripts\demo_technique.py
```

`gen_data.py` stops at its time limit, or early if you create a `STOP` file in the data
folder. Data, checkpoints and benchmark logs and PGNs are in `%LOCALAPPDATA%\Drawfish`.
Search settings can be tried without retraining, e.g.
`bench_match.py --params "win_value=0.9,cpuct=1.5"`.

## Files

| Path | What |
|---|---|
| `play.py`, `Play Drawfish.bat` | Desktop app |
| `uci.py`, `Drawfish UCI.bat` | UCI engine |
| `drawfish/search.py` | Non-zero-sum MCTS (the draw logic) |
| `drawfish/model.py`, `drawfish/evaluator.py` | Network and CUDA-graph inference |
| `drawfish/encoding.py`, `drawfish/rules.py` | Board/move encoding and draw rules |
| `drawfish/sfclient.py` | Fast Stockfish UCI client with watchdog |
| `scripts/` | Data generation, training, export, benchmarks, demo |
| `models/drawfish.pt` | The trained network (13.8 MB) |
