"""Drawfish UCI front-end: load it in any UCI chess GUI (Arena, Cute Chess, Banksia, ...)
via "Drawfish UCI.bat"."""
import math
import sys
import threading
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chess  # noqa: E402

from drawfish.engine import DrawfishEngine, budget_from_clock  # noqa: E402
from drawfish.paths import DEFAULT_MODEL  # noqa: E402
from drawfish.rules import Game  # noqa: E402
from drawfish.search import SearchParams  # noqa: E402

NAME = "Drawfish 1.0"


def out(s):
    sys.stdout.write(s + "\n")
    sys.stdout.flush()


class UCI:
    def __init__(self):
        self.params = SearchParams()
        self.model_path = str(DEFAULT_MODEL)
        self.engine = None
        self.game = Game()
        self.thread = None
        self.stop_event = threading.Event()
        self.default_movetime = 2000
        self.fixed_nodes = 0

    def ensure_engine(self):
        if self.engine is None:
            self.engine = DrawfishEngine(self.model_path, params=self.params)

    def info(self, st):
        if not st or not st.get("wdl"):
            return
        w, d, l = st["wdl"]
        e = min(max(w + 0.5 * d, 1e-4), 1 - 1e-4)
        cp = int(round(400 * math.log10(e / (1 - e))))
        pv = " ".join(m.uci() for m in st["pv"])
        depth = max(1, len(st["pv"]))
        out(f"info depth {depth} nodes {st['nodes']} nps {int(st['nps'])} time {int(st['time'] * 1000)} "
            f"score cp {cp} wdl {int(w * 1000)} {int(d * 1000)} {int(l * 1000)} pv {pv}")

    def go(self, tokens):
        args = {}
        i = 0
        while i < len(tokens):
            t = tokens[i]
            if t in ("wtime", "btime", "winc", "binc", "movestogo", "movetime", "nodes", "depth"):
                args[t] = int(tokens[i + 1])
                i += 2
            else:
                args[t] = True
                i += 1
        nodes = args.get("nodes") or (self.fixed_nodes or None)
        movetime = None
        if "infinite" in args or "ponder" in args:
            movetime = 3600 * 24.0
        elif "movetime" in args:
            movetime = max(0.01, args["movetime"] / 1000.0 - 0.03)
        elif "wtime" in args or "btime" in args:
            movetime = budget_from_clock(self.game.board, args.get("wtime"), args.get("btime"),
                                         args.get("winc", 0), args.get("binc", 0), args.get("movestogo"))
        elif nodes is None:
            movetime = self.default_movetime / 1000.0
        self.stop_event.clear()
        game = self.game

        def run():
            move, st = self.engine.think(game, nodes=nodes, movetime=movetime,
                                         stop_event=self.stop_event, info_cb=self.info)
            self.info(st)
            out(f"bestmove {move.uci() if move else '0000'}")

        self.thread = threading.Thread(target=run, daemon=True)
        self.thread.start()

    def wait(self):
        if self.thread is not None:
            self.thread.join()
            self.thread = None

    def setoption(self, line):
        rest = line[len("setoption name "):]
        name, _, value = rest.partition(" value ")
        name, value = name.strip().lower(), value.strip()
        if name == "mode":
            self.params.mode = "win" if value.lower() == "win" else "draw"
            if self.engine:
                self.engine.new_game()
        elif name == "movetime":
            self.default_movetime = int(value)
        elif name == "nodes":
            self.fixed_nodes = int(value)
        elif name == "cpuct":
            self.params.cpuct = int(value) / 100.0
        elif name == "drawbalance":
            self.params.gamma = int(value) / 100.0
        elif name == "winvalue":
            self.params.win_value = int(value) / 100.0
        elif name == "drawprior":
            self.params.draw_prior = int(value) / 100.0
        elif name == "modelpath":
            self.model_path = value
            self.engine = None

    def position(self, tokens):
        if tokens[0] == "startpos":
            g = Game()
            rest = tokens[1:]
        else:
            k = tokens.index("moves") if "moves" in tokens else len(tokens)
            g = Game(" ".join(tokens[1:k]))
            rest = tokens[k:]
        if rest and rest[0] == "moves":
            for u in rest[1:]:
                g.push(chess.Move.from_uci(u))
        self.game = g

    def loop(self):
        for raw in sys.stdin:
            # tolerate a UTF-8 byte-order mark (raw or mis-decoded) on the first line
            line = raw.strip().lstrip("﻿").removeprefix("ï»¿")
            if not line:
                continue
            tok = line.split()
            cmd = tok[0]
            if cmd == "uci":
                out(f"id name {NAME}")
                out("id author Drawfish project")
                out("option name Mode type combo default Draw var Draw var Win")
                out(f"option name MoveTime type spin default {self.default_movetime} min 10 max 600000")
                out("option name Nodes type spin default 0 min 0 max 100000000")
                out(f"option name CPuct type spin default {int(self.params.cpuct * 100)} min 10 max 1000")
                out(f"option name DrawBalance type spin default {int(self.params.gamma * 100)} min 0 max 100")
                out(f"option name WinValue type spin default {round(self.params.win_value * 100)} min 0 max 100")
                out(f"option name DrawPrior type spin default {round(self.params.draw_prior * 100)} min 0 max 100")
                out(f"option name ModelPath type string default {self.model_path}")
                out("uciok")
            elif cmd == "isready":
                self.ensure_engine()
                out("readyok")
            elif cmd == "setoption":
                self.setoption(line)
            elif cmd == "ucinewgame":
                self.stop_event.set()
                self.wait()
                self.ensure_engine()
                self.engine.new_game()
                self.game = Game()
            elif cmd == "position":
                self.stop_event.set()
                self.wait()
                self.position(tok[1:])
            elif cmd == "go":
                self.stop_event.set()
                self.wait()
                self.ensure_engine()
                self.go(tok[1:])
            elif cmd == "stop":
                self.stop_event.set()
                self.wait()
            elif cmd == "ponderhit":
                pass
            elif cmd == "quit":
                self.stop_event.set()
                self.wait()
                break
            elif cmd == "d":
                out(str(self.game.board))
                out(self.game.board.fen())


if __name__ == "__main__":
    UCI().loop()
