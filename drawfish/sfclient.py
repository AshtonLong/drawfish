"""Minimal, fast UCI client for Stockfish (teacher for data generation and a
sparring partner for benchmarks). Talks over raw pipes instead of
chess.engine to avoid the asyncio overhead on millions of short searches."""
import subprocess
import sys
import threading
import time

MATE_CP = 20000


class SearchLine:
    __slots__ = ("move", "cp", "wdl", "depth")

    def __init__(self, move, cp, wdl, depth):
        self.move, self.cp, self.wdl, self.depth = move, cp, wdl, depth


class Stockfish:
    """If Stockfish ever stops answering for `timeout` seconds (e.g. it wedged under memory
    pressure), a watchdog kills it, so callers get a RuntimeError instead of hanging."""

    def __init__(self, path, threads=1, hash_mb=16, options=None, low_priority=False, timeout=60.0):
        flags = 0
        if sys.platform == "win32":
            flags = subprocess.CREATE_NO_WINDOW
            if low_priority:
                flags |= subprocess.BELOW_NORMAL_PRIORITY_CLASS
        self.p = subprocess.Popen([str(path)], stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                  stderr=subprocess.DEVNULL, text=True, bufsize=1,
                                  creationflags=flags)
        self.timeout = timeout
        self.deadline = None
        threading.Thread(target=self._watchdog, daemon=True).start()
        self._send("uci")
        self._wait("uciok")
        self.multipv = 1
        self.set_option("Threads", threads)
        self.set_option("Hash", hash_mb)
        self.set_option("UCI_ShowWDL", "true")
        for k, v in (options or {}).items():
            self.set_option(k, v)
        self.ready()

    def _watchdog(self):
        while self.p.poll() is None:
            time.sleep(1.0)
            d = self.deadline
            if d is not None and time.monotonic() > d:
                self.p.kill()
                return

    def _send(self, s):
        self.p.stdin.write(s + "\n")
        self.p.stdin.flush()

    def _wait(self, token):
        self.deadline = time.monotonic() + self.timeout
        try:
            while True:
                line = self.p.stdout.readline()
                if not line:
                    raise RuntimeError("Stockfish exited unexpectedly")
                if line.startswith(token):
                    return line
        finally:
            self.deadline = None

    def set_option(self, name, value):
        self._send(f"setoption name {name} value {value}")

    def ready(self):
        self._send("isready")
        self._wait("readyok")

    def new_game(self):
        self._send("ucinewgame")
        self.ready()

    def search(self, fen=None, moves=(), nodes=None, depth=None, movetime=None, multipv=1):
        """Run a search. Returns (bestmove_uci, {multipv_index: SearchLine})."""
        if multipv != self.multipv:
            self.set_option("MultiPV", multipv)
            self.multipv = multipv
        pos = f"position fen {fen}" if fen else "position startpos"
        if moves:
            pos += " moves " + " ".join(moves)
        self._send(pos)
        go = "go"
        if nodes:
            go += f" nodes {nodes}"
        if depth:
            go += f" depth {depth}"
        if movetime:
            go += f" movetime {movetime}"
        self._send(go)
        lines = {}
        readline = self.p.stdout.readline
        limit = self.timeout + (movetime or 0) / 1000.0
        self.deadline = time.monotonic() + limit
        try:
            return self._read_search(readline, lines)
        finally:
            self.deadline = None

    @staticmethod
    def _read_search(readline, lines):
        while True:
            line = readline()
            if not line:
                raise RuntimeError("Stockfish exited unexpectedly")
            if line.startswith("bestmove"):
                best = line.split()[1]
                return best, lines
            if not line.startswith("info") or " pv " not in line or " score " not in line:
                continue
            t = line.split()
            if "lowerbound" in t or "upperbound" in t:
                continue
            k = int(t[t.index("multipv") + 1]) if "multipv" in t else 1
            i = t.index("score")
            if t[i + 1] == "cp":
                cp = int(t[i + 2])
            else:
                m = int(t[i + 2])
                cp = (MATE_CP - abs(m)) if m > 0 else -(MATE_CP - abs(m))
            j = t.index("wdl")
            wdl = (int(t[j + 1]), int(t[j + 2]), int(t[j + 3]))
            d = int(t[t.index("depth") + 1])
            lines[k] = SearchLine(t[t.index("pv") + 1], cp, wdl, d)

    def close(self):
        try:
            self._send("quit")
            self.p.wait(timeout=5)
        except Exception:
            self.p.kill()
