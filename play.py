"""Play chess against Drawfish, a neural-network engine whose only goal is a draw.

Launch with "Play Drawfish.bat" (or: python play.py [--model path]).
Draw rules: stalemate, insufficient material, threefold repetition. Move counts never
draw a game (no fifty-move rule).
"""
import argparse
import queue
import random
import sys
import threading
import tkinter as tk
import traceback
from pathlib import Path
from tkinter import messagebox, simpledialog, ttk

sys.path.insert(0, str(Path(__file__).resolve().parent))
import chess  # noqa: E402
import chess.pgn  # noqa: E402

from drawfish.paths import DEFAULT_MODEL  # noqa: E402
from drawfish.rules import DRAW, Game  # noqa: E402

LIGHT, DARK = "#EFD9B4", "#B48764"
LAST_LIGHT, LAST_DARK = "#D9D46A", "#B5A93E"
SEL_COLOR = "#7FA650"
CHECK_COLOR = "#E0533F"
DOT_COLOR = "#4B6B32"
BAR_ENGINE, BAR_DRAW, BAR_HUMAN = "#C0504D", "#E8B33A", "#4F81BD"
FILLED = {chess.KING: "♚", chess.QUEEN: "♛", chess.ROOK: "♜",
          chess.BISHOP: "♝", chess.KNIGHT: "♞", chess.PAWN: "♟"}
OUTLINE = {chess.KING: "♔", chess.QUEEN: "♕", chess.ROOK: "♖",
           chess.BISHOP: "♗", chess.KNIGHT: "♘", chess.PAWN: "♙"}
PIECE_FONT = "Segoe UI Symbol" if sys.platform == "win32" else "DejaVu Sans"


class App:
    def __init__(self, root: tk.Tk, model_path: str):
        self.root = root
        self.model_path = model_path
        root.title("Drawfish — the chess engine that wants a draw")
        root.configure(bg="#2B2B2B")
        self.sq = max(52, min(88, (root.winfo_screenheight() - 170) // 8))
        self.flipped = False
        self.human = chess.WHITE
        self.game = Game()
        self.engine = None
        self.engine_lock = threading.Lock()
        self.q = queue.Queue()
        self.gen = 0
        self.stop_event = threading.Event()
        self.thinking = False
        self.selected = None
        self.drag = None
        self.last_stats = None
        self.game_over_shown = False
        self.draw_agreed = False
        self._build_ui()
        self.redraw()
        self.set_status("Loading the neural network…")
        threading.Thread(target=self._load_engine, daemon=True).start()
        root.after(40, self.poll)

    # ------------------------------------------------------------------ UI layout
    def _build_ui(self):
        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass
        bg, fg = "#2B2B2B", "#EAEAEA"
        style.configure("TFrame", background=bg)
        style.configure("TLabel", background=bg, foreground=fg, font=("Segoe UI", 10))
        style.configure("Title.TLabel", font=("Segoe UI Semibold", 18), foreground="#F2C14E")
        style.configure("Status.TLabel", font=("Segoe UI Semibold", 11))
        style.configure("Small.TLabel", font=("Segoe UI", 9), foreground="#A8A8A8")
        style.configure("TButton", font=("Segoe UI", 10), padding=5)
        style.configure("TRadiobutton", background=bg, foreground=fg, font=("Segoe UI", 10))
        style.configure("TLabelframe", background=bg, foreground=fg)
        style.configure("TLabelframe.Label", background=bg, foreground="#F2C14E",
                        font=("Segoe UI Semibold", 10))

        outer = ttk.Frame(self.root, padding=10)
        outer.pack(fill="both", expand=True)
        size = self.sq * 8
        board_fr = ttk.Frame(outer)
        board_fr.grid(row=0, column=0, sticky="n")
        self.canvas = tk.Canvas(board_fr, width=size + 22, height=size + 22, bg=bg,
                                highlightthickness=0)
        self.canvas.pack()
        self.canvas.bind("<ButtonPress-1>", self.on_press)
        self.canvas.bind("<B1-Motion>", self.on_motion)
        self.canvas.bind("<ButtonRelease-1>", self.on_release)

        side = ttk.Frame(outer, padding=(14, 0, 0, 0))
        side.grid(row=0, column=1, sticky="ns")
        ttk.Label(side, text="Drawfish", style="Title.TLabel").pack(anchor="w")
        ttk.Label(side, text="A neural-network engine whose only goal is a draw.",
                  style="Small.TLabel").pack(anchor="w", pady=(0, 8))
        self.status = ttk.Label(side, text="", style="Status.TLabel", wraplength=330)
        self.status.pack(anchor="w", pady=(0, 8))

        evf = ttk.LabelFrame(side, text="What Drawfish expects", padding=8)
        evf.pack(fill="x", pady=4)
        self.bar = tk.Canvas(evf, width=320, height=22, bg="#444", highlightthickness=0)
        self.bar.pack(anchor="w")
        self.eval_label = ttk.Label(evf, text="—", wraplength=320)
        self.eval_label.pack(anchor="w", pady=(4, 0))
        self.pv_label = ttk.Label(evf, text="", style="Small.TLabel", wraplength=320)
        self.pv_label.pack(anchor="w")

        gf = ttk.LabelFrame(side, text="New game", padding=8)
        gf.pack(fill="x", pady=4)
        self.color_var = tk.StringVar(value="white")
        row = ttk.Frame(gf)
        row.pack(anchor="w")
        ttk.Label(row, text="You play:").pack(side="left")
        for txt, val in (("White", "white"), ("Black", "black"), ("Random", "random")):
            ttk.Radiobutton(row, text=txt, value=val, variable=self.color_var).pack(side="left", padx=4)
        ttk.Button(gf, text="Start new game", command=self.new_game).pack(anchor="w", pady=(6, 0))

        sf = ttk.LabelFrame(side, text="Engine settings", padding=8)
        sf.pack(fill="x", pady=4)
        self.time_var = tk.DoubleVar(value=2.0)
        self.time_label = ttk.Label(sf, text="")
        self.time_label.pack(anchor="w")
        ttk.Scale(sf, from_=0.5, to=15.0, variable=self.time_var, length=300,
                  command=lambda _=None: self._time_text()).pack(anchor="w")
        self._time_text()
        row = ttk.Frame(sf)
        row.pack(anchor="w", pady=(6, 0))
        ttk.Label(row, text="Engine goal:").pack(side="left")
        self.mode_var = tk.StringVar(value="Draw (Drawfish)")
        mode = ttk.Combobox(row, textvariable=self.mode_var, state="readonly", width=22,
                            values=["Draw (Drawfish)", "Win (ordinary engine)"])
        mode.pack(side="left", padx=6)
        mode.bind("<<ComboboxSelected>>", lambda e: self.change_mode())

        bf = ttk.Frame(side)
        bf.pack(fill="x", pady=6)
        for i, (txt, cmd) in enumerate((("Undo", self.undo), ("Offer draw", self.offer_draw),
                                        ("Flip board", self.flip), ("Load FEN…", self.load_fen),
                                        ("Copy PGN", self.copy_pgn))):
            ttk.Button(bf, text=txt, command=cmd).grid(row=i // 3, column=i % 3, padx=2, pady=2, sticky="ew")

        mf = ttk.LabelFrame(side, text="Moves", padding=6)
        mf.pack(fill="both", expand=True, pady=4)
        self.moves_text = tk.Text(mf, width=40, height=10, bg="#1F1F1F", fg="#EAEAEA",
                                  font=("Consolas", 10), relief="flat", wrap="word")
        self.moves_text.pack(fill="both", expand=True)
        self.moves_text.configure(state="disabled")
        ttk.Label(side, text="Draws: stalemate, insufficient material, threefold repetition.\n"
                             "Move counts never draw (no fifty-move rule).",
                  style="Small.TLabel").pack(anchor="w", pady=(6, 0))

        self.root.bind("<Control-z>", lambda e: self.undo())
        self.root.bind("<Control-n>", lambda e: self.new_game())
        self.root.bind("<f>", lambda e: self.flip())
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

    def _time_text(self):
        v = round(self.time_var.get() * 2) / 2
        self.time_var.set(v)
        self.time_label.configure(text=f"Thinking time per move: {v:.1f} s")

    # ------------------------------------------------------------------ engine plumbing
    def _load_engine(self):
        try:
            from drawfish.engine import DrawfishEngine
            eng = DrawfishEngine(self.model_path)
            # Warm up (CUDA graphs, cuDNN autotune) so the first real move is not slow.
            eng.think(Game(), nodes=200)
            eng.new_game()
            self.q.put(("loaded", eng))
        except Exception:
            self.q.put(("error", -1, traceback.format_exc()))

    def poll(self):
        try:
            while True:
                msg = self.q.get_nowait()
                kind = msg[0]
                if kind == "loaded":
                    self.engine = msg[1]
                    self.update_all()
                    self.maybe_engine_move()
                elif kind == "mode_set":
                    self.maybe_engine_move()
                elif kind == "info" and msg[1] == self.gen:
                    self.show_stats(msg[2], live=True)
                elif kind == "move" and msg[1] == self.gen:
                    self.thinking = False
                    move, st = msg[2], msg[3]
                    if move is not None and move in self.game.board.legal_moves:
                        self.game.push(move)
                        if st:
                            self.show_stats(st, live=False)
                    self.update_all()
                elif kind == "error":
                    self.thinking = False
                    messagebox.showerror("Drawfish error", msg[2])
                    if self.engine is None:
                        self.set_status("Could not load the engine (see error).")
        except queue.Empty:
            pass
        self.root.after(40, self.poll)

    def maybe_engine_move(self):
        if self.engine is None or self.thinking or self.outcome():
            return
        if self.game.board.turn == self.human:
            return
        self.gen += 1
        gen = self.gen
        self.stop_event = threading.Event()
        stop = self.stop_event
        g = self.game.copy()
        movetime = float(self.time_var.get())
        self.thinking = True
        self.set_status("Drawfish is thinking…")

        def run():
            try:
                with self.engine_lock:
                    if stop.is_set():
                        return
                    mv, st = self.engine.think(
                        g, movetime=movetime, stop_event=stop,
                        info_cb=lambda s: self.q.put(("info", gen, s)))
                self.q.put(("move", gen, mv, st))
            except Exception:
                self.q.put(("error", gen, traceback.format_exc()))

        threading.Thread(target=run, daemon=True).start()

    def cancel_engine(self):
        self.stop_event.set()
        self.gen += 1
        self.thinking = False

    # ------------------------------------------------------------------ actions
    def new_game(self):
        self.cancel_engine()
        c = self.color_var.get()
        self.human = {"white": chess.WHITE, "black": chess.BLACK}.get(c, random.choice([chess.WHITE, chess.BLACK]))
        self.flipped = self.human == chess.BLACK
        self.game = Game()
        self.selected = None
        self.clear_stats()
        self.game_over_shown = False
        self.draw_agreed = False
        self._new_engine_game()
        self.update_all()
        self.maybe_engine_move()

    def _new_engine_game(self):
        if self.engine is not None:
            def reset():
                with self.engine_lock:
                    self.engine.new_game()
            threading.Thread(target=reset, daemon=True).start()

    def change_mode(self):
        self.cancel_engine()
        if self.engine is None:
            return
        mode = "win" if self.mode_var.get().startswith("Win") else "draw"

        def apply():
            with self.engine_lock:
                self.engine.params.mode = mode
                self.engine.new_game()
            self.q.put(("mode_set",))
        threading.Thread(target=apply, daemon=True).start()
        self.update_all()

    def undo(self):
        self.cancel_engine()
        b = self.game.board
        if not b.move_stack:
            return
        self.game.pop()
        if b.move_stack and b.turn != self.human:
            self.game.pop()
        self.selected = None
        self.game_over_shown = False
        self.draw_agreed = False
        self.update_all()
        self.maybe_engine_move()

    def outcome(self):
        if self.draw_agreed:
            return DRAW, "agreement"
        return self.game.outcome()

    def offer_draw(self):
        """A draw-maximizing engine accepts every draw offer."""
        if self.engine is None or self.outcome():
            return
        self.cancel_engine()
        self.draw_agreed = True
        self.update_all()

    def flip(self):
        self.flipped = not self.flipped
        self.redraw()

    def load_fen(self):
        fen = simpledialog.askstring("Load FEN", "Paste a FEN position:", parent=self.root)
        if not fen:
            return
        try:
            b = chess.Board(fen.strip())
            if not b.is_valid():
                raise ValueError("illegal position")
        except Exception as e:
            messagebox.showerror("Invalid FEN", str(e))
            return
        self.cancel_engine()
        self.game = Game(b.fen())
        self.selected = None
        self.clear_stats()
        self.game_over_shown = False
        self.draw_agreed = False
        self._new_engine_game()
        self.update_all()
        self.maybe_engine_move()

    def pgn(self):
        g = chess.pgn.Game()
        g.headers["Event"] = "Drawfish game"
        g.headers["White"] = "You" if self.human == chess.WHITE else "Drawfish"
        g.headers["Black"] = "You" if self.human == chess.BLACK else "Drawfish"
        if self.game.start_fen != chess.STARTING_FEN:
            g.setup(chess.Board(self.game.start_fen))
        node = g
        for m in self.game.board.move_stack:
            node = node.add_variation(m)
        out = self.outcome()
        g.headers["Result"] = out[0] if out else "*"
        return str(g)

    def copy_pgn(self):
        self.root.clipboard_clear()
        self.root.clipboard_append(self.pgn())
        self.set_status("PGN copied to the clipboard.")

    def on_close(self):
        self.stop_event.set()
        self.root.destroy()

    # ------------------------------------------------------------------ board input
    def square_at(self, x, y):
        x -= 11
        y -= 11
        if not (0 <= x < self.sq * 8 and 0 <= y < self.sq * 8):
            return None
        f, r = int(x // self.sq), 7 - int(y // self.sq)
        if self.flipped:
            f, r = 7 - f, 7 - r
        return chess.square(f, r)

    def xy(self, square):
        f, r = chess.square_file(square), chess.square_rank(square)
        if self.flipped:
            f, r = 7 - f, 7 - r
        return 11 + f * self.sq, 11 + (7 - r) * self.sq

    def human_can_move(self):
        return (self.engine is not None and not self.thinking and not self.outcome()
                and self.game.board.turn == self.human)

    def on_press(self, ev):
        s = self.square_at(ev.x, ev.y)
        if s is None or not self.human_can_move():
            return
        p = self.game.board.piece_at(s)
        if self.selected is not None and s != self.selected and self.targets(self.selected).get(s):
            self.try_move(self.selected, s)
            return
        if p is not None and p.color == self.human:
            self.selected = s
            self.redraw()
            x, y = ev.x, ev.y
            self.drag = (s, self.canvas.create_text(x, y, text=FILLED[p.piece_type],
                                                    font=(PIECE_FONT, int(self.sq * 0.62)),
                                                    fill="white" if p.color else "black"))
        else:
            self.selected = None
            self.redraw()

    def on_motion(self, ev):
        if self.drag:
            self.canvas.coords(self.drag[1], ev.x, ev.y)

    def on_release(self, ev):
        if not self.drag:
            return
        origin, item = self.drag
        self.drag = None
        self.canvas.delete(item)
        s = self.square_at(ev.x, ev.y)
        if s is not None and s != origin and self.targets(origin).get(s):
            self.try_move(origin, s)
        else:
            self.redraw()

    def targets(self, origin):
        t = {}
        for m in self.game.board.legal_moves:
            if m.from_square == origin:
                t.setdefault(m.to_square, []).append(m)
        return t

    def try_move(self, origin, dest):
        moves = self.targets(origin).get(dest, [])
        if not moves:
            return
        move = moves[0]
        if len(moves) > 1:  # promotion
            piece = self.ask_promotion()
            if piece is None:
                self.redraw()
                return
            move = next(m for m in moves if m.promotion == piece)
        self.selected = None
        self.game.push(move)
        self.update_all()
        self.maybe_engine_move()

    def ask_promotion(self):
        win = tk.Toplevel(self.root)
        win.title("Promote to")
        win.transient(self.root)
        win.grab_set()
        choice = {"p": None}
        for pt in (chess.QUEEN, chess.ROOK, chess.BISHOP, chess.KNIGHT):
            def pick(pt=pt):
                choice["p"] = pt
                win.destroy()
            tk.Button(win, text=FILLED[pt], font=(PIECE_FONT, 30), width=2, command=pick).pack(side="left", padx=4, pady=4)
        self.root.wait_window(win)
        return choice["p"]

    # ------------------------------------------------------------------ rendering
    def redraw(self):
        c, sq, b = self.canvas, self.sq, self.game.board
        c.delete("all")
        last = b.move_stack[-1] if b.move_stack else None
        check_sq = b.king(b.turn) if b.is_check() else None
        targets = self.targets(self.selected) if self.selected is not None else {}
        for s in chess.SQUARES:
            x, y = self.xy(s)
            light = (chess.square_file(s) + chess.square_rank(s)) % 2 == 1
            color = LIGHT if light else DARK
            if last and s in (last.from_square, last.to_square):
                color = LAST_LIGHT if light else LAST_DARK
            if s == self.selected:
                color = SEL_COLOR
            if s == check_sq:
                color = CHECK_COLOR
            c.create_rectangle(x, y, x + sq, y + sq, fill=color, outline="")
        for i in range(8):
            f = 7 - i if self.flipped else i
            r = i if self.flipped else 7 - i
            c.create_text(11 + i * sq + sq / 2, 11 + 8 * sq + 6, text="abcdefgh"[f],
                          fill="#BDBDBD", font=("Segoe UI", 8))
            c.create_text(5, 11 + i * sq + sq / 2, text=str(r + 1), fill="#BDBDBD", font=("Segoe UI", 8))
        fsize = int(sq * 0.62)
        for s, p in b.piece_map().items():
            if self.drag and s == self.drag[0]:
                continue
            x, y = self.xy(s)
            cx, cy = x + sq / 2, y + sq / 2 + sq * 0.02
            if p.color == chess.WHITE:
                c.create_text(cx, cy, text=FILLED[p.piece_type], font=(PIECE_FONT, fsize), fill="white")
                c.create_text(cx, cy, text=OUTLINE[p.piece_type], font=(PIECE_FONT, fsize), fill="black")
            else:
                c.create_text(cx, cy, text=FILLED[p.piece_type], font=(PIECE_FONT, fsize), fill="black")
        for t in targets:
            x, y = self.xy(t)
            r = sq * (0.16 if b.piece_at(t) is None else 0.42)
            if b.piece_at(t) is None:
                c.create_oval(x + sq / 2 - r, y + sq / 2 - r, x + sq / 2 + r, y + sq / 2 + r,
                              fill=DOT_COLOR, outline="")
            else:
                c.create_oval(x + sq / 2 - r, y + sq / 2 - r, x + sq / 2 + r, y + sq / 2 + r,
                              outline=DOT_COLOR, width=3)
        if self.drag:
            c.tag_raise(self.drag[1])

    def update_moves(self):
        b = chess.Board(self.game.start_fen)
        parts = []
        for m in self.game.board.move_stack:
            san = b.san(m)
            if b.turn == chess.WHITE:
                parts.append(f"{b.fullmove_number}. {san}")
            elif not parts:
                parts.append(f"{b.fullmove_number}... {san}")
            else:
                parts[-1] += f" {san}"
            b.push(m)
        self.moves_text.configure(state="normal")
        self.moves_text.delete("1.0", "end")
        self.moves_text.insert("end", "\n".join(parts))
        self.moves_text.see("end")
        self.moves_text.configure(state="disabled")

    def show_stats(self, st, live):
        if not st or not st.get("wdl"):
            return
        self.last_stats = st
        w, d, l = st["wdl"]
        self.bar.delete("all")
        width = 320
        x = 0
        for frac, col in ((w, BAR_ENGINE), (d, BAR_DRAW), (l, BAR_HUMAN)):
            seg = width * max(0.0, frac)
            self.bar.create_rectangle(x, 0, x + seg, 22, fill=col, outline="")
            x += seg
        self.bar.create_text(width / 2, 11, text=f"draw {d * 100:.0f}%", fill="black",
                             font=("Segoe UI Semibold", 9))
        self.eval_label.configure(text=f"Draw {d * 100:.1f}%   ·   Drawfish wins {w * 100:.1f}%"
                                       f"   ·   you win {l * 100:.1f}%")
        extra = st.get("note") or (f"{st['nodes']:,} nodes, {st['nps']:,.0f}/s" if st.get("nodes") else "")
        pv = ""
        if st.get("pv") and not st.get("note"):
            try:
                b = self.game.board.copy()
                if not live:  # the move has already been played: show the line from before it
                    b.pop()
                pv = b.variation_san(st["pv"][:8])
            except Exception:
                pv = ""
        self.pv_label.configure(text=(extra + ("\nexpects: " + pv if pv else "")))

    def clear_stats(self):
        self.last_stats = None
        self.bar.delete("all")
        self.eval_label.configure(text="—")
        self.pv_label.configure(text="")

    def set_status(self, txt):
        self.status.configure(text=txt)

    def update_all(self):
        self.redraw()
        self.update_moves()
        out = self.outcome()
        if self.engine is None:
            return
        if out:
            res, reason = out
            if res == DRAW:
                msg = f"Draw by {reason}. Drawfish got exactly what it wanted."
            else:
                winner = chess.WHITE if res == "1-0" else chess.BLACK
                if winner == self.human:
                    msg = f"You win by {reason}! You broke Drawfish's defence."
                else:
                    msg = f"Drawfish won by {reason} — it did not want to, but you left it no draw."
            self.set_status("Game over: " + msg)
            if not self.game_over_shown:
                self.game_over_shown = True
                self.root.after(150, lambda: messagebox.showinfo("Game over", msg))
        elif self.thinking:
            self.set_status("Drawfish is thinking…")
        elif self.game.board.turn == self.human:
            self.set_status("Your move" + (" (you are in check)" if self.game.board.is_check() else "") + ".")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    args = ap.parse_args()
    if not Path(args.model).exists():
        tk.Tk().withdraw()
        messagebox.showerror("Drawfish", f"Model file not found:\n{args.model}\n\nTrain one first (see README).")
        return
    try:
        from ctypes import windll
        windll.shcore.SetProcessDpiAwareness(1)
    except Exception:
        pass
    root = tk.Tk()
    App(root, args.model)
    root.mainloop()


if __name__ == "__main__":
    main()
