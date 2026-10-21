"""Batched network inference for the search, using CUDA graphs on the GPU (Windows kernel
launch overhead otherwise dominates small batches: ~11 ms -> ~3 ms for 32 positions).

Two buffer slots allow the search to pipeline: launch() enqueues a batch asynchronously
(pinned memory, non-blocking copies) and returns a handle; the search selects the next
batch on the CPU while the GPU works, then calls wait(handle)."""
import numpy as np
import torch

from .encoding import NUM_MOVES
from .model import DrawNet, load_model


class _Graphed(torch.nn.Module):
    """Network + per-row policy blend: logits = sel * draw head + (1 - sel) * best-move head.
    Without the draw head (every sel = 0, the default) it is skipped: the blend would return
    the best-move logits exactly."""

    def __init__(self, net: DrawNet, draw_head=True):
        super().__init__()
        self.net = net
        self.draw_head = draw_head

    def forward(self, feat, sel):
        out = self.net(feat, draw_head=self.draw_head)
        if not self.draw_head:
            return torch.softmax(out["wdl"], 1), out["adv"], out["pbest"]
        w = sel[:, None]
        logits = w * out["pdraw"] + (1.0 - w) * out["pbest"]
        return torch.softmax(out["wdl"], 1), out["adv"], logits


class _Slot:
    """A captured graph for one batch size plus its pinned host staging buffers. Graphs share
    one memory pool: they all run in order on the same stream, and each keeps its own outputs."""

    def __init__(self, mod, size, device, pool):
        self.size = size
        self.feat = torch.zeros((size, 14), dtype=torch.int64, device=device)
        self.sel = torch.zeros(size, device=device)
        s = torch.cuda.Stream()
        s.wait_stream(torch.cuda.current_stream())
        with torch.cuda.stream(s), torch.no_grad():
            for _ in range(3):
                mod(self.feat, self.sel)
        torch.cuda.current_stream().wait_stream(s)
        self.graph = torch.cuda.CUDAGraph()
        with torch.no_grad(), torch.cuda.graph(self.graph, pool=pool):
            self.out = mod(self.feat, self.sel)
        self.h_feat = torch.zeros((size, 14), dtype=torch.int64).pin_memory()
        self.h_sel = torch.zeros(size).pin_memory()
        self.h_wdl = torch.zeros((size, 3)).pin_memory()
        self.h_adv = torch.zeros(size).pin_memory()
        self.h_logits = torch.zeros((size, NUM_MOVES)).pin_memory()
        self.event = torch.cuda.Event()


class Evaluator:
    # Batch sizes with a captured graph; a batch is padded to the next one. The search's
    # batches vary in size (cache hits and finished games need no evaluation), so the steps
    # are fine.
    SIZES = (1, 4, 8, 16, 24, 32, 48, 64, 96, 128)

    def __init__(self, model_path=None, net: DrawNet | None = None, device=None, use_graphs=True):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if net is None:
            net = load_model(model_path, self.device)
        net = net.to(self.device).eval()
        self.mod = _Graphed(net)
        self.mod_best = _Graphed(net, draw_head=False)
        self.use_graphs = use_graphs and self.device.type == "cuda"
        self.slots = {}
        self.pool = torch.cuda.graph_pool_handle() if self.use_graphs else None
        self.turn = 0
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

    def _slot(self, size, k, draw_head):
        key = (size, k, draw_head)
        if key not in self.slots:
            mod = self.mod if draw_head else self.mod_best
            self.slots[key] = _Slot(mod, size, self.device, self.pool)
        return self.slots[key]

    @torch.no_grad()
    def launch(self, feats: np.ndarray, sel: np.ndarray):
        """Enqueue a batch; returns a handle for wait(). Alternates between two slots, so at
        most two batches may be in flight and each must be waited on before the next-but-one
        launch."""
        b = feats.shape[0]
        if not self.use_graphs or b > self.SIZES[-1]:
            f = torch.from_numpy(feats.view(np.int64)).to(self.device)
            s = torch.from_numpy(sel.astype(np.float32)).to(self.device)
            wdl, adv, lg = (self.mod if sel.any() else self.mod_best)(f, s)
            return ("eager", wdl.cpu().numpy(), adv.cpu().numpy(), lg.cpu().numpy())
        size = next(z for z in self.SIZES if z >= b)
        self.turn ^= 1
        slot = self._slot(size, self.turn, bool(sel.any()))
        slot.h_feat[:b].numpy()[:] = feats.view(np.int64)
        slot.h_sel[:b].numpy()[:] = sel
        if b < size:
            slot.h_feat[b:].zero_()
        slot.feat.copy_(slot.h_feat, non_blocking=True)
        slot.sel.copy_(slot.h_sel, non_blocking=True)
        slot.graph.replay()
        wdl, adv, lg = slot.out
        slot.h_wdl[:b].copy_(wdl[:b], non_blocking=True)
        slot.h_adv[:b].copy_(adv[:b], non_blocking=True)
        slot.h_logits[:b].copy_(lg[:b], non_blocking=True)
        slot.event.record()
        return ("graph", slot, b)

    def wait(self, handle):
        if handle[0] == "eager":
            return handle[1], handle[2], handle[3]
        _, slot, b = handle
        slot.event.synchronize()
        return (slot.h_wdl[:b].numpy().copy(), slot.h_adv[:b].numpy().copy(),
                slot.h_logits[:b].numpy().copy())

    def __call__(self, feats: np.ndarray, sel: np.ndarray):
        """Synchronous evaluation: (wdl (B,3), adv (B,), logits (B,4168)) as numpy."""
        return self.wait(self.launch(feats, sel))
