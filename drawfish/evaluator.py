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
    """Network + per-row policy blend: logits = sel * draw head + (1 - sel) * best-move head."""

    def __init__(self, net: DrawNet):
        super().__init__()
        self.net = net

    def forward(self, feat, sel):
        out = self.net(feat)
        w = sel[:, None]
        logits = w * out["pdraw"] + (1.0 - w) * out["pbest"]
        return torch.softmax(out["wdl"], 1), out["adv"], logits


class _Slot:
    """A captured graph for one batch size plus its pinned host staging buffers."""

    def __init__(self, mod, size, device):
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
        with torch.no_grad(), torch.cuda.graph(self.graph):
            self.out = mod(self.feat, self.sel)
        self.h_feat = torch.zeros((size, 14), dtype=torch.int64).pin_memory()
        self.h_sel = torch.zeros(size).pin_memory()
        self.h_wdl = torch.zeros((size, 3)).pin_memory()
        self.h_adv = torch.zeros(size).pin_memory()
        self.h_logits = torch.zeros((size, NUM_MOVES)).pin_memory()
        self.event = torch.cuda.Event()


class Evaluator:
    SIZES = (1, 8, 16, 32, 64, 128)

    def __init__(self, model_path=None, net: DrawNet | None = None, device=None, use_graphs=True):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        if net is None:
            net = load_model(model_path, self.device)
        self.mod = _Graphed(net.to(self.device).eval())
        self.use_graphs = use_graphs and self.device.type == "cuda"
        self.slots = {}
        self.turn = 0
        if self.device.type == "cuda":
            torch.backends.cudnn.benchmark = True

    def _slot(self, size, k):
        key = (size, k)
        if key not in self.slots:
            self.slots[key] = _Slot(self.mod, size, self.device)
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
            wdl, adv, lg = self.mod(f, s)
            return ("eager", wdl.cpu().numpy(), adv.cpu().numpy(), lg.cpu().numpy())
        size = next(z for z in self.SIZES if z >= b)
        self.turn ^= 1
        slot = self._slot(size, self.turn)
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
