"""Edge-level GNN (static or temporal), pure PyTorch. Each flow = an edge to classify."""
import argparse
import copy
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.features.load_data import load_config
from src.models.rf import evaluate, log_result


class MPLayer(nn.Module):
    """Mean-aggregation message passing over both edge directions, using edge features."""
    def __init__(self, h, e):
        super().__init__()
        self.m_in = nn.Sequential(nn.Linear(2 * h + e, h), nn.ReLU())
        self.m_out = nn.Sequential(nn.Linear(2 * h + e, h), nn.ReLU())
        self.upd = nn.Sequential(nn.Linear(3 * h, h), nn.ReLU())

    def forward(self, h, ei, ea):
        src, dst = ei
        n = h.size(0)
        a_in = torch.zeros_like(h).index_add_(0, dst, self.m_in(torch.cat([h[src], h[dst], ea], 1)))
        a_out = torch.zeros_like(h).index_add_(0, src, self.m_out(torch.cat([h[dst], h[src], ea], 1)))
        d_in = torch.bincount(dst, minlength=n).clamp(min=1).unsqueeze(1).float()
        d_out = torch.bincount(src, minlength=n).clamp(min=1).unsqueeze(1).float()
        return h + self.upd(torch.cat([h, a_in / d_in, a_out / d_out], 1))


class EdgeGNN(nn.Module):
    def __init__(self, node_in, edge_in, h=64, temporal=False):
        super().__init__()
        self.temporal = temporal
        self.enc = nn.Sequential(nn.Linear(node_in, h), nn.ReLU())
        self.mp1, self.mp2 = MPLayer(h, edge_in), MPLayer(h, edge_in)
        if temporal:
            self.gru = nn.GRUCell(h, h)
        self.eenc = nn.Sequential(nn.Linear(edge_in, h), nn.ReLU())
        self.cls = nn.Sequential(nn.Linear(3 * h, h), nn.ReLU(), nn.Linear(h, 1))

    def forward(self, g, state):
        ei, ea = g["edge_index"], g["edge_attr"]
        h = self.mp2(self.mp1(self.enc(g["x"]), ei, ea), ei, ea)
        if self.temporal:
            state = self.gru(h, state)      # memory carried across windows
            z = state
        else:
            z = h
        logit = self.cls(torch.cat([z[ei[0]], z[ei[1]], self.eenc(ea)], 1)).squeeze(1)
        return logit, state


@torch.no_grad()
def predict(model, warm, snaps, n, H, dev):
    """Run warm-up snapshots (to build state), then collect predictions on snaps."""
    model.eval()
    state = torch.zeros(n, H, device=dev)
    if model.temporal:
        for g in warm:
            _, state = model(g, state)
    P, Y, C = [], [], []
    for g in snaps:
        logit, state = model(g, state)
        P.append(torch.sigmoid(logit).cpu())
        Y.append(g["y"].cpu())
        C.append(g["cat"].cpu())
    return (torch.cat(P).numpy(), torch.cat(Y).numpy(), torch.cat(C).numpy())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["static", "temporal"], required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--chunk", type=int, default=16)   # snapshots per optimizer step (BPTT length)
    ap.add_argument("--warm", type=int, default=100)   # train snapshots used to warm up state at eval
    a = ap.parse_args()

    cfg = load_config()
    seed, H, lr = cfg["seed"], cfg["model"]["hidden_dim"], cfg["model"]["lr"]
    torch.manual_seed(seed)
    np.random.seed(seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("Device:", dev)

    blob = torch.load("data/processed/snapshots.pt", weights_only=False)
    snaps = sorted(blob["snapshots"], key=lambda g: g["t"])
    meta = blob["meta"]
    tr = [g for g in snaps if g["split"] == "train"]
    va = [g for g in snaps if g["split"] == "val"]
    te = [g for g in snaps if g["split"] == "test"]

    # Normalise node features with TRAIN statistics only
    allx = torch.cat([g["x"] for g in tr])
    mu, sd = allx.mean(0), allx.std(0) + 1e-6
    for g in snaps:
        g["x"] = (g["x"] - mu) / sd
        for k in ("x", "edge_index", "edge_attr", "y", "cat"):
            g[k] = g[k].to(dev)

    n, node_in, edge_in = tr[0]["x"].shape[0], tr[0]["x"].shape[1], tr[0]["edge_attr"].shape[1]
    model = EdgeGNN(node_in, edge_in, H, temporal=(a.model == "temporal")).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)

    pos = float(sum(g["y"].sum() for g in tr)) / float(sum(g["y"].numel() for g in tr))
    pw = torch.tensor(((1 - pos) / pos) ** 0.5, device=dev)   # softened class weight
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw)
    print(f"Model: {a.model} | params: {sum(p.numel() for p in model.parameters()):,} | pos_weight {pw:.2f}")

    best_f1, best_sd = -1, None
    for ep in range(1, a.epochs + 1):
        model.train()
        t0, tot = time.time(), 0.0
        state = torch.zeros(n, H, device=dev)
        for i in range(0, len(tr), a.chunk):
            state = state.detach()
            loss = 0
            for g in tr[i:i + a.chunk]:
                logit, state = model(g, state)
                loss = loss + lossf(logit, g["y"].float())
            loss = loss / len(tr[i:i + a.chunk])
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item()
        p, y, _ = predict(model, tr[-a.warm:], va, n, H, dev)
        m = evaluate(y, p)
        print(f"ep {ep:02d} | loss {tot / (len(tr) / a.chunk):.4f} | val F1 {m['f1']:.4f} "
              f"P {m['precision']:.3f} R {m['recall']:.3f} | {time.time() - t0:.0f}s")
        if m["f1"] > best_f1:
            best_f1, best_sd = m["f1"], copy.deepcopy(model.state_dict())

    # Final evaluation with the best-on-val weights
    name = "static_gnn" if a.model == "static" else "tgnn"
    trained_now = best_sd is not None
    if best_sd is None:   # --epochs 0: reuse the weights saved earlier
        best_sd = torch.load(f"results/{name}.pth", map_location=dev)
    model.load_state_dict(best_sd)
    pv, yv, _ = predict(model, tr[-a.warm:], va, n, H, dev)
    mv = evaluate(yv, pv)
    p, y, c = predict(model, (tr[-a.warm:] if a.model == "temporal" else []) + va, te, n, H, dev)
    mt = evaluate(y, p)
    Path("results/preds").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f"results/preds/{name}_val.npz", y=yv, p=pv)
    np.savez_compressed(f"results/preds/{name}_test.npz", y=y, p=p)

    print("\nBEST", name)
    print("val ", {k: round(float(v), 4) for k, v in mv.items()})
    print("test", {k: round(float(v), 4) for k, v in mt.items()})
    print("\nTest recall per attack category:")
    for i, cn in enumerate(meta["cat_names"]):
        mask = (c == i) & (y == 1)
        if cn != "Normal" and mask.sum() > 0:
            print(f"  {cn:15s} n={int(mask.sum()):6d}  recall={(p[mask] >= 0.5).mean():.3f}")
    if trained_now:
        for sp, m in [("val", mv), ("test", mt)]:
            log_result({"time": datetime.now().isoformat(timespec="seconds"),
                        "model": name, "split": sp, "seed": seed, **m})
        torch.save(best_sd, f"results/{name}.pth")


if __name__ == "__main__":
    main()