"""Adversarial (evasion-aware) training for the static / temporal edge GNN.

Train-set attack flows are perturbed with the same feature-space emulator used
for testing. Test data is never used to build the augmentation.
"""
import os
import argparse
import copy
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from src.attacker.perturb import LEVELS, add_padding, perturb_flows
from src.eval.evasion import inv, node_x
from src.features.load_data import load_config
from src.models.gnn import EdgeGNN, predict
from src.models.rf import evaluate, log_result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", choices=["static", "temporal"], required=True)
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--chunk", type=int, default=16)
    ap.add_argument("--warm", type=int, default=100)
    ap.add_argument("--p_clean", type=float, default=0.3)   # share of clean chunks
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
    feats, mu, sd = meta["edge_feats"], meta["mu"], meta["sd"]
    col = {c: i for i, c in enumerate(feats)}
    ib, idb = col["sbytes"], col["dbytes"]
    tr = [g for g in snaps if g["split"] == "train"]
    va = [g for g in snaps if g["split"] == "val"]
    te = [g for g in snaps if g["split"] == "test"]
    n, edge_in = tr[0]["x"].shape[0], len(feats)

    allx = torch.cat([g["x"] for g in tr])
    xmu, xsd = allx.mean(0), allx.std(0) + 1e-6

    def to_dev(x, ei, Z, y, cat, w=None):
        if w is None:
            w = np.ones(len(y), dtype=np.float32)
        return {"x": ((torch.as_tensor(x) - xmu) / xsd).to(dev),
                "edge_index": torch.as_tensor(ei).long().to(dev),
                "edge_attr": torch.as_tensor(Z).float().to(dev),
                "y": torch.as_tensor(y).long().to(dev),
                "cat": torch.as_tensor(cat).to(dev),
                "w": torch.as_tensor(w).float().to(dev)}

    def clean(g):
        return to_dev(g["x"], g["edge_index"], g["edge_attr"], g["y"], g["cat"])

    # attacker's imitation target: benign TRAIN flows only
    rng0 = np.random.default_rng(seed)
    ref = []
    for g in tr:
        Zg, yg = g["edge_attr"].numpy(), g["y"].numpy()
        b = np.where(yg == 0)[0]
        if len(b):
            ref.append(Zg[rng0.choice(b, min(100, len(b)), replace=False)])
    ref = np.concatenate(ref)

    def evasive(g, lc, rng):
        Z, y = g["edge_attr"].numpy(), g["y"].numpy()
        ei, cat = g["edge_index"].numpy(), g["cat"].numpy()
        Z = perturb_flows(Z, y, lc, ref, col, rng)
        if lc.get("pad"):
            ei2, Z2, y2, cat2 = add_padding(ei, Z, y, cat, lc["pad"], rng)
        else:
            ei2, Z2, y2, cat2 = ei, Z, y, cat
        raw = inv(Z2[:, [ib, idb]].astype(np.float64) * sd[[ib, idb]] + mu[[ib, idb]])
        x = torch.from_numpy(node_x(ei2, np.clip(raw[:, 0], 0, None),
                                    np.clip(raw[:, 1], 0, None), n))
        w = np.zeros(len(y2), dtype=np.float32)
        w[:len(y)] = 1.0                      # padded edges get zero loss weight
        return to_dev(x, ei2, Z2, y2, cat2, w)

    tr_dev = [clean(g) for g in tr]
    va_dev = [clean(g) for g in va]
    te_dev = [clean(g) for g in te]
    warm_dev = tr_dev[-a.warm:]

    # fixed evasive validation set (strongest level) for model selection
    rv = np.random.default_rng(seed + 999)
    va_ev = [evasive(g, LEVELS["L3_detector"], rv) for g in va]
    mask_ev = torch.cat([g["w"] for g in va_ev]).cpu().numpy() > 0

    temporal = a.model == "temporal"
    name = "tgnn_adv" if temporal else "static_gnn_adv"
    name += os.environ.get("TG_TAG", "")
    model = EdgeGNN(6, edge_in, H, temporal=temporal).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    pos = float(sum(g["y"].sum() for g in tr_dev)) / float(sum(g["y"].numel() for g in tr_dev))
    pw = torch.tensor(((1 - pos) / pos) ** 0.5, device=dev)
    lossf = nn.BCEWithLogitsLoss(pos_weight=pw, reduction="none")
    levels = [k for k in LEVELS if k != "clean"]
    print(f"Model: {name} | params: {sum(p.numel() for p in model.parameters()):,}")

    best, best_sd = -1.0, None
    for ep in range(1, a.epochs + 1):
        model.train()
        rng = np.random.default_rng(seed + ep)
        t0, tot = time.time(), 0.0
        state = torch.zeros(n, H, device=dev)
        for i in range(0, len(tr), a.chunk):
            state = state.detach()
            lvl = None if rng.random() < a.p_clean else levels[rng.integers(len(levels))]
            idx = range(i, min(i + a.chunk, len(tr)))
            loss = 0
            for j in idx:
                g = tr_dev[j] if lvl is None else evasive(tr[j], LEVELS[lvl], rng)
                logit, state = model(g, state)
                l = lossf(logit, g["y"].float())
                loss = loss + (l * g["w"]).sum() / g["w"].sum()
            loss = loss / len(idx)
            opt.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            opt.step()
            tot += loss.item()
        pc, yc, _ = predict(model, warm_dev, va_dev, n, H, dev)
        f1c = evaluate(yc, pc)["f1"]
        pe, ye, _ = predict(model, warm_dev, va_ev, n, H, dev)
        f1e = evaluate(ye[mask_ev], pe[mask_ev])["f1"]
        score = (f1c + f1e) / 2
        print(f"ep {ep:02d} | loss {tot / (len(tr) / a.chunk):.4f} | val clean F1 {f1c:.4f} "
              f"| val L3 F1 {f1e:.4f} | {time.time() - t0:.0f}s")
        if score > best:
            best, best_sd = score, copy.deepcopy(model.state_dict())

    model.load_state_dict(best_sd)
    pv, yv, _ = predict(model, warm_dev, va_dev, n, H, dev)
    pt, yt, _ = predict(model, (warm_dev if temporal else []) + va_dev, te_dev, n, H, dev)
    Path("results/preds").mkdir(parents=True, exist_ok=True)
    np.savez_compressed(f"results/preds/{name}_val.npz", y=yv, p=pv)
    np.savez_compressed(f"results/preds/{name}_test.npz", y=yt, p=pt)
    torch.save(best_sd, f"results/{name}.pth")
    mv, mt = evaluate(yv, pv), evaluate(yt, pt)
    print("\nBEST", name)
    print("val ", {k: round(float(v), 4) for k, v in mv.items()})
    print("test", {k: round(float(v), 4) for k, v in mt.items()})
    for sp, m in [("val", mv), ("test", mt)]:
        log_result({"time": datetime.now().isoformat(timespec="seconds"),
                    "model": name, "split": sp, "seed": seed, **m})


if __name__ == "__main__":
    main()