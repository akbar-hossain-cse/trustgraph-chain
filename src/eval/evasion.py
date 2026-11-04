"""Evaluate RF / static GNN / temporal GNN under feature-space evasion.
Thresholds are fixed from CLEAN validation data (defender has not seen the evasion)."""
import joblib
import numpy as np
import pandas as pd
import torch

from src.attacker.perturb import CONTROLLABLE, LEVELS, add_padding, perturb_flows
from src.eval.threshold import best_threshold
from src.features.load_data import load_config
from src.models.gnn import EdgeGNN, predict
from src.models.rf import evaluate

MODELS = ["rf", "static_gnn", "tgnn"]


def inv(v):
    """Inverse of signed log1p."""
    return np.sign(v) * np.expm1(np.abs(v))


def node_x(ei, sb, db, n):
    """Same node features as build_snapshots.py."""
    s, d = ei[0], ei[1]
    pairs = np.unique(s * n + d)
    return np.log1p(np.stack([
        np.bincount(s, minlength=n),
        np.bincount(d, minlength=n),
        np.bincount(s, weights=sb, minlength=n) + np.bincount(d, weights=db, minlength=n),
        np.bincount(s, weights=db, minlength=n) + np.bincount(d, weights=sb, minlength=n),
        np.bincount(pairs // n, minlength=n),
        np.bincount(pairs % n, minlength=n),
    ], axis=1)).astype(np.float32)


def main():
    cfg = load_config()
    seed, H = cfg["seed"], cfg["model"]["hidden_dim"]
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    blob = torch.load("data/processed/snapshots.pt", weights_only=False)
    snaps = sorted(blob["snapshots"], key=lambda g: g["t"])
    meta = blob["meta"]
    feats, mu, sd = meta["edge_feats"], meta["mu"], meta["sd"]
    col = {c: i for i, c in enumerate(feats)}
    ctrl_idx = [col[c] for c in CONTROLLABLE]
    ib, idb = col["sbytes"], col["dbytes"]
    tr = [g for g in snaps if g["split"] == "train"]
    va = [g for g in snaps if g["split"] == "val"]
    te = [g for g in snaps if g["split"] == "test"]
    n = tr[0]["x"].shape[0]

    # node-feature normalisation: TRAIN statistics (same as gnn.py)
    allx = torch.cat([g["x"] for g in tr])
    xmu, xsd = allx.mean(0), allx.std(0) + 1e-6

    def dev_snap(x, ei, Z, y, cat):
        return {"x": ((torch.as_tensor(x) - xmu) / xsd).to(dev),
                "edge_index": torch.as_tensor(ei).long().to(dev),
                "edge_attr": torch.as_tensor(Z).float().to(dev),
                "y": torch.as_tensor(y).long().to(dev),
                "cat": torch.as_tensor(cat).to(dev)}

    warm = [dev_snap(g["x"], g["edge_index"], g["edge_attr"], g["y"], g["cat"])
            for g in tr[-100:] + va]

    # models and CLEAN-val thresholds
    rf = joblib.load("results/rf.joblib")
    nets = {}
    for name, temporal in [("static_gnn", False), ("tgnn", True)]:
        m = EdgeGNN(6, len(feats), H, temporal=temporal).to(dev)
        m.load_state_dict(torch.load(f"results/{name}.pth", map_location=dev))
        m.eval()
        nets[name] = m
    thr = {}
    for name in MODELS:
        v = np.load(f"results/preds/{name}_val.npz")
        thr[name] = best_threshold(v["y"], v["p"])
    print("Thresholds (from clean val):", {k: round(v, 3) for k, v in thr.items()})

    # benign TRAIN flows = the attacker's imitation target
    rng0 = np.random.default_rng(seed)
    ref = []
    for g in tr:
        Zg, yg = g["edge_attr"].numpy(), g["y"].numpy()
        b = np.where(yg == 0)[0]
        if len(b):
            ref.append(Zg[rng0.choice(b, min(100, len(b)), replace=False)])
    ref = np.concatenate(ref)

    rows = []
    for level, lc in LEVELS.items():
        rng = np.random.default_rng(seed)
        snaps_dev, Zrf, y_rf, orig = [], [], [], []
        for g in te:
            Z, y = g["edge_attr"].numpy(), g["y"].numpy()
            ei, cat = g["edge_index"].numpy(), g["cat"].numpy()
            if not lc:
                x, Z2, ei2, y2, cat2 = g["x"], Z, ei, y, cat
            else:
                Z = perturb_flows(Z, y, lc, ref, col, rng)
                if lc.get("pad"):
                    ei2, Z2, y2, cat2 = add_padding(ei, Z, y, cat, lc["pad"], rng)
                else:
                    ei2, Z2, y2, cat2 = ei, Z, y, cat
                raw = inv(Z2[:, [ib, idb]].astype(np.float64) * sd[[ib, idb]] + mu[[ib, idb]])
                x = torch.from_numpy(node_x(ei2, np.clip(raw[:, 0], 0, None),
                                            np.clip(raw[:, 1], 0, None), n))
            snaps_dev.append(dev_snap(x, ei2, Z2, y2, cat2))
            Zrf.append(Z)
            y_rf.append(y)
            o = np.zeros(len(y2), dtype=bool)
            o[:len(y)] = True                      # original edges only
            orig.append(o)

        y_o, mask = np.concatenate(y_rf), np.concatenate(orig)
        R = inv(np.concatenate(Zrf).astype(np.float64) * sd + mu)
        R[:, ctrl_idx] = np.clip(R[:, ctrl_idx], 0, None)
        probs = {"rf": rf.predict_proba(pd.DataFrame(R, columns=feats))[:, 1]}
        for name, m in nets.items():
            p, _, _ = predict(m, warm if m.temporal else [], snaps_dev, n, H, dev)
            probs[name] = p[mask]

        for name in MODELS:
            met = evaluate(y_o, probs[name], thr[name])
            rows.append({"level": level, "model": name,
                         **{k: float(v) for k, v in met.items()}})
        print(f"done: {level}")

    df = pd.DataFrame(rows)
    df.to_csv("results/evasion_results.csv", index=False)
    for metric in ["recall", "f1", "fpr"]:
        print(f"\n{metric.upper()}")
        print(df.pivot(index="model", columns="level", values=metric)
                .reindex(index=MODELS, columns=list(LEVELS)).round(4).to_string())


if __name__ == "__main__":
    main()