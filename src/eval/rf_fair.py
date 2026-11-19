"""Fair RF baselines under feature-space evasion.
rf_hard: trained only on features the attacker cannot forge (constant under feature forgery).
rf_ctx : all edge features plus source/destination host features (graph context)."""
import numpy as np
import torch
from sklearn.ensemble import RandomForestClassifier

from src.attacker.perturb import (HELDOUT, LEVELS, CONTROLLABLE, SRC_EXTRA,
                                  add_padding, build_groups, perturb_flows)
from src.eval.evasion import inv, node_x
from src.eval.threshold import best_threshold
from src.features.load_data import load_config
from src.models.rf import evaluate

cfg = load_config()
seed = cfg["seed"]
blob = torch.load("data/processed/snapshots.pt", weights_only=False)
snaps = sorted(blob["snapshots"], key=lambda g: g["t"])
meta = blob["meta"]
feats, mu, sd = meta["edge_feats"], meta["mu"], meta["sd"]
col = {c: i for i, c in enumerate(feats)}
tr = [g for g in snaps if g["split"] == "train"]
va = [g for g in snaps if g["split"] == "val"]
te = [g for g in snaps if g["split"] == "test"]
n = tr[0]["x"].shape[0]
ib, idb = col["sbytes"], col["dbytes"]

HARD = [f for f in feats if f not in CONTROLLABLE + SRC_EXTRA]
hard_idx = [col[f] for f in HARD]
print(f"hardened RF uses {len(HARD)} features: {HARD}\n")


def rows(Z, ei, x):
    return np.hstack([Z, x[ei[0]], x[ei[1]]]).astype(np.float32)


def stack(gs):
    X = np.concatenate([rows(g["edge_attr"].numpy(), g["edge_index"].numpy(), g["x"].numpy()) for g in gs])
    y = np.concatenate([g["y"].numpy() for g in gs])
    return X, y


Xtr, ytr = stack(tr)
Xva, yva = stack(va)
models = {}
for name, cols in [("rf_hard", hard_idx), ("rf_ctx", list(range(Xtr.shape[1])))]:
    rf = RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=-1, random_state=seed)
    rf.fit(Xtr[:, cols], ytr)
    thr = best_threshold(yva, rf.predict_proba(Xva[:, cols])[:, 1])
    models[name] = (rf, cols, thr)
    print(f"trained {name} (threshold {thr:.3f})")

rng0 = np.random.default_rng(seed)
ref = []
for g in tr:
    Zg, yg = g["edge_attr"].numpy(), g["y"].numpy()
    b = np.where(yg == 0)[0]
    if len(b):
        ref.append(Zg[rng0.choice(b, min(100, len(b)), replace=False)])
ref = np.concatenate(ref)
groups = build_groups(ref, col)

res = {m: {} for m in models}
for level, lc in {**LEVELS, **HELDOUT}.items():
    rng = np.random.default_rng(seed)
    parts, ys = [], []
    for g in te:
        Z, y = g["edge_attr"].numpy(), g["y"].numpy()
        ei, cat = g["edge_index"].numpy(), g["cat"].numpy()
        if not lc:
            x, Z2, ei2 = g["x"].numpy(), Z, ei
        else:
            Z = perturb_flows(Z, y, lc, ref, col, rng, groups)
            if lc.get("pad"):
                ei2, Z2, y2, _ = add_padding(ei, Z, y, cat, lc["pad"], rng)
            else:
                ei2, Z2, y2 = ei, Z, y
            raw = inv(Z2[:, [ib, idb]].astype(np.float64) * sd[[ib, idb]] + mu[[ib, idb]])
            x = node_x(ei2, np.clip(raw[:, 0], 0, None), np.clip(raw[:, 1], 0, None), n)
        k = len(y)
        parts.append(rows(Z2[:k], ei2[:, :k], x))
        ys.append(y)
    X, yy = np.concatenate(parts), np.concatenate(ys)
    for name, (rf, cols, thr) in models.items():
        m = evaluate(yy, rf.predict_proba(X[:, cols])[:, 1], thr)
        res[name][level] = m
    print("done:", level)

levels = list(LEVELS) + list(HELDOUT)
for metric in ["recall", "f1"]:
    print(f"\n{metric.upper()}")
    print(f"{'model':10s} " + " ".join(f"{l[:11]:>11s}" for l in levels))
    for name in models:
        print(f"{name:10s} " + " ".join(f"{res[name][l][metric]:11.4f}" for l in levels))
print("\nFPR (clean):", {k: round(float(res[k]['clean']['fpr']), 4) for k in models})