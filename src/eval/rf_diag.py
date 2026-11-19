"""Diagnose why RF recall rises under evasion.
Replace groups of features of TEST attack flows with benign values and measure RF recall."""
import joblib
import numpy as np
import pandas as pd
import torch

from src.attacker.perturb import CONTROLLABLE, perturb_flows
from src.eval.evasion import inv
from src.eval.threshold import best_threshold
from src.features.load_data import load_config

seed = load_config()["seed"]
blob = torch.load("data/processed/snapshots.pt", weights_only=False)
snaps = sorted(blob["snapshots"], key=lambda g: g["t"])
meta = blob["meta"]
feats, mu, sd = meta["edge_feats"], meta["mu"], meta["sd"]
col = {c: i for i, c in enumerate(feats)}
UNCTRL = [f for f in feats if f not in CONTROLLABLE]
tr = [g for g in snaps if g["split"] == "train"]
te = [g for g in snaps if g["split"] == "test"]

rng0 = np.random.default_rng(seed)
ref = []
for g in tr:
    Zg, yg = g["edge_attr"].numpy(), g["y"].numpy()
    b = np.where(yg == 0)[0]
    if len(b):
        ref.append(Zg[rng0.choice(b, min(100, len(b)), replace=False)])
ref = np.concatenate(ref)

Z = np.concatenate([g["edge_attr"].numpy() for g in te])
y = np.concatenate([g["y"].numpy() for g in te])

rf = joblib.load("results/rf.joblib")
v = np.load("results/preds/rf_val.npz")
thr = best_threshold(v["y"], v["p"])


def recall(Zp):
    R = inv(Zp.astype(np.float64) * sd + mu)
    R[:, [col[c] for c in CONTROLLABLE]] = np.clip(R[:, [col[c] for c in CONTROLLABLE]], 0, None)
    p = rf.predict_proba(pd.DataFrame(R, columns=feats))[:, 1]
    return float((p[y == 1] >= thr).mean())


settings = [("clean", None, None)]
for a in [0.25, 0.5, 0.75, 1.0]:
    settings.append((f"controllable features, alpha {a}", CONTROLLABLE, a))
settings.append(("uncontrollable features, alpha 1.0", UNCTRL, 1.0))
settings.append(("ALL features, alpha 1.0 (sanity)", CONTROLLABLE + UNCTRL, 1.0))

print(f"RF threshold {thr:.3f} | attack flows {int((y == 1).sum()):,}\n")
for name, cols, a in settings:
    if cols is None:
        r = recall(Z)
    else:
        Zp = perturb_flows(Z, y, {"alpha": a, "cols": cols}, ref, col,
                           np.random.default_rng(seed))
        r = recall(Zp)
    print(f"{name:42s} RF recall {r:.4f}")