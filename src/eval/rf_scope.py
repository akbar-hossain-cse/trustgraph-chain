"""RF robustness vs how many features the attacker controls.
Mimicry (alpha) on test attack flows; recall at a clean-val threshold."""
import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import RandomForestClassifier

from src.attacker.perturb import CONTROLLABLE, perturb_flows
from src.eval.evasion import inv
from src.eval.threshold import best_threshold
from src.features.load_data import load_config, load_raw, time_split
from src.models.rf import encode

cfg = load_config()
seed = cfg["seed"]
tr, va, te = time_split(load_raw(), cfg["data"]["train_ratio"], cfg["data"]["val_ratio"])
tr, va, te, feats = encode(tr, va, te)
CTRL = [f for f in feats if f in CONTROLLABLE]

rf_all = joblib.load("results/rf.joblib")
rf_ctrl = RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=-1, random_state=seed)
rf_ctrl.fit(tr[CTRL], tr["Label"])
thr_all = best_threshold(va["Label"].values, rf_all.predict_proba(va[feats])[:, 1])
thr_ctrl = best_threshold(va["Label"].values, rf_ctrl.predict_proba(va[CTRL])[:, 1])

blob = torch.load("data/processed/snapshots.pt", weights_only=False)
snaps = sorted(blob["snapshots"], key=lambda g: g["t"])
meta = blob["meta"]
mu, sd = meta["mu"], meta["sd"]
col = {c: i for i, c in enumerate(meta["edge_feats"])}
trs = [g for g in snaps if g["split"] == "train"]
tes = [g for g in snaps if g["split"] == "test"]

rng0 = np.random.default_rng(seed)
ref = []
for g in trs:
    Zg, yg = g["edge_attr"].numpy(), g["y"].numpy()
    b = np.where(yg == 0)[0]
    if len(b):
        ref.append(Zg[rng0.choice(b, min(100, len(b)), replace=False)])
ref = np.concatenate(ref)
Z = np.concatenate([g["edge_attr"].numpy() for g in tes])
y = np.concatenate([g["y"].numpy() for g in tes])
fn = meta["edge_feats"]


def raw_df(Zp):
    R = inv(Zp.astype(np.float64) * sd + mu)
    ci = [col[c] for c in CONTROLLABLE]
    R[:, ci] = np.clip(R[:, ci], 0, None)
    return pd.DataFrame(R, columns=fn)


def recalls(Zp):
    d = raw_df(Zp)
    a = rf_all.predict_proba(d[feats])[:, 1][y == 1] >= thr_all
    c = rf_ctrl.predict_proba(d[CTRL])[:, 1][y == 1] >= thr_ctrl
    return a.mean(), c.mean()


print(f"{'attacker scope / alpha':34s} {'RF all (31)':>12s} {'RF ctrl-only (13)':>18s}")
ra, rc = recalls(Z)
print(f"{'clean':34s} {ra:12.4f} {rc:18.4f}")
for scope, cols in [("controls 13 features", CONTROLLABLE), ("controls all 31 features", fn)]:
    for a in [0.25, 0.5, 0.75, 1.0]:
        Zp = perturb_flows(Z, y, {"alpha": a, "cols": cols}, ref, col, np.random.default_rng(seed))
        ra, rc = recalls(Zp)
        print(f"{scope + ', alpha ' + str(a):34s} {ra:12.4f} {rc:18.4f}")