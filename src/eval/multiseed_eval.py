"""Evaluate all configurations over several seeds under feature-space evasion.
Attacker is fixed (seed from config); only model seeds vary. Threshold: each model's own clean val."""
import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from scipy import stats

from src.attacker.perturb import (CONTROLLABLE, HELDOUT, LEVELS, add_padding,
                                  build_groups, perturb_flows)
from src.eval.evasion import inv, node_x
from src.eval.threshold import best_threshold
from src.features.load_data import load_config
from src.models.gnn import EdgeGNN, predict
from src.models.rf import evaluate

CONFIGS = {
    "static": ("static_gnn", False, 2),
    "temporal": ("tgnn", True, 2),
    "static_L0": ("static_gnn_L0", False, 0),
    "temporal_L0": ("tgnn_L0", True, 0),
    "static_adv": ("static_gnn_adv", False, 2),
    "temporal_adv": ("tgnn_adv", True, 2),
}
SHOW = ["clean", "L3_detector", "H3_matched_pad", "S1_src16_a0.50", "S2_src16_a1.00"]
PAIRS = [("temporal", "static"), ("static", "static_L0"), ("temporal", "temporal_L0"),
         ("temporal_L0", "static_L0"),
         ("static_adv", "static"), ("temporal_adv", "temporal")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[42, 1, 2, 3, 4])
    ap.add_argument("--no-rf", action="store_true")
    a = ap.parse_args()

    cfg = load_config()
    base_seed, H = cfg["seed"], cfg["model"]["hidden_dim"]
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

    entries = []
    for cname, (base, temporal, layers) in CONFIGS.items():
        for s in a.seeds:
            stem = base if s == base_seed else f"{base}_s{s}"
            if not (Path(f"results/{stem}.pth").exists() and Path(f"results/preds/{stem}_val.npz").exists()):
                continue
            m = EdgeGNN(6, len(feats), H, temporal=temporal, layers=layers).to(dev)
            m.load_state_dict(torch.load(f"results/{stem}.pth", map_location=dev))
            m.eval()
            v = np.load(f"results/preds/{stem}_val.npz")
            entries.append({"cfg": cname, "seed": s, "net": m, "thr": best_threshold(v["y"], v["p"])})
    if not a.no_rf:
        for s in a.seeds:
            stem = "rf" if s == base_seed else f"rf_s{s}"
            if not (Path(f"results/{stem}.joblib").exists() and Path(f"results/preds/{stem}_val.npz").exists()):
                continue
            v = np.load(f"results/preds/{stem}_val.npz")
            entries.append({"cfg": "rf", "seed": s, "rf": joblib.load(f"results/{stem}.joblib"),
                            "thr": best_threshold(v["y"], v["p"])})
    counts = pd.Series([e["cfg"] for e in entries]).value_counts()
    print("models found per config:", dict(counts))
    has_rf = any("rf" in e for e in entries)

    rng0 = np.random.default_rng(base_seed)
    ref = []
    for g in tr:
        Zg, yg = g["edge_attr"].numpy(), g["y"].numpy()
        b = np.where(yg == 0)[0]
        if len(b):
            ref.append(Zg[rng0.choice(b, min(100, len(b)), replace=False)])
    ref = np.concatenate(ref)
    groups = build_groups(ref, col)

    rows = []
    for level, lc in {**LEVELS, **HELDOUT}.items():
        rng = np.random.default_rng(base_seed)
        snaps_dev, Zrf, y_rf, orig = [], [], [], []
        for g in te:
            Z, y = g["edge_attr"].numpy(), g["y"].numpy()
            ei, cat = g["edge_index"].numpy(), g["cat"].numpy()
            if not lc:
                x, Z2, ei2, y2, cat2 = g["x"], Z, ei, y, cat
            else:
                Z = perturb_flows(Z, y, lc, ref, col, rng, groups)
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
            o[:len(y)] = True
            orig.append(o)
        y_o, mask = np.concatenate(y_rf), np.concatenate(orig)
        if has_rf:
            R = inv(np.concatenate(Zrf).astype(np.float64) * sd + mu)
            R[:, ctrl_idx] = np.clip(R[:, ctrl_idx], 0, None)
            dfR = pd.DataFrame(R, columns=feats)
        for e in entries:
            if "rf" in e:
                p = e["rf"].predict_proba(dfR)[:, 1]
            else:
                p, _, _ = predict(e["net"], warm if e["net"].temporal else [], snaps_dev, n, H, dev)
                p = p[mask]
            met = evaluate(y_o, p, e["thr"])
            rows.append({"cfg": e["cfg"], "seed": e["seed"], "level": level,
                         **{k: float(v) for k, v in met.items()}})
        print("done:", level, flush=True)

    df = pd.DataFrame(rows)
    df.to_csv("results/multiseed_results.csv", index=False)
    order = [c for c in ["rf"] + list(CONFIGS) if c in set(df.cfg)]
    for metric in ["recall", "f1"]:
        print(f"\n{metric.upper()}  (mean ± std over seeds)")
        print(f"{'model':13s}{'n':>2s}  " + "".join(f"{l[:14]:>16s}" for l in SHOW))
        for c in order:
            sub = df[df.cfg == c]
            cells = ""
            for l in SHOW:
                v = sub[sub.level == l][metric].to_numpy()
                sdv = v.std(ddof=1) if len(v) > 1 else float("nan")
                cells += f"{v.mean():9.4f}±{sdv:5.3f} "
            print(f"{c:13s}{sub.seed.nunique():2d}  {cells}")

    tests = []
    for A, B in PAIRS:
        for l in SHOW:
            fa = df[(df.cfg == A) & (df.level == l)].set_index("seed").f1
            fb = df[(df.cfg == B) & (df.level == l)].set_index("seed").f1
            common = fa.index.intersection(fb.index)
            if len(common) < 3:
                continue
            d = fa[common] - fb[common]
            _, p = stats.ttest_rel(fa[common], fb[common])
            tests.append((A, B, l, d.mean(), p, len(common)))
    bonf = 0.05 / max(len(tests), 1)
    print(f"\nPaired F1 difference (A minus B), paired t-test across seeds")
    print(f"{len(tests)} tests run; Bonferroni threshold p < {bonf:.4f} (marked *)")
    for A, B, l, dm, p, k in tests:
        star = "*" if p < bonf else " "
        print(f"{A:12s} - {B:12s} | {l:15s} diff {dm:+.4f}  p={p:.3f} {star} n={k}")

    print("\nFPR on clean data (mean ± std over seeds)")
    for c in order:
        v = df[(df.cfg == c) & (df.level == "clean")].fpr.to_numpy()
        sdv = v.std(ddof=1) if len(v) > 1 else float("nan")
        print(f"{c:13s} {v.mean():.4f} ± {sdv:.4f}")


if __name__ == "__main__":
    main()