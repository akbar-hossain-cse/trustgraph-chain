"""Train every configuration for several seeds. Resumable: finished runs are skipped.
Weights go to results/{name}_s{seed}.pth; logs to results/logs/."""
import argparse
import os
import subprocess
import sys
from pathlib import Path

import joblib
import numpy as np
from sklearn.ensemble import RandomForestClassifier

from src.features.load_data import load_config, load_raw, time_split
from src.models.rf import encode

GNN = [
    ("static_gnn", ["-m", "src.models.gnn", "--model", "static"]),
    ("tgnn", ["-m", "src.models.gnn", "--model", "temporal"]),
    ("static_gnn_L0", ["-m", "src.models.gnn", "--model", "static", "--layers", "0"]),
    ("tgnn_L0", ["-m", "src.models.gnn", "--model", "temporal", "--layers", "0"]),
]
ADV = [
    ("static_gnn_adv", ["-m", "src.models.gnn_adv", "--model", "static"]),
    ("tgnn_adv", ["-m", "src.models.gnn_adv", "--model", "temporal"]),
]


def run_gnn(seeds, runs):
    Path("results/logs").mkdir(parents=True, exist_ok=True)
    for s in seeds:
        env = os.environ.copy()
        env["TG_SEED"], env["TG_TAG"] = str(s), f"_s{s}"
        for stem, argv in runs:
            if Path(f"results/{stem}_s{s}.pth").exists():
                print(f"skip {stem} seed {s} (done)")
                continue
            log = Path(f"results/logs/{stem}_s{s}.log")
            print(f"training {stem} seed {s} ...", flush=True)
            with open(log, "w") as f:
                subprocess.run([sys.executable] + argv + ["--epochs", "20"],
                               env=env, stdout=f, stderr=subprocess.STDOUT, check=True)
            line = [l for l in log.read_text().splitlines() if l.startswith("test")]
            print("   ", line[-1] if line else "(no test line)", flush=True)


def run_rf(seeds):
    todo = [s for s in seeds if not Path(f"results/rf_s{s}.joblib").exists()]
    if not todo:
        return
    cfg = load_config()
    tr, va, te = time_split(load_raw(), cfg["data"]["train_ratio"], cfg["data"]["val_ratio"])
    tr, va, te, feats = encode(tr, va, te)
    for s in todo:
        rf = RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=-1, random_state=s)
        rf.fit(tr[feats], tr["Label"])
        for name, part in [("val", va), ("test", te)]:
            np.savez_compressed(f"results/preds/rf_s{s}_{name}.npz", y=part["Label"].values,
                                p=rf.predict_proba(part[feats])[:, 1].astype(np.float32))
        joblib.dump(rf, f"results/rf_s{s}.joblib")
        print(f"trained rf seed {s}", flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, nargs="+", default=[1, 2, 3, 4])
    ap.add_argument("--adv", action="store_true", help="also train adversarial models (slow)")
    ap.add_argument("--skip-rf", action="store_true")
    a = ap.parse_args()
    run_gnn(a.seeds, GNN + (ADV if a.adv else []))
    if not a.skip_rf:
        run_rf(a.seeds)
    print("all done")