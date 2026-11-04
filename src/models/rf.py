"""Random Forest baseline on UNSW-NB15 flow features (time-based split)."""
import csv
import time
from datetime import datetime
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (confusion_matrix, f1_score, precision_score,
                             recall_score, roc_auc_score)

from src.features.load_data import (CATEGORICAL, COLUMNS, ID_COLS,
                                    load_config, load_raw, time_split)

# Excluded: IDs, timestamps, ports (dataset-specific shortcuts),
# and ct_* counters (derived from surrounding connections -> leakage risk).
EXCLUDE = set(ID_COLS + ["Label", "Stime", "Ltime", "sport", "dsport"]
              + [c for c in COLUMNS if c.startswith("ct_")])
NUMERIC = [c for c in COLUMNS if c not in EXCLUDE and c not in CATEGORICAL]


def encode(train, val, test):
    """Ordinal-encode categoricals using TRAIN categories only (unseen -> -1)."""
    train, val, test = train.copy(), val.copy(), test.copy()
    for c in CATEGORICAL:
        cats = {v: i for i, v in enumerate(sorted(train[c].unique()))}
        for d in (train, val, test):
            d[c + "_enc"] = d[c].map(cats).fillna(-1).astype(int)
    feats = NUMERIC + [c + "_enc" for c in CATEGORICAL]
    return train, val, test, feats


def evaluate(y_true, y_prob, thr=0.5):
    y_pred = (y_prob >= thr).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    return {
        "precision": precision_score(y_true, y_pred, zero_division=0),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "auc": roc_auc_score(y_true, y_prob),
        "fpr": fp / (fp + tn),
    }


def log_result(row, path="results/metrics.csv"):
    Path(path).parent.mkdir(exist_ok=True)
    new = not Path(path).exists()
    with open(path, "a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=row.keys())
        if new:
            w.writeheader()
        w.writerow(row)


if __name__ == "__main__":
    cfg = load_config()
    seed = cfg["seed"]
    df = load_raw()
    train, val, test = time_split(df, cfg["data"]["train_ratio"],
                                  cfg["data"]["val_ratio"])
    train, val, test, feats = encode(train, val, test)
    print(f"Using {len(feats)} features: {feats}")

    t0 = time.time()
    rf = RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=-1,
                                random_state=seed)
    rf.fit(train[feats], train["Label"])
    print(f"Trained in {time.time() - t0:.0f}s")

    Path("results/preds").mkdir(parents=True, exist_ok=True)
    for name, part in [("val", val), ("test", test)]:
        prob = rf.predict_proba(part[feats])[:, 1]
        m = evaluate(part["Label"].values, prob)
        print(name, {k: round(float(v), 4) for k, v in m.items()})
        log_result({"time": datetime.now().isoformat(timespec="seconds"),
                    "model": "rf", "split": name, "seed": seed, **m})
        np.savez_compressed(f"results/preds/{'rf'}_{name}.npz",
                            y=part["Label"].values, p=prob.astype(np.float32))
    joblib.dump(rf, "results/rf.joblib")   # needed later for the evasion tests