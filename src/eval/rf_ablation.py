"""RF with groups of features removed (clean test, threshold tuned on val)."""
from sklearn.ensemble import RandomForestClassifier

from src.attacker.perturb import CONTROLLABLE
from src.eval.threshold import best_threshold
from src.features.load_data import load_config, load_raw, time_split
from src.models.rf import encode, evaluate

cfg = load_config()
seed = cfg["seed"]
tr, va, te = time_split(load_raw(), cfg["data"]["train_ratio"], cfg["data"]["val_ratio"])
tr, va, te, feats = encode(tr, va, te)

TTL = ["sttl", "dttl"]
TCP = ["swin", "dwin", "stcpb", "dtcpb"]
sets = {
    "all features": feats,
    "no TTL": [f for f in feats if f not in TTL],
    "no TTL, no TCP window/seq": [f for f in feats if f not in TTL + TCP],
    "controllable features only": [f for f in feats if f in CONTROLLABLE],
}

print(f"{'setting':32s} {'#feat':>5s} {'prec':>7s} {'recall':>7s} {'f1':>7s} {'auc':>7s} {'fpr':>7s}")
for name, f in sets.items():
    rf = RandomForestClassifier(n_estimators=100, max_depth=20, n_jobs=-1, random_state=seed)
    rf.fit(tr[f], tr["Label"])
    thr = best_threshold(va["Label"].values, rf.predict_proba(va[f])[:, 1])
    m = evaluate(te["Label"].values, rf.predict_proba(te[f])[:, 1], thr)
    print(f"{name:32s} {len(f):5d} {m['precision']:7.4f} {m['recall']:7.4f} "
          f"{m['f1']:7.4f} {m['auc']:7.4f} {float(m['fpr']):7.4f}")