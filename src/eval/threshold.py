"""Pick the best-F1 threshold on VAL for each model, then report on TEST."""
from datetime import datetime

import numpy as np
from sklearn.metrics import precision_recall_curve

from src.features.load_data import load_config
from src.models.rf import evaluate, log_result

MODELS = ["rf", "static_gnn", "tgnn"]


def best_threshold(y, p):
    prec, rec, thr = precision_recall_curve(y, p)
    f1 = 2 * prec[:-1] * rec[:-1] / (prec[:-1] + rec[:-1] + 1e-12)
    return float(thr[np.argmax(f1)])


if __name__ == "__main__":
    seed = load_config()["seed"]
    print(f"{'model':12s} {'thr':>6s} {'prec':>7s} {'recall':>7s} {'f1':>7s} {'auc':>7s} {'fpr':>7s}")
    for name in MODELS:
        v = np.load(f"results/preds/{name}_val.npz")
        t = np.load(f"results/preds/{name}_test.npz")
        thr = best_threshold(v["y"], v["p"])        # chosen on VAL only
        m = evaluate(t["y"], t["p"], thr)           # applied to TEST
        print(f"{name:12s} {thr:6.3f} {m['precision']:7.4f} {m['recall']:7.4f} "
              f"{m['f1']:7.4f} {m['auc']:7.4f} {float(m['fpr']):7.4f}")
        log_result({"time": datetime.now().isoformat(timespec="seconds"),
                    "model": name + "_tuned", "split": "test", "seed": seed, **m})