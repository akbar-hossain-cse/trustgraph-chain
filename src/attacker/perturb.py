"""Feature-space evasion emulation (lab use only).

Only edits feature rows of existing synthetic benchmark flows. No traffic is
generated and no attack tooling is produced. Works in the standardised
signed-log space used by the graph snapshots.
"""
import numpy as np

TIMING = ["dur", "Sload", "Dload", "Sjit", "Djit", "Sintpkt", "Dintpkt"]
VOLUME = ["sbytes", "dbytes", "Spkts", "Dpkts", "smeansz", "dmeansz"]
CONTROLLABLE = TIMING + VOLUME

LEVELS = {
    "clean": {},
    "L1_naive": {"noise": 0.5, "cols": CONTROLLABLE},
    "L2_timing": {"alpha": 0.75, "cols": TIMING},
    "L3_detector": {"alpha": 0.75, "cols": CONTROLLABLE, "pad": 0.5},
    "sweep_a0.25": {"alpha": 0.25, "cols": CONTROLLABLE},
    "sweep_a0.50": {"alpha": 0.50, "cols": CONTROLLABLE},
    "sweep_a0.75": {"alpha": 0.75, "cols": CONTROLLABLE},
}


def perturb_flows(Z, y, lc, ref, col, rng):
    """Modify ATTACK rows only. Z: (edges, feats) float32. ref: benign train rows."""
    Z = Z.copy()
    idx = np.where(y == 1)[0]
    if len(idx) == 0:
        return Z
    cols = [col[c] for c in lc["cols"]]
    sel = np.ix_(idx, cols)
    if "noise" in lc:
        Z[sel] += rng.normal(0, lc["noise"], (len(idx), len(cols))).astype(np.float32)
    if "alpha" in lc:
        a = lc["alpha"]
        pick = ref[rng.integers(0, len(ref), len(idx))]      # random benign flow per attack flow
        Z[sel] = (1 - a) * Z[sel] + a * pick[:, cols]
    return Z


def add_padding(ei, Z, y, cat, frac, rng):
    """Add benign-looking extra connections on attacker->victim pairs.
    Padded edges are appended AFTER the original edges."""
    ia, ib = np.where(y == 1)[0], np.where(y == 0)[0]
    k = int(frac * len(ia))
    if k == 0 or len(ib) == 0:
        return ei, Z, y, cat
    a = ia[rng.integers(0, len(ia), k)]
    b = ib[rng.integers(0, len(ib), k)]
    return (np.concatenate([ei, ei[:, a]], axis=1),
            np.concatenate([Z, Z[b]]),
            np.concatenate([y, np.ones(k, dtype=y.dtype)]),
            np.concatenate([cat, cat[a]]))