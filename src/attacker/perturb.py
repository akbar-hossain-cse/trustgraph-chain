"""Feature-space evasion emulation (lab use only).

Only edits feature rows of existing synthetic benchmark flows. No traffic is
generated and no attack tooling is produced. Works in the standardised
signed-log space used by the graph snapshots.
"""
import numpy as np

TIMING = ["dur", "Sload", "Dload", "Sjit", "Djit", "Sintpkt", "Dintpkt"]
VOLUME = ["sbytes", "dbytes", "Spkts", "Dpkts", "smeansz", "dmeansz"]
CONTROLLABLE = TIMING + VOLUME
MATCH_COLS = ["proto_enc", "state_enc", "service_enc"]
SRC_EXTRA = ["sttl", "swin", "stcpb"]

# Used for testing AND adversarial training
LEVELS = {
    "clean": {},
    "L1_naive": {"noise": 0.5, "cols": CONTROLLABLE},
    "L2_timing": {"alpha": 0.75, "cols": TIMING},
    "L3_detector": {"alpha": 0.75, "cols": CONTROLLABLE, "pad": 0.5},
    "sweep_a0.25": {"alpha": 0.25, "cols": CONTROLLABLE},
    "sweep_a0.50": {"alpha": 0.50, "cols": CONTROLLABLE},
    "sweep_a0.75": {"alpha": 0.75, "cols": CONTROLLABLE},
}

# Held-out attacker: never used for training. Mimics a benign "twin" with the
# same protocol, state and service, so the feature combination stays plausible.
HELDOUT = {
    "H1_matched_a0.50": {"alpha": 0.50, "cols": CONTROLLABLE, "match": True},
    "H2_matched_a0.75": {"alpha": 0.75, "cols": CONTROLLABLE, "match": True},
    "H3_matched_pad": {"alpha": 0.75, "cols": CONTROLLABLE, "match": True, "pad": 0.5},
    "S1_src16_a0.50": {"alpha": 0.50, "cols": CONTROLLABLE + SRC_EXTRA},
    "S2_src16_a1.00": {"alpha": 1.00, "cols": CONTROLLABLE + SRC_EXTRA},
}


def build_groups(ref, col):
    """Index benign reference rows by (proto, state, service)."""
    idx = [col[c] for c in MATCH_COLS]
    keys = np.round(ref[:, idx], 4)
    groups = {}
    for i, k in enumerate(map(tuple, keys)):
        groups.setdefault(k, []).append(i)
    return {k: np.array(v) for k, v in groups.items()}


def perturb_flows(Z, y, lc, ref, col, rng, groups=None):
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
        if lc.get("match") and groups is not None:
            mcols = [col[c] for c in MATCH_COLS]
            keys = np.round(Z[np.ix_(idx, mcols)], 4)
            pick = np.empty((len(idx), Z.shape[1]), dtype=Z.dtype)
            for r, k in enumerate(map(tuple, keys)):
                g = groups.get(k)
                j = g[rng.integers(len(g))] if g is not None else rng.integers(len(ref))
                pick[r] = ref[j]
        else:
            pick = ref[rng.integers(0, len(ref), len(idx))]
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