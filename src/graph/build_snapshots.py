"""Build per-window graph snapshots: edge-level labels, behaviour-only node features."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from src.features.load_data import load_config, load_raw, time_split
from src.models.rf import encode


def signed_log(a):
    return np.sign(a) * np.log1p(np.abs(a))


def build():
    cfg = load_config()
    W = cfg["graph"]["window_seconds"]

    df = load_raw()
    tr, va, te = time_split(df, cfg["data"]["train_ratio"], cfg["data"]["val_ratio"])
    tr, va, te, feats = encode(tr, va, te)
    df = pd.concat([tr, va, te], ignore_index=True)
    df["split"] = ["train"] * len(tr) + ["val"] * len(va) + ["test"] * len(te)

    # Edge features: same flow features as RF, scaled with TRAIN statistics only
    X = signed_log(df[feats].to_numpy(dtype=np.float64))
    is_tr = (df["split"] == "train").to_numpy()
    mu, sd = X[is_tr].mean(0), X[is_tr].std(0) + 1e-6
    X = ((X - mu) / sd).astype(np.float32)

    # IPs are used ONLY to index nodes, never as features
    hosts = sorted(set(df.srcip) | set(df.dstip))
    hid = {h: i for i, h in enumerate(hosts)}
    n = len(hosts)
    src = df.srcip.map(hid).to_numpy()
    dst = df.dstip.map(hid).to_numpy()
    sb = df.sbytes.to_numpy(float)
    db = df.dbytes.to_numpy(float)
    y = df.Label.to_numpy()
    cat_names = sorted(df.attack_cat.unique())
    cat = df.attack_cat.map({c: i for i, c in enumerate(cat_names)}).to_numpy()
    split = df["split"].to_numpy()
    win = (df.Stime // W).astype(np.int64).to_numpy()

    snaps = []
    for w, ix in sorted(df.groupby(win).indices.items()):
        s, d = src[ix], dst[ix]
        pairs = np.unique(s * n + d)
        node_x = np.log1p(np.stack([
            np.bincount(s, minlength=n),                                  # out flows
            np.bincount(d, minlength=n),                                  # in flows
            np.bincount(s, weights=sb[ix], minlength=n)
            + np.bincount(d, weights=db[ix], minlength=n),                # bytes sent
            np.bincount(s, weights=db[ix], minlength=n)
            + np.bincount(d, weights=sb[ix], minlength=n),                # bytes received
            np.bincount(pairs // n, minlength=n),                         # unique out-peers
            np.bincount(pairs % n, minlength=n),                          # unique in-peers
        ], axis=1)).astype(np.float32)

        snaps.append({
            "t": int(w * W),
            "split": split[ix[0]],
            "x": torch.from_numpy(node_x),
            "edge_index": torch.from_numpy(np.stack([s, d])).long(),
            "edge_attr": torch.from_numpy(X[ix]),
            "y": torch.from_numpy(y[ix]).long(),
            "cat": torch.from_numpy(cat[ix]).to(torch.int8),
        })

    meta = {"hosts": hosts, "cat_names": cat_names, "edge_feats": feats,
            "mu": mu, "sd": sd, "window_seconds": W}
    Path("data/processed").mkdir(parents=True, exist_ok=True)
    torch.save({"snapshots": snaps, "meta": meta}, "data/processed/snapshots.pt")

    print(f"Hosts (nodes): {n} | node feats: 6 | edge feats: {len(feats)}")
    for sp in ["train", "val", "test"]:
        S = [g for g in snaps if g["split"] == sp]
        edges = np.array([g["y"].numel() for g in S])
        atk = np.array([int(g["y"].sum() > 0) for g in S])
        atk_edges = sum(int(g["y"].sum()) for g in S)
        print(f"{sp}: {len(S)} snapshots | mean edges {edges.mean():.0f} | "
              f"windows with attack {atk.mean():.2f} | attack edge ratio "
              f"{atk_edges / edges.sum():.3f}")


if __name__ == "__main__":
    build()