"""Load UNSW-NB15, clean it, and make a time-based train/val/test split."""
from pathlib import Path

import pandas as pd
import yaml

RAW_DIR = Path("data/raw")

# The 4 raw CSV files have NO header, so we define the 49 column names here.
COLUMNS = [
    "srcip", "sport", "dstip", "dsport", "proto", "state", "dur", "sbytes",
    "dbytes", "sttl", "dttl", "sloss", "dloss", "service", "Sload", "Dload",
    "Spkts", "Dpkts", "swin", "dwin", "stcpb", "dtcpb", "smeansz", "dmeansz",
    "trans_depth", "res_bdy_len", "Sjit", "Djit", "Stime", "Ltime",
    "Sintpkt", "Dintpkt", "tcprtt", "synack", "ackdat", "is_sm_ips_ports",
    "ct_state_ttl", "ct_flw_http_mthd", "is_ftp_login", "ct_ftp_cmd",
    "ct_srv_src", "ct_srv_dst", "ct_dst_ltm", "ct_src_ltm",
    "ct_src_dport_ltm", "ct_dst_sport_ltm", "ct_dst_src_ltm",
    "attack_cat", "Label",
]
CATEGORICAL = ["proto", "state", "service"]
ID_COLS = ["srcip", "dstip", "attack_cat"]


def load_config(path="configs/base.yaml"):
    with open(path) as f:
        return yaml.safe_load(f)


def load_raw(raw_dir=RAW_DIR, nrows_per_file=None):
    """Read UNSW-NB15_1..4.csv, clean types, sort by start time."""
    files = sorted(Path(raw_dir).glob("UNSW-NB15_[1-4].csv"))
    if not files:
        raise FileNotFoundError(f"No UNSW-NB15_*.csv found in {raw_dir}")

    parts = []
    for f in files:
        print(f"Reading {f.name} ...")
        parts.append(pd.read_csv(f, header=None, names=COLUMNS,
                                 low_memory=False, nrows=nrows_per_file))
    df = pd.concat(parts, ignore_index=True)

    # IPs and categorical columns: strings, stripped
    for c in ["srcip", "dstip"] + CATEGORICAL:
        df[c] = df[c].astype(str).str.strip()
    df["service"] = df["service"].replace("-", "none")

    # attack_cat: empty means normal traffic
    df["attack_cat"] = df["attack_cat"].fillna("Normal").astype(str).str.strip()
    df["attack_cat"] = df["attack_cat"].replace({"Backdoors": "Backdoor"})

    # Everything else numeric (ports can contain hex values -> NaN -> -1)
    num_cols = [c for c in COLUMNS if c not in ID_COLS + CATEGORICAL]
    for c in num_cols:
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df[num_cols] = df[num_cols].fillna(-1)
    df["Label"] = df["Label"].astype(int)

    # Timeline: sort by flow start time (needed for time-based split + graph)
    df = df.sort_values("Stime", kind="stable").reset_index(drop=True)
    return df


def time_split(df, train_ratio=0.7, val_ratio=0.1):
    """Split by time order (no shuffling) so test is strictly in the future."""
    n = len(df)
    t_end = int(n * train_ratio)
    v_end = int(n * (train_ratio + val_ratio))
    return df.iloc[:t_end], df.iloc[t_end:v_end], df.iloc[v_end:]


if __name__ == "__main__":
    cfg = load_config()
    df = load_raw()
    train, val, test = time_split(df, cfg["data"]["train_ratio"],
                                  cfg["data"]["val_ratio"])

    print(f"\nTotal flows: {len(df):,}")
    print(f"Unique hosts: {pd.concat([df.srcip, df.dstip]).nunique():,}")
    print("\nAttack categories:\n", df["attack_cat"].value_counts())
    for name, part in [("train", train), ("val", val), ("test", test)]:
        t0 = pd.to_datetime(part.Stime.min(), unit="s")
        t1 = pd.to_datetime(part.Stime.max(), unit="s")
        print(f"\n{name}: {len(part):,} flows | attack ratio "
              f"{part.Label.mean():.3f} | {t0} -> {t1}")