"""Sanity-check the downloaded data and cache it as Parquet in data/interim/.

Expected (from the AMLworld paper / Elliptic KDD'19 paper):
  HI-Small : ~5.08M transactions, ~515K accounts, laundering rate ~0.10% (1 in ~981)
  Elliptic : 203,769 txs, 234,355 edges, 49 time steps, 4,545 illicit / 42,019 licit
"""
from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
INTERIM = ROOT / "data" / "interim"
INTERIM.mkdir(parents=True, exist_ok=True)

AML_COLS = [
    "timestamp", "from_bank", "from_account", "to_bank", "to_account",
    "amount_received", "receiving_currency", "amount_paid",
    "payment_currency", "payment_format", "is_laundering",
]


def find(folder: Path, name: str) -> Path:
    hits = list(folder.rglob(name))
    if not hits:
        sys.exit(f"Missing {name} under {folder}. Run scripts/download_data.py first.")
    return hits[0]


def check_aml() -> None:
    print("=== IBM AMLworld HI-Small ===")
    path = find(RAW / "ibm_aml", "HI-Small_Trans.csv")
    df = pd.read_csv(
        path, header=0, names=AML_COLS,
        dtype={"from_bank": "int32", "to_bank": "int32",
               "from_account": "string", "to_account": "string",
               "is_laundering": "int8"},
    )
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y/%m/%d %H:%M")
    # Bank+account is the true account identity in AMLworld
    df["src"] = df["from_bank"].astype(str) + "_" + df["from_account"]
    df["dst"] = df["to_bank"].astype(str) + "_" + df["to_account"]

    n, pos = len(df), int(df["is_laundering"].sum())
    accts = pd.concat([df["src"], df["dst"]]).nunique()
    print(f"rows            : {n:,}")
    print(f"unique accounts : {accts:,}")
    print(f"laundering      : {pos:,}  ({pos / n:.4%}, 1 in {n // max(pos, 1):,})")
    print(f"time range      : {df.timestamp.min()} -> {df.timestamp.max()}")
    print("payment formats :", df.payment_format.value_counts().to_dict())

    out = INTERIM / "hi_small_trans.parquet"
    df.to_parquet(out, index=False)
    print(f"saved -> {out.relative_to(ROOT)}\n")


def check_elliptic() -> None:
    print("=== Elliptic ===")
    base = RAW / "elliptic"
    feats = pd.read_csv(find(base, "elliptic_txs_features.csv"), header=None)
    feats.columns = ["txId", "time_step"] + [f"f{i}" for i in range(feats.shape[1] - 2)]
    classes = pd.read_csv(find(base, "elliptic_txs_classes.csv"))
    edges = pd.read_csv(find(base, "elliptic_txs_edgelist.csv"))

    classes["label"] = classes["class"].map({"1": 1, "2": 0, "unknown": -1}).astype("int8")
    nodes = feats.merge(classes[["txId", "label"]], on="txId", how="left")

    print(f"nodes      : {len(nodes):,}   features: {feats.shape[1] - 2}")
    print(f"edges      : {len(edges):,}")
    print(f"time steps : {nodes.time_step.nunique()}")
    print("labels     :", nodes.label.map({1: "illicit", 0: "licit", -1: "unknown"})
          .value_counts().to_dict())

    nodes.to_parquet(INTERIM / "elliptic_nodes.parquet", index=False)
    edges.to_parquet(INTERIM / "elliptic_edges.parquet", index=False)
    print("saved -> data/interim/elliptic_{nodes,edges}.parquet\n")


if __name__ == "__main__":
    check_aml()
    check_elliptic()
    print("All good. Data is ready for EDA.")
