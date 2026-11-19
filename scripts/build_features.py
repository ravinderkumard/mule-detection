"""Build per-window account feature tables from the HI-Small parquet cache.

Reads  data/interim/hi_small_trans.parquet (from scripts/verify_data.py)
Writes data/processed/features_{train,val,test}.parquet  (index = account, features + labels)
"""
from __future__ import annotations

from muledetect.features import (
    build_split_tables, laundering_rates, load_transactions, mule_rates, temporal_split,
)
from muledetect.paths import PROCESSED, ROOT
from muledetect.schemas import validate_features, validate_transactions


def main() -> None:
    df = validate_transactions(load_transactions(), "hi_small_trans")
    print(f"{len(df):,} transactions")
    print(laundering_rates(temporal_split(df)).to_string(), "\n")

    tables = build_split_tables(df)
    print(mule_rates(tables).to_string(), "\n")

    PROCESSED.mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        out = PROCESSED / f"features_{name}.parquet"
        flat = validate_features(table.reset_index(), f"features_{name}")  # stop before writing
        flat.to_parquet(out, index=False)
        print(f"saved {len(table):,} accounts x {table.shape[1]} cols -> {out.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
