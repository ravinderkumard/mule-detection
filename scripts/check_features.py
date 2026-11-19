"""Sanity report for data/processed/features_{train,val,test}.parquet.

Per window: shape, mule count, NaN share (columns with NaN only), infinite values.
Train only: mule vs normal medians ranked by |median gap| / IQR, and a range table.
Exit code 1 if any infinite value is found.
"""
from __future__ import annotations

import sys

import numpy as np
import pandas as pd

from muledetect.paths import PROCESSED

SPLITS = ("train", "val", "test")


def main() -> int:
    tabs = {s: pd.read_parquet(PROCESSED / f"features_{s}.parquet").set_index("account") for s in SPLITS}
    feats = [c for c in tabs["train"].columns if c not in ("is_mule", "mule_role")]
    n_inf_total = 0

    for s, t in tabs.items():
        x = t[feats]
        nan = x.isna().mean()
        nan = nan[nan > 0]
        n_inf = int(np.isinf(x.to_numpy(dtype=float)).sum())
        n_inf_total += n_inf
        print(f"\n== {s}: shape {t.shape}, mules {int(t.is_mule.sum()):,} ({t.is_mule.mean():.3%})")
        print("   columns identical to train:", list(t.columns) == list(tabs["train"].columns))
        print(f"   infinite values: {n_inf}")
        if len(nan):
            print("   NaN share:\n" + nan.map("{:.1%}".format).to_string())
        else:
            print("   NaN share: none")

    t = tabs["train"]
    mule, norm = t[t.is_mule == 1][feats], t[t.is_mule == 0][feats]
    med_m, med_n = mule.median(), norm.median()
    iqr = t[feats].quantile(0.75) - t[feats].quantile(0.25)
    scale = iqr.where(iqr > 0, t[feats].std()).where(lambda v: v > 0, 1.0).fillna(1.0)
    rep = pd.DataFrame({"normal": med_n, "mule": med_m, "score": (med_m - med_n).abs() / scale})
    print("\n== train: mule vs normal medians, ranked by |gap| / IQR")
    print(rep.sort_values("score", ascending=False).map(lambda v: f"{v:.4g}").to_string())

    rng = t[feats].describe(percentiles=[0.01, 0.5, 0.99]).T[["min", "1%", "50%", "99%", "max"]]
    print("\n== train: ranges")
    print(rng.map(lambda v: f"{v:.4g}").to_string())
    return 1 if n_inf_total else 0


if __name__ == "__main__":
    sys.exit(main())
