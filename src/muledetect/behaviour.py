"""Day 6 behaviour features: velocity, pass-through, payment mix and fan shape.

``behaviour_features(tx, core)`` returns one row per account, aligned to ``core.index``.
Everything is computed from the one window passed in (no cross-window leakage).

Conventions
  * Event = one (account, transaction) pair. A normal txn is an event for src and for dst;
    a self-transfer (src == dst) is ONE event for that account.
  * Share/count columns are 0 when undefined; hold, gap and diversity columns are NaN when
    undefined (LightGBM handles NaN). No infinities are produced.
  * ``tx`` needs the Day-5 USD columns only through ``core`` (``amount_in`` / ``amount_out``).
  * Name note: core already has ``active_hours`` (= span in hours), so the distinct counts
    here are called ``n_active_hours`` / ``n_active_days``.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

VELOCITY = [
    "active_span_hours", "n_active_days", "n_active_hours", "txns_per_active_day",
    "txns_per_active_hour", "max_txns_in_hour", "burst_ratio", "gap_min_hours", "gap_median_hours",
]
PASSTHROUGH = ["in_out_amount_ratio", "out_after_in_share", "hold_median_hours", "out_within_24h_share"]
MIX_OTHER = ["cross_ccy_share", "round_amount_share", "self_transfers"]
FAN = ["fan_out_diversity", "fan_in_diversity", "fan_in_out_ratio"]


def fmt_col(fmt: str) -> str:
    return "fmt_out_" + fmt.lower().replace(" ", "_")


def _events(tx: pd.DataFrame) -> pd.DataFrame:
    cents = np.round(tx["amount_paid"] * 100).astype("int64")
    flags = pd.DataFrame({
        "timestamp": tx["timestamp"],
        "cross": (tx.payment_currency != tx.receiving_currency).astype("int8"),
        "round": ((tx["amount_paid"] >= 100) & (cents % 10_000 == 0)).astype("int8"),
    })
    not_self = (tx.src != tx.dst).to_numpy()
    return pd.concat([
        flags.assign(account=tx["src"].to_numpy()),
        flags[not_self].assign(account=tx.loc[not_self, "dst"].to_numpy()),
    ], ignore_index=True).sort_values(["account", "timestamp"], kind="stable")


def _velocity(ev: pd.DataFrame) -> pd.DataFrame:
    g = ev.groupby("account", sort=False)
    n = g.size()
    span = (g["timestamp"].max() - g["timestamp"].min()).dt.total_seconds() / 3600

    per_hour = ev.assign(h=ev["timestamp"].dt.floor("h")).groupby(["account", "h"]).size()
    n_hours = per_hour.groupby(level=0).size()
    max_hour = per_hour.groupby(level=0).max()
    n_days = ev.assign(d=ev["timestamp"].dt.floor("D")).groupby("account")["d"].nunique()

    gap = ev.groupby("account")["timestamp"].diff().dt.total_seconds() / 3600  # NaN on first row
    gg = gap.groupby(ev["account"])
    tph = n / n_hours
    return pd.DataFrame({
        "active_span_hours": span, "n_active_days": n_days, "n_active_hours": n_hours,
        "txns_per_active_day": n / n_days, "txns_per_active_hour": tph,
        "max_txns_in_hour": max_hour, "burst_ratio": max_hour / tph,
        "gap_min_hours": gg.min(), "gap_median_hours": gg.median(),   # NaN for 1-event accounts
    })


def _hold_times(tx: pd.DataFrame) -> pd.DataFrame:
    """Per outgoing txn: hours since the account's latest earlier incoming txn (as-of join)."""
    t = tx[tx.src != tx.dst]
    outs = t[["src", "timestamp"]].rename(columns={"src": "account"}).sort_values("timestamp")
    ins = (t[["dst", "timestamp"]].rename(columns={"dst": "account"})
           .assign(t_in=lambda d: d["timestamp"]).sort_values("timestamp"))
    m = pd.merge_asof(outs, ins, on="timestamp", by="account", direction="backward")
    m["hold"] = (m["timestamp"] - m["t_in"]).dt.total_seconds() / 3600
    n_out = m.groupby("account").size()
    return pd.DataFrame({
        "out_after_in_share": m["hold"].notna().groupby(m["account"]).sum() / n_out,
        "hold_median_hours": m.groupby("account")["hold"].median(),
        "out_within_24h_share": (m["hold"] <= 24).groupby(m["account"]).sum() / n_out,
    })


def behaviour_features(tx: pd.DataFrame, core: pd.DataFrame, formats: list[str] | None = None) -> pd.DataFrame:
    """All Day-6 features for the accounts in ``core`` (same index, same row order).

    ``formats``: payment formats to emit ``fmt_out_*`` columns for (default: those seen in ``tx``).
    Pass the train window's list so every window gets identical columns.
    """
    ev = _events(tx)
    out = _velocity(ev).join(_hold_times(tx), how="outer")

    # pass-through balance, from core's USD totals
    hi, lo = np.maximum(core.amount_in, core.amount_out), np.minimum(core.amount_in, core.amount_out)
    out["in_out_amount_ratio"] = (lo / hi.where(hi > 0, 1.0)).where(hi > 0, 0.0)

    # mix
    g = ev.groupby("account")
    out["cross_ccy_share"] = g["cross"].mean()
    out["round_amount_share"] = g["round"].mean()
    out["self_transfers"] = (tx.src == tx.dst).groupby(tx["src"]).sum()

    fmts = sorted(tx["payment_format"].unique()) if formats is None else list(formats)
    ct = pd.crosstab(tx["src"], tx["payment_format"]).reindex(columns=fmts, fill_value=0)
    share = ct.div(ct.sum(axis=1), axis=0)
    share.columns = [fmt_col(f) for f in fmts]
    out = out.join(share, how="outer")

    out = out.reindex(core.index)
    share_cols = ["out_after_in_share", "out_within_24h_share", "cross_ccy_share",
                  "round_amount_share", "self_transfers", *share.columns]
    out[share_cols] = out[share_cols].fillna(0.0)

    # fan shape from core (NaN when the side has no transactions)
    out["fan_out_diversity"] = core.cp_out / core.txns_out.where(core.txns_out > 0)
    out["fan_in_diversity"] = core.cp_in / core.txns_in.where(core.txns_in > 0)
    out["fan_in_out_ratio"] = (core.cp_in + 1) / (core.cp_out + 1)

    cols = [*VELOCITY, *PASSTHROUGH, *share.columns, *MIX_OTHER, *FAN]
    return out[cols].replace([np.inf, -np.inf], np.nan)


def align_columns(tables: dict[str, pd.DataFrame], reference: str = "train") -> dict[str, pd.DataFrame]:
    """Give every table exactly the reference table's columns, in its order.

    A ``fmt_out_*`` column missing from a window is added as 0 (format never used there).
    Any other missing column is an error; extra columns are dropped.
    """
    ref = list(tables[reference].columns)
    aligned = {}
    for name, t in tables.items():
        missing = [c for c in ref if c not in t.columns]
        bad = [c for c in missing if not c.startswith("fmt_out_")]
        if bad:
            raise KeyError(f"{name}: missing non-format columns {bad}")
        aligned[name] = t.assign(**{c: 0.0 for c in missing})[ref]
    return aligned
