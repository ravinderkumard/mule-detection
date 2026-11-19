"""Temporal split, laundering rates, account-level core features and labels for HI-Small.

Input is the transaction table cached by scripts/verify_data.py
(data/interim/hi_small_trans.parquet): one row per transaction with
``timestamp, src, dst, from_bank, to_bank, amount_paid, amount_received, payment_format, is_laundering``.

Leakage rule: features and labels for a split are computed only from that split's
own transactions, so nothing from the future leaks into an earlier window.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from muledetect.behaviour import align_columns, behaviour_features
from muledetect.paths import INTERIM

TRANS_PATH = INTERIM / "hi_small_trans.parquet"
SPLITS = ("train", "val", "test")

CORE_FEATURES = [
    "txns_out", "txns_in", "txns", "amount_out", "amount_in",
    "cp_out", "cp_in", "counterparties", "banks_out", "banks_in", "banks",
    "avg_amount_out", "avg_amount_in", "max_amount_out", "max_amount_in",
    "in_out_ratio", "amount_ratio", "active_hours", "txns_per_hour",
]


def load_transactions(path=TRANS_PATH) -> pd.DataFrame:
    return pd.read_parquet(path)


def estimate_usd_rates(txns: pd.DataFrame, anchor: str = "US Dollar") -> pd.Series:
    """USD value of one unit of each currency, estimated from cross-currency transactions.

    A cross-currency row says ``amount_received * u[recv] == amount_paid * u[pay]``, so
    ``log u[pay] - log u[recv] = log(amount_received / amount_paid)``. The median ratio per
    currency pair is taken (robust to rounding noise), then the log-rates are solved by
    least squares weighted by sqrt(pair count), with ``u[anchor] = 1``. Pairs that never
    trade with USD are pinned down through the other currencies.
    """
    x = txns[txns.receiving_currency != txns.payment_currency]
    x = x[(x.amount_paid > 0) & (x.amount_received > 0)]
    pairs = (x.assign(log_r=np.log(x.amount_received / x.amount_paid))
             .groupby(["payment_currency", "receiving_currency"])["log_r"]
             .agg(["median", "size"]).reset_index())

    currencies = sorted(set(pairs.payment_currency) | set(pairs.receiving_currency) | {anchor})
    idx = {c: i for i, c in enumerate(currencies)}
    A = np.zeros((len(pairs) + 1, len(currencies)))
    A[np.arange(len(pairs)), pairs.payment_currency.map(idx)] = 1
    A[np.arange(len(pairs)), pairs.receiving_currency.map(idx)] = -1
    b = np.append(pairs["median"].to_numpy(), 0.0)
    w = np.append(np.sqrt(pairs["size"].to_numpy()), 1e3)  # heavy weight pins the anchor
    A[-1, idx[anchor]] = 1
    log_u, *_ = np.linalg.lstsq(A * w[:, None], b * w, rcond=None)
    return pd.Series(np.exp(log_u), index=currencies, name="usd_per_unit")


def add_usd_amounts(txns: pd.DataFrame, rates: pd.Series) -> pd.DataFrame:
    """Add ``amount_paid_usd`` / ``amount_received_usd`` columns using ``rates``."""
    return txns.assign(
        amount_paid_usd=txns.amount_paid * txns.payment_currency.map(rates),
        amount_received_usd=txns.amount_received * txns.receiving_currency.map(rates),
    )


def temporal_split(df: pd.DataFrame, train_frac: float = 0.6, val_frac: float = 0.2) -> dict[str, pd.DataFrame]:
    """Split by time: the earliest ``train_frac`` of rows is train, the next ``val_frac`` val, the rest test.

    Cut points are row-count quantiles of ``timestamp`` so each split has a predictable size.
    Rows sharing a cut timestamp all go to the later split, keeping windows disjoint.
    """
    if not (0 < train_frac and train_frac + val_frac < 1):
        raise ValueError("need 0 < train_frac and train_frac + val_frac < 1")
    ts = df["timestamp"]
    t_val = ts.quantile(train_frac)
    t_test = ts.quantile(train_frac + val_frac)
    return {
        "train": df[ts < t_val],
        "val": df[(ts >= t_val) & (ts < t_test)],
        "test": df[ts >= t_test],
    }


def laundering_rates(splits: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per-split transaction count, laundering count and rate, plus the time window covered."""
    rows = {}
    for name, part in splits.items():
        n, pos = len(part), int(part["is_laundering"].sum())
        rows[name] = {
            "txns": n, "laundering": pos, "rate": pos / n if n else np.nan,
            "start": part["timestamp"].min(), "end": part["timestamp"].max(),
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def build_labels(txns: pd.DataFrame) -> pd.DataFrame:
    """Account-level labels from the window's laundering transactions.

    ``is_mule``  : account sends or receives at least one laundering transaction.
    ``mule_role``: pass_through (both), sender_only, receiver_only, or not_mule.
    """
    sent = txns.groupby("src")["is_laundering"].sum().rename("n_sent_laund")
    recv = txns.groupby("dst")["is_laundering"].sum().rename("n_recv_laund")
    labels = pd.concat([sent, recv], axis=1).fillna(0).astype(int)
    labels.index.name = "account"

    has_out, has_in = labels.n_sent_laund > 0, labels.n_recv_laund > 0
    labels["is_mule"] = (has_out | has_in).astype("int8")
    labels["mule_role"] = pd.Series(
        np.select([has_out & has_in, has_out, has_in],
                  ["pass_through", "sender_only", "receiver_only"], default="not_mule"),
        index=labels.index,
    ).astype("category")
    return labels[["is_mule", "mule_role"]]


def mule_rates(tables: dict[str, pd.DataFrame]) -> pd.DataFrame:
    """Per-window account count, mule count, mule rate and role breakdown (from ``is_mule``/``mule_role``)."""
    rows = {}
    for name, t in tables.items():
        roles = t["mule_role"].value_counts()
        rows[name] = {
            "accounts": len(t), "mules": int(t["is_mule"].sum()), "rate": t["is_mule"].mean(),
            **{r: int(roles.get(r, 0)) for r in ("pass_through", "sender_only", "receiver_only")},
        }
    return pd.DataFrame.from_dict(rows, orient="index")


def build_core_features(txns: pd.DataFrame) -> pd.DataFrame:
    """One row per account seen in ``txns``: volume, amounts, counterparties, timing.

    ``txns`` must carry ``amount_paid_usd`` / ``amount_received_usd`` (see ``add_usd_amounts``).

    Ratios use +1 smoothing so accounts with no inbound or outbound side stay finite.
    """
    out = txns.groupby("src").agg(
        txns_out=("dst", "size"), amount_out=("amount_paid_usd", "sum"),
        avg_amount_out=("amount_paid_usd", "mean"), max_amount_out=("amount_paid_usd", "max"),
        cp_out=("dst", "nunique"), banks_out=("to_bank", "nunique"),
    )
    inn = txns.groupby("dst").agg(
        txns_in=("src", "size"), amount_in=("amount_received_usd", "sum"),
        avg_amount_in=("amount_received_usd", "mean"), max_amount_in=("amount_received_usd", "max"),
        cp_in=("src", "nunique"), banks_in=("from_bank", "nunique"),
    )
    out.index.name = inn.index.name = "account"
    feat = out.join(inn, how="outer").fillna(0)

    # distinct counterparties and counterparty banks regardless of direction
    pairs = pd.concat([
        txns[["src", "dst", "to_bank"]].set_axis(["account", "cp", "bank"], axis=1),
        txns[["dst", "src", "from_bank"]].set_axis(["account", "cp", "bank"], axis=1),
    ]).drop_duplicates()
    feat["counterparties"] = pairs.groupby("account").size()
    feat["banks"] = pairs.drop_duplicates(["account", "bank"]).groupby("account").size()
    feat["txns"] = feat.txns_out + feat.txns_in

    feat["in_out_ratio"] = (feat.txns_in + 1) / (feat.txns_out + 1)
    feat["amount_ratio"] = (feat.amount_in + 1) / (feat.amount_out + 1)

    # activity span in hours across both directions
    ts = pd.concat([
        txns[["src", "timestamp"]].set_axis(["account", "timestamp"], axis=1),
        txns[["dst", "timestamp"]].set_axis(["account", "timestamp"], axis=1),
    ]).groupby("account")["timestamp"].agg(["min", "max"])
    feat["active_hours"] = (ts["max"] - ts["min"]).dt.total_seconds() / 3600
    feat["txns_per_hour"] = feat.txns / (feat.active_hours + 1)

    return feat[CORE_FEATURES]


def build_account_table(txns: pd.DataFrame, behaviour: bool = True, formats: list[str] | None = None) -> pd.DataFrame:
    """Core features (+ Day-6 behaviour features if ``behaviour``) joined with labels for one window.

    ``formats``: payment formats for the ``fmt_out_*`` columns (see ``behaviour_features``).
    With ``behaviour=False`` the output is exactly the Day-5 table.
    """
    core = build_core_features(txns)
    parts = [core]
    if behaviour:
        parts.append(behaviour_features(txns, core, formats))
    return pd.concat(parts, axis=1).join(build_labels(txns))


def build_split_tables(df: pd.DataFrame, train_frac: float = 0.6, val_frac: float = 0.2,
                       behaviour: bool = True) -> dict[str, pd.DataFrame]:
    """Temporal split, then one account table per split (features + labels from that split only).

    USD rates are estimated on the train window and applied to every split. With ``behaviour``,
    the payment-format list comes from train and all tables are aligned to train's columns.
    """
    splits = temporal_split(df, train_frac, val_frac)
    rates = estimate_usd_rates(splits["train"])
    formats = sorted(splits["train"]["payment_format"].unique())
    tables = {name: build_account_table(add_usd_amounts(part, rates), behaviour, formats)
              for name, part in splits.items()}
    return align_columns(tables) if behaviour else tables
