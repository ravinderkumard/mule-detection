"""Data contracts (pandera) for the transaction table and the per-split feature tables.

Bad data should fail loudly at the boundary, not silently inside the model. ``validate_*``
runs the schema eagerly (all failures collected) and raises ``DataContractError`` with a
readable summary of which column and check failed.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
import pandera.pandas as pa

from muledetect.features import CORE_FEATURES

PAYMENT_FORMATS = ["ACH", "Bitcoin", "Cash", "Cheque", "Credit Card", "Reinvestment", "Wire"]

# Behaviour features: shares/fractions bounded in [0, 1]; the rest only need to be >= 0.
SHARE_FEATURES = [
    "in_out_amount_ratio", "out_after_in_share", "out_within_24h_share",
    "cross_ccy_share", "round_amount_share", "fan_out_diversity", "fan_in_diversity",
]
# Undefined for accounts with a single txn / no outflow / no inflow: NaN allowed (never inf).
NULLABLE_FEATURES = [
    "gap_min_hours", "gap_median_hours", "hold_median_hours",
    "fan_out_diversity", "fan_in_diversity",
]
NONNEG_FEATURES = [
    "n_active_days", "n_active_hours", "txns_per_active_day", "txns_per_active_hour",
    "max_txns_in_hour", "burst_ratio", "gap_min_hours", "gap_median_hours",
    "hold_median_hours", "self_transfers", "fan_in_out_ratio", "active_span_hours",
]
MULE_ROLES = ["not_mule", "pass_through", "sender_only", "receiver_only"]


class DataContractError(ValueError):
    """Raised when a table violates its schema; the message lists every failed check."""


def _finite(s: pd.Series) -> pd.Series:
    return ~np.isinf(s.astype("float64"))


def _num(nullable: bool = False, **kw) -> pa.Column:
    return pa.Column(float, nullable=nullable, coerce=True,
                     checks=[pa.Check(_finite, element_wise=False, error="contains +/-inf"),
                             *([pa.Check.ge(kw["ge"])] if "ge" in kw else []),
                             *([pa.Check.le(kw["le"])] if "le" in kw else [])])


TRANSACTION_SCHEMA = pa.DataFrameSchema(
    {
        "timestamp": pa.Column("datetime64[us]", nullable=False, coerce=True),
        "from_bank": pa.Column(int, pa.Check.ge(0), coerce=True),
        "to_bank": pa.Column(int, pa.Check.ge(0), coerce=True),
        "src": pa.Column(str, pa.Check.str_length(min_value=1), nullable=False),
        "dst": pa.Column(str, pa.Check.str_length(min_value=1), nullable=False),
        "amount_paid": _num(ge=0),
        "amount_received": _num(ge=0),
        "payment_currency": pa.Column(str, nullable=False),
        "receiving_currency": pa.Column(str, nullable=False),
        "payment_format": pa.Column(str, pa.Check.isin(PAYMENT_FORMATS), nullable=False),
        "is_laundering": pa.Column(int, pa.Check.isin([0, 1]), coerce=True),
    },
    strict=False,
    name="transactions",
)

_FEATURE_COLUMNS: dict[str, pa.Column] = {"account": pa.Column(str, nullable=False)}
for _c in CORE_FEATURES:
    _FEATURE_COLUMNS[_c] = _num(ge=0)
for _c in NONNEG_FEATURES:
    _FEATURE_COLUMNS[_c] = _num(nullable=_c in NULLABLE_FEATURES, ge=0)
for _c in SHARE_FEATURES:
    _FEATURE_COLUMNS[_c] = _num(nullable=_c in NULLABLE_FEATURES, ge=0, le=1)
# fmt_out_<format> shares: which formats appear can differ per window, so match by regex.
_FEATURE_COLUMNS["fmt_out_.*"] = pa.Column(
    float, pa.Check.in_range(0, 1), regex=True, required=False, nullable=False, coerce=True)
_FEATURE_COLUMNS["is_mule"] = pa.Column(int, pa.Check.isin([0, 1]), coerce=True)
_FEATURE_COLUMNS["mule_role"] = pa.Column(str, pa.Check.isin(MULE_ROLES), coerce=True)

FEATURE_SCHEMA = pa.DataFrameSchema(
    _FEATURE_COLUMNS,
    checks=[
        pa.Check(lambda d: d["account"].is_unique, error="account ids are not unique"),
        pa.Check(lambda d: (d["txns"] >= 1).all(), error="account with zero transactions"),
        pa.Check(lambda d: (d["txns"] == d["txns_in"] + d["txns_out"]).all(),
                 error="txns != txns_in + txns_out"),
    ],
    strict=False,
    name="features",
)


def _validate(schema: pa.DataFrameSchema, df: pd.DataFrame, what: str) -> pd.DataFrame:
    try:
        return schema.validate(df, lazy=True)
    except pa.errors.SchemaErrors as e:
        fc = e.failure_cases[["column", "check", "failure_case"]].drop_duplicates("check") \
            if len(e.failure_cases) else e.failure_cases
        lines = [f"  - column={r.column!r} check={r.check!s} e.g. {r.failure_case!r}"
                 for r in fc.itertuples()]
        raise DataContractError(
            f"{what} failed schema '{schema.name}' ({len(e.failure_cases)} failure cases):\n"
            + "\n".join(lines)) from None


def validate_transactions(df: pd.DataFrame, what: str = "transactions") -> pd.DataFrame:
    return _validate(TRANSACTION_SCHEMA, df, what)


def validate_features(df: pd.DataFrame, what: str = "features") -> pd.DataFrame:
    """``df`` must have ``account`` as a column (not the index)."""
    return _validate(FEATURE_SCHEMA, df, what)
