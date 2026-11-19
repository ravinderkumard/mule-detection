import numpy as np
import pandas as pd
import pytest

from muledetect.features import build_account_table
from muledetect.schemas import (
    DataContractError, validate_features, validate_transactions,
)
from test_features import make_tx


@pytest.fixture
def tx():
    return make_tx([
        ("A", "B", "2022-09-01 00:00", 100.0, "ACH"),
        ("B", "C", "2022-09-01 01:00", 90.0, "Wire"),
        ("C", "A", "2022-09-02 00:00", 50.0, "Cheque"),
    ])


@pytest.fixture
def feats(tx):
    return build_account_table(tx).reset_index()


def test_valid_transactions_pass(tx):
    validate_transactions(tx)


@pytest.mark.parametrize("col,bad", [
    ("amount_paid", -1.0),
    ("payment_format", "Carrier Pigeon"),
    ("is_laundering", 2),
    ("src", ""),
    ("amount_received", np.inf),
])
def test_bad_transactions_raise(tx, col, bad):
    tx[col] = tx[col].astype(object) if isinstance(bad, str) else tx[col]
    tx.loc[0, col] = bad
    with pytest.raises(DataContractError, match="transactions"):
        validate_transactions(tx)


def test_missing_transaction_column_raises(tx):
    with pytest.raises(DataContractError, match="timestamp"):
        validate_transactions(tx.drop(columns="timestamp"))


def test_null_timestamp_raises(tx):
    tx.loc[0, "timestamp"] = pd.NaT
    with pytest.raises(DataContractError):
        validate_transactions(tx)


def test_valid_features_pass(feats):
    validate_features(feats)


def test_infinity_in_features_raises(feats):
    feats.loc[0, "amount_ratio"] = np.inf
    with pytest.raises(DataContractError, match="inf"):
        validate_features(feats)


def test_duplicate_account_raises(feats):
    with pytest.raises(DataContractError, match="unique"):
        validate_features(pd.concat([feats, feats.iloc[[0]]], ignore_index=True))


def test_share_out_of_range_raises(feats):
    feats.loc[0, "round_amount_share"] = 1.5
    with pytest.raises(DataContractError, match="round_amount_share"):
        validate_features(feats)


def test_negative_count_raises(feats):
    feats.loc[0, "txns_out"] = -1
    with pytest.raises(DataContractError):
        validate_features(feats)


def test_nan_allowed_only_in_nullable_columns(feats):
    feats.loc[0, "gap_min_hours"] = np.nan
    validate_features(feats)
    feats.loc[0, "txns_in"] = np.nan
    with pytest.raises(DataContractError):
        validate_features(feats)


def test_bad_label_raises(feats):
    feats["mule_role"] = feats["mule_role"].astype(str)
    feats.loc[0, "mule_role"] = "banker"
    with pytest.raises(DataContractError, match="mule_role"):
        validate_features(feats)
