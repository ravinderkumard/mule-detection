import numpy as np
import pandas as pd
import pytest

from muledetect.behaviour import align_columns, behaviour_features, fmt_col
from muledetect.features import build_core_features, build_account_table


def make_tx(rows):
    """rows: (src, dst, 'YYYY-mm-dd HH:MM', amount, fmt[, pay_ccy, recv_ccy])"""
    recs = []
    for r in rows:
        src, dst, ts, amt, fmt = r[:5]
        pc, rc = (r[5], r[6]) if len(r) > 5 else ("USD", "USD")
        recs.append(dict(
            timestamp=pd.Timestamp(ts), src=src, dst=dst, from_bank=1, to_bank=2,
            amount_paid=amt, amount_received=amt, amount_paid_usd=amt, amount_received_usd=amt,
            payment_currency=pc, receiving_currency=rc, payment_format=fmt, is_laundering=0))
    return pd.DataFrame(recs)


def run(tx, formats=None):
    core = build_core_features(tx)
    return behaviour_features(tx, core, formats)


def test_pass_through_receive_then_send_one_hour_later():
    tx = make_tx([("x", "m", "2022-09-01 10:00", 100, "ACH"),
                  ("m", "y", "2022-09-01 11:00", 100, "ACH")])
    f = run(tx).loc["m"]
    assert f.in_out_amount_ratio == 1
    assert f.hold_median_hours == pytest.approx(1)
    assert f.out_within_24h_share == 1
    assert f.out_after_in_share == 1


def test_single_transaction_gap_nan_and_no_crash():
    f = run(make_tx([("a", "b", "2022-09-01 10:00", 50, "Wire")]))
    assert f.loc["a", ["gap_min_hours", "gap_median_hours"]].isna().all()
    assert f.loc["b", ["gap_min_hours", "gap_median_hours"]].isna().all()
    assert np.isnan(f.loc["a", "hold_median_hours"])      # a sends but never received
    assert np.isnan(f.loc["a", "fan_in_diversity"])       # no incoming txns
    assert f.loc["b", "fmt_out_wire"] == 0                # b never sends -> 0, not NaN


def test_self_transfer_is_counted():
    tx = make_tx([("a", "a", "2022-09-01 10:00", 10, "Reinvestment"),
                  ("a", "b", "2022-09-01 11:00", 20, "ACH")])
    f = run(tx)
    assert f.loc["a", "self_transfers"] == 1
    assert f.loc["b", "self_transfers"] == 0


def test_fmt_out_shares_sum_to_one_or_zero():
    tx = make_tx([("a", "b", "2022-09-01 10:00", 10, "ACH"),
                  ("a", "c", "2022-09-01 11:00", 20, "Wire"),
                  ("a", "c", "2022-09-01 12:00", 30, "Wire")])
    f = run(tx)
    cols = [c for c in f.columns if c.startswith("fmt_out_")]
    assert f.loc["a", cols].sum() == pytest.approx(1)
    assert f.loc["a", fmt_col("Wire")] == pytest.approx(2 / 3)
    assert f.loc["b", cols].sum() == 0 and f.loc["c", cols].sum() == 0


def test_no_infinite_values_and_index_aligned():
    tx = make_tx([("a", "a", "2022-09-01 10:00", 0.0, "ACH"),
                  ("a", "b", "2022-09-01 10:00", 0.0, "ACH"),
                  ("c", "d", "2022-09-02 10:00", 100, "ACH", "USD", "EUR")])
    core = build_core_features(tx)
    f = behaviour_features(tx, core)
    assert not np.isinf(f.to_numpy(dtype=float)).any()
    assert f.index.equals(core.index)


def test_velocity_round_and_cross_ccy():
    tx = make_tx([("a", "b", "2022-09-01 10:00", 200, "ACH", "USD", "EUR"),
                  ("a", "b", "2022-09-01 10:30", 150, "ACH"),
                  ("a", "b", "2022-09-02 10:00", 300, "ACH")])
    f = run(tx).loc["a"]
    assert f.active_span_hours == pytest.approx(24)
    assert f.n_active_days == 2 and f.n_active_hours == 2  # 10:00 and 10:30 share an hour
    assert f.max_txns_in_hour == 2
    assert f.burst_ratio == pytest.approx(2 / 1.5)   # txns_per_active_hour = 3/2
    assert f.gap_min_hours == pytest.approx(0.5)
    assert f.round_amount_share == pytest.approx(2 / 3)   # 200 and 300, not 150
    assert f.cross_ccy_share == pytest.approx(1 / 3)


def test_column_alignment_when_format_missing_from_a_window():
    train = make_tx([("a", "b", "2022-09-01 10:00", 10, "ACH"), ("b", "c", "2022-09-01 11:00", 5, "Wire")])
    val = make_tx([("a", "b", "2022-09-05 10:00", 10, "ACH")])        # no Wire
    fmts = sorted(train.payment_format.unique())
    tabs = {"train": build_account_table(train, formats=fmts),
            "val": build_account_table(val, formats=fmts)}
    tabs["val"] = tabs["val"].drop(columns=["fmt_out_wire"])           # simulate a format-less build
    out = align_columns(tabs)
    assert list(out["val"].columns) == list(out["train"].columns)
    assert (out["val"]["fmt_out_wire"] == 0).all()
    with pytest.raises(KeyError):
        align_columns({"train": tabs["train"], "val": tabs["val"].drop(columns=["txns"])})


def test_behaviour_flag_off_gives_day5_table():
    tx = make_tx([("a", "b", "2022-09-01 10:00", 10, "ACH")])
    t = build_account_table(tx, behaviour=False)
    assert "burst_ratio" not in t.columns and "txns" in t.columns
