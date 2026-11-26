import numpy as np
import pandas as pd

from muledetect.evaluation import PlattCalibrator, rule_flags, rules_score, top_k_metrics


def test_top_k_metrics_counts_hits():
    y = np.array([1, 0, 1, 0, 0, 1])
    s = np.array([0.9, 0.8, 0.7, 0.2, 0.1, 0.05])
    m = top_k_metrics(y, s, [2, 3])
    assert m[2]["hits"] == 1 and m[2]["precision"] == 0.5
    assert m[3]["hits"] == 2 and abs(m[3]["recall"] - 2 / 3) < 1e-9


def test_rules_rank_by_rules_fired_then_volume():
    df = pd.DataFrame({"a": [5, 5, 0, 5], "b": [5, 0, 0, 5], "amount_out": [1.0, 9.0, 99.0, 5.0]})
    rules = {"r1": {"a": 1}, "r2": {"b": 1}}
    assert rule_flags(df, rules).sum(axis=1).tolist() == [2, 1, 0, 2]
    s = rules_score(df, rules)
    assert s[3] > s[0] > s[1] > s[2]   # two rules > one rule; ties broken by volume; no alert last
    assert s[2] == 0                    # no rule fired -> not an alert, whatever the volume


def test_platt_is_monotone():
    rng = np.random.default_rng(0)
    p = rng.uniform(0.001, 0.9, 2000)
    y = (rng.uniform(size=2000) < p / 2).astype(int)
    out = PlattCalibrator().fit(p, y).predict(np.sort(p))
    assert (np.diff(out) >= 0).all()
