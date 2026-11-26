# Day 8 — Baseline model and business evaluation

## Summary
Day 8 adds the business view of the model: a hand-written rules baseline standing in for the bank's existing
rules engine, a tuned LightGBM on the same features, and a test-window comparison at fixed alert budgets
(top 100 / 500 / 1,000 accounts). The pipeline's `train` stage now also fits a probability calibrator, and the
`evaluate` stage writes a Markdown report, a precision-recall plot and a metrics file, and logs all of them to MLflow.
On the test window LightGBM catches 100 / 443 / 624 mules at budgets of 100 / 500 / 1,000, against 22 / 43 / 57
for the rules (4.5x–10.9x recall lift), and PR-AUC is 0.338 vs 0.015.

## Run
```
make train evaluate      # or: python -m muledetect.pipeline train evaluate
make ui                  # MLflow UI -> experiment mule-detection
```
Needs the features from `make features`. Outputs: `reports/evaluation.md`, `reports/pr_curve.png`,
`reports/pr_curve_test.csv`, `reports/metrics.json`.

## Rules baseline
Four rules, each an AND of `column >= threshold` conditions, defined in `configs/pipeline.yaml` under `rules`:

| Rule | Condition |
|---|---|
| high_passthrough_24h | `out_within_24h_share >= 0.5` and `txns_out >= 3` |
| many_counterparties | `counterparties >= 10` |
| balanced_in_out | `in_out_amount_ratio >= 0.8`, `txns_in >= 2`, `txns_out >= 2` |
| rapid_burst | `max_txns_in_hour >= 5` |

- An account is an alert if at least one rule fires. The queue is ordered by number of rules fired, then by USD volume
  (`amount_out`), the way a triage queue is usually sorted. On test the rules alert on 62,662 accounts (17.6%).
- Thresholds were set by looking at the train and validation windows only, never test.
- "New counterparties" is approximated by distinct counterparties inside the window; the dataset has no earlier history.

## LightGBM
- Same 43 model columns as Day 7. Early stopping on validation average precision (patience 100).
- **Class weighting:** `scale_pos_weight=2`. Selected on validation PR-AUC from a small sweep (weights 1–50, several tree sizes).
  Weights of 10 or more stopped after 1–5 trees (val PR-AUC 0.04–0.12); weight 5 was unstable; weights 1–3 trained
  for 400–1,600 trees (val PR-AUC about 0.25). This was the cause of the Day 7 one-tree model.
- Final params: 7 leaves, `min_child_samples=500`, learning rate 0.03, up to 2,000 trees; stopped at 1,038.
- **Calibration:** Platt (sigmoid) on validation log-odds, saved to `models/calibrator.joblib`. It is monotone, so it
  cannot change the ranking or any alert-budget number.

## Results (test window: 356,396 accounts, 2,201 mules, base rate 0.618%)
| Budget | Rules caught | Rules precision | LightGBM caught | LightGBM precision | LightGBM recall | Recall lift |
|---|---|---|---|---|---|---|
| 100 | 22 | 22.0% | 100 | 100.0% | 4.5% | 4.5x |
| 500 | 43 | 8.6% | 443 | 88.6% | 20.1% | 10.3x |
| 1,000 | 57 | 5.7% | 624 | 62.4% | 28.4% | 10.9x |

| Ranking quality | LightGBM | Rules | Random |
|---|---|---|---|
| PR-AUC | 0.338 | 0.015 | 0.006 |
| ROC-AUC | 0.907 | 0.607 | 0.5 |

Validation PR-AUC is 0.253, so test is higher than validation; the windows differ in mule mix (see Day 6).

| Calibration (test) | Raw | Calibrated |
|---|---|---|
| Brier score | 0.00478 | 0.00477 |
| Expected calibration error | 0.00076 | 0.00136 |

## What each MLflow run records
Everything from Day 7 (config params, git tags, data hashes), plus for `evaluate`:
- **Metrics:** per model PR-AUC, ROC-AUC, and hits / precision / recall at each budget (`test_lgbm_*`, `test_rules_*`),
  recall lift and extra mules caught per budget, calibration metrics, rules alert count.
- **Artifacts:** `evaluation/evaluation.md`, `evaluation/pr_curve.png`, `evaluation/pr_curve_test.csv`;
  the model and calibrator under `model/` when `train` ran.

## Verification
- 26 tests pass (3 new in `tests/test_evaluation.py`: top-K counting, rules ordering, calibrator monotonicity).
- `evaluate` re-run on the same trained model gives identical test metrics. Retraining was not repeated, so
  train-to-train reproducibility is not checked here.

## Caveats
- Platt calibration did not help on test: ECE rose from 0.00076 to 0.00136, because the validation window's mule rate
  (0.65%) differs from test's. The raw probabilities were already close to the true rate, so use them unless the
  calibrator is refit on fresher data. Early stopping and calibration share the validation split.
- The rules are deliberately simple, so the lift partly reflects how weak they are. Real bank rules are tuned and
  would narrow the gap. HI-Small is synthetic, so absolute numbers will not transfer; the method does.
- With only 2,201 test mules, one mule changes precision at K=100 by one point.
- The hyperparameter sweep was run ad hoc and is not in the repo; only its chosen result (with a comment) is in the config.
- `fmt_out_reinvestment` and `self_transfers` stay excluded from the model columns (train-only artefacts, Day 6).

## Files
`src/muledetect/evaluation.py`, `src/muledetect/pipeline.py` (train + evaluate stages), `configs/pipeline.yaml`
(`model`, `evaluate`, `rules`, new `paths`), `Makefile` (`clean`), `tests/test_evaluation.py`.
