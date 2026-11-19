# Day 6 — Behaviour features (velocity, pass-through, mix, fan)

## Summary
Day 6 adds 26 behaviour features to the Day-5 account tables (45 features in total), built per window
from that window's transactions only: velocity (how long, how often and how bursty an account is active),
pass-through (whether and how fast incoming money is forwarded), payment mix (outgoing format shares,
cross-currency, round amounts, self-transfers) and fan shape (counterparty diversity). The whole build
runs in about 20 seconds on the full HI-Small data, with no infinite values and `NaN` kept for undefined
values. The strongest stable signals are velocity (`burst_ratio`, `active_span_hours`, AUC ≈ 0.66–0.70 on
both train and test) and outgoing ACH share (≈ 0.70); the pass-through and fan features carry real but
unstable signal (some flip direction between train and test), and round amounts carry none on this
synthetic data. The main caveats are that "Reinvestment" self-transfers exist only in the train window
(so those features are train artefacts), that counts and amounts are extremely heavy-tailed, and that
much of the velocity signal is a proxy for transaction volume.

Code: `src/muledetect/behaviour.py` → `behaviour_features(tx, core, formats=None)`, one row per account,
aligned to `core.index`. `build_account_table(txns, behaviour=True)` / `build_split_tables(df, behaviour=True)`
in `features.py` join it between core features and labels; `behaviour=False` gives exactly the Day-5 table.
Everything is computed per window only (no cross-window leakage).

Run: `python scripts/build_features.py` (~20 s on full HI-Small), `python scripts/check_features.py`
(sanity report, exit 1 on infinities), `pytest tests`.
Output: 45 features + `is_mule`, `mule_role` = 47 columns; val and test have exactly train's columns in
train's order (`align_columns`; a payment format missing from a window becomes 0).

## Conventions
- **Event** = one (account, txn) pair; a self-transfer (src == dst) is **one** event (Day-5 `txns` counts
  it as two; `cp_*` include it).
- Shares and counts: 0 when undefined. Hold, gap, diversity: **NaN** when undefined (LightGBM handles NaN).
  No infinities.
- Name clash: core already has `active_hours` (= span in hours), so the distinct counts are
  `n_active_hours` / `n_active_days`. `active_span_hours` duplicates core `active_hours` (kept per spec).

## 1. Velocity
| Feature | Meaning |
|---|---|
| `active_span_hours` | last − first event |
| `n_active_days`, `n_active_hours` | distinct calendar days / clock hours with activity |
| `txns_per_active_day`, `txns_per_active_hour` | events / distinct days / hours |
| `max_txns_in_hour`, `burst_ratio` | busiest hour; that divided by `txns_per_active_hour` (≥ 1) |
| `gap_min_hours`, `gap_median_hours` | time between consecutive events (NaN if one event) |

## 2. Pass-through
| Feature | Meaning |
|---|---|
| `in_out_amount_ratio` | min(in, out) / max(in, out) USD (0 if both 0), from core totals |
| `out_after_in_share` | share of outgoing txns with an earlier incoming one |
| `hold_median_hours` | median hours from the latest earlier incoming txn to each outgoing txn (`merge_asof`; NaN if never) |
| `out_within_24h_share` | share of outgoing txns sent within 24 h of money arriving |

Self-transfers are excluded from hold times.

## 3. Mix
`fmt_out_<format>` (share of **outgoing** txns per payment format, 0 if the account never sends),
`cross_ccy_share` (events with different payment/receiving currency), `round_amount_share`
(`amount_paid` ≥ 100 and a multiple of 100, original currency), `self_transfers` (count).

## 4. Fan shape (from core)
`fan_out_diversity` = cp_out / txns_out, `fan_in_diversity` = cp_in / txns_in (NaN if no such txns),
`fan_in_out_ratio` = (cp_in + 1) / (cp_out + 1).

## Findings (single-feature ROC-AUC, train / test; NaN scored as lowest)
- **Velocity is the strongest family and stable:** `burst_ratio` 0.70 / 0.66, `active_span_hours` 0.70 / 0.69,
  `n_active_hours` 0.69 / 0.64. Mules are active over longer spans (median 116 h vs 41 h) and in
  more distinct hours and days; they are also burstier. These are partly a proxy for volume (`txns` 14 vs 5).
- **`fmt_out_ach`** 0.69 / 0.70: mules send far more by ACH. Medians 0.20 vs 0.
- **Pass-through:** `out_after_in_share` 0.66 / 0.64 and `hold_median_hours` 0.65 / 0.64 are stable, but
  much of that is *whether* the account ever forwards (73–76% of accounts have a NaN hold). Among accounts that
  do forward, the median hold is similar (12.8 h normal vs 13.4 h mule). `in_out_amount_ratio` **flips**
  (0.49 train, 0.71 test).
- **Fan diversity is unstable:** `fan_in_diversity` 0.54 / 0.71, `fan_out_diversity` 0.40 / 0.56 (flip).
  `fan_in_out_ratio` 0.59 / 0.54. Role-driven (receiver-only vs sender-only mules pull opposite ways).
- **Mix:** `cross_ccy_share` 0.59 / 0.63 and `self_transfers` 0.64 / 0.63 (mules are just more active; see
  below). `round_amount_share` is useless in this synthetic data (AUC 0.50; median 0 for everyone).
  `fmt_out_cheque` / `fmt_out_credit_card` flip.
- **Drift to remember:** "Reinvestment" (all self-transfers) exists only in the train window: 481K txns in
  train, none in val/test, so `fmt_out_reinvestment` and most of `self_transfers` are train artefacts.
  Candidates to drop. NaN rates also shift between windows (`fan_out_diversity` NaN: 3.9% train, 44.8% val).
- **Tails and redundancy:** counts and amounts are extremely heavy-tailed (max 93,951 txns, `amount_out` up to
  $36B; the two biggest hubs are mules), so log-transform for non-tree models. Near-duplicates:
  `active_hours` ≡ `active_span_hours`, `cp_*` ≈ `banks_*`.

## Review notes (differences from the original spec)
- Day-5 names kept (`temporal_split`, `build_core_features`, `build_labels`, `build_account_table`,
  `build_split_tables`); the `behaviour` flag was added to the last two.
- My earlier per-task modules (velocity/passthrough/mix/fan_round) and their extra features
  (burstiness, net_flow_share, both-direction payment shares, `n_formats`, fixed-sentinel delays) were
  replaced by this single module. Sentinel-filled delays became NaN.
