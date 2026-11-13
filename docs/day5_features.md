# Day 5 — Temporal split, USD rates, account features and labels (HI-Small)

Everything added on Day 5 lives in two files:

- `src/muledetect/features.py` — the reusable logic.
- `scripts/build_features.py` — a thin runner that builds and saves the tables.

Output: `data/processed/features_{train,val,test}.parquet` (git-ignored, regenerate with
`python scripts/build_features.py`).

---

## 1. The goal in one paragraph

HI-Small is a table of ~5.08M transactions. The model we want predicts, **per account**,
whether it is a money mule. So Day 5 turns "one row per transaction" into "one row per
account", does it separately for three time windows (train / val / test), and makes sure
nothing from the future leaks into the past.

```
hi_small_trans.parquet (5.08M txns)
   -> temporal_split        -> train / val / test transaction windows
   -> estimate_usd_rates    -> USD value of each currency (from train only)
   -> add_usd_amounts       -> amounts comparable across currencies
   -> build_core_features   -> one row per account: 19 numeric features
   -> build_labels          -> is_mule + mule_role for each account
   -> features_{split}.parquet
```

---

## 2. Step by step

### 2.1 Temporal split — `temporal_split`
- 60 / 20 / 20 **by transaction volume**: cut points are the 60% and 80% row-count
  quantiles of `timestamp`.
- Rows sharing a cut timestamp all go to the later split, so windows never overlap.
- Result:

| Split | Transactions | Share | Window |
|---|---|---|---|
| train | 3,046,861 | 60.0% | 2022-09-01 00:00 → 2022-09-06 13:35 |
| val | 1,015,602 | 20.0% | 2022-09-06 13:36 → 2022-09-08 16:11 |
| test | 1,015,882 | 20.0% | 2022-09-08 16:12 → 2022-09-18 16:18 |

- The windows are very different in calendar length (≈5.5, 2 and 10 days) because
  transaction volume is front-loaded. If equal calendar time were wanted, only the cut
  points would change.
- **Why temporal, not random:** a random split would put transactions from the same
  laundering scheme in both train and test, so the model could memorise the scheme and
  scores would be inflated. In production you always predict the future from the past.

### 2.2 Laundering rate per window — `laundering_rates`
Transaction-level rate (laundering txns / all txns) plus the window start/end.

| Split | Laundering txns | Rate |
|---|---|---|
| train | 2,297 | 0.075% |
| val | 1,082 | 0.107% |
| test | 1,798 | 0.177% |

The rate climbs from train to test: **distribution drift**. Expect a model to see
a harder/different test set than train.

### 2.3 Estimating USD exchange rates — `estimate_usd_rates`, `add_usd_amounts`
**Problem.** `amount_paid` is in the payer's currency (15 currencies incl. Bitcoin, Yen,
Rupee…). Summing or averaging raw amounts across accounts mixes units — 1 BTC and 1 Yen
would count the same. The dataset has no exchange-rate table.

**Key observation.** ~72K transactions (1.4%) are cross-currency: they carry both
`amount_paid` (in currency A) and `amount_received` (in currency B). Their ratio *is* the
exchange rate A→B. The rates are constant over the whole period (e.g. USD→EUR = 0.8534 on
every day; the only variation is rounding).

**Method.**
1. For each cross-currency row compute `log(amount_received / amount_paid)`.
2. Take the **median per currency pair** (robust to rounding noise and outliers).
3. Each pair gives one equation: `log u[pay] − log u[recv] = log ratio`, where `u[c]` is the
   USD value of one unit of currency `c`.
4. Solve all equations together by **weighted least squares** (weight = √pair count) in log
   space, pinning `u[USD] = 1`. Currencies that never trade with USD directly
   (e.g. Yuan↔Rupee) are determined through the other currencies.
5. `add_usd_amounts` adds `amount_paid_usd` and `amount_received_usd`.

**Result (USD per 1 unit).**

| Currency | USD | Currency | USD |
|---|---|---|---|
| Bitcoin | 11,758 | Saudi Riyal | 0.267 |
| UK Pound | 1.292 | Brazil Real | 0.177 |
| Euro | 1.172 | Yuan | 0.149 |
| Swiss Franc | 1.093 | Mexican Peso | 0.0473 |
| Canadian Dollar | 0.758 | Rupee | 0.0136 |
| Australian Dollar | 0.708 | Ruble | 0.0129 |
| Shekel | 0.296 | Yen | 0.0095 |

**Validation.** Converting both sides of each cross-currency transaction to USD gives a
median mismatch of 0.008%; about 1% of those rows differ by more than 5% (small amounts
where rounding dominates).

**Leakage hygiene.** In `build_split_tables` the rates are estimated on the **train window
only** and then applied to val and test. They differ from full-data rates by at most 0.33%,
so this is cheap insurance, and it mirrors production (you would not know future rates).

**Bug this fixed.** My first version of the core features summed raw amounts across
currencies. Switching to USD changed the numbers materially (e.g. median outgoing amount
for train mules ≈ $39.6K vs ≈ $5.3K for normal accounts).

### 2.4 Core account features — `build_core_features`
One row per account that appears in the window. All computed from that window's
transactions only. `CORE_FEATURES` lists the 19 columns:

| Group | Columns | Meaning |
|---|---|---|
| Volume | `txns_out`, `txns_in`, `txns` | Number of outgoing / incoming / total transactions |
| USD amounts (out) | `amount_out`, `avg_amount_out`, `max_amount_out` | Sum / mean / max sent, in USD |
| USD amounts (in) | `amount_in`, `avg_amount_in`, `max_amount_in` | Sum / mean / max received, in USD |
| Counterparties | `cp_out`, `cp_in`, `counterparties` | Distinct recipients / senders / either direction |
| Banks | `banks_out`, `banks_in`, `banks` | Distinct recipient banks / sender banks / either direction |
| Ratios | `in_out_ratio`, `amount_ratio` | (in+1)/(out+1) for counts and USD amounts |
| Timing | `active_hours`, `txns_per_hour` | Hours between first and last transaction; txns / (hours + 1) |

Design notes:
- **Account identity = bank + account** (`src` / `dst` keys, e.g. `10057_803A115E0`), as in
  the AMLworld data; account numbers alone are not unique.
- **Smoothing:** the `+1` in ratios keeps accounts with no inbound or outbound side finite
  (no divide-by-zero, no infinities in the model).
- **Missing side = 0:** an account that only sends has `txns_in = 0`, `amount_in = 0`, etc.
  (outer join + fillna(0)). There are no NaNs in the output.
- **Counterparty banks are *other* banks.** An account's own bank is not a feature.
- **Sanity invariant checked:** `max(banks_out, banks_in) <= banks <= counterparties`.
- **Signal:** mules touch more parties — median counterparties 4 vs 2, banks 4 vs 2 (train).

### 2.5 Labels — `build_labels`
- `is_mule = 1` if the account **sends or receives at least one laundering transaction in
  that window**.
- `mule_role`: `pass_through` (both sends and receives laundering money), `sender_only`,
  `receiver_only`, or `not_mule`.
- Same definitions as notebook `02_patterns_and_labels`.
- I deliberately did **not** carry over `in_pattern` / pattern-type columns from the
  notebook. They come from `HI-Small_Patterns.txt`, which is derived from the labels —
  using them as features would be label leakage.

### 2.6 Per-window mule rates — `mule_rates`

| Window | Accounts | Mules | Rate | Pass-through | Sender only | Receiver only |
|---|---|---|---|---|---|---|
| train | 512,713 | 3,170 | 0.618% | 305 | 1,281 | 1,584 |
| val | 236,106 | 1,533 | 0.649% | 148 | 621 | 764 |
| test | 356,396 | 2,201 | 0.618% | 325 | 876 | 1,000 |

Findings:
- Account-level mule rate is **stable (~0.62–0.65%)** even though the transaction-level
  laundering rate rose 2.3× → test mules each carry more laundering transactions.
- **Labels are per-window, not permanent.** Only 6.0% of test mules were also mules in
  train (15.7% of val mules were mules in train), yet ~97.6% of test mules *do* appear in
  train — mostly looking normal there. So the model cannot just memorise train mules.
- The same account can appear in several windows; each window gets its own row, features
  and label. They are not linked.

### 2.7 Orchestration — `build_split_tables`, `scripts/build_features.py`
`build_split_tables(df)` = split → estimate rates on train → add USD columns → build
feature + label table per window. The script prints the split and mule-rate tables, then
writes `data/processed/features_{train,val,test}.parquet` (account as a regular column,
19 features + `is_mule` + `mule_role`; 512,713 / 236,106 / 356,396 rows).

---

## 3. Public API of `muledetect.features`

| Function | Purpose |
|---|---|
| `load_transactions()` | Read the cached transactions parquet |
| `temporal_split(df, train_frac, val_frac)` | Dict of train/val/test transaction frames |
| `laundering_rates(splits)` | Per-split txn rate and window |
| `estimate_usd_rates(txns)` | USD value per currency from cross-currency rows |
| `add_usd_amounts(txns, rates)` | Add `*_usd` amount columns |
| `build_core_features(txns)` | Account features (needs the USD columns) |
| `build_labels(txns)` | `is_mule`, `mule_role` |
| `build_account_table(txns)` | Features joined with labels |
| `mule_rates(tables)` | Per-window account-level mule rates |
| `build_split_tables(df)` | Whole pipeline, per split |

Note: `build_core_features` fails on a raw table; call `add_usd_amounts` first.

---

## 4. Interview perspective

### How to tell the story (60 seconds)
"I framed mule detection as account-level classification on IBM AMLworld. I split the
5M transactions 60/20/20 by time, not randomly, to avoid leakage. I built account features
from each window only — volumes, counterparties, banks, timing. Amounts were in 15
currencies with no rate table, so I recovered USD rates from cross-currency transactions
using a log-space weighted least-squares fit, fitted on train only. Labels are defined
per window, and I measured that the mule rate is stable at 0.6% while label identity
changes a lot between windows."

### Likely questions and strong answers
- **Why a temporal split?** Real deployment predicts the future from the past; random
  splits let the same laundering scheme appear on both sides and inflate metrics.
- **Where could leakage still hide?** Pattern file columns (derived from labels),
  exchange rates or any statistic fitted on the full data, and features computed over a
  window that includes the label period. I handled each: patterns excluded, rates from
  train only, features/labels per window.
- **How did you handle currencies?** Inferred rates from cross-currency rows; median per
  pair for robustness; least squares in log space because rates compose multiplicatively;
  validated to 0.008% median error.
- **Why log space and median?** Rates multiply along paths (A→B→C), so logs make it a
  linear system; median resists rounding and outlier rows.
- **Why not just use the raw amounts?** Units differ by orders of magnitude (BTC ≈ $11.7K,
  Yen ≈ $0.0095); sums would be dominated by currency, not behaviour.
- **Why is the class imbalance a problem?** ~0.6% of accounts are mules (≈1 in 160), so
  accuracy is meaningless; use PR-AUC, recall at fixed precision, and class weights or
  resampling. Report metrics on the test window only.
- **What does drift mean here?** Transaction laundering rate rose 0.075% → 0.177%; the
  model trained on early data sees a different test distribution; so monitor and consider
  retraining.
- **Why are mule labels per window?** An account is a mule only when it handles laundering
  money in that period; most test mules look normal in train, so memorisation fails and
  behavioural features are needed.
- **What are weaknesses?** Rates assumed constant (true here, may not be in real data);
  features are aggregates and ignore graph structure (next step: GNN or graph features);
  accounts are not linked across windows; the mule definition counts any account touching
  a laundering transaction, including victims or intermediaries.
- **How would this scale?** All steps are group-by aggregations; for larger variants
  (HI-Medium/Large) move to Polars/DuckDB or Spark and avoid the Python concat for pairs.

### Numbers worth memorising
5,078,345 txns · 515K accounts · 15 currencies · 72K cross-currency rows (1.4%) ·
60/20/20 split · txn laundering rate 0.075% → 0.107% → 0.177% · mule rate ≈ 0.62%
(3,170 / 1,533 / 2,201 mules) · 19 features · 6.0% of test mules were train mules.

---

## 5. How to reproduce
```bash
python scripts/verify_data.py      # once: writes data/interim/hi_small_trans.parquet
python scripts/build_features.py    # writes data/processed/features_{train,val,test}.parquet
```
