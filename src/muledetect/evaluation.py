"""Business evaluation: rules baseline, alert-budget metrics, calibration, PR curve.

Accounts are ranked by a score; an investigator team can only review the top K per period
(the *alert budget*). We therefore report recall / precision at fixed K, and the lift of the
model over the hand-written rules a bank already runs.
"""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, brier_score_loss, precision_recall_curve, roc_auc_score


# --------------------------------------------------------------------------- rules baseline
def rule_flags(df: pd.DataFrame, rules: dict[str, dict]) -> pd.DataFrame:
    """Boolean column per rule. Each rule is ``{col: threshold}`` pairs, ANDed; a column named
    ``min_<col>`` means ``df[col] >= threshold`` (only ``>=`` rules are used)."""
    flags = {}
    for name, conds in rules.items():
        ok = pd.Series(True, index=df.index)
        for col, thr in conds.items():
            ok &= df[col] >= thr
        flags[name] = ok
    return pd.DataFrame(flags)


def rules_score(df: pd.DataFrame, rules: dict[str, dict], tiebreak: str = "amount_out") -> np.ndarray:
    """Rank key for the rules engine: number of rules fired, ties broken by USD volume moved
    (how a triage queue is usually ordered). Accounts with 0 rules fired are not alerts.
    Returned as one float so the generic top-K code can be used for model and rules alike."""
    n = rule_flags(df, rules).sum(axis=1).to_numpy(dtype=float)
    vol = df[tiebreak].to_numpy(dtype=float)
    rank = pd.Series(vol).rank(pct=True, method="average").to_numpy()   # in (0, 1]
    return n + rank * 0.5 * (n > 0)                                      # never lifts 0 above 1


# --------------------------------------------------------------------------- metrics
def top_k_metrics(y: np.ndarray, score: np.ndarray, ks: list[int]) -> dict[int, dict[str, float]]:
    """Recall / precision / hits / lift over random at the K highest-scored accounts."""
    y = np.asarray(y)
    order = np.argsort(-score, kind="stable")
    base = y.mean()
    out = {}
    for k in ks:
        hits = float(y[order[:k]].sum())
        out[k] = {"hits": hits, "precision": hits / k, "recall": hits / y.sum(),
                  "lift_vs_random": hits / k / base}
    return out


def expected_calibration_error(y: np.ndarray, prob: np.ndarray, bins: int = 10) -> float:
    """ECE with equal-count bins (positives are rare, so equal-width bins would be nearly empty)."""
    order = np.argsort(prob)
    err = 0.0
    for idx in np.array_split(order, bins):
        err += len(idx) / len(y) * abs(y[idx].mean() - prob[idx].mean())
    return float(err)


# --------------------------------------------------------------------------- calibration
class PlattCalibrator:
    """Sigmoid calibration on the model's log-odds, fit on validation. Strictly monotone, so
    it changes the probabilities but never the ranking (alert-budget metrics are unaffected)."""

    def fit(self, raw_prob: np.ndarray, y: np.ndarray) -> "PlattCalibrator":
        self.lr_ = LogisticRegression(C=1e6, max_iter=1000).fit(self._logit(raw_prob), y)
        return self

    def predict(self, raw_prob: np.ndarray) -> np.ndarray:
        return self.lr_.predict_proba(self._logit(raw_prob))[:, 1]

    @staticmethod
    def _logit(p: np.ndarray) -> np.ndarray:
        p = np.clip(np.asarray(p, dtype=float), 1e-7, 1 - 1e-7)
        return np.log(p / (1 - p)).reshape(-1, 1)


# --------------------------------------------------------------------------- full evaluation
def evaluate_scores(y: np.ndarray, scores: dict[str, np.ndarray], ks: list[int]) -> dict[str, dict]:
    """Per-scorer PR-AUC, ROC-AUC and top-K metrics."""
    res = {}
    for name, s in scores.items():
        res[name] = {"pr_auc": float(average_precision_score(y, s)), "roc_auc": float(roc_auc_score(y, s)),
                     "top_k": top_k_metrics(y, s, ks)}
    return res


def calibration_summary(y: np.ndarray, raw: np.ndarray, cal: np.ndarray) -> dict[str, float]:
    return {"brier_raw": float(brier_score_loss(y, raw)), "brier_calibrated": float(brier_score_loss(y, cal)),
            "ece_raw": expected_calibration_error(y, raw), "ece_calibrated": expected_calibration_error(y, cal),
            "mean_prob_raw": float(raw.mean()), "mean_prob_calibrated": float(cal.mean()),
            "base_rate": float(y.mean())}


def pr_curve_figure(y: np.ndarray, scores: dict[str, np.ndarray], ks: list[int], title: str):
    """Precision-recall curves for every scorer, with the top-K operating points marked."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 5))
    colours = {"lightgbm": "#1f77b4", "rules": "#d62728"}
    for name, s in scores.items():
        prec, rec, _ = precision_recall_curve(y, s)
        ax.plot(rec, prec, label=f"{name} (PR-AUC {average_precision_score(y, s):.3f})",
                color=colours.get(name), lw=1.8)
        m = top_k_metrics(y, s, ks)
        ax.scatter([m[k]["recall"] for k in ks], [m[k]["precision"] for k in ks], color=colours.get(name),
                   s=28, zorder=3)
        for k in ks:
            ax.annotate(f"{k}", (m[k]["recall"], m[k]["precision"]), textcoords="offset points",
                        xytext=(4, 4), fontsize=7, color=colours.get(name))
    ax.axhline(y.mean(), ls="--", color="grey", lw=1, label=f"random ({y.mean():.4f})")
    ax.set(xlabel="Recall", ylabel="Precision", title=title, xlim=(0, 1), ylim=(0, 1.02))
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    return fig


def render_report(cfg: dict, res: dict, cal: dict, ks: list[int], n: int, n_mules: int,
                  best_iter: int, val_pr_auc: float) -> str:
    """Markdown evaluation report (test window)."""
    lg, ru = res["lightgbm"], res["rules"]
    rows = []
    for k in ks:
        a, b = lg["top_k"][k], ru["top_k"][k]
        lift_r = a["recall"] / b["recall"] if b["recall"] else float("inf")
        rows.append(f"| {k:,} | {b['hits']:.0f} | {b['precision']:.1%} | {b['recall']:.1%} | "
                    f"{a['hits']:.0f} | {a['precision']:.1%} | {a['recall']:.1%} | "
                    f"{a['hits'] - b['hits']:+.0f} | {lift_r:.2f}x |")
    rule_lines = "\n".join(f"- **{name}**: " + " and ".join(f"`{c} >= {t}`" for c, t in conds.items())
                           for name, conds in cfg["rules"]["definitions"].items())
    n_alerts = res["rules"]["n_alerts"]
    better = cal["ece_calibrated"] < cal["ece_raw"]
    cal_verdict = ("On test it lowers calibration error." if better else
                   "On test it did **not** help (ECE rose): the validation window's mule rate differs from test's "
                   "(0.65% vs 0.62%), so the fitted offset does not carry over. With weight 2 the raw probabilities "
                   "were already close to the true rate; use the raw ones unless the rates are re-fit on fresher data.")
    return f"""# Day 8 - Baseline model and business evaluation

Test window: {n:,} accounts, {n_mules:,} mules ({n_mules / n:.3%} base rate). Trained on the earliest 60% of
time, early-stopped and calibrated on the next 20%, scored here on the final 20% (never seen).

## Headline
Judged by what an investigation team catches for a fixed workload, LightGBM vs. the rules engine:

| Alert budget (accounts reviewed) | Rules: mules caught | Rules precision | Rules recall | LightGBM: mules caught | LightGBM precision | LightGBM recall | Extra mules caught | Recall lift |
|---|---|---|---|---|---|---|---|---|
{chr(10).join(rows)}

Overall ranking quality: PR-AUC **{lg['pr_auc']:.3f}** (LightGBM) vs **{ru['pr_auc']:.3f}** (rules) vs
{n_mules / n:.4f} for random. ROC-AUC {lg['roc_auc']:.3f} vs {ru['roc_auc']:.3f}.

![PR curve](pr_curve.png)

## Rules baseline (stands in for the bank's existing rules engine)
Fires when an account trips at least one rule; accounts are ordered by number of rules tripped, then by USD volume.
{rule_lines}

On the test window the rules alert on {n_alerts:,} accounts ({n_alerts / n:.1%}), far more than any team can
review, so the order inside the queue is what matters. Thresholds were hand-set by looking at the train/validation
windows only. If the budget exceeds the number of alerts, the rules have nothing more to rank.

## LightGBM
- Early stopping on validation average precision (patience {cfg['model']['early_stopping_rounds']}); stopped at iteration {best_iter}
  (validation PR-AUC {val_pr_auc:.3f}).
- Class imbalance: `scale_pos_weight={cfg['model']['params']['scale_pos_weight']}`. On this temporal split heavy
  weighting was *harmful*: weights of 10 or more stopped at 1-5 trees (val PR-AUC 0.04-0.12) while weight 1-2 trained
  properly (a validation-only sweep; see the comment in `configs/pipeline.yaml`).
- Calibration: Platt (sigmoid) on validation log-odds. It is monotone, so ranking and the table above are unchanged.
  {cal_verdict}

| Calibration (test) | Raw | Calibrated |
|---|---|---|
| Brier score | {cal['brier_raw']:.5f} | {cal['brier_calibrated']:.5f} |
| Expected calibration error | {cal['ece_raw']:.5f} | {cal['ece_calibrated']:.5f} |
| Mean predicted probability (true rate {cal['base_rate']:.5f}) | {cal['mean_prob_raw']:.5f} | {cal['mean_prob_calibrated']:.5f} |

## How to read this
Investigators care about mules caught per unit of review work. At a budget of {ks[0]:,} accounts the model hands
them {lg['top_k'][ks[0]]['hits']:.0f} mules vs {ru['top_k'][ks[0]]['hits']:.0f} from the rules queue;
at {ks[-1]:,} it is {lg['top_k'][ks[-1]]['hits']:.0f} vs {ru['top_k'][ks[-1]]['hits']:.0f}.

## Caveats
- Early stopping and Platt calibration share the validation split, so calibration metrics on validation would be
  optimistic; the figures above are on test.
- Test mules: {n_mules:,}. At K=100 one extra mule moves precision by 1 point, so small gaps there are noise.
- Window features: "new counterparties" is the number of distinct counterparties inside the window (no earlier
  history is available in this dataset), so the rule is a proxy.
- HI-Small is synthetic; absolute numbers will not transfer to a real bank, the comparison method does.
"""
