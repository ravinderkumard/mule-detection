"""Config-driven pipeline: ingest -> validate -> features -> train -> evaluate.

Every setting comes from ``configs/pipeline.yaml``. Each stage reads and writes named files
(the paths in the config), so any stage can be rerun alone. Every invocation is one MLflow
run that records the full config, the sha256 of the raw CSV and of every file the stages
read or wrote, and the git commit (+ whether the tree was dirty).

    python -m muledetect.pipeline all
    python -m muledetect.pipeline features train evaluate
    python -m muledetect.pipeline --config configs/pipeline.yaml train
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import yaml

from muledetect.paths import ROOT

AML_COLS = [
    "timestamp", "from_bank", "from_account", "to_bank", "to_account",
    "amount_received", "receiving_currency", "amount_paid",
    "payment_currency", "payment_format", "is_laundering",
]
SPLITS = ("train", "val", "test")
DEFAULT_CONFIG = ROOT / "configs" / "pipeline.yaml"


# --------------------------------------------------------------------------- config / paths
def load_config(path: Path = DEFAULT_CONFIG) -> dict:
    return yaml.safe_load(Path(path).read_text())


def p(cfg: dict, key: str) -> Path:
    return ROOT / cfg["paths"][key]


def feature_path(cfg: dict, split: str) -> Path:
    return p(cfg, "features_dir") / f"features_{split}.parquet"


def stage_io(cfg: dict) -> dict[str, dict[str, list[Path]]]:
    """Named inputs/outputs per stage (also what gets hashed into the MLflow run)."""
    feats = [feature_path(cfg, s) for s in SPLITS]
    return {
        "ingest": {"in": [p(cfg, "raw_csv")], "out": [p(cfg, "interim")]},
        "validate": {"in": [p(cfg, "interim")], "out": [p(cfg, "validation_report")]},
        "features": {"in": [p(cfg, "interim")], "out": feats},
        "train": {"in": feats[:2], "out": [p(cfg, "model")]},
        "evaluate": {"in": [p(cfg, "model"), *feats[1:]], "out": [p(cfg, "metrics")]},
    }


def _need(path: Path, stage: str, producer: str) -> Path:
    if not path.exists():
        sys.exit(f"[{stage}] missing input {path.relative_to(ROOT)} - run stage '{producer}' first")
    return path


# --------------------------------------------------------------------------- stages
def ingest(cfg: dict) -> None:
    """Raw HI-Small CSV -> typed parquet with src/dst account ids (bank + account)."""
    df = pd.read_csv(
        _need(p(cfg, "raw_csv"), "ingest", "download_data.py"), header=0, names=AML_COLS,
        dtype={"from_bank": "int32", "to_bank": "int32", "from_account": "string",
               "to_account": "string", "is_laundering": "int8"})
    df["timestamp"] = pd.to_datetime(df["timestamp"], format="%Y/%m/%d %H:%M")
    df["src"] = df["from_bank"].astype(str) + "_" + df["from_account"]
    df["dst"] = df["to_bank"].astype(str) + "_" + df["to_account"]
    out = p(cfg, "interim")
    out.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(out, index=False)
    print(f"ingest: {len(df):,} transactions -> {out.relative_to(ROOT)}")


def validate(cfg: dict) -> None:
    """Fail loudly (DataContractError) if the transactions break the schema."""
    from muledetect.schemas import validate_transactions
    df = pd.read_parquet(_need(p(cfg, "interim"), "validate", "ingest"))
    validate_transactions(df, "hi_small_trans")
    report = {"rows": len(df), "laundering": int(df.is_laundering.sum()),
              "accounts": int(pd.concat([df.src, df.dst]).nunique()),
              "start": str(df.timestamp.min()), "end": str(df.timestamp.max()), "passed": True}
    out = p(cfg, "validation_report")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(f"validate: ok {report}")


def features(cfg: dict) -> None:
    from muledetect.features import build_split_tables
    from muledetect.schemas import validate_features
    df = pd.read_parquet(_need(p(cfg, "interim"), "features", "ingest"))
    tables = build_split_tables(df, cfg["split"]["train_frac"], cfg["split"]["val_frac"],
                                cfg["features"]["behaviour"])
    p(cfg, "features_dir").mkdir(parents=True, exist_ok=True)
    for name, table in tables.items():
        flat = validate_features(table.reset_index(), f"features_{name}")
        flat.to_parquet(feature_path(cfg, name), index=False)
        print(f"features: {name} {len(flat):,} accounts, mules {int(flat.is_mule.sum()):,}")


def _xy(cfg: dict, split: str, stage: str) -> tuple[pd.DataFrame, pd.Series]:
    t = pd.read_parquet(_need(feature_path(cfg, split), stage, "features"))
    return t[cfg["features"]["model_columns"]], t[cfg["features"]["target"]]


def train(cfg: dict) -> None:
    import lightgbm as lgb
    X_tr, y_tr = _xy(cfg, "train", "train")
    X_va, y_va = _xy(cfg, "val", "train")
    model = lgb.LGBMClassifier(**cfg["model"]["params"], random_state=cfg["seed"], verbose=-1)
    model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], eval_metric="average_precision",
              callbacks=[lgb.early_stopping(cfg["model"]["early_stopping_rounds"], verbose=False)])
    out = p(cfg, "model")
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, out)
    print(f"train: best_iteration={model.best_iteration_} -> {out.relative_to(ROOT)}")


def evaluate(cfg: dict) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score
    model = joblib.load(_need(p(cfg, "model"), "evaluate", "train"))
    metrics: dict[str, float] = {"best_iteration": int(model.best_iteration_)}
    for split in ("val", "test"):
        X, y = _xy(cfg, split, "evaluate")
        score = model.predict_proba(X)[:, 1]
        metrics[f"{split}_pr_auc"] = float(average_precision_score(y, score))
        metrics[f"{split}_roc_auc"] = float(roc_auc_score(y, score))
        metrics[f"{split}_base_rate"] = float(y.mean())
        order = np.argsort(-score, kind="stable")
        for frac in cfg["evaluate"]["top_k_fracs"]:
            k = max(1, int(round(frac * len(y))))
            hits = float(y.to_numpy()[order[:k]].sum())
            metrics[f"{split}_recall_top{frac:g}"] = hits / float(y.sum())
            metrics[f"{split}_precision_top{frac:g}"] = hits / k
    out = p(cfg, "metrics")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(metrics, indent=2))
    print("evaluate:", json.dumps(metrics, indent=2))
    return metrics


STAGES = {"ingest": ingest, "validate": validate, "features": features,
          "train": train, "evaluate": evaluate}


# --------------------------------------------------------------------------- lineage / MLflow
def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def git_info() -> dict[str, str]:
    def run(*a: str) -> str:
        return subprocess.run(["git", *a], cwd=ROOT, capture_output=True, text=True).stdout.strip()
    return {"git.commit": run("rev-parse", "HEAD") or "unknown",
            "git.branch": run("rev-parse", "--abbrev-ref", "HEAD"),
            # dirty = tracked or untracked source changes outside ignored data/artifacts
            "git.dirty": str(bool(run("status", "--porcelain"))).lower()}


def flatten(d: dict, prefix: str = "") -> dict[str, str]:
    out = {}
    for k, v in d.items():
        key = f"{prefix}{k}"
        if isinstance(v, dict):
            out.update(flatten(v, key + "."))
        else:
            out[key] = str(v)[:500]  # MLflow param value limit is 500 chars (list -> its repr)
    return out


def run_stages(cfg: dict, config_path: Path, stages: list[str]) -> None:
    import mlflow

    io = stage_io(cfg)
    mlflow.set_tracking_uri(f"sqlite:///{ROOT / cfg['mlflow']['db']}")
    mlflow.set_experiment(cfg["mlflow"]["experiment"])
    with mlflow.start_run(run_name="+".join(stages)):
        mlflow.set_tags({**git_info(), "stages": ",".join(stages),
                         "config.sha256": hashlib.sha256(Path(config_path).read_bytes()).hexdigest()})
        mlflow.log_params({k: v for k, v in flatten(cfg).items() if not k.startswith("mlflow.")})
        mlflow.log_artifact(str(config_path), "config")
        raw = p(cfg, "raw_csv")
        if raw.exists():
            mlflow.set_tag("data.raw_csv.sha256", sha256(raw))

        manifest: dict[str, dict] = {}
        for s in stages:
            for path in io[s]["in"]:  # hash inputs before the stage runs
                if path.exists():
                    manifest.setdefault(str(path.relative_to(ROOT)), {"sha256": sha256(path), "role": f"input:{s}"})
            t0 = time.time()
            result = STAGES[s](cfg)
            mlflow.log_metric(f"time_{s}_s", round(time.time() - t0, 1))
            for path in io[s]["out"]:
                manifest[str(path.relative_to(ROOT))] = {"sha256": sha256(path), "role": f"output:{s}"}
            if s == "evaluate":
                mlflow.log_metrics({k: v for k, v in result.items()})
        mlflow.log_dict(manifest, "data_manifest.json")
        if "train" in stages:
            mlflow.log_artifact(str(p(cfg, "model")), "model")
        print(f"\nMLflow run {mlflow.active_run().info.run_id} (mlflow ui --backend-store-uri "
              f"sqlite:///{cfg['mlflow']['db']})")


def main(argv: list[str] | None = None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    ap.add_argument("stages", nargs="+", choices=[*STAGES, "all"])
    ap.add_argument("--config", type=Path, default=DEFAULT_CONFIG)
    a = ap.parse_args(argv)
    stages = list(STAGES) if "all" in a.stages else [s for s in STAGES if s in a.stages]
    run_stages(load_config(a.config), a.config.resolve(), stages)


if __name__ == "__main__":
    main()
