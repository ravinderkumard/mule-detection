# Day 7 — Reproducible pipeline and experiment tracking

## Summary
Day 7 turns the loose scripts into one config-driven pipeline: `ingest → validate → features → train → evaluate`.
Every setting (paths, split fractions, model feature list, LightGBM parameters, seed) lives in
`configs/pipeline.yaml`. Each stage reads and writes named files, so any stage can be rerun alone, and a
Makefile wraps it (`make all` rebuilds everything from the raw CSV in about 40 s). Every invocation is one
local MLflow run that records the config, the sha256 of the raw CSV, the hash of every file the stages
touched, and the git commit. Any result can be traced back to the exact data, code and settings that
produced it.

## Run
```
make all                 # one MLflow run covering every stage
make features            # or ingest / validate / train / evaluate, each its own run
make ui                  # MLflow UI at http://127.0.0.1:5000 (experiment: mule-detection)
python -m muledetect.pipeline features train evaluate   # same thing without make
```
Needs `data/raw/ibm_aml/HI-Small_Trans.csv` (`scripts/download_data.py`). `make clean` deletes derived files only.

## Stages (inputs → outputs, all paths from the config)
| Stage | Reads | Writes |
|---|---|---|
| ingest | raw CSV | `data/interim/hi_small_trans.parquet` (typed, `src`/`dst` = bank_account) |
| validate | interim parquet | `reports/validation.json` (raises `DataContractError` on schema breach) |
| features | interim parquet | `data/processed/features_{train,val,test}.parquet` (schema-validated before writing) |
| train | features train + val | `models/lgbm.joblib` (LightGBM, early stopping on val average precision) |
| evaluate | model + features val/test | `reports/metrics.json` (PR-AUC, ROC-AUC, recall/precision at top 1% and 5%) |

A stage with a missing input exits with a message naming the stage to run first.

## What each MLflow run records
- **Config:** every value as a param, plus the YAML itself as an artifact and its sha256 as a tag.
- **Code:** `git.commit`, `git.branch`, `git.dirty` tags.
- **Data:** `data.raw_csv.sha256` tag, and `data_manifest.json` with the sha256 of every input and output file of the stages run.
- **Results:** stage timings, evaluate metrics, and the model artifact when `train` ran.

## Verification
- `make all` run twice: identical raw-CSV, interim, features and metrics hashes, test PR-AUC 0.1655 both times.
  Only the pickled model's bytes differ between runs (metrics identical).
- `git.dirty` is `true` until the Day 7 files are committed; a run from a clean tree records `false`.
- 23 existing tests still pass.

## Caveats
- **Degenerate model:** early stopping on val average precision stops at iteration 1 (one tree; val PR-AUC 0.118,
  test 0.165). This reflects the train→later-window instability from Day 6, not a pipeline bug. Fix by changing
  the early-stopping metric or using a fixed number of rounds in the config.
- `fmt_out_reinvestment` and `self_transfers` are excluded from the model columns (train-only artefacts).
- Elliptic is not part of the pipeline (`scripts/verify_data.py` still handles it).
- `make` does not track file dependencies; choose which stages to rerun after a change.

## Files
`configs/pipeline.yaml`, `src/muledetect/pipeline.py`, `Makefile`; `mlflow` and `pyyaml` added to `requirements.txt`;
`mlflow.db` and `mlartifacts/` git-ignored.
