# Reproducible pipeline. Settings live in configs/pipeline.yaml.
# Each target is one MLflow run; `make all` is a single run covering every stage.
PY ?= .venv/bin/python
PIPE = $(PY) -m muledetect.pipeline

.PHONY: all ingest validate features train evaluate ui test clean
all:      ; $(PIPE) all
ingest:   ; $(PIPE) ingest
validate: ; $(PIPE) validate
features: ; $(PIPE) features
train:    ; $(PIPE) train
evaluate: ; $(PIPE) evaluate
ui:       ; $(PY) -m mlflow ui --backend-store-uri sqlite:///mlflow.db
test:     ; $(PY) -m pytest tests
# Removes derived data/models only (raw CSV and MLflow history are kept).
clean:
	rm -f data/interim/hi_small_trans.parquet data/processed/features_*.parquet \
	      models/lgbm.joblib reports/validation.json reports/metrics.json
