# World Cricket ML Foundations

This project is a Stage-1 practical ML foundations capstone built around men's T20I cricket team data. It turns a rolling two-year quarterly-updated slice of international match history into a production-style learning system: data ingestion, data quality checks, leakage protection, regression, classification, clustering, anomaly detection, text-based proxy experiments, model packaging, and a small FastAPI serving layer.

The problem framing is intentionally market-relevant: which teams look strongest for the next global event cycle, which established teams show downfall risk, and which underdog teams look capable of surprising the field.

## Why This Project Works For Stage-1

One dataset family supports every major Stage-1 concept:

- supervised learning: predict `future_win_rate_next_5` and `dominant_next_cycle`
- unsupervised learning: segment teams and flag anomalous team states
- self-supervised proxy: learn text embeddings from generated match narratives
- RL scope: contextual-bandit-style toss decision simulation
- leakage, splits, thresholds, calibration, business cost, error analysis, and packaging

This is not a notebook dump. The repository is organized like a small ML system.

## Data Scope

- Format: men's T20 internationals
- Team universe: curated top-20 T20I nations used by the project configuration
- History policy: rolling two-year window over the last eight completed quarters
- Refresh cadence: rerun data build and training every 3 months
- Current generated window on this machine: `2024-04-01` to `2026-03-31`
- Raw source: Cricsheet JSON archive
- Generated rows depend on the current quarterly window and the teams found in the archive

## Repository Layout

```text
world-cricket-ml-foundations/
|-- data/
|-- artifacts/
|-- docs/
|-- scripts/
|-- src/world_cricket_ml/
|   |-- api/
|   |-- app/
|   |-- domain/
|   |-- training/
|   |-- analysis/
|   |-- experiments/
|   `-- serving/
`-- tests/
```

## Run Order

```powershell
python scripts\fetch_data.py
python scripts\train_models.py
python -m pytest -q
python scripts\run_api.py
```

Swagger is available at `http://127.0.0.1:8000/docs` after the API starts.

## Main Deliverables

- Dataset build outputs in `data/raw/`
- Classification artifacts in `artifacts/classification/`
- Regression artifacts in `artifacts/regression/`
- Business framing in `artifacts/business/business_decisions.json`
- Data quality and leakage audits in `artifacts/data_quality/` and `artifacts/leakage/`
- Team segmentation and anomaly outputs in `artifacts/unsupervised/`
- Advanced Stage-1 experiments in `artifacts/self_supervised/`, `artifacts/reinforcement_learning/`, `artifacts/boosting/`, `artifacts/llm_benchmark/`, and `artifacts/onnx/`

## Learning Path

1. Read `docs/architecture.md`
2. Read `docs/coverage-map.md`
3. Run the data build script
4. Inspect `data/raw/world_cricket_snapshots.csv`
5. Run training and inspect artifacts folder by folder
6. Read `docs/stage1-implementation-map.md`
7. Use the API to inspect team-level predictions

## Data Sources

- Cricsheet downloads: https://cricsheet.org/downloads/
- ICC rankings landing page used as the project's team-universe reference point: https://www.icc-cricket.com/rankings/team-rankings/mens/t20i
