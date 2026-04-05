# P1 Customer Health ML System

This is the Stage 1 capstone project for practical ML foundations. It is one bounded business system designed to cover the full Stage 1 surface with working code, not disconnected demos.

The project simulates a SaaS customer-health workflow and implements:

- churn classification
- revenue-change regression
- customer segmentation with clustering
- anomaly detection
- optional self-supervised text-embedding baseline
- evaluation, threshold tuning, calibration reporting, and slice analysis
- model packaging with `joblib`
- a small FastAPI inference service

## Why this project fits Stage 1

One domain covers nearly all Stage 1 topics naturally:

- structured features: spend, logins, tickets, tenure, usage
- categorical features: plan, region, industry, contract type
- time-derived features: snapshot date, days since last activity
- text basics: support notes
- supervised learning: churn and revenue prediction
- unsupervised learning: segmentation and anomaly detection
- self-supervised learning: optional sentence-embedding workflow
- production basics: pipeline, packaging, and serving

## Repo shape

```text
P1-customer-health-ml-system/
├── data/
│   ├── raw/
│   └── processed/
├── artifacts/
│   ├── classification/
│   ├── regression/
│   ├── unsupervised/
│   └── self_supervised/
├── docs/
├── scripts/
├── src/p1_customer_health/
│   ├── api/
│   └── ml/
└── tests/
```

## Local run

```bash
uv sync --dev
uv run python scripts/generate_data.py
uv run python scripts/train_models.py
uv run uvicorn p1_customer_health.api.main:app --reload
```

## Main outputs

- `artifacts/classification/model.joblib`
- `artifacts/classification/metrics.json`
- `artifacts/classification/slice_analysis.csv`
- `artifacts/classification/calibration_bins.csv`
- `artifacts/regression/model.joblib`
- `artifacts/regression/metrics.json`
- `artifacts/unsupervised/segmenter.joblib`
- `artifacts/unsupervised/anomaly_detector.joblib`
- `artifacts/self_supervised/status.json`
- `artifacts/boosting/status.json`

## Core API endpoints

- `GET /health`
- `POST /v1/predict/customer-health`

## What the project teaches

- how to define features, labels, and time-safe splits
- how to compare regression and classification models honestly
- how thresholds and calibration change business decisions
- how imbalance changes metric choice and operating policy
- how feature engineering and sklearn pipelines fit together
- how to run slice analysis and basic failure analysis
- how to package and serve a fitted artifact

## Stage 1 lesson coverage

See [docs/coverage-map.md](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/docs/coverage-map.md) for the full lesson-to-project mapping.
