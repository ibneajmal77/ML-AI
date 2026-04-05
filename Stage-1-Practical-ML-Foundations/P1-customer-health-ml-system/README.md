# P1 Customer Health ML System

This is the Stage 1 capstone project for practical ML foundations. It is one bounded business system designed to cover the full Stage 1 surface with working code, not disconnected demos.

The project simulates a SaaS customer-health workflow and implements:

- churn classification
- revenue-change regression
- customer segmentation with clustering
- anomaly detection
- optional self-supervised text-embedding baseline
- evaluation, threshold tuning, calibration reporting, and slice analysis
- business framing, data-quality auditing, leakage checks, and failure taxonomy
- model packaging with `joblib`
- ONNX export path
- contextual-bandit RL simulation for retention actions
- classical-vs-LLM-style text benchmark
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
- `artifacts/classification/calibrated_model.joblib`
- `artifacts/classification/failure_taxonomy_summary.csv`
- `artifacts/classification/fit_diagnosis.csv`
- `artifacts/regression/model.joblib`
- `artifacts/regression/metrics.json`
- `artifacts/regression/fit_diagnosis.csv`
- `artifacts/unsupervised/segmenter.joblib`
- `artifacts/unsupervised/anomaly_detector.joblib`
- `artifacts/self_supervised/status.json`
- `artifacts/boosting/status.json`
- `artifacts/business/business_decisions.json`
- `artifacts/data_quality/data_quality_report.json`
- `artifacts/leakage/leakage_report.json`
- `artifacts/llm_benchmark/llm_vs_classical_benchmark.json`
- `artifacts/reinforcement_learning/contextual_bandit_report.json`
- `artifacts/onnx/status.json`

## Core API endpoints

- `GET /health`
- `POST /v1/predict/customer-health`

## What the project teaches

- how to define features, labels, and time-safe splits
- how to code the rules-vs-ML business decision explicitly
- how to compare regression and classification models honestly
- how thresholds and calibration change business decisions
- how resampling, class weights, and calibration differ in implementation
- how imbalance changes metric choice and operating policy
- how feature engineering and sklearn pipelines fit together
- how to run slice analysis, failure taxonomy, and fit diagnosis
- how to audit data quality and detect leakage risks
- how an RL-style retention policy can be simulated without replacing the main predictive workflow
- how to compare classical text ML against an LLM-style baseline
- how to package and serve a fitted artifact

## Stage 1 lesson coverage

See [docs/coverage-map.md](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/docs/coverage-map.md) for the full lesson-to-project mapping.
