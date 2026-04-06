# Coverage Map

This file maps Stage 1 lesson topics to concrete implementation areas in the capstone.

| Lesson | Topic | Where it is implemented |
|---|---|---|
| `1.1` | What ML is in practical business terms | `src/p1_customer_health/analysis/business_framing.py` and `artifacts/business/business_decisions.json` |
| `1.2` | Supervised / unsupervised / self-supervised / RL scope | classification, regression, clustering, self-supervised benchmark, and contextual bandit in `src/p1_customer_health/training` and `src/p1_customer_health/experiments` |
| `1.3` | Features, labels, splits, leakage, data quality | `src/p1_customer_health/domain/dataset.py`, `src/p1_customer_health/analysis/audit.py`, and `src/p1_customer_health/analysis/leakage.py` |
| `1.4` | Regression | `src/p1_customer_health/training/regression.py` |
| `1.5` | Classification, thresholds, calibration, business cost | `src/p1_customer_health/training/classification.py` |
| `1.6` | Logistic regression | `src/p1_customer_health/training/classification.py` |
| `1.7` | Trees, random forest, gradient boosting | `src/p1_customer_health/training/classification.py` |
| `1.8` | XGBoost and LightGBM | `src/p1_customer_health/experiments/boosting.py` and `artifacts/boosting/status.json` |
| `1.9` | Evaluation metrics | `src/p1_customer_health/training/metrics.py` plus classification/regression outputs |
| `1.10` | Class imbalance, sampling, class weights, threshold tuning | `src/p1_customer_health/training/classification.py` |
| `1.11` | Feature engineering | `src/p1_customer_health/domain/dataset.py` and `src/p1_customer_health/training/preprocessing.py` |
| `1.12` | Overfitting, underfitting, bias-variance diagnosis | fit diagnosis written by `classification.py` and `regression.py` |
| `1.13` | Error analysis, slice analysis, failure taxonomy | `src/p1_customer_health/analysis/failure_taxonomy.py` and classification slice outputs |
| `1.14` | Classical ML vs LLMs | `src/p1_customer_health/experiments/llm_benchmark.py` |
| `1.15` | Scikit-learn pipelines | `src/p1_customer_health/training/preprocessing.py` and task trainers |
| `1.16` | Packaging and serving basics, joblib, ONNX, REST | `src/p1_customer_health/serving/prediction_service.py`, `src/p1_customer_health/api/main.py`, and `src/p1_customer_health/experiments/onnx_export.py` |
