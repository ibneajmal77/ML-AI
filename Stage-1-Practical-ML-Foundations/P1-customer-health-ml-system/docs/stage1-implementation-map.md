# Stage 1 Implementation Map

This file shows exactly how the Stage 1 lessons are implemented in the capstone project, what to read first, what to run, and which artifacts to inspect.

---

## How To Use This Project For Learning

### Read in this order

1. [README.md](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/README.md)
2. [docs/architecture.md](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/docs/architecture.md)
3. [src/p1_customer_health/ml/features.py](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/src/p1_customer_health/ml/features.py)
4. [src/p1_customer_health/ml/train.py](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/src/p1_customer_health/ml/train.py)
5. [src/p1_customer_health/api/main.py](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/src/p1_customer_health/api/main.py)

### Run in this order

1. `python scripts/generate_data.py`
2. `python scripts/train_models.py`
3. `pytest -q`
4. `uvicorn p1_customer_health.api.main:app --reload`

### Inspect in this order

1. classification metrics
2. regression metrics
3. slice analysis
4. failure taxonomy
5. leakage report
6. data-quality report
7. LLM benchmark
8. RL report
9. ONNX status

---

## Topic-By-Topic Stage 1 Map

| Lesson | Topic | Main implementation | Main output |
|---|---|---|---|
| `1.1` | What ML is in practical business terms | [business_framing.py](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/src/p1_customer_health/ml/business_framing.py) | [business_decisions.json](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/artifacts/business/business_decisions.json) |
| `1.2` | Supervised / unsupervised / self-supervised / RL scope | [train.py](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/src/p1_customer_health/ml/train.py), [rl.py](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/src/p1_customer_health/ml/rl.py) | [contextual_bandit_report.json](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/artifacts/reinforcement_learning/contextual_bandit_report.json), [status.json](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/artifacts/self_supervised/status.json) |
| `1.3` | Features, labels, splits, leakage, data quality | [features.py](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/src/p1_customer_health/ml/features.py), [audit.py](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/src/p1_customer_health/ml/audit.py), [leakage.py](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/src/p1_customer_health/ml/leakage.py) | [data_quality_report.json](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/artifacts/data_quality/data_quality_report.json), [leakage_report.json](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/artifacts/leakage/leakage_report.json) |
| `1.4` | Regression | [train.py](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/src/p1_customer_health/ml/train.py) | [regression/metrics.json](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/artifacts/regression/metrics.json) |
| `1.5` | Classification, thresholds, calibration, business cost | [train.py](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/src/p1_customer_health/ml/train.py) | [classification/metrics.json](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/artifacts/classification/metrics.json), [calibrated_model.joblib](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/artifacts/classification/calibrated_model.joblib) |
| `1.6` | Logistic regression | [train.py](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/src/p1_customer_health/ml/train.py) | included in [classification/metrics.json](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/artifacts/classification/metrics.json) |
| `1.7` | Trees, random forest, gradient boosting | [train.py](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/src/p1_customer_health/ml/train.py) | included in [classification/metrics.json](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/artifacts/classification/metrics.json) |
| `1.8` | XGBoost and LightGBM | [train.py](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/src/p1_customer_health/ml/train.py) | [boosting/status.json](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/artifacts/boosting/status.json) |
| `1.9` | Evaluation metrics | [train.py](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/src/p1_customer_health/ml/train.py) | [classification/metrics.json](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/artifacts/classification/metrics.json), [regression/metrics.json](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/artifacts/regression/metrics.json) |
| `1.10` | Class imbalance, sampling, class weights, threshold tuning | [train.py](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/src/p1_customer_health/ml/train.py) | included in [classification/metrics.json](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/artifacts/classification/metrics.json) |
| `1.11` | Feature engineering | [features.py](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/src/p1_customer_health/ml/features.py), [train.py](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/src/p1_customer_health/ml/train.py) | model artifacts and metric outputs |
| `1.12` | Overfitting, underfitting, bias-variance diagnosis | [train.py](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/src/p1_customer_health/ml/train.py) | [classification/fit_diagnosis.csv](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/artifacts/classification/fit_diagnosis.csv), [regression/fit_diagnosis.csv](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/artifacts/regression/fit_diagnosis.csv) |
| `1.13` | Error analysis, slice analysis, failure taxonomy | [failure_taxonomy.py](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/src/p1_customer_health/ml/failure_taxonomy.py), [train.py](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/src/p1_customer_health/ml/train.py) | [slice_analysis.csv](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/artifacts/classification/slice_analysis.csv), [failure_taxonomy_summary.csv](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/artifacts/classification/failure_taxonomy_summary.csv) |
| `1.14` | When classical ML beats LLMs | [llm_benchmark.py](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/src/p1_customer_health/ml/llm_benchmark.py) | [llm_vs_classical_benchmark.json](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/artifacts/llm_benchmark/llm_vs_classical_benchmark.json) |
| `1.15` | Scikit-learn pipelines | [train.py](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/src/p1_customer_health/ml/train.py) | [model.joblib](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/artifacts/classification/model.joblib), [model.joblib](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/artifacts/regression/model.joblib) |
| `1.16` | Packaging and serving basics, joblib, ONNX, REST | [api/main.py](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/src/p1_customer_health/api/main.py), [onnx_export.py](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/src/p1_customer_health/ml/onnx_export.py) | [dense_status.json](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/artifacts/onnx/dense_status.json) |

---

## What To Inspect For Each Lesson

### `1.1`
- [business_decisions.json](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/artifacts/business/business_decisions.json)

### `1.2`
- [contextual_bandit_report.json](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/artifacts/reinforcement_learning/contextual_bandit_report.json)
- [status.json](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/artifacts/self_supervised/status.json)
- [customer_segments.csv](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/artifacts/unsupervised/customer_segments.csv)

### `1.3`
- [data_quality_report.json](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/artifacts/data_quality/data_quality_report.json)
- [leakage_report.json](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/artifacts/leakage/leakage_report.json)

### `1.4`
- [regression/metrics.json](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/artifacts/regression/metrics.json)

### `1.5`
- [classification/metrics.json](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/artifacts/classification/metrics.json)
- [calibration_bins.csv](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/artifacts/classification/calibration_bins.csv)
- [calibrated_calibration_bins.csv](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/artifacts/classification/calibrated_calibration_bins.csv)

### `1.6` to `1.10`
- [classification/metrics.json](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/artifacts/classification/metrics.json)
- [boosting/status.json](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/artifacts/boosting/status.json)

### `1.11`
- [features.py](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/src/p1_customer_health/ml/features.py)

### `1.12`
- [classification/fit_diagnosis.csv](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/artifacts/classification/fit_diagnosis.csv)
- [regression/fit_diagnosis.csv](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/artifacts/regression/fit_diagnosis.csv)

### `1.13`
- [slice_analysis.csv](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/artifacts/classification/slice_analysis.csv)
- [failure_taxonomy_summary.csv](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/artifacts/classification/failure_taxonomy_summary.csv)

### `1.14`
- [llm_vs_classical_benchmark.json](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/artifacts/llm_benchmark/llm_vs_classical_benchmark.json)

### `1.15`
- [classification/model.joblib](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/artifacts/classification/model.joblib)
- [regression/model.joblib](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/artifacts/regression/model.joblib)

### `1.16`
- [api/main.py](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/src/p1_customer_health/api/main.py)
- [dense_status.json](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/artifacts/onnx/dense_status.json)

---

## Strict Verification Status

- data generation ran
- full training pipeline ran
- artifacts were generated
- tests passed

See also:
- [docs/coverage-map.md](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/docs/coverage-map.md)
