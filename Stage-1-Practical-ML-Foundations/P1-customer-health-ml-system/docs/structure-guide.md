# Structure Guide

This guide explains the repository using enterprise-style responsibilities, not by the order in which the files were written.

## Top-Level Folders

### `scripts/`

Human-facing entry points only.

- [generate_data.py](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/scripts/generate_data.py)
- [train_models.py](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/scripts/train_models.py)
- [run_api.py](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/scripts/run_api.py)

### `src/p1_customer_health/domain/`

Business data contract and dataset-building logic.

- [dataset.py](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/src/p1_customer_health/domain/dataset.py)
- [synthetic_data.py](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/src/p1_customer_health/domain/synthetic_data.py)

### `src/p1_customer_health/training/`

Offline model-building flow.

- [preprocessing.py](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/src/p1_customer_health/training/preprocessing.py)
- [metrics.py](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/src/p1_customer_health/training/metrics.py)
- [classification.py](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/src/p1_customer_health/training/classification.py)
- [regression.py](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/src/p1_customer_health/training/regression.py)
- [unsupervised.py](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/src/p1_customer_health/training/unsupervised.py)
- [orchestration.py](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/src/p1_customer_health/training/orchestration.py)

### `src/p1_customer_health/analysis/`

Reports, audits, and structured interpretation.

- [business_framing.py](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/src/p1_customer_health/analysis/business_framing.py)
- [audit.py](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/src/p1_customer_health/analysis/audit.py)
- [leakage.py](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/src/p1_customer_health/analysis/leakage.py)
- [failure_taxonomy.py](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/src/p1_customer_health/analysis/failure_taxonomy.py)

### `src/p1_customer_health/experiments/`

Bounded advanced tracks that are useful but not required for basic API serving.

- [boosting.py](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/src/p1_customer_health/experiments/boosting.py)
- [self_supervised.py](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/src/p1_customer_health/experiments/self_supervised.py)
- [llm_benchmark.py](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/src/p1_customer_health/experiments/llm_benchmark.py)
- [rl.py](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/src/p1_customer_health/experiments/rl.py)
- [onnx_export.py](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/src/p1_customer_health/experiments/onnx_export.py)

### `src/p1_customer_health/serving/`

Inference-time orchestration.

- [prediction_service.py](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/src/p1_customer_health/serving/prediction_service.py)

### `src/p1_customer_health/api/`

HTTP surface only.

- [main.py](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/src/p1_customer_health/api/main.py)
- [schemas.py](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/src/p1_customer_health/api/schemas.py)

### `src/p1_customer_health/app/`

Runtime settings and app-level configuration.

- [settings.py](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/src/p1_customer_health/app/settings.py)

### `src/p1_customer_health/ml/`

Compatibility wrappers only. Keep new work out of this package.
