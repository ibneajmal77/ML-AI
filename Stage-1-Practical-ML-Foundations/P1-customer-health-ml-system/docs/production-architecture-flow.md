# Production Architecture Flow

This project should be read as a production-style ML system with a clear separation between offline work and online serving.

## The Two Main Flows

1. Offline flow
2. Online flow

Keep them separate in your head. Most confusion comes from mixing them.

## 1. Offline Flow

```text
synthetic/raw data
    ->
domain validation
    ->
time-safe split
    ->
shared preprocessing
    ->
task training
    ->
analysis and diagnostics
    ->
artifact packaging
```

### Main files

- [generate_data.py](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/scripts/generate_data.py)
- [train_models.py](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/scripts/train_models.py)
- [dataset.py](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/src/p1_customer_health/domain/dataset.py)
- [synthetic_data.py](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/src/p1_customer_health/domain/synthetic_data.py)
- [orchestration.py](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/src/p1_customer_health/training/orchestration.py)
- [classification.py](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/src/p1_customer_health/training/classification.py)
- [regression.py](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/src/p1_customer_health/training/regression.py)
- [unsupervised.py](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/src/p1_customer_health/training/unsupervised.py)
- all report and experiment modules under:
  - [analysis](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/src/p1_customer_health/analysis)
  - [experiments](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/src/p1_customer_health/experiments)

## 2. Online Flow

```text
request
    ->
schema validation
    ->
settings lookup
    ->
prediction service loads artifacts
    ->
inference across saved models
    ->
combined API response
```

### Main files

- [run_api.py](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/scripts/run_api.py)
- [settings.py](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/src/p1_customer_health/app/settings.py)
- [prediction_service.py](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/src/p1_customer_health/serving/prediction_service.py)
- [schemas.py](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/src/p1_customer_health/api/schemas.py)
- [main.py](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/src/p1_customer_health/api/main.py)

## The Most Important Boundary

```text
offline model building
        vs
online artifact serving
```

Offline owns:

- dataset creation/loading
- training
- evaluation
- reporting
- packaging

Online owns:

- request validation
- artifact loading
- prediction orchestration
- response contract
