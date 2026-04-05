# Stage 1 Coverage Map

| Lesson | Topic | Where it appears in the project |
|---|---|---|
| `1.1` | What ML is in practical business terms | `src/p1_customer_health/ml/business_framing.py` and `artifacts/business/business_decisions.json` |
| `1.2` | Supervised / unsupervised / self-supervised / RL scope | classifier, regressor, clustering, self-supervised embedding path, and contextual bandit in `src/p1_customer_health/ml/rl.py` |
| `1.3` | Features, labels, splits, leakage, data quality | `src/p1_customer_health/ml/features.py`, `src/p1_customer_health/ml/audit.py`, and `src/p1_customer_health/ml/leakage.py` |
| `1.4` | Regression | revenue-change model |
| `1.5` | Classification thresholds and calibration | threshold tuning, calibration reports, and calibrated classifier artifact |
| `1.6` | Logistic regression | baseline classifier |
| `1.7` | Trees, random forests, gradient boosting | classifier/regressor comparisons |
| `1.8` | XGBoost and LightGBM | boosting benchmark path in `src/p1_customer_health/ml/train.py` and `artifacts/boosting/status.json` |
| `1.9` | Evaluation metrics | metrics report JSON for classification and regression |
| `1.10` | Class imbalance | class weighting, oversampling, undersampling, PR-AUC, and cost-based threshold selection |
| `1.11` | Feature engineering | numeric, categorical, time, and text preprocessing |
| `1.12` | Overfitting / underfitting / bias-variance | train-vs-val metrics and `fit_diagnosis.csv` reports |
| `1.13` | Error analysis and slices | slice analysis plus failure taxonomy outputs |
| `1.14` | Classical ML vs LLMs | implemented comparison in `src/p1_customer_health/ml/llm_benchmark.py` |
| `1.15` | sklearn pipelines | packaged preprocessing + model artifacts |
| `1.16` | Packaging and serving basics | `joblib` artifacts, FastAPI app, and ONNX export path |

## RL note

RL is implemented as a bounded contextual-bandit workflow for retention actions, not as the core churn model. That keeps the main business problem realistic while still implementing the RL family in a way that makes sense for Stage 1.
