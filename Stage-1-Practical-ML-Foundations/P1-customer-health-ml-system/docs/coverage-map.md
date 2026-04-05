# Stage 1 Coverage Map

| Lesson | Topic | Where it appears in the project |
|---|---|---|
| `1.1` | What ML is in practical business terms | README, architecture, churn-vs-rule framing |
| `1.2` | Supervised / unsupervised / self-supervised / RL scope | classifier, regressor, clustering, optional embeddings, explicit "no RL" decision |
| `1.3` | Features, labels, splits, leakage, data quality | `src/p1_customer_health/ml/features.py`, time split in training, synthetic data checks |
| `1.4` | Regression | revenue-change model |
| `1.5` | Classification thresholds and calibration | classifier metrics, threshold tuning, calibration bins |
| `1.6` | Logistic regression | baseline classifier |
| `1.7` | Trees, random forests, gradient boosting | classifier/regressor comparisons |
| `1.8` | XGBoost and LightGBM | optional boosting benchmark in `src/p1_customer_health/ml/train.py` and `artifacts/boosting/status.json` |
| `1.9` | Evaluation metrics | metrics report JSON for classification and regression |
| `1.10` | Class imbalance | class weighting, PR-AUC, cost-based threshold selection |
| `1.11` | Feature engineering | numeric, categorical, time, and text preprocessing |
| `1.12` | Overfitting / underfitting / bias-variance | train-vs-val metrics and model comparison |
| `1.13` | Error analysis and slices | slice analysis CSV and false-positive / false-negative review |
| `1.14` | Classical ML vs LLMs | explicit bounded text baseline instead of generative solution |
| `1.15` | sklearn pipelines | packaged preprocessing + model artifacts |
| `1.16` | Packaging and serving basics | `joblib` artifacts and FastAPI app |

## Why RL is not in the implementation

The project includes RL in scope analysis only, not as a forced implementation. That is intentional. Retention actions here are a prediction problem first, not a sequential control problem. Forcing RL into this capstone would make it less realistic, not more senior.
