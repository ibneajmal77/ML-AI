# Coverage Map

| Lesson | Topic | Where it is implemented |
|---|---|---|
| `1.1` | What ML is in practical business terms | `src/world_cricket_ml/analysis/business_framing.py` and `artifacts/business/business_decisions.json` |
| `1.2` | Supervised / unsupervised / self-supervised / RL scope | `training/`, `training/unsupervised.py`, `experiments/self_supervised.py`, `experiments/rl.py` |
| `1.3` | Features, labels, splits, leakage, data quality | `domain/dataset.py`, `analysis/audit.py`, `analysis/leakage.py` (incl. mutual information scan), `training/splits.py` (two-way and three-way time splits) |
| `1.4` | Regression — predicting values, evaluating error | `training/regression.py` (ElasticNet, DecisionTree, RF, GBR; MAE, RMSE, R², MAPE) |
| `1.5` | Classification, thresholds, calibration, business cost | `training/classification.py` (threshold sweep 0.1–0.9, CalibratedClassifierCV, business cost function, `artifacts/classification/calibration_curve.csv`) |
| `1.6` | Logistic regression | `training/classification.py` (plain LR + class_weight="balanced" variant) |
| `1.7` | Trees, random forests, gradient boosting | `training/classification.py` and `training/regression.py` (DecisionTree → RandomForest → GradientBoosting complexity ladder) |
| `1.8` | XGBoost and LightGBM | `experiments/boosting.py` (classifier AND regressor benchmarks for both libraries) |
| `1.9` | Evaluation metrics | `training/metrics.py` (accuracy, precision, recall, F1, ROC-AUC, PR-AUC, MAPE) |
| `1.10` | Class imbalance, sampling, class weights, threshold tuning | `training/classification.py` (random oversampling, class_weight="balanced", SMOTE via imbalanced-learn [optional], threshold sweep) |
| `1.11` | Feature engineering | `domain/dataset.py` and `training/preprocessing.py` (rolling windows, rest days, opponent strength, temporal features, OHE, StandardScaler) |
| `1.12` | Overfitting, underfitting, bias-variance diagnosis | `artifacts/classification/fit_diagnosis.csv`, `artifacts/regression/fit_diagnosis.csv`, train/test gap column, diagnosis label per model; see also `1.12-Overfitting-Underfitting-Bias-Variance-Diagnosis.md` |
| `1.13` | Error analysis, slice analysis, failure taxonomy | `analysis/failure_taxonomy.py`, `artifacts/classification/slice_analysis.csv`, `artifacts/classification/failure_taxonomy_summary.csv` |
| `1.14` | When classical ML beats LLMs | `experiments/llm_benchmark.py` (TF-IDF text proxy vs structured tabular model ROC-AUC comparison) |
| `1.15` | Scikit-learn pipelines | `training/preprocessing.py`, `training/classification.py`, `training/regression.py` (all models in `Pipeline`; `TimeSeriesSplit` cross-validation) |
| `1.16` | Packaging and serving basics, joblib, ONNX, REST | `serving/prediction_service.py`, `api/main.py`, `experiments/onnx_export.py`, `artifacts/model_registry.json` |

## New artifacts added beyond base coverage

| Artifact | Lesson | What it demonstrates |
|---|---|---|
| `artifacts/classification/feature_importance.csv` | 1.7, 1.8 | Which features the winning model relies on; basis for explainability and drift monitoring |
| `artifacts/regression/feature_importance.csv` | 1.7, 1.8 | Same for the regression model |
| `artifacts/classification/calibration_curve.csv` | 1.5 | Raw vs calibrated probability reliability — verifies that CalibratedClassifierCV improved probability estimates |
| `artifacts/model_registry.json` | 1.16 | Consolidated model metadata: best model, metrics, data window, training timestamp |
| `artifacts/leakage/leakage_report.json` → `mutual_info_scan` | 1.3 | MI-based leakage detection catches future-information proxy features that name checks miss |
| `artifacts/regression/fit_diagnosis.csv` → `test_mape` | 1.4 | MAPE as a percentage-based business-readable error metric |

## Optional dependency groups

| Group | Install command | What it unlocks |
|---|---|---|
| `boosting` | `pip install -e ".[boosting]"` | XGBoost + LightGBM classifier and regressor benchmark (`artifacts/boosting/status.json`) |
| `onnx` | `pip install -e ".[onnx]"` | ONNX model export (`artifacts/onnx/dominance_model.onnx`) |
| `imbalanced` | `pip install -e ".[imbalanced]"` | SMOTE oversampling model variant (`smote_random_forest` in classification comparison) |
| `all` | `pip install -e ".[all]"` | Everything above plus dev/test tools |
