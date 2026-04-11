# Stage-1 Implementation Map

## Read First

1. `README.md`
2. `docs/architecture.md`
3. `src/world_cricket_ml/domain/dataset.py`
4. `src/world_cricket_ml/training/orchestration.py`
5. `src/world_cricket_ml/serving/prediction_service.py`

## Run First

```powershell
python scripts\fetch_data.py
python scripts\train_models.py
python -m pytest -q
python scripts\run_api.py
```

## Deliverable Map

| Deliverable | Purpose |
|---|---|
| `data/raw/world_cricket_team_matches.csv` | raw team-perspective match table filtered to the rolling two-year quarterly window |
| `data/raw/world_cricket_snapshots.csv` | model-ready training snapshots |
| `data/raw/dataset_metadata.json` | current date window and quarterly refresh policy |
| `artifacts/data_quality/data_quality_report.json` | missingness, duplicates, label prevalence |
| `artifacts/leakage/leakage_report.json` | blocked features and implemented time-split policy |
| `artifacts/classification/metrics.json` | thresholded classifier comparison and selected serving artifact |
| `artifacts/regression/metrics.json` | regression model comparison |
| `artifacts/unsupervised/team_segmentation.csv` | cluster assignments and anomaly scores |
| `artifacts/business/business_decisions.json` | dominance, downfall, underdog watchlists |
| `artifacts/llm_benchmark/llm_vs_classical_benchmark.json` | true ROC AUC comparison between structured ML and text-only proxy |
| `artifacts/reinforcement_learning/contextual_bandit_report.json` | RL-scope toss decision proxy |
