# Learning Flow

## Phase 1: Understand the problem

Read the README and business framing output first. The goal is not just predicting winners. The goal is learning how one sports dataset can support many ML tasks.

## Phase 2: Understand the data contract

Open `src/world_cricket_ml/domain/dataset.py` and trace how raw Cricsheet JSON becomes snapshot rows. Focus on:

- why each match becomes two team rows
- why features use past information only
- why targets are future-looking
- why the project now uses a rolling two-year window refreshed every quarter
- why time splits matter more than random splits here

## Phase 3: Understand supervised ML

Study `training/classification.py` and `training/regression.py` together.

- logistic regression is the linear baseline
- tree ensembles capture non-linear interactions
- thresholds and business cost matter as much as raw probability scores
- regression and classification can answer different business questions on the same data
- a shorter rolling history window makes the system more sensitive to current team momentum

## Phase 4: Understand diagnostics

Read the artifacts after training.

- `metrics.json` shows model tradeoffs
- `fit_diagnosis.csv` shows possible underfitting or variance issues
- `slice_analysis.csv` shows where the classifier behaves differently across teams and form buckets
- `failure_taxonomy_summary.csv` turns wrong predictions into categories you can debug

## Phase 5: Understand advanced scope

Use the experiments folder to understand where Stage-1 expands beyond basic supervised learning.

- self-supervised proxy: text embeddings from unlabeled match narratives
- boosting: optional XGBoost and LightGBM benchmark
- RL proxy: toss decision as a contextual bandit exercise
- classical vs LLM: tabular structure still wins for many operational tasks
- ONNX export: packaging path for deployment-minded workflows
