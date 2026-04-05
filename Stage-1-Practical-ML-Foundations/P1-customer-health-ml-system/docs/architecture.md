# Architecture

## Domain

This project models SaaS customer health at a monthly account snapshot level.

Each row represents one account snapshot with:

- numeric usage and spend signals
- categorical account context
- time-derived fields
- short support-note text
- a churn label
- a future revenue-change target

## ML workstreams

### 1. Churn classification

- target: `churned_30d`
- key decisions: threshold tuning, PR-AUC focus, calibration review, slice analysis

### 2. Revenue-change regression

- target: `revenue_change_next_30d`
- key decisions: MAE and RMSE, prediction intervals as a later extension

### 3. Customer segmentation and anomaly detection

- clustering: `KMeans`
- dimensionality reduction: `PCA`
- anomaly detection: `IsolationForest`

### 4. Optional self-supervised text workflow

- embedding source: `sentence-transformers` if installed
- use case: note-text-only churn baseline for comparison

## Serving boundary

The API loads packaged artifacts and returns:

- churn score
- churn label using the chosen threshold
- predicted revenue change
- segment id
- anomaly flag

## Production shape

This project stays intentionally bounded:

- one domain
- one primary dataset family
- one API surface
- multiple ML tasks from the same business system

That keeps it realistic enough for jobs without becoming a fake all-in-one tutorial.
