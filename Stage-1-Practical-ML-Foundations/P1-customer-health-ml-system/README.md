# P1 Customer Health ML System

This is the Stage 1 capstone for practical ML foundations. It is structured as a small production-style ML system, not a notebook dump and not a fake web app with many empty endpoints.

The business setting is a SaaS customer-health workflow. One dataset family supports multiple realistic ML tasks:

- churn classification
- revenue-change regression
- customer segmentation
- anomaly detection
- text-only self-supervised benchmark
- boosting benchmark with XGBoost and LightGBM
- classical-ML-vs-LLM-style comparison
- contextual-bandit retention simulation
- packaging and API serving

## Production Architecture

The repository is organized around the actual system flow:

```text
P1-customer-health-ml-system/
|-- data/
|-- artifacts/
|-- docs/
|-- scripts/
|-- src/p1_customer_health/
|   |-- api/              # FastAPI routes and request/response schemas
|   |-- app/              # runtime settings
|   |-- domain/           # dataset contract and synthetic data generation
|   |-- training/         # offline model building and shared training utilities
|   |-- analysis/         # audits, leakage checks, failure taxonomy, business framing
|   |-- experiments/      # bounded advanced tracks: boosting, RL, LLM, ONNX, embeddings
|   |-- serving/          # inference-time artifact loading and prediction orchestration
|   `-- ml/               # thin compatibility wrappers only
`-- tests/
```

## How To Think About The System

There are two different flows:

1. Offline flow
   - generate or load data
   - validate dataset and split by time
   - train models
   - run diagnostics and reports
   - save artifacts
2. Online flow
   - receive API request
   - validate request schema
   - load saved artifacts
   - run inference
   - return combined prediction response

That split is the core architecture. The project is mainly an offline ML system with a small serving layer on top.

## Entry Points

- [generate_data.py](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/scripts/generate_data.py)
- [train_models.py](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/scripts/train_models.py)
- [run_api.py](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/scripts/run_api.py)

## Local Run

```powershell
python scripts\generate_data.py
python scripts\train_models.py
pytest -q
python scripts\run_api.py
```

Swagger:

- `http://127.0.0.1:8000/docs`

## Best Docs To Read First

1. [architecture.md](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/docs/architecture.md)
2. [production-architecture-flow.md](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/docs/production-architecture-flow.md)
3. [structure-guide.md](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/docs/structure-guide.md)
4. [learning-flow.md](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/docs/learning-flow.md)
5. [stage1-implementation-map.md](C:/Users/ibnea/source/ML-AI/Stage-1-Practical-ML-Foundations/P1-customer-health-ml-system/docs/stage1-implementation-map.md)
