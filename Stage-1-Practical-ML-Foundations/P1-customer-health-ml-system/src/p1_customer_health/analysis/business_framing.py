from __future__ import annotations

from pathlib import Path

from p1_customer_health.utils import ensure_dir, write_json


def write_business_decision_workflow(output_dir: Path) -> None:
    ensure_dir(output_dir)
    payload = {
        "project_domain": "saas_customer_health",
        "decision_workflow": [
            {
                "question": "Should this be ML at all?",
                "answer": "Yes for churn scoring and revenue forecasting because the patterns are repetitive, measurable, and depend on many interacting signals.",
            },
            {
                "question": "Where should rules still win?",
                "answer": "Hard policy conditions such as account lock, unpaid invoices, or known contractual overrides should stay rule-based.",
            },
            {
                "question": "Which learning paradigms fit this system?",
                "answer": "Supervised for churn and revenue, unsupervised for segmentation and anomaly detection, self-supervised for note embeddings, RL only for later retention-action sequencing.",
            },
            {
                "question": "What production shape fits the business timing?",
                "answer": "Batch scoring first, API serving second, because many retention workflows do not need synchronous inference everywhere.",
            },
        ],
    }
    write_json(output_dir / "business_decisions.json", payload)
