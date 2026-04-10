import pandas as pd
import pytest

from pathlib import Path

from world_cricket_ml.experiments.llm_benchmark import run_llm_vs_classical_benchmark
from world_cricket_ml.utils import read_json


@pytest.fixture()
def sample_frame() -> pd.DataFrame:
    rows = [
        {
            "match_date": pd.Timestamp("2024-01-01") + pd.Timedelta(days=idx),
            "future_matches_available": 3,
            "match_text": f"Team form summary {idx}",
            "dominant_next_cycle": idx % 2,
        }
        for idx in range(20)
    ]
    return pd.DataFrame(rows)


def test_llm_benchmark_preserves_passed_roc_auc(sample_frame: pd.DataFrame, tmp_path: Path) -> None:
    artifact_root = tmp_path / "artifacts"
    run_llm_vs_classical_benchmark(sample_frame, classical_auc=0.8123, artifact_root=artifact_root)
    payload = read_json(artifact_root / "llm_benchmark" / "llm_vs_classical_benchmark.json")
    assert payload["classical_tabular_roc_auc"] == 0.8123
    assert 0.0 <= payload["text_only_proxy_roc_auc"] <= 1.0
