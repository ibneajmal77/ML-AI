from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict

# Resolves to the project root (P1-customer-health-ml-system/) regardless of CWD
_PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="P1_")

    dataset_path: Path = _PROJECT_ROOT / "data/raw/customer_health.csv"
    classifier_artifact: Path = _PROJECT_ROOT / "artifacts/classification/model.joblib"
    regressor_artifact: Path = _PROJECT_ROOT / "artifacts/regression/model.joblib"
    segmenter_artifact: Path = _PROJECT_ROOT / "artifacts/unsupervised/segmenter.joblib"
    anomaly_artifact: Path = _PROJECT_ROOT / "artifacts/unsupervised/anomaly_detector.joblib"


@lru_cache
def get_settings() -> Settings:
    return Settings()
