from functools import lru_cache
from pathlib import Path

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_prefix="P1_")

    dataset_path: Path = Path("data/raw/customer_health.csv")
    classifier_artifact: Path = Path("artifacts/classification/model.joblib")
    regressor_artifact: Path = Path("artifacts/regression/model.joblib")
    segmenter_artifact: Path = Path("artifacts/unsupervised/segmenter.joblib")
    anomaly_artifact: Path = Path("artifacts/unsupervised/anomaly_detector.joblib")


@lru_cache
def get_settings() -> Settings:
    return Settings()
