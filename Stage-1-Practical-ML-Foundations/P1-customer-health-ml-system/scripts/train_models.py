from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from p1_customer_health.training.orchestration import train_all


def main() -> None:
    train_all(
        dataset_path=PROJECT_ROOT / "data/raw/customer_health.csv",
        artifact_root=PROJECT_ROOT / "artifacts",
    )


if __name__ == "__main__":
    main()
