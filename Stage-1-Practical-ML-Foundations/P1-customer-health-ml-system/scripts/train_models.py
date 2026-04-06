from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from p1_customer_health.training.orchestration import train_all


def main() -> None:
    train_all(
        dataset_path=Path("data/raw/customer_health.csv"),
        artifact_root=Path("artifacts"),
    )


if __name__ == "__main__":
    main()
