from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from world_cricket_ml.training.orchestration import train_all


def main() -> None:
    train_all(PROJECT_ROOT)


if __name__ == "__main__":
    main()
