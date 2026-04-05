from pathlib import Path

from p1_customer_health.ml.train import train_all


def main() -> None:
    train_all(
        dataset_path=Path("data/raw/customer_health.csv"),
        artifact_root=Path("artifacts"),
    )


if __name__ == "__main__":
    main()
