from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = PROJECT_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from p1_customer_health.ml.data_generation import generate_customer_health_data


def main() -> None:
    output_path = Path("data/raw/customer_health.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_customer_health_data(n_samples=5000, seed=42)
    df.to_csv(output_path, index=False)
    print(f"wrote {len(df)} rows to {output_path}")


if __name__ == "__main__":
    main()
