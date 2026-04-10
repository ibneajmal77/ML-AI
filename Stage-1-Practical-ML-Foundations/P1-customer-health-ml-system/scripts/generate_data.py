from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "src"))

from p1_customer_health.domain.synthetic_data import generate_customer_health_data


def main() -> None:
    output_path = PROJECT_ROOT / "data/raw/customer_health.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df = generate_customer_health_data(n_samples=5000, seed=42)
    df.to_csv(output_path, index=False)
    print(f"Wrote {len(df)} rows to {output_path}")


if __name__ == "__main__":
    main()
