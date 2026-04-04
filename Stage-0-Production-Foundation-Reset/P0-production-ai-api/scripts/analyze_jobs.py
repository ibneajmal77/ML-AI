from __future__ import annotations

import sqlite3
from pathlib import Path

import numpy as np
import pandas as pd


def main() -> None:
    db_path = Path("data/app.db")
    if not db_path.exists():
        raise SystemExit("Database does not exist yet. Run the app and create some jobs first.")

    with sqlite3.connect(db_path) as connection:
        frame = pd.read_sql_query("SELECT * FROM jobs", connection)

    if frame.empty:
        print("No jobs found.")
        return

    frame["processing_seconds"] = (
        pd.to_datetime(frame["updated_at"]) - pd.to_datetime(frame["created_at"])
    ).dt.total_seconds()
    frame["estimated_cost_usd"] = frame["estimated_cost_usd"].astype(np.float32)

    print("Jobs by status:")
    print(frame["status"].value_counts())
    print()
    print("Average estimated cost by status:")
    print(frame.groupby("status")["estimated_cost_usd"].mean())


if __name__ == "__main__":
    main()

