from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

from world_cricket_ml.utils import write_json

log = logging.getLogger(__name__)


def build_data_quality_report(df: pd.DataFrame, artifact_root: Path) -> dict:
    log.info("Building data quality report for %d rows × %d columns", len(df), df.shape[1])
    report = {
        "row_count": int(len(df)),
        "column_count": int(df.shape[1]),
        "duplicate_rows": int(df.duplicated().sum()),
        "missing_share": {column: round(float(value), 4) for column, value in df.isna().mean().items()},
        "label_prevalence": {
            "dominant_next_cycle": round(float(df["dominant_next_cycle"].mean()), 4),
            "downfall_next_cycle": round(float(df["downfall_next_cycle"].mean()), 4),
        },
        "team_counts": df["team"].value_counts().sort_index().to_dict(),
    }
    dominance_rate = report["label_prevalence"]["dominant_next_cycle"]
    log.info(
        "  duplicates=%d  dominant_rate=%.1f%%  downfall_rate=%.1f%%",
        report["duplicate_rows"],
        100 * dominance_rate,
        100 * report["label_prevalence"]["downfall_next_cycle"],
    )
    if dominance_rate < 0.10:
        log.warning("Dominant label prevalence is %.1f%% — severe class imbalance.", 100 * dominance_rate)
    write_json(artifact_root / "data_quality" / "data_quality_report.json", report)
    return report
