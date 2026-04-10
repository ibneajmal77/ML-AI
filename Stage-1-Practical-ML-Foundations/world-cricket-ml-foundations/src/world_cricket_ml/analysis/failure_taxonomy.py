from __future__ import annotations

import logging
from pathlib import Path

import pandas as pd

log = logging.getLogger(__name__)


def build_failure_taxonomy(scored: pd.DataFrame, artifact_root: Path) -> None:
    error_rows = scored[scored["dominant_next_cycle"] != scored["predicted_label"]].copy()
    total_errors = len(error_rows)
    log.info("Failure taxonomy: %d errors out of %d predictions", total_errors, len(scored))

    error_rows["failure_bucket"] = "other"
    error_rows.loc[
        (error_rows["rolling_5_win_rate"] < 0.4) & (error_rows["predicted_label"] == 1),
        "failure_bucket",
    ] = "false_hype_on_bad_recent_form"
    error_rows.loc[
        (error_rows["opponent_rolling_5_win_rate"] > 0.65) & (error_rows["predicted_label"] == 0),
        "failure_bucket",
    ] = "missed_strong_schedule_effect"
    error_rows.loc[error_rows["rest_days"] > 21, "failure_bucket"] = "calendar_gap_shift"

    summary = (
        error_rows.groupby("failure_bucket")
        .size()
        .reset_index(name="count")
        .sort_values("count", ascending=False)
    )
    for _, row in summary.iterrows():
        log.info("  %-40s %d (%.1f%%)", row["failure_bucket"], row["count"], 100 * row["count"] / max(total_errors, 1))

    error_rows.to_csv(artifact_root / "classification" / "failure_taxonomy.csv", index=False)
    summary.to_csv(artifact_root / "classification" / "failure_taxonomy_summary.csv", index=False)
