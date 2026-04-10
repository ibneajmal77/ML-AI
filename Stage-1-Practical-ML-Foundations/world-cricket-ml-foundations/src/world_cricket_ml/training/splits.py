from __future__ import annotations

import logging

import pandas as pd

log = logging.getLogger(__name__)


def time_split(df: pd.DataFrame, quantile: float = 0.8) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Chronological two-way split: oldest *quantile* fraction → train, remainder → test.

    Both halves are returned as copies to prevent accidental mutation of the source frame.
    """
    cutoff = pd.Timestamp(df["match_date"].quantile(quantile))
    train = df[df["match_date"] < cutoff].copy()
    test = df[df["match_date"] >= cutoff].copy()
    log.debug(
        "time_split: cutoff=%s  train=%d rows  test=%d rows",
        cutoff.date(),
        len(train),
        len(test),
    )
    return train, test


def three_way_time_split(
    df: pd.DataFrame,
    train_frac: float = 0.6,
    val_frac: float = 0.2,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Chronological three-way split: train / validation / test.

    Use a separate validation set for model selection and hyperparameter tuning,
    and keep the test set untouched until final evaluation.  Touching the test
    set repeatedly during development inflates reported metrics.

    Parameters
    ----------
    df:
        DataFrame with a ``match_date`` column used for ordering.
    train_frac:
        Fraction of rows allocated to training (default 60 %).
    val_frac:
        Fraction of rows allocated to validation (default 20 %).
        The test set receives the remaining rows (default 20 %).

    Returns
    -------
    (train_df, val_df, test_df) — three non-overlapping chronological slices, each a copy.
    """
    if not (0 < train_frac < 1 and 0 < val_frac < 1 and train_frac + val_frac < 1):
        raise ValueError(
            f"train_frac={train_frac} and val_frac={val_frac} must be positive and sum to less than 1."
        )
    train_cutoff = pd.Timestamp(df["match_date"].quantile(train_frac))
    val_cutoff = pd.Timestamp(df["match_date"].quantile(train_frac + val_frac))

    train = df[df["match_date"] < train_cutoff].copy()
    val = df[(df["match_date"] >= train_cutoff) & (df["match_date"] < val_cutoff)].copy()
    test = df[df["match_date"] >= val_cutoff].copy()
    log.debug(
        "three_way_time_split: train_cutoff=%s  val_cutoff=%s  train=%d  val=%d  test=%d",
        train_cutoff.date(),
        val_cutoff.date(),
        len(train),
        len(val),
        len(test),
    )
    return train, val, test
