# src/scoring/meta_scores.py
"""
Composite seasonality scores combining ACF, P2M, and STL metrics.

This module provides the canonical implementation used across the pipeline
and notebooks.

NOTE: Do not duplicate this logic elsewhere.
"""

from typing import List, Optional

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def add_meta_scores(
    df: pd.DataFrame,
    metrics: Optional[List[str]] = None,
    weights: Optional[List[float]] = None,
    eps: float = 1e-6,
) -> pd.DataFrame:
    """
    Add linear, geometric, and harmonic seasonality meta-scores.

    Formulas:
    ---------
    Linear:    Σ(weight_i × metric_i)
            - Interpretable, no imbalance penalty

    Geometric: exp(Σ(weight_i × log(metric_i)))
            - Moderately penalizes weak metrics
            - Rewards balance
            - One weak metric pull down moderately

    Harmonic:  1 / Σ(weight_i / metric_i)
            - Strong penalty for imbalance (conservative)
            - One weak metric drags the score the most

    Parameters:
    -----------
    df : DataFrame
        Must contain normalized metric columns (values in [0,1])
    metrics : list of str
        Column names to combine. Default: ['acf_lag_val', 'p2m_val', 'stl_strength']
    weights : list of float
        Optional weights (must sum to 1). Default: equal weights.
    eps : float
        Small constant to avoid log(0) and division by zero

    Returns:
    --------
    DataFrame
        Original df with three added columns:
        - seasonality_score_linear
        - seasonality_score_geom
        - seasonality_score_harmonic

    Examples:
    ---------
    >>> df = pd.DataFrame(
    dict(
    ticker=['AAPL', 'MSFT'],
    acf_lag_val=[0.8, 0.6],
    p2m_val=[0.7, 0.9],
    stl_strength=[0.9, 0.5]))
    >>> df_scored = add_meta_scores(df)
    >>> df_scored['seasonality_score_linear']
    0    0.800
    1    0.667
    dtype: float64

    Notes:
    ------
    - Metrics should be pre-normalized to [0,1] via MinMaxScaler per interval/freq.
    - Harmonic mean is most conservative; use for risk-averse ticker selection.
    - Geometric mean balances interpretability and robustness.
    """
    if metrics is None:
        metrics = ["acf_lag_val", "p2m_val", "stl_strength"]

    n = len(metrics)
    if weights is None:
        weights = [1 / n] * n

    if len(weights) != n:
        raise ValueError(
            f"Length of weights ({len(weights)}) must match " f"length of metrics ({n})"
        )

    if not np.isclose(sum(weights), 1.0):
        raise ValueError(f"Weights must sum to 1, got {sum(weights)}")

    # Clip to avoid numerical issues
    vals = df[metrics].clip(lower=eps)

    # Linear: weighted sum (most interpretable)
    df["seasonality_score_linear"] = sum(w * vals[m] for w, m in zip(weights, metrics))

    # Geometric: exp(weighted log-sum)
    df["seasonality_score_geom"] = np.exp(
        sum(w * np.log(vals[m]) for w, m in zip(weights, metrics))
    )

    # Harmonic: 1 / weighted(1/x)
    df["seasonality_score_harmonic"] = 1 / sum(
        w / vals[m] for w, m in zip(weights, metrics)
    )

    return df


def normalize_metrics_by_group(
    df: pd.DataFrame,
    group_cols: List[str],
    metric_cols: List[str],
) -> pd.DataFrame:
    """
    Apply MinMaxScaler normalization within groups.

    Typically used to normalize metrics per (interval, freq) combination
    before computing meta-scores.

    Parameters:
    -----------
    df : DataFrame
        Raw metrics DataFrame
    group_cols : list of str
        Columns defining groups (e.g., ['interval', 'freq'])
    metric_cols : list of str
        Columns to normalize within each group

    Returns:
    --------
    DataFrame
        Normalized metrics (values in [0,1] per group)

    Examples:
    ---------
    >>> df = pd.DataFrame(dict(
    interval=['1d', '1d', '1wk', '1wk'], acf_lag_val=[0.5, 0.8, 0.3, 0.6]))
    >>> normalize_metrics_by_group(df, ['interval'], ['acf_lag_val'])
       interval  acf_lag_val
    0       1d     0.000000
    1       1d     1.000000
    2      1wk     0.000000
    3      1wk     1.000000
    """
    normalized_groups = []
    for _, group in df.groupby(group_cols):
        group_copy = group.copy()
        scaler = MinMaxScaler()
        group_copy[metric_cols] = scaler.fit_transform(group[metric_cols])
        normalized_groups.append(group_copy)

    return pd.concat(normalized_groups, ignore_index=True)
