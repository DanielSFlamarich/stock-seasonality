import numpy as np
import pandas as pd
import pytest

from src.metrics.anova_seasonality import compute_anova_seasonality


def test_valid_series_returns_f_ratio():
    index = pd.date_range(start="2023-01-01", periods=365, freq="D")
    seasonal_indexer = index.month
    seasonal = 5 * np.sin(2 * np.pi * seasonal_indexer / 12)
    noise = np.random.normal(0, 1, size=365)
    series = pd.Series(seasonal + noise, index=index)

    f_ratio = compute_anova_seasonality(
        series, pd.Series(seasonal_indexer, index=index)
    )
    assert f_ratio > 1  # should detect some seasonality


def test_mismatched_lengths_raise():
    index = pd.date_range("2023-01-01", periods=30)
    series = pd.Series(np.random.randn(30), index=index)
    groups = pd.Series(np.arange(20))
    with pytest.raises(ValueError):
        compute_anova_seasonality(series, groups)


def test_non_series_input_raises():
    index = pd.date_range("2023-01-01", periods=30)
    series = np.random.randn(30)
    groups = pd.Series(index.month, index=index)
    with pytest.raises(TypeError):
        compute_anova_seasonality(series, groups)


def test_zero_within_variance_raises():
    index = pd.date_range("2023-01-01", periods=12, freq="ME")
    series = pd.Series([5.0] * 12, index=index)
    groups = pd.Series(index.month, index=index)
    with pytest.raises(ValueError):
        compute_anova_seasonality(series, groups)
