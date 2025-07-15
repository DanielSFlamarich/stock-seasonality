import numpy as np
import pandas as pd
import pytest

from src.metrics.quadratic_fit import compute_quadratic_fit_r2


def test_valid_quadratic_fit_returns_r2():
    index = pd.date_range(start="2023-01-01", periods=365, freq="D")
    seasonal_indexer = index.dayofyear
    seasonal = -0.01 * (seasonal_indexer - 180) ** 2 + 100
    noise = np.random.normal(0, 5, size=365)
    series = pd.Series(seasonal + noise, index=index)

    r2 = compute_quadratic_fit_r2(series, pd.Series(seasonal_indexer, index=index))
    assert 0 <= r2 <= 1
    assert r2 > 0.5  # expect decent fit for a quadratic pattern


def test_mismatched_lengths_raise():
    index = pd.date_range("2023-01-01", periods=30)
    series = pd.Series(np.random.randn(30), index=index)
    groups = pd.Series(np.arange(20))
    with pytest.raises(ValueError):
        compute_quadratic_fit_r2(series, groups)


def test_non_series_input_raises():
    index = pd.date_range("2023-01-01", periods=30)
    series = np.random.randn(30)
    groups = pd.Series(index.day, index=index)
    with pytest.raises(TypeError):
        compute_quadratic_fit_r2(series, groups)


def test_too_few_seasonal_units_raises():
    index = pd.date_range("2023-01-01", periods=3)
    series = pd.Series(np.random.randn(3), index=index)
    groups = pd.Series([1, 1, 2], index=index)
    with pytest.raises(ValueError):
        compute_quadratic_fit_r2(series, groups)
