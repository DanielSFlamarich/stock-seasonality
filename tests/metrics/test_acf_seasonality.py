import numpy as np
import pandas as pd
import pytest

from src.metrics.acf_seasonality import compute_acf_seasonality


def test_valid_series_returns_acf():
    np.random.seed(42)
    index = pd.date_range(start="2022-01-01", periods=300, freq="D")
    seasonal = 5 * np.sin(2 * np.pi * index.dayofyear / 30)
    noise = np.random.normal(0, 1, size=300)
    series = pd.Series(seasonal + noise, index=index)

    lag = 30
    acf_val = compute_acf_seasonality(series, lag=lag)
    assert -1 <= acf_val <= 1
    assert acf_val > 0.2  # should reflect detectable seasonality


def test_non_series_input_raises():
    arr = np.random.randn(100)
    with pytest.raises(TypeError):
        compute_acf_seasonality(arr, lag=12)


def test_non_datetime_index_raises():
    series = pd.Series(np.random.randn(100), index=range(100))
    with pytest.raises(TypeError):
        compute_acf_seasonality(series, lag=12)


def test_too_short_series_raises():
    index = pd.date_range(start="2023-01-01", periods=10, freq="D")
    series = pd.Series(np.random.randn(10), index=index)
    with pytest.raises(ValueError):
        compute_acf_seasonality(series, lag=12)


def test_constant_series_raises():
    index = pd.date_range(start="2023-01-01", periods=50, freq="D")
    series = pd.Series(3.14, index=index)
    with pytest.raises(ValueError):
        compute_acf_seasonality(series, lag=10)
