import numpy as np
import pandas as pd
import pytest

from src.metrics.stl_strength import compute_stl_strength


def test_valid_series_returns_strength():
    index = pd.date_range(start="2022-01-01", periods=730, freq="D")
    seasonal = 10 * np.sin(2 * np.pi * index.dayofyear / 365)
    noise = np.random.normal(0, 1, size=730)
    series = pd.Series(seasonal + noise, index=index)

    strength = compute_stl_strength(series, period=365)
    assert 0 <= strength <= 1
    assert strength > 0.5  # should have moderate to strong seasonality


def test_non_datetime_index_raises():
    index = pd.RangeIndex(start=0, stop=100)
    series = pd.Series(np.random.randn(100), index=index)

    with pytest.raises(TypeError):
        compute_stl_strength(series, period=12)


def test_non_series_input_raises():
    arr = np.random.randn(100)
    with pytest.raises(TypeError):
        compute_stl_strength(arr, period=12)


def test_constant_series_raises():
    index = pd.date_range("2022-01-01", periods=100, freq="D")
    series = pd.Series(5.0, index=index)
    with pytest.raises(ValueError):
        compute_stl_strength(series, period=12)


def test_too_short_series_raises():
    index = pd.date_range("2022-01-01", periods=10, freq="D")
    series = pd.Series(np.random.randn(10), index=index)
    with pytest.raises(ValueError):
        compute_stl_strength(series, period=12)
