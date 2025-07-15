import numpy as np
import pandas as pd
import pytest

from src.metrics.periodogram_strength import compute_periodogram_strength


def test_valid_series_returns_ratio():
    np.random.seed(42)
    index = pd.date_range(start="2022-01-01", periods=200, freq="D")
    seasonal = 5 * np.sin(2 * np.pi * index.dayofyear / 30)
    noise = np.random.normal(0, 1, size=200)
    series = pd.Series(seasonal + noise, index=index)

    ratio = compute_periodogram_strength(series)
    assert ratio > 1.5  # peak should exceed background noise


def test_non_series_input_raises():
    arr = np.random.randn(100)
    with pytest.raises(TypeError):
        compute_periodogram_strength(arr)


def test_non_datetime_index_raises():
    series = pd.Series(np.random.randn(100), index=range(100))
    with pytest.raises(TypeError):
        compute_periodogram_strength(series)


def test_too_short_series_raises():
    index = pd.date_range("2023-01-01", periods=10, freq="D")
    series = pd.Series(np.random.randn(10), index=index)
    with pytest.raises(ValueError):
        compute_periodogram_strength(series)


def test_constant_series_raises():
    index = pd.date_range("2023-01-01", periods=50, freq="D")
    series = pd.Series(3.14, index=index)
    with pytest.raises(ValueError):
        compute_periodogram_strength(series)
