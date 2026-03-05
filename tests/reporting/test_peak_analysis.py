# tests/reporting/test_peak_analysis.py

"""
Unit tests for reporting.peak_analysis.

Based on the actual public API:

    compute_peak_stats(series: pd.Series, freq: str, ...) -> dict
        series must have a DatetimeIndex, sorted ascending, no NaNs.

    summarise_peaks(df_prices: pd.DataFrame, freqs=None, ...) -> pd.DataFrame
        df_prices must have columns: ticker, close, and either a 'date' column
        or a DatetimeIndex.

All tests use synthetic data only. Real-ticker tests belong in
test_peak_analysis_integration.py and are marked @pytest.mark.integration.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from reporting.peak_analysis import (
    FREQ_MIN_DISTANCE,
    MIN_PEAKS_FOR_GAPS,
    compute_peak_stats,
    summarise_peaks,
)

SUPPORTED_FREQS = list(FREQ_MIN_DISTANCE.keys())  # ['W', 'ME', 'QE', 'YE']


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _make_series(
    n: int,
    cycles: float,
    *,
    amplitude: float = 10.0,
    baseline: float = 100.0,
    seed: int = 42,
) -> pd.Series:
    """
    Sinusoidal daily price series with a DatetimeIndex.
    `cycles` full oscillations over n samples.
    """
    np.random.default_rng(seed)
    dates = pd.date_range("2018-01-01", periods=n, freq="D")
    t = np.linspace(0, 2 * np.pi * cycles, n)
    values = baseline + amplitude * np.sin(t)
    return pd.Series(values, index=dates, name="close")


def _flat_series(n: int = 300) -> pd.Series:
    """Constant-price series — IQR == 0, no peaks possible."""
    dates = pd.date_range("2018-01-01", periods=n, freq="D")
    return pd.Series(np.full(n, 50.0), index=dates, name="close")


def _make_prices_df(
    n: int, cycles: float, ticker: str = "TEST", as_index: bool = False
) -> pd.DataFrame:
    """DataFrame version of _make_series for summarise_peaks tests."""
    series = _make_series(n, cycles)
    df = series.reset_index()
    df.columns = ["date", "close"]
    df["ticker"] = ticker
    if as_index:
        df = df.set_index("date")
    return df


# ---------------------------------------------------------------------------
# compute_peak_stats — return shape and types
# ---------------------------------------------------------------------------


class TestComputePeakStatsSchema:
    def test_returns_dict_with_correct_keys(self):
        series = _make_series(500, cycles=8.0)
        result = compute_peak_stats(series, freq="ME")
        assert set(result.keys()) == {
            "peak_count",
            "mean_peak_gap_days",
            "std_peak_gap_days",
        }

    def test_peak_count_is_int(self):
        series = _make_series(500, cycles=8.0)
        result = compute_peak_stats(series, freq="ME")
        assert isinstance(result["peak_count"], (int, np.integer))

    def test_gap_fields_are_float_or_nan(self):
        series = _make_series(500, cycles=8.0)
        result = compute_peak_stats(series, freq="ME")
        for key in ("mean_peak_gap_days", "std_peak_gap_days"):
            val = result[key]
            assert isinstance(val, float), f"{key} must be float, got {type(val)}"


# ---------------------------------------------------------------------------
# compute_peak_stats — sine wave detects peaks
# ---------------------------------------------------------------------------


class TestSineWavePeakDetection:
    @pytest.mark.parametrize("freq", SUPPORTED_FREQS)
    def test_multi_cycle_sine_detects_peaks(self, freq):
        """Enough cycles → at least one peak detected."""
        min_dist = FREQ_MIN_DISTANCE[freq]
        n = min_dist * 30  # 30× the minimum spacing
        cycles = 10.0
        series = _make_series(n, cycles=cycles)
        result = compute_peak_stats(series, freq=freq)
        assert (
            result["peak_count"] >= 1
        ), f"Expected ≥1 peak for freq={freq}, got {result['peak_count']}"

    @pytest.mark.parametrize("freq", SUPPORTED_FREQS)
    def test_two_or_more_peaks_yield_positive_mean_gap(self, freq):
        min_dist = FREQ_MIN_DISTANCE[freq]
        n = min_dist * 30
        series = _make_series(n, cycles=10.0)
        result = compute_peak_stats(series, freq=freq)
        if result["peak_count"] >= MIN_PEAKS_FOR_GAPS:
            assert result["mean_peak_gap_days"] > 0

    @pytest.mark.parametrize("freq", SUPPORTED_FREQS)
    def test_std_gap_is_non_negative(self, freq):
        min_dist = FREQ_MIN_DISTANCE[freq]
        n = min_dist * 30
        series = _make_series(n, cycles=10.0)
        result = compute_peak_stats(series, freq=freq)
        if result["peak_count"] >= MIN_PEAKS_FOR_GAPS:
            assert result["std_peak_gap_days"] >= 0


# ---------------------------------------------------------------------------
# compute_peak_stats — flat series (IQR == 0 branch)
# ---------------------------------------------------------------------------


class TestFlatSeries:
    @pytest.mark.parametrize("freq", SUPPORTED_FREQS)
    def test_flat_series_peak_count_zero(self, freq):
        """IQR=0 → _empty_stats() → peak_count=0."""
        result = compute_peak_stats(_flat_series(), freq=freq)
        assert result["peak_count"] == 0

    @pytest.mark.parametrize("freq", SUPPORTED_FREQS)
    def test_flat_series_gaps_are_nan(self, freq):
        """IQR=0 → both gap fields NaN."""
        result = compute_peak_stats(_flat_series(), freq=freq)
        assert np.isnan(result["mean_peak_gap_days"])
        assert np.isnan(result["std_peak_gap_days"])


# ---------------------------------------------------------------------------
# compute_peak_stats — fewer than MIN_PEAKS_FOR_GAPS (< 2 peaks → NaN gaps)
# ---------------------------------------------------------------------------


class TestFewPeaksNaNGaps:
    def test_single_peak_mean_gap_is_nan(self):
        """
        A single isolated spike → peak_count=1 → mean_peak_gap_days must be
        NaN, not 0. Returning 0 would falsely imply a gap was measured.
        """
        n = 200
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        values = np.full(n, 50.0)
        values[100] = 200.0  # one isolated spike well above IQR threshold
        series = pd.Series(values, index=dates)

        result = compute_peak_stats(series, freq="ME")
        if result["peak_count"] == 1:
            assert np.isnan(
                result["mean_peak_gap_days"]
            ), "mean_peak_gap_days must be NaN when peak_count < MIN_PEAKS_FOR_GAPS"

    def test_single_peak_std_gap_is_nan(self):
        """std_peak_gap_days must also be NaN (not 0.0) for a single peak."""
        n = 200
        dates = pd.date_range("2020-01-01", periods=n, freq="D")
        values = np.full(n, 50.0)
        values[100] = 200.0
        series = pd.Series(values, index=dates)

        result = compute_peak_stats(series, freq="ME")
        if result["peak_count"] == 1:
            assert np.isnan(
                result["std_peak_gap_days"]
            ), "std_peak_gap_days must be NaN, not 0.0, when no gaps exist"

    def test_zero_peaks_gaps_are_nan(self):
        """Zero peaks → gap fields NaN (via _empty_stats path)."""
        result = compute_peak_stats(_flat_series(), freq="QE")
        assert np.isnan(result["mean_peak_gap_days"])
        assert np.isnan(result["std_peak_gap_days"])


# ---------------------------------------------------------------------------
# compute_peak_stats — minimum distance floor
# ---------------------------------------------------------------------------


class TestMinimumDistanceRespected:
    @pytest.mark.parametrize("freq,min_dist", FREQ_MIN_DISTANCE.items())
    def test_mean_gap_at_least_min_distance(self, freq, min_dist):
        """
        For a daily series the minimum sample distance == minimum calendar-day
        gap. mean_peak_gap_days must be >= min_dist when ≥2 peaks detected.
        """
        n = min_dist * 30
        series = _make_series(n, cycles=10.0)
        result = compute_peak_stats(series, freq=freq)
        if result["peak_count"] >= MIN_PEAKS_FOR_GAPS:
            assert result["mean_peak_gap_days"] >= min_dist, (
                f"freq={freq}: mean gap {result['mean_peak_gap_days']:.1f} "
                f"< min_dist={min_dist}"
            )


# ---------------------------------------------------------------------------
# compute_peak_stats — unsupported freq raises ValueError
# ---------------------------------------------------------------------------


class TestUnsupportedFreq:
    @pytest.mark.parametrize("bad_freq", ["D", "BM", "Q", "M", "Y"])
    def test_unsupported_freq_raises_value_error(self, bad_freq):
        series = _make_series(300, cycles=4.0)
        with pytest.raises(ValueError):
            compute_peak_stats(series, freq=bad_freq)


# ---------------------------------------------------------------------------
# summarise_peaks — input validation
# ---------------------------------------------------------------------------


class TestSummarisePeaksValidation:
    def test_missing_close_raises(self):
        df = _make_prices_df(300, 4.0).rename(columns={"close": "price"})
        with pytest.raises(ValueError):
            summarise_peaks(df)

    def test_missing_ticker_raises(self):
        df = _make_prices_df(300, 4.0).drop(columns=["ticker"])
        with pytest.raises(ValueError):
            summarise_peaks(df)

    def test_missing_date_raises(self):
        """No 'date' column and no DatetimeIndex → ValueError."""
        df = _make_prices_df(300, 4.0).rename(columns={"date": "timestamp"})
        with pytest.raises(ValueError):
            summarise_peaks(df)

    def test_unsupported_freq_in_list_raises(self):
        df = _make_prices_df(300, 4.0)
        with pytest.raises(ValueError):
            summarise_peaks(df, freqs=["D"])

    def test_mixed_freqs_one_unsupported_raises(self):
        df = _make_prices_df(300, 4.0)
        with pytest.raises(ValueError):
            summarise_peaks(df, freqs=["ME", "BADFREQ"])


# ---------------------------------------------------------------------------
# summarise_peaks — date normalisation (column vs DatetimeIndex)
# ---------------------------------------------------------------------------


class TestSummarisePeaksDateFormats:
    def test_date_as_column(self):
        df = _make_prices_df(500, cycles=8.0, as_index=False)
        assert "date" in df.columns
        result = summarise_peaks(df, freqs=["ME"])
        assert len(result) == 1

    def test_date_as_datetimeindex(self):
        df = _make_prices_df(500, cycles=8.0, as_index=True)
        assert isinstance(df.index, pd.DatetimeIndex)
        assert "date" not in df.columns
        result = summarise_peaks(df, freqs=["ME"])
        assert len(result) == 1

    def test_column_and_index_give_same_peak_count(self):
        df_col = _make_prices_df(500, cycles=8.0, as_index=False)
        df_idx = _make_prices_df(500, cycles=8.0, as_index=True)
        r_col = summarise_peaks(df_col, freqs=["ME"])
        r_idx = summarise_peaks(df_idx, freqs=["ME"])
        assert r_col["peak_count"].iloc[0] == r_idx["peak_count"].iloc[0]


# ---------------------------------------------------------------------------
# summarise_peaks — output schema and shape
# ---------------------------------------------------------------------------


class TestSummarisePeaksOutput:
    @pytest.fixture()
    def two_ticker_df(self) -> pd.DataFrame:
        return pd.concat(
            [
                _make_prices_df(365 * 4, cycles=16.0, ticker="AAA"),
                _make_prices_df(365 * 4, cycles=16.0, ticker="BBB"),
            ],
            ignore_index=True,
        )

    def test_output_has_required_columns(self, two_ticker_df):
        result = summarise_peaks(two_ticker_df)
        for col in (
            "ticker",
            "freq",
            "peak_count",
            "mean_peak_gap_days",
            "std_peak_gap_days",
        ):
            assert col in result.columns

    def test_one_row_per_ticker_freq(self, two_ticker_df):
        result = summarise_peaks(two_ticker_df)
        assert result.duplicated(subset=["ticker", "freq"]).sum() == 0

    def test_all_default_freqs_present(self, two_ticker_df):
        result = summarise_peaks(two_ticker_df)
        assert set(result["freq"].unique()) == set(SUPPORTED_FREQS)

    def test_both_tickers_present(self, two_ticker_df):
        result = summarise_peaks(two_ticker_df)
        assert set(result["ticker"].unique()) == {"AAA", "BBB"}

    def test_custom_freqs_respected(self, two_ticker_df):
        result = summarise_peaks(two_ticker_df, freqs=["ME", "QE"])
        assert set(result["freq"].unique()) == {"ME", "QE"}

    def test_peak_count_dtype_integer(self, two_ticker_df):
        result = summarise_peaks(two_ticker_df)
        assert pd.api.types.is_integer_dtype(result["peak_count"])

    def test_gap_columns_dtype_float(self, two_ticker_df):
        result = summarise_peaks(two_ticker_df)
        for col in ("mean_peak_gap_days", "std_peak_gap_days"):
            assert pd.api.types.is_float_dtype(result[col])
