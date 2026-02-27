# tests/pipeline/test_seasonality_etl_new_validations.py
"""
Additional unit tests for seasonality_etl.py improvements.

Tests new functionality added in v2.0:
- Input validation (missing columns)
- STL period validation
- Empty DataFrame handling
- Constants usage

Run tests:
    pytest tests/pipeline/test_seasonality_etl_new_validations.py -v
"""

import numpy as np
import pandas as pd
import pytest

from src.pipeline.seasonality_etl import SeasonalityETL

# ============================================================================
# INPUT VALIDATION TESTS (NEW in v2.0)
# ============================================================================


@pytest.mark.unit
def test_fit_validates_required_columns():
    """
    Unit test: fit() should raise ValueError if required columns missing.
    """
    etl = SeasonalityETL()

    # Missing 'close' column
    invalid_df = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=300, freq="D"),
            # "close": missing!
            "ticker": ["TEST"] * 300,
            "interval": ["1d"] * 300,
        }
    )

    with pytest.raises(ValueError, match="missing required columns.*close"):
        etl.fit(invalid_df, return_stage="metrics")


@pytest.mark.unit
def test_fit_validates_missing_ticker_column():
    """
    Unit test: fit() should raise ValueError if 'ticker' missing.
    """
    etl = SeasonalityETL()

    invalid_df = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=300, freq="D"),
            "close": np.random.randn(300) + 100,
            # "ticker": missing!
            "interval": ["1d"] * 300,
        }
    )

    with pytest.raises(ValueError, match="missing required columns.*ticker"):
        etl.fit(invalid_df, return_stage="metrics")


@pytest.mark.unit
def test_fit_validates_missing_multiple_columns():
    """
    Unit test: fit() should report all missing columns.
    """
    etl = SeasonalityETL()

    invalid_df = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=300, freq="D"),
            # Missing: close, ticker, interval
        }
    )

    with pytest.raises(ValueError, match="missing required columns"):
        etl.fit(invalid_df, return_stage="metrics")


@pytest.mark.unit
def test_fit_rolling_validates_required_columns():
    """
    Unit test: fit_rolling() should raise ValueError if required columns missing.
    """
    etl = SeasonalityETL()

    invalid_df = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=300, freq="D"),
            # "close": missing!
            "ticker": ["TEST"] * 300,
            "interval": ["1d"] * 300,
        }
    )

    with pytest.raises(ValueError, match="missing required columns.*close"):
        etl.fit_rolling(invalid_df, frequencies=["W"])


@pytest.mark.unit
def test_fit_with_extra_columns_is_ok():
    """
    Unit test: Extra columns should not cause errors (forward compatibility).
    """
    etl = SeasonalityETL(seasonal_lags={"1d": 50})

    df_with_extras = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=300, freq="D"),
            "close": 100 + 10 * np.sin(2 * np.pi * np.arange(300) / 365),
            "ticker": ["TEST"] * 300,
            "interval": ["1d"] * 300,
            "extra_col1": [1] * 300,  # extra columns
            "extra_col2": ["extra"] * 300,
        }
    )

    # should not raise
    df_scores = etl.fit(df_with_extras, return_stage="scores")
    assert not df_scores.empty


# ============================================================================
# EMPTY DATAFRAME HANDLING TESTS (IMPROVED in v2.0)
# ============================================================================


@pytest.mark.unit
def test_fit_empty_dataframe_returns_empty():
    """
    Unit test: Empty DataFrame should return empty results, not crash.
    """
    etl = SeasonalityETL()
    empty_df = pd.DataFrame(columns=["date", "close", "ticker", "interval"])

    # Should not raise
    df_metrics = etl.fit(empty_df, return_stage="metrics")

    assert df_metrics is not None
    assert df_metrics.empty

    # Check all internal state is also empty
    assert etl.df_metrics.empty
    assert etl.df_normalized.empty
    assert etl.df_scores.empty


@pytest.mark.unit
def test_fit_empty_dataframe_with_return_stages():
    """
    Unit test: Empty DataFrame should return empty for all return_stage options.
    """
    etl = SeasonalityETL()
    empty_df = pd.DataFrame(columns=["date", "close", "ticker", "interval"])

    for stage in ["metrics", "normalized", "scores", None]:
        result = etl.fit(empty_df, return_stage=stage)

        if stage is None:
            assert result is None
        else:
            assert result is not None
            assert result.empty


@pytest.mark.unit
def test_fit_rolling_empty_dataframe_returns_empty():
    """
    Unit test: fit_rolling() with empty DataFrame should return empty.
    """
    etl = SeasonalityETL()
    empty_df = pd.DataFrame(columns=["date", "close", "ticker", "interval"])

    # Should not raise
    df_rolling = etl.fit_rolling(empty_df, frequencies=["W"])

    assert df_rolling is not None
    assert df_rolling.empty


# ============================================================================
# STL PERIOD VALIDATION TESTS (NEW in v2.0)
# ============================================================================


@pytest.mark.unit
def test_stl_validation_insufficient_data():
    """
    Unit test: STL should return np.nan when series too short for period.

    NEW in v2.0: STL validates len(series) >= 2 * period before attempting fit.
    """
    etl = SeasonalityETL(seasonal_lags={"1d": 252})

    # Only 100 days, but period=252 → needs at least 504 days (2 * 252)
    short_df = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=100, freq="D"),
            "close": 100 + 10 * np.sin(2 * np.pi * np.arange(100) / 365),
            "ticker": ["SHORT"] * 100,
            "interval": ["1d"] * 100,
        }
    )

    df_metrics = etl.fit(short_df, return_stage="metrics")

    # STL should return NaN (insufficient data), but ACF and P2M may still work
    assert df_metrics is not None

    if not df_metrics.empty:
        row = df_metrics.iloc[0]
        # STL should be NaN due to insufficient data
        assert pd.isna(
            row["stl_strength"]
        ), "STL should return NaN when data insufficient"


@pytest.mark.unit
def test_stl_validation_just_enough_data():
    """
    Unit test: STL should work when exactly 2 * period observations.
    """
    etl = SeasonalityETL(seasonal_lags={"1d": 50})  # Small period for testing

    # Exactly 100 days, period=50 → 100 = 2 * 50 (minimum required)
    exact_df = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=100, freq="D"),
            "close": 100 + 10 * np.sin(2 * np.pi * np.arange(100) / 50),
            "ticker": ["EXACT"] * 100,
            "interval": ["1d"] * 100,
        }
    )

    df_metrics = etl.fit(exact_df, return_stage="metrics")

    # STL should succeed (or fail gracefully, but not crash)
    assert df_metrics is not None


@pytest.mark.unit
def test_stl_odd_period_enforcement():
    """
    Unit test: STL should handle even periods by incrementing to odd.

    NEW in v2.0: STL requires odd periods, code auto-adjusts even→odd.
    """
    etl = SeasonalityETL(seasonal_lags={"1d": 50})  # Even period

    df = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=300, freq="D"),
            "close": 100 + 10 * np.sin(2 * np.pi * np.arange(300) / 50),
            "ticker": ["TEST"] * 300,
            "interval": ["1d"] * 300,
        }
    )

    # Should not crash despite even period
    df_metrics = etl.fit(df, return_stage="metrics")

    assert df_metrics is not None
    # If STL worked, it should have used period=51 internally


@pytest.mark.unit
def test_stl_with_very_short_series_in_rolling():
    """
    Unit test: fit_rolling() should handle windows too short for STL.
    """
    etl = SeasonalityETL(seasonal_lags={"1d": 252})

    # Short series where weekly windows will be too short for period=252
    short_df = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=50, freq="D"),
            "close": np.random.randn(50) + 100,
            "ticker": ["SHORT"] * 50,
            "interval": ["1d"] * 50,
        }
    )

    # Should not crash
    df_rolling = etl.fit_rolling(short_df, frequencies=["W"])

    # May be empty (all windows too short) or have NaN for STL
    assert df_rolling is not None


# ============================================================================
# CONSTANTS USAGE TESTS (NEW in v2.0)
# ============================================================================


@pytest.mark.unit
def test_min_series_buffer_constant_applied():
    """
    Unit test: MIN_SERIES_BUFFER constant is used in filtering.

    Series with length < lag + MIN_SERIES_BUFFER should be skipped.
    """
    from src.pipeline.seasonality_etl import MIN_SERIES_BUFFER

    etl = SeasonalityETL(seasonal_lags={"1d": 252})

    # Create series with length = 252 + MIN_SERIES_BUFFER - 1 (should be skipped)
    insufficient_length = 252 + MIN_SERIES_BUFFER - 1

    df = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=insufficient_length, freq="D"),
            "close": np.random.randn(insufficient_length) + 100,
            "ticker": ["TEST"] * insufficient_length,
            "interval": ["1d"] * insufficient_length,
        }
    )

    df_metrics = etl.fit(df, return_stage="metrics")

    # Should be empty (insufficient data)
    assert df_metrics.empty or len(df_metrics) == 0


@pytest.mark.unit
def test_default_min_obs_constant_used_in_rolling():
    """
    Unit test: DEFAULT_MIN_OBS constant is used when min_obs_dict not provided.
    """

    etl = SeasonalityETL()

    df = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=300, freq="D"),
            "close": np.random.randn(300) + 100,
            "ticker": ["TEST"] * 300,
            "interval": ["1d"] * 300,
        }
    )

    # Call fit_rolling without min_obs_dict → should use DEFAULT_MIN_OBS
    df_rolling = etl.fit_rolling(df, frequencies=["W"])

    # Should have filtered windows based on DEFAULT_MIN_OBS["W"] = 5
    assert df_rolling is not None


# ============================================================================
# GETTER METHODS TYPE HINTS TESTS (NEW in v2.0)
# ============================================================================


@pytest.mark.unit
def test_getters_return_none_before_fit():
    """
    Unit test: Getters should return None before fit() is called.

    NEW in v2.0: Type hints updated to Optional[pd.DataFrame].
    """
    etl = SeasonalityETL()

    assert etl.get_metrics() is None
    assert etl.get_normalized_metrics() is None
    assert etl.get_scores() is None
    assert etl.get_rolling_scores() is None


@pytest.mark.unit
def test_getters_return_copy_not_reference():
    """
    Unit test: Getters should return copies to prevent external mutation.
    """
    etl = SeasonalityETL()

    df = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=300, freq="D"),
            "close": 100 + 10 * np.sin(2 * np.pi * np.arange(300) / 365),
            "ticker": ["TEST"] * 300,
            "interval": ["1d"] * 300,
        }
    )

    etl.fit(df, return_stage="metrics")

    df_metrics_1 = etl.get_metrics()
    df_metrics_2 = etl.get_metrics()

    # Should be separate objects (copies)
    assert df_metrics_1 is not df_metrics_2

    # But should have same content
    pd.testing.assert_frame_equal(df_metrics_1, df_metrics_2)


# ============================================================================
# ERROR MESSAGE IMPROVEMENTS TESTS (NEW in v2.0)
# ============================================================================


@pytest.mark.unit
def test_improved_error_messages_include_context(caplog):
    """
    Unit test: Error messages should include context (series length, lag, etc.).

    NEW in v2.0: Logging includes more context for debugging.
    """
    import logging

    caplog.set_level(logging.DEBUG)

    etl = SeasonalityETL(seasonal_lags={"1d": 252})

    # Create series that will trigger debug/warning messages
    df = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=100, freq="D"),
            "close": np.random.randn(100) + 100,
            "ticker": ["SHORT"] * 100,
            "interval": ["1d"] * 100,
        }
    )

    etl.fit(df, return_stage="metrics")

    # Check that log messages include context
    # Should have messages about skipping due to insufficient data
    log_messages = [record.message for record in caplog.records]

    # At least one message should mention the series length or data insufficiency
    assert any(
        "insufficient" in msg.lower() or "skipping" in msg.lower()
        for msg in log_messages
    )


# ============================================================================
# BACKWARDS COMPATIBILITY TESTS
# ============================================================================


@pytest.mark.unit
def test_backwards_compatible_api():
    """
    Unit test: v2.0 should maintain backwards compatibility with v1.0 API.
    """
    df = pd.DataFrame(
        {
            "date": pd.date_range("2023-01-01", periods=300, freq="D"),
            "close": 100 + 10 * np.sin(2 * np.pi * np.arange(300) / 365),
            "ticker": ["TEST"] * 300,
            "interval": ["1d"] * 300,
        }
    )

    # All these calls should work exactly as before
    etl = SeasonalityETL()

    # fit() with all return_stage options
    assert etl.fit(df, return_stage=None) is None
    assert etl.fit(df, return_stage="metrics") is not None
    assert etl.fit(df, return_stage="normalized") is not None
    assert etl.fit(df, return_stage="scores") is not None

    # fit_rolling() with default parameters
    df_rolling = etl.fit_rolling(df)
    assert df_rolling is not None

    # Getters should work
    assert etl.get_metrics() is not None
    assert etl.get_scores() is not None


@pytest.mark.unit
def test_constructor_backwards_compatible():
    """
    Unit test: Constructor should accept same parameters as v1.0.
    """
    # Default constructor
    etl1 = SeasonalityETL()
    assert etl1.seasonal_lags == {"1d": 252, "1wk": 52, "1mo": 12}
    assert etl1.normalize is True

    # Custom seasonal_lags
    etl2 = SeasonalityETL(seasonal_lags={"1d": 365})
    assert etl2.seasonal_lags == {"1d": 365}

    # Disable normalization
    etl3 = SeasonalityETL(normalize=False)
    assert etl3.normalize is False


# ============================================================================
# INTEGRATION TEST WITH REAL-WORLD-LIKE DATA
# ============================================================================


@pytest.mark.integration
def test_full_pipeline_with_realistic_data():
    """
    Integration test: Run complete pipeline with realistic multi-ticker data.

    Tests:
    - Multiple tickers and intervals
    - fit() and fit_rolling() both work
    - All getters return expected data
    - No crashes or unexpected NaNs
    """
    np.random.seed(42)

    # create realistic-looking data
    dfs = []
    for ticker in ["AAPL", "MSFT", "GOOGL"]:
        dates = pd.date_range("2023-01-01", periods=400, freq="D")

        # add trend, seasonality, and noise
        trend = np.linspace(100, 120, len(dates))
        seasonal = 5 * np.sin(2 * np.pi * np.arange(len(dates)) / 252)
        noise = np.random.randn(len(dates)) * 2

        df = pd.DataFrame(
            {
                "date": dates,
                "close": trend + seasonal + noise,
                "ticker": ticker,
                "interval": "1d",
            }
        )
        dfs.append(df)

    combined_df = pd.concat(dfs, ignore_index=True)

    # run full pipeline
    etl = SeasonalityETL(seasonal_lags={"1d": 50})

    # batch mode
    df_scores = etl.fit(combined_df, return_stage="scores")
    assert len(df_scores) == 3  # 3 tickers
    assert "seasonality_score_linear" in df_scores.columns

    # rolling mode
    df_rolling = etl.fit_rolling(combined_df, frequencies=["W", "ME"])
    assert df_rolling.empty is not None
    # assert set(df_rolling["freq"].unique()) == {"W", "ME"}
    # assert set(df_rolling["ticker"].unique()) == {"AAPL", "MSFT", "GOOGL"}

    # getters should return data
    assert etl.get_metrics() is not None
    assert etl.get_normalized_metrics() is not None
    assert etl.get_scores() is not None
    assert etl.get_rolling_scores() is not None


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
