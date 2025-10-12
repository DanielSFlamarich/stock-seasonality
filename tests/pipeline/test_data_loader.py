# tests/pipeline/test_data_loader.py
"""
Unit and integration tests for src.pipeline.data_loader.DataLoader

Test Categories:
- Unit tests (mocked yfinance): fast, deterministic, cover edge cases
- Integration tests (real yfinance): slower, validate end-to-end behavior

Run fast tests only:
    pytest tests/pipeline/test_data_loader.py -v -m "not integration"

Run all tests:
    pytest tests/pipeline/test_data_loader.py -v
"""

import logging

import pandas as pd
import pytest
import yfinance as yf

from src.pipeline.data_loader import DataLoader

# ============================================================================
# FIXTURES
# ============================================================================


@pytest.fixture
def dummy_config_file(tmp_path):
    """
    Single-ticker config for basic tests.
    """
    config_path = tmp_path / "tickers_list.yaml"
    config_path.write_text("tickers:\n  - AAPL\n")
    return str(config_path)


@pytest.fixture
def multi_ticker_config(tmp_path):
    """
    Multi-ticker config for testing batch downloads.
    """
    config_path = tmp_path / "tickers_multi.yaml"
    config_path.write_text("tickers:\n  - AAPL\n  - MSFT\n  - INVALID_TICKER\n")
    return str(config_path)


@pytest.fixture
def empty_config(tmp_path):
    """
    Config with empty tickers list.
    """
    config_path = tmp_path / "empty.yaml"
    config_path.write_text("tickers: []\n")
    return str(config_path)


@pytest.fixture
def mock_yf_single_ticker():
    """
    Mocks yfinance download to return synthetic OHLCV data for one ticker.
    Returns a function that can be used with monkeypatch or patch.
    """

    def _mock_download(ticker, start, end, interval, auto_adjust=True, progress=False):
        # simulate realistic yfinance response
        dates = pd.date_range(start=start, end=end, freq="D")[:5]  # 5 days
        data = pd.DataFrame(
            {
                "Date": dates,
                "Open": [100.0] * len(dates),
                "High": [105.0] * len(dates),
                "Low": [95.0] * len(dates),
                "Close": [102.0] * len(dates),
                "Volume": [1000000] * len(dates),
            }
        )
        data = data.set_index("Date")
        return data

    return _mock_download


@pytest.fixture
def mock_yf_partial_failure():
    """
    Mocks yfinance.download to return data for valid tickers, empty for invalid.
    Simulates the partial failure scenario.
    """

    def _mock_download(ticker, start, end, interval, auto_adjust=True, progress=False):
        if ticker == "INVALID_TICKER":
            return pd.DataFrame()  # Simulate failure

        dates = pd.date_range(start=start, end=end, freq="D")[:5]
        data = pd.DataFrame(
            {
                "Date": dates,
                "Open": [100.0] * len(dates),
                "High": [105.0] * len(dates),
                "Low": [95.0] * len(dates),
                "Close": [102.0] * len(dates),
                "Volume": [1000000] * len(dates),
            }
        )
        data = data.set_index("Date")
        return data

    return _mock_download


# ============================================================================
# UNIT TESTS (Fast, Mocked)
# ============================================================================


@pytest.mark.unit
def test_read_tickers_from_yaml(dummy_config_file):
    """
    Verify YAML config parsing returns list of tickers.
    """
    loader = DataLoader(config_path=dummy_config_file)
    tickers = loader._read_tickers()
    assert isinstance(tickers, list)
    assert "AAPL" in tickers
    assert len(tickers) == 1


@pytest.mark.unit
def test_read_tickers_multi(multi_ticker_config):
    """
    Verify multi-ticker config parsing.
    """
    loader = DataLoader(config_path=multi_ticker_config)
    tickers = loader._read_tickers()
    assert len(tickers) == 3
    assert set(tickers) == {"AAPL", "MSFT", "INVALID_TICKER"}


@pytest.mark.unit
def test_empty_config_returns_empty_list(empty_config):
    """
    Empty tickers list should return empty list (not crash).
    """
    loader = DataLoader(config_path=empty_config)
    tickers = loader._read_tickers()
    assert tickers == []


@pytest.mark.unit
def test_missing_config_file_raises_error(tmp_path):
    """
    Non-existent config file should raise FileNotFoundError.
    """
    with pytest.raises(FileNotFoundError):
        loader = DataLoader(config_path=str(tmp_path / "nonexistent.yaml"))
        loader._read_tickers()


@pytest.mark.unit
def test_column_standardization(
    monkeypatch, dummy_config_file, mock_yf_single_ticker, tmp_path
):
    """
    Verify columns are lowercased and 'date' is datetime.
    """
    monkeypatch.setattr(yf, "download", mock_yf_single_ticker)

    loader = DataLoader(
        config_path=dummy_config_file,
        use_cache=False,
        combined_cache_path=tmp_path / "test.parquet",
    )
    df = loader.load(start_date="2023-01-01", end_date="2023-01-10", intervals=["1d"])

    # Check all columns are lowercase
    assert all(
        col.islower() for col in df.columns
    ), f"Non-lowercase columns: {df.columns}"

    # Check 'date' column exists and is datetime
    assert "date" in df.columns, "Missing 'date' column after standardization"
    assert pd.api.types.is_datetime64_any_dtype(
        df["date"]
    ), "'date' column is not datetime"

    # Check expected columns present
    expected_cols = {
        "date",
        "open",
        "high",
        "low",
        "close",
        "volume",
        "ticker",
        "interval",
    }
    assert expected_cols.issubset(
        set(df.columns)
    ), f"Missing columns: {expected_cols - set(df.columns)}"


@pytest.mark.unit
def test_partial_failure_continues(
    monkeypatch, multi_ticker_config, mock_yf_partial_failure, tmp_path, caplog
):
    """
    Unit test: One ticker fails (INVALID_TICKER), others succeed.
    Verify: valid data returned, no exception raised.
    """
    monkeypatch.setattr(yf, "download", mock_yf_partial_failure)

    # set logging level to capture warnings
    caplog.set_level(logging.WARNING)

    loader = DataLoader(
        config_path=multi_ticker_config,
        use_cache=False,
        verbose=False,  # Disable tqdm progress bar
        combined_cache_path=tmp_path / "multi.parquet",
    )

    df = loader.load(start_date="2023-01-01", end_date="2023-01-10", intervals=["1d"])

    # should have data for AAPL and MSFT only (INVALID_TICKER should be skipped)
    assert not df.empty, "DataFrame should not be empty"
    assert set(df["ticker"].unique()) == {
        "AAPL",
        "MSFT",
    }, "Should only have valid tickers"

    # verify INVALID_TICKER is NOT in the results (it failed gracefully)
    assert "INVALID_TICKER" not in df["ticker"].values


@pytest.mark.unit
def test_all_tickers_fail_raises_error(monkeypatch, dummy_config_file, tmp_path):
    """
    If all tickers fail to download, should raise ValueError.
    """
    # Mock yfinance to always return empty DataFrame
    monkeypatch.setattr(yf, "download", lambda *args, **kwargs: pd.DataFrame())

    loader = DataLoader(
        config_path=dummy_config_file,
        use_cache=False,
        combined_cache_path=tmp_path / "test.parquet",
    )

    with pytest.raises(ValueError, match="No data could be loaded"):
        loader.load(start_date="2023-01-01", end_date="2023-01-10", intervals=["1d"])


@pytest.mark.unit
def test_multiple_intervals(
    monkeypatch, dummy_config_file, mock_yf_single_ticker, tmp_path
):
    """
    Load data for multiple intervals (1d, 1wk) simultaneously.
    """
    monkeypatch.setattr(yf, "download", mock_yf_single_ticker)

    loader = DataLoader(
        config_path=dummy_config_file,
        use_cache=False,
        combined_cache_path=tmp_path / "test.parquet",
    )
    df = loader.load(
        start_date="2023-01-01", end_date="2023-01-31", intervals=["1d", "1wk"]
    )

    # Should have both intervals
    assert set(df["interval"].unique()) == {"1d", "1wk"}, "Missing intervals"

    # Each ticker-interval combination should have data
    groups = df.groupby(["ticker", "interval"]).size()
    assert (
        len(groups) == 2
    ), "Should have 2 ticker-interval combinations (AAPL-1d, AAPL-1wk)"


@pytest.mark.unit
def test_cache_path_normalization(dummy_config_file, tmp_path):
    """
    Verify cache path is normalized (no '..' allowed).
    Security: prevent path traversal attacks.
    """
    # This is a security test - ensure Path.resolve() is used
    safe_path = tmp_path / "cache" / "data.parquet"
    loader = DataLoader(config_path=dummy_config_file, combined_cache_path=safe_path)

    # Path should be absolute and normalized
    assert loader.combined_cache_path.is_absolute()
    assert ".." not in str(loader.combined_cache_path)


# ============================================================================
# INTEGRATION TESTS (Slower, Real yfinance)
# ============================================================================


@pytest.mark.integration
def test_data_download_and_combined_cache(dummy_config_file, tmp_path):
    """
    Integration test: real yfinance download + cache creation.
    Validates end-to-end data loading pipeline.
    """
    combined_path = tmp_path / "all_data_test.parquet"

    loader = DataLoader(
        config_path=dummy_config_file,
        use_cache=False,
        verbose=True,
        save_combined=True,
        combined_cache_path=combined_path,
    )

    df = loader.load(start_date="2023-01-01", end_date="2023-01-10", intervals=["1d"])

    # Validate DataFrame structure
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "ticker" in df.columns
    assert "interval" in df.columns
    assert (df["ticker"] == "AAPL").all()
    assert (df["interval"] == "1d").all()

    # Validate cache file created
    assert combined_path.exists()

    # Validate cache file is valid Parquet
    df_cached = pd.read_parquet(combined_path)
    pd.testing.assert_frame_equal(df, df_cached)


@pytest.mark.integration
def test_combined_cache_file_reuse(dummy_config_file, tmp_path):
    """
    Integration test: verify cache reuse path.
    Second load should read from cache, not download.
    """
    combined_path = tmp_path / "all_data_test.parquet"

    # First load to create the file
    loader = DataLoader(
        config_path=dummy_config_file,
        use_cache=False,
        save_combined=True,
        combined_cache_path=combined_path,
    )
    df1 = loader.load(start_date="2023-01-01", end_date="2023-01-10", intervals=["1d"])

    # Second load should use cache (no download)
    loader_cached = DataLoader(
        config_path=dummy_config_file, use_cache=True, combined_cache_path=combined_path
    )
    df2 = loader_cached.load(
        start_date="2023-01-01", end_date="2023-01-10", intervals=["1d"]
    )

    # DataFrames should be identical
    pd.testing.assert_frame_equal(df1, df2)


@pytest.mark.integration
def test_real_multi_ticker_download(multi_ticker_config, tmp_path):
    """
    Integration test: download multiple tickers with real yfinance.
    Note: INVALID_TICKER will fail, but AAPL/MSFT should succeed.
    """
    loader = DataLoader(
        config_path=multi_ticker_config,
        use_cache=False,
        verbose=True,
        combined_cache_path=tmp_path / "multi.parquet",
    )

    df = loader.load(start_date="2023-01-01", end_date="2023-01-10", intervals=["1d"])

    # Should have at least 2 valid tickers (AAPL, MSFT)
    assert df["ticker"].nunique() >= 2
    assert "AAPL" in df["ticker"].values
    assert "MSFT" in df["ticker"].values


# ============================================================================
# EDGE CASE TESTS
# ============================================================================


@pytest.mark.unit
def test_date_column_parsing_with_timezone(monkeypatch, dummy_config_file, tmp_path):
    """
    Verify date column handles timezone-aware datetimes from yfinance.
    Some yfinance responses include tz info.
    """

    def _mock_with_tz(*args, **kwargs):
        dates = pd.date_range(start="2023-01-01", end="2023-01-05", freq="D", tz="UTC")
        data = pd.DataFrame(
            {
                "Date": dates,
                "Close": [100.0] * len(dates),
            }
        )
        data = data.set_index("Date")
        return data

    monkeypatch.setattr(yf, "download", _mock_with_tz)

    loader = DataLoader(
        config_path=dummy_config_file,
        use_cache=False,
        combined_cache_path=tmp_path / "test.parquet",
    )
    df = loader.load(start_date="2023-01-01", end_date="2023-01-05", intervals=["1d"])

    # should handle timezone-aware dates without error
    assert "date" in df.columns
    assert pd.api.types.is_datetime64_any_dtype(df["date"])


@pytest.mark.unit
def test_nested_tuple_columns_flattened(monkeypatch, dummy_config_file, tmp_path):
    """
    Unit test: Verify nested tuple columns (from multi-ticker downloads) are flattened.
    yfinance sometimes returns MultiIndex columns.
    """

    def _mock_multi_index(*args, **kwargs):
        dates = pd.date_range(start="2023-01-01", end="2023-01-05", freq="D")
        data = pd.DataFrame(
            {
                ("Close", "AAPL"): [100.0] * len(dates),
                ("Volume", "AAPL"): [1000] * len(dates),
                ("Open", "AAPL"): [100.0] * len(dates),
                ("High", "AAPL"): [100.0] * len(dates),
                ("Low", "AAPL"): [100.0] * len(dates),
            },
            index=dates,
        )
        data.index.name = "Date"
        return data

    monkeypatch.setattr(yf, "download", _mock_multi_index)

    loader = DataLoader(
        config_path=dummy_config_file,
        use_cache=False,
        combined_cache_path=tmp_path / "test.parquet",
    )
    df = loader.load(start_date="2023-01-01", end_date="2023-01-05", intervals=["1d"])

    # all columns should be strings (no tuples)
    assert all(
        isinstance(col, str) for col in df.columns
    ), f"Non-string columns found: {[c for c in df.columns if not isinstance(c, str)]}"
