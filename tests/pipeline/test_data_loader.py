# tests/pipeline/test_data_loader.py

from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import pytest

from src.pipeline.data_loader import (
    MAX_RETRIES,
    DataLoader,
)

# fixtures


@pytest.fixture
def temp_config_file(tmp_path):
    """
    Create a minimal valid tickers YAML config.
    """
    config = tmp_path / "tickers_list.yaml"
    config.write_text("tickers:\n  - AAPL\n  - MSFT\n  - GOOGL\n")
    return config


@pytest.fixture
def mock_yfinance_data():
    """
    Realistic yfinance DataFrame with standard columns.
    """
    dates = pd.date_range("2023-01-01", periods=5, freq="D")
    df = pd.DataFrame(
        {
            "Open": np.random.uniform(100, 200, 5),
            "High": np.random.uniform(100, 200, 5),
            "Low": np.random.uniform(100, 200, 5),
            "Close": np.random.uniform(100, 200, 5),
            "Volume": np.random.randint(1_000_000, 10_000_000, 5),
        },
        index=dates,
    )
    df.index.name = "Date"
    return df


# Constructor validation


@pytest.mark.unit
def test_constructor_with_valid_config(temp_config_file):
    """
    Test that constructor succeeds with valid config.
    """
    loader = DataLoader(config_path=str(temp_config_file), use_cache=False)
    # compare raw strings (avoid macOS /var -> /private/var symlink mismatch)
    assert loader.config_path == str(temp_config_file)
    assert loader.stats == {"success": 0, "failed": 0, "total": 0}
    # no cache path yet — computed lazily in load()
    assert loader.combined_cache_path is None


@pytest.mark.unit
def test_constructor_missing_config_raises():
    """
    Test that constructor raises FileNotFoundError for missing config.
    """
    with pytest.raises(FileNotFoundError, match="Config file not found"):
        DataLoader(config_path="nonexistent/path.yaml")


@pytest.mark.unit
def test_constructor_with_directory_as_config(tmp_path):
    """
    Test that constructor raises ValueError when config_path is a directory.
    """
    with pytest.raises(ValueError, match="Config path is not a file"):
        DataLoader(config_path=str(tmp_path))


@pytest.mark.unit
def test_constructor_with_explicit_cache_path(temp_config_file, tmp_path):
    """
    Test that explicit cache path overrides auto-computation.
    """
    cache_path = tmp_path / "my_cache.parquet"
    loader = DataLoader(
        config_path=str(temp_config_file),
        combined_cache_path=str(cache_path),
    )
    assert loader.combined_cache_path == cache_path


# YAML validation


@pytest.mark.unit
def test_read_tickers_valid(temp_config_file):
    loader = DataLoader(config_path=str(temp_config_file), use_cache=False)
    tickers = loader._read_tickers()
    assert tickers == ["AAPL", "MSFT", "GOOGL"]


@pytest.mark.unit
def test_read_tickers_missing_key(tmp_path):
    config = tmp_path / "bad.yaml"
    config.write_text("something_else:\n  - AAPL\n")
    loader = DataLoader(config_path=str(config), use_cache=False)
    with pytest.raises(ValueError, match="must contain 'tickers' key"):
        loader._read_tickers()


@pytest.mark.unit
def test_read_tickers_empty_list(tmp_path):
    config = tmp_path / "empty.yaml"
    config.write_text("tickers: []\n")
    loader = DataLoader(config_path=str(config), use_cache=False)
    with pytest.raises(ValueError, match="cannot be empty"):
        loader._read_tickers()


@pytest.mark.unit
def test_read_tickers_non_string_items(tmp_path):
    config = tmp_path / "bad_types.yaml"
    config.write_text("tickers:\n  - 123\n  - AAPL\n")
    loader = DataLoader(config_path=str(config), use_cache=False)
    with pytest.raises(ValueError, match="must be strings"):
        loader._read_tickers()


# date validation


@pytest.mark.unit
def test_validate_date_valid():
    DataLoader._validate_date("2023-01-01")  # should not raise


@pytest.mark.unit
def test_validate_date_invalid():
    with pytest.raises(ValueError, match="Invalid date format"):
        DataLoader._validate_date("not-a-date")


@pytest.mark.unit
def test_validate_date_wrong_format():
    with pytest.raises(ValueError, match="Invalid date format"):
        DataLoader._validate_date("01/01/2023")


# interval validation


@pytest.mark.unit
def test_validate_intervals_valid():
    DataLoader._validate_intervals(["1d", "1wk"])  # should not raise


@pytest.mark.unit
def test_validate_intervals_invalid():
    with pytest.raises(ValueError, match="Invalid intervals"):
        DataLoader._validate_intervals(["1d", "2y"])


# cache hash


@pytest.mark.unit
def test_cache_hash_order_independent():
    """
    Same tickers in different order produce the same hash.
    """
    h1 = DataLoader._compute_cache_hash(
        ["AAPL", "MSFT"], ["1d"], "2022-01-01", "2023-01-01"
    )
    h2 = DataLoader._compute_cache_hash(
        ["MSFT", "AAPL"], ["1d"], "2022-01-01", "2023-01-01"
    )
    assert h1 == h2


@pytest.mark.unit
def test_cache_hash_different_intervals():
    """
    Different intervals produce different hashes.
    """
    h1 = DataLoader._compute_cache_hash(["AAPL"], ["1d"], "2022-01-01", "2023-01-01")
    h2 = DataLoader._compute_cache_hash(["AAPL"], ["1wk"], "2022-01-01", "2023-01-01")
    assert h1 != h2


@pytest.mark.unit
def test_cache_hash_different_dates():
    """
    Different date ranges produce different hashes.
    """
    h1 = DataLoader._compute_cache_hash(["AAPL"], ["1d"], "2022-01-01", "2023-01-01")
    h2 = DataLoader._compute_cache_hash(["AAPL"], ["1d"], "2020-01-01", "2023-01-01")
    assert h1 != h2


@pytest.mark.unit
def test_cache_hash_length():
    h = DataLoader._compute_cache_hash(["AAPL"], ["1d"], "2022-01-01", "2023-01-01")
    assert len(h) == 8


# cache freshness


@pytest.mark.unit
def test_is_cache_fresh_when_no_path_set(temp_config_file):
    """
    Cache is not fresh when no path has been computed yet.
    """
    loader = DataLoader(config_path=str(temp_config_file), use_cache=True)
    assert loader._is_cache_fresh() is False


@pytest.mark.unit
def test_is_cache_fresh_nonexistent(temp_config_file):
    """Cache is not fresh when file doesn't exist."""
    loader = DataLoader(
        config_path=str(temp_config_file),
        use_cache=True,
        combined_cache_path="/tmp/nonexistent_cache.parquet",
    )
    assert loader._is_cache_fresh() is False


@pytest.mark.unit
def test_is_cache_fresh_recent_file(temp_config_file, tmp_path):
    """
    A recently created cache file is fresh.
    """
    cache_file = tmp_path / "fresh_cache.parquet"
    cache_file.touch()

    loader = DataLoader(
        config_path=str(temp_config_file),
        use_cache=True,
        combined_cache_path=str(cache_file),
    )
    assert loader._is_cache_fresh() is True


# build cache path


@pytest.mark.unit
def test_build_cache_path_includes_dates(temp_config_file):
    """Cache filename encodes the date range."""
    loader = DataLoader(config_path=str(temp_config_file), use_cache=False)
    path = loader._build_cache_path(["AAPL"], ["1d"], "2022-01-01", "2023-01-01")
    assert "2022-01-01" in path.name
    assert "2023-01-01" in path.name
    assert path.suffix == ".parquet"


@pytest.mark.unit
def test_build_cache_path_different_params_differ(temp_config_file):
    """
    Different query params produce different filenames.
    """
    loader = DataLoader(config_path=str(temp_config_file), use_cache=False)
    p1 = loader._build_cache_path(["AAPL"], ["1d"], "2022-01-01", "2023-01-01")
    p2 = loader._build_cache_path(["AAPL"], ["1d"], "2020-01-01", "2023-01-01")
    assert p1 != p2


# retry logic (mocked)


@pytest.mark.unit
@patch("src.pipeline.data_loader.yf.download")
def test_retry_success_first_attempt(
    mock_download, temp_config_file, mock_yfinance_data
):
    mock_download.return_value = mock_yfinance_data
    loader = DataLoader(config_path=str(temp_config_file), use_cache=False)
    result = loader._download_with_retry("AAPL", "2023-01-01", "2023-01-10", "1d")

    assert result is not None
    assert not result.empty
    assert mock_download.call_count == 1


@pytest.mark.unit
@patch("src.pipeline.data_loader.time.sleep")
@patch("src.pipeline.data_loader.yf.download")
def test_retry_success_after_failures(
    mock_download, mock_sleep, temp_config_file, mock_yfinance_data
):
    mock_download.side_effect = [
        Exception("Network error"),
        Exception("Timeout"),
        mock_yfinance_data,
    ]
    loader = DataLoader(config_path=str(temp_config_file), use_cache=False)
    result = loader._download_with_retry("AAPL", "2023-01-01", "2023-01-10", "1d")

    assert result is not None
    assert not result.empty
    assert mock_download.call_count == 3
    assert mock_sleep.call_count == 2


@pytest.mark.unit
@patch("src.pipeline.data_loader.time.sleep")
@patch("src.pipeline.data_loader.yf.download")
def test_retry_all_failures_returns_none(mock_download, mock_sleep, temp_config_file):
    mock_download.side_effect = Exception("Persistent error")
    loader = DataLoader(config_path=str(temp_config_file), use_cache=False)
    result = loader._download_with_retry("AAPL", "2023-01-01", "2023-01-10", "1d")

    assert result is None
    assert mock_download.call_count == MAX_RETRIES
    assert mock_sleep.call_count == MAX_RETRIES - 1


@pytest.mark.unit
@patch("src.pipeline.data_loader.time.sleep")
@patch("src.pipeline.data_loader.yf.download")
def test_retry_empty_dataframe_retries(mock_download, mock_sleep, temp_config_file):
    mock_download.return_value = pd.DataFrame()
    loader = DataLoader(config_path=str(temp_config_file), use_cache=False)
    result = loader._download_with_retry("BAD", "2023-01-01", "2023-01-10", "1d")

    assert result is None
    assert mock_download.call_count == MAX_RETRIES


# load method (mocked)


@pytest.mark.unit
@patch("src.pipeline.data_loader.yf.download")
def test_load_default_end_date(mock_download, temp_config_file, mock_yfinance_data):
    mock_download.return_value = mock_yfinance_data
    loader = DataLoader(
        config_path=str(temp_config_file), use_cache=False, save_combined=False
    )
    df = loader.load(start_date="2023-01-01")

    assert not df.empty
    assert "ticker" in df.columns
    assert "interval" in df.columns
    assert "date" in df.columns


@pytest.mark.unit
@patch("src.pipeline.data_loader.yf.download")
def test_load_sets_cache_path_from_params(
    mock_download, temp_config_file, mock_yfinance_data
):
    """
    load() computes a cache path that includes the date range.
    """
    mock_download.return_value = mock_yfinance_data
    loader = DataLoader(
        config_path=str(temp_config_file), use_cache=False, save_combined=False
    )
    assert loader.combined_cache_path is None

    loader.load(start_date="2022-01-01", end_date="2023-01-01")

    assert loader.combined_cache_path is not None
    assert "2022-01-01" in loader.combined_cache_path.name
    assert "2023-01-01" in loader.combined_cache_path.name


@pytest.mark.unit
@patch("src.pipeline.data_loader.time.sleep")
@patch("src.pipeline.data_loader.yf.download")
def test_load_statistics_tracking(
    mock_download, mock_sleep, temp_config_file, mock_yfinance_data
):
    # AAPL succeeds, MSFT fails (3x), GOOGL succeeds
    mock_download.side_effect = [
        mock_yfinance_data,
        Exception("Error"),
        Exception("Error"),
        Exception("Error"),
        mock_yfinance_data,
    ]
    loader = DataLoader(
        config_path=str(temp_config_file), use_cache=False, save_combined=False
    )
    loader.load(start_date="2023-01-01", end_date="2023-01-10")

    assert loader.stats["success"] == 2
    assert loader.stats["failed"] == 1
    assert loader.stats["total"] == 3


@pytest.mark.unit
@patch("src.pipeline.data_loader.time.sleep")
@patch("src.pipeline.data_loader.yf.download")
def test_load_all_failures_raises(mock_download, mock_sleep, temp_config_file):
    mock_download.side_effect = Exception("Everything is broken")
    loader = DataLoader(
        config_path=str(temp_config_file), use_cache=False, save_combined=False
    )
    with pytest.raises(ValueError, match="No data could be loaded"):
        loader.load(start_date="2023-01-01")


@pytest.mark.unit
def test_load_invalid_date_raises(temp_config_file):
    loader = DataLoader(config_path=str(temp_config_file), use_cache=False)
    with pytest.raises(ValueError, match="Invalid date format"):
        loader.load(start_date="bad-date")


@pytest.mark.unit
def test_load_invalid_interval_raises(temp_config_file):
    loader = DataLoader(config_path=str(temp_config_file), use_cache=False)
    with pytest.raises(ValueError, match="Invalid intervals"):
        loader.load(start_date="2023-01-01", intervals=["2y"])


# integration tests (real API)


@pytest.mark.integration
def test_integration_download_single_ticker(temp_config_file):
    """
    Smoke test: download a small date range for one ticker.
    """
    config = Path(temp_config_file).parent / "single.yaml"
    config.write_text("tickers:\n  - SPY\n")

    loader = DataLoader(
        config_path=str(config), use_cache=False, save_combined=False, verbose=True
    )
    df = loader.load(start_date="2024-01-01", end_date="2024-01-10")

    assert not df.empty
    assert "close" in df.columns
    assert "ticker" in df.columns
    assert df["ticker"].iloc[0] == "SPY"
    assert loader.stats["success"] >= 1
