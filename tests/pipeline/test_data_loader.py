# tests/pipeline/test_data_loader.py

import pandas as pd
import pytest
import yfinance as yf

from src.pipeline.data_loader import DataLoader

"""
Unit tests for the DataLoader class in src/pipeline/data_loader.py.

Covers:
- Configuration loading from YAML.
- Ticker data downloading with caching enabled.
- Cache file creation and reuse.
- Gracefully handling of download failures or empty responses.

Relies on monkeypatching yfinance.download for isolation and speed.
"""


@pytest.fixture
def dummy_config_file(tmp_path):
    config_path = tmp_path / "tickers_list.yaml"
    config_path.write_text("tickers:\n  - AAPL\n")
    return str(config_path)


def test_load_config_reads_yaml(dummy_config_file):
    loader = DataLoader(config_path=dummy_config_file)
    tickers = loader.load_config()
    assert isinstance(tickers, list)
    assert "AAPL" in tickers


def test_data_download_and_cache(monkeypatch, dummy_config_file):
    # use a small date range to keep the test fast
    loader = DataLoader(config_path=dummy_config_file, use_cache=True, verbose=True)

    df = loader.load(start_date="2023-01-01", end_date="2023-01-10", intervals=["1d"])
    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "ticker" in df.columns
    assert "interval" in df.columns
    assert df["ticker"].nunique() == 1
    assert df["interval"].unique().tolist() == ["1d"]


def test_cache_file_written(tmp_path, dummy_config_file):
    loader = DataLoader(config_path=dummy_config_file, use_cache=True)
    loader.load(
        start_date="2023-01-01", end_date="2023-01-10", intervals=["1d"]
    )  # no need to assign to df

    cached_files = list(loader.cache_dir.glob("AAPL_*.parquet"))
    assert any(
        f.exists() for f in cached_files
    ), "Expected a cached parquet file to be created."


def test_fallback_no_data(monkeypatch, dummy_config_file):
    loader = DataLoader(config_path=dummy_config_file, use_cache=False)

    # monkeypatching yfinance to simulate empty data
    monkeypatch.setattr(yf, "download", lambda *args, **kwargs: pd.DataFrame())

    with pytest.raises(ValueError):
        loader.load(start_date="2023-01-01", end_date="2023-01-10", intervals=["1d"])
