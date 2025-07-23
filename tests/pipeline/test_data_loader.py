# tests/pipeline/test_data_loader.py

import pandas as pd
import pytest
import yfinance as yf

from src.pipeline.data_loader import DataLoader


@pytest.fixture
def dummy_config_file(tmp_path):
    config_path = tmp_path / "tickers_list.yaml"
    config_path.write_text("tickers:\n  - AAPL\n")
    return str(config_path)


def test_read_tickers_from_yaml(dummy_config_file):
    loader = DataLoader(config_path=dummy_config_file)
    tickers = loader._read_tickers()
    assert isinstance(tickers, list)
    assert "AAPL" in tickers


def test_data_download_and_combined_cache(monkeypatch, dummy_config_file, tmp_path):
    combined_path = tmp_path / "all_data_test.parquet"

    loader = DataLoader(
        config_path=dummy_config_file,
        use_cache=False,
        verbose=True,
        save_combined=True,
        combined_cache_path=combined_path,
    )

    df = loader.load(start_date="2023-01-01", end_date="2023-01-10", intervals=["1d"])

    assert isinstance(df, pd.DataFrame)
    assert not df.empty
    assert "ticker" in df.columns
    assert "interval" in df.columns
    assert (df["ticker"] == "AAPL").all()
    assert (df["interval"] == "1d").all()
    assert combined_path.exists()


def test_combined_cache_file_reuse(dummy_config_file, tmp_path):
    combined_path = tmp_path / "all_data_test.parquet"

    # First load to create the file
    loader = DataLoader(
        config_path=dummy_config_file,
        use_cache=False,
        save_combined=True,
        combined_cache_path=combined_path,
    )
    df1 = loader.load(start_date="2023-01-01", end_date="2023-01-10", intervals=["1d"])

    # Second load should use cache
    loader_cached = DataLoader(
        config_path=dummy_config_file, use_cache=True, combined_cache_path=combined_path
    )
    df2 = loader_cached.load(
        start_date="2023-01-01", end_date="2023-01-10", intervals=["1d"]
    )

    pd.testing.assert_frame_equal(df1, df2)


def test_fallback_no_data(monkeypatch, dummy_config_file):
    loader = DataLoader(config_path=dummy_config_file, use_cache=False)

    # Simulate yfinance returning an empty DataFrame
    monkeypatch.setattr(yf, "download", lambda *args, **kwargs: pd.DataFrame())

    with pytest.raises(ValueError):
        loader.load(start_date="2023-01-01", end_date="2023-01-10", intervals=["1d"])
