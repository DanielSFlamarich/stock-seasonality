import numpy as np
import pandas as pd
import pytest

from src.pipeline.seasonality_etl import SeasonalityETL


@pytest.fixture
def dummy_df():
    dates = pd.date_range(start="2023-01-01", periods=400)
    data = {
        "date": dates,
        "close": np.sin(2 * np.pi * dates.dayofyear / 365)
        + np.random.normal(0, 0.1, len(dates)),
        "ticker": ["TEST"] * len(dates),
        "interval": ["1d"] * len(dates),
    }
    return pd.DataFrame(data)


# ---------- Tests for Batch Mode (fit) ----------


def test_return_metrics(dummy_df):
    etl = SeasonalityETL()
    df_metrics = etl.fit(dummy_df, return_stage="metrics")
    assert "acf_lag_val" in df_metrics.columns
    assert not df_metrics.empty


def test_return_normalized(dummy_df):
    etl = SeasonalityETL()
    df_norm = etl.fit(dummy_df, return_stage="normalized")
    assert df_norm[["acf_lag_val", "p2m_val", "stl_strength"]].max().max() <= 1
    assert df_norm[["acf_lag_val", "p2m_val", "stl_strength"]].min().min() >= 0


def test_return_scores(dummy_df):
    etl = SeasonalityETL()
    df_scores = etl.fit(dummy_df, return_stage="scores")
    assert all(
        col in df_scores.columns
        for col in [
            "seasonality_score_linear",
            "seasonality_score_geom",
            "seasonality_score_harmonic",
        ]
    )


# ---------- Tests for Rolling Window Mode (fit_rolling) ----------


def test_fit_rolling_output_structure(dummy_df):
    etl = SeasonalityETL()
    df_rolling = etl.fit_rolling(dummy_df, frequencies=["W"])
    assert not df_rolling.empty
    assert {"window_start", "seasonality_score_linear"}.issubset(df_rolling.columns)


def test_fit_rolling_normalization(dummy_df):
    etl = SeasonalityETL(normalize=True)
    df_rolling = etl.fit_rolling(dummy_df, frequencies=["ME"])
    for col in ["acf_lag_val", "p2m_val", "stl_strength"]:
        assert df_rolling[col].min() >= 0
        assert df_rolling[col].max() <= 1


def test_fit_rolling_multiple_frequencies(dummy_df):
    etl = SeasonalityETL()
    df_rolling = etl.fit_rolling(dummy_df, frequencies=["W", "ME"])
    assert set(df_rolling["freq"].unique()) == {"W", "ME"}
