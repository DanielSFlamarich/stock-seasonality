import pandas as pd

from src.visualization.synthetic_data_generator import (
    generate_perfect_seasonality,
    generate_perfect_seasonality_all_intervals,
)


def test_generate_perfect_seasonality_output():
    df = generate_perfect_seasonality(freq="YE", seed=123)
    assert isinstance(df, pd.DataFrame)
    assert {"date", "close", "ticker"}.issubset(df.columns)
    assert pd.api.types.is_datetime64_any_dtype(df["date"])
    assert df["ticker"].nunique() == 1
    assert df["ticker"].iloc[0] == "PERFECT.SYN"


def test_generate_all_intervals_consistency():
    df = generate_perfect_seasonality_all_intervals(seed=123)
    assert isinstance(df, pd.DataFrame)
    assert {"date", "close", "ticker"}.issubset(df.columns)
    assert df["ticker"].nunique() == 1
    assert df["ticker"].unique()[0] == "PERFECT.SYN"
